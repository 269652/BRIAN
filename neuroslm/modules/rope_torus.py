"""RoPE-on-a-Torus: rotary positional encoding on T^n = (S^1)^n.

Classical RoPE (Su et al. 2021, "RoFormer") rotates each 2D slice of the
embedding by an angle proportional to position; the frequencies form a
geometric ladder, which yields a relative-position bias under the dot
product:

    <RoPE(q, m), RoPE(k, n)> = <q, R_{n-m} k>

This module generalises that to a **multi-frequency torus**: every 2D
slice gets its OWN base period drawn from a configurable schedule, and
positions are taken modulo each slice's period independently. The
geometry is the product manifold T^n = S^1 × S^1 × … × S^1.

Why a torus? Sequence position is naturally cyclic at multiple scales
(token, sentence, paragraph, document). Classical RoPE bakes a single
exponential-decay frequency schedule into the rotation; the torus
formulation lets the schedule be data-driven (e.g. a learnable period
per slice) and makes the wrap-around explicit instead of relying on the
exponential's vanishing tail.

References
~~~~~~~~~~
* Su, Lu, Pan, Murtadha, Wen, Liu — "RoFormer: Enhanced Transformer with
  Rotary Position Embedding", *Neurocomputing* 568 (2024).
* "Manifold positional encoding" — the product-of-circles construction
  is folklore; the closest published treatment is Bronstein et al.,
  *Geometric Deep Learning* (2021), §5.4.

Implementation: tested by ``tests/test_rope_torus.py`` (15 contracts).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# Tiny epsilon for numerical safety in periods (avoid zero division).
_PERIOD_EPS = 1e-6


def build_torus_periods(
    n_pairs: int,
    *,
    base: float = 10000.0,
    schedule: str = "geometric",
) -> torch.Tensor:
    """Return the per-slice period vector of length ``n_pairs``.

    Each entry is the angular period (in tokens) of one 2D rotation
    slice. Schedules:

    ``"geometric"`` (classical RoPE):
        ``period_i = 2π · base^(2i / d)`` for ``i = 0..n_pairs-1`` where
        ``d = 2·n_pairs``. The shortest period sits at slice 0.

    ``"linear"``:
        ``period_i = (i+1) · base / n_pairs`` — uniformly spaced periods,
        useful for ablations.

    ``"harmonic"``:
        ``period_i = base / (i+1)`` — 1/k spacing, denser at short
        periods.

    Args:
        n_pairs: number of 2D rotation slices (= d_model / 2).
        base: scale parameter; classical RoPE uses 10000.
        schedule: ``"geometric"`` | ``"linear"`` | ``"harmonic"``.
    """
    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be positive, got {n_pairs}")
    if schedule == "geometric":
        d = 2.0 * n_pairs
        i = torch.arange(n_pairs, dtype=torch.float32)
        periods = 2.0 * math.pi * (base ** (2.0 * i / d))
    elif schedule == "linear":
        i = torch.arange(1, n_pairs + 1, dtype=torch.float32)
        periods = i * (base / float(n_pairs))
    elif schedule == "harmonic":
        i = torch.arange(1, n_pairs + 1, dtype=torch.float32)
        periods = base / i
    else:
        raise ValueError(
            f"unknown schedule {schedule!r}; "
            f"expected geometric|linear|harmonic"
        )
    # Floor periods so the cos/sin tables are well-defined.
    return periods.clamp_min(_PERIOD_EPS)


def build_torus_cos_sin(
    seq_len: int,
    periods: torch.Tensor,
    *,
    device=None,
    dtype=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for every (position, slice) pair.

    Returns ``(cos, sin)`` each shaped ``(seq_len, n_pairs)``. The
    angle of slice ``j`` at position ``p`` is

        θ_{p,j} = 2π · (p mod period_j) / period_j

    which is the canonical S^1 coordinate. The ``mod`` ensures the
    angle wraps inside each slice's period — that's the "torus" part.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    # (T, 1) / (1, n_pairs) → (T, n_pairs) with broadcasting.
    p = periods.to(device=device, dtype=torch.float32)
    # Modular reduction keeps the angle in [0, 2π) for every slice.
    # We use fmod (which preserves sign) on positives, equivalent to %
    # but autograd-friendly.
    phase = (pos.unsqueeze(-1).fmod(p.unsqueeze(0))) / p.unsqueeze(0)
    theta = 2.0 * math.pi * phase
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    if dtype is not None:
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
    return cos, sin


def apply_rope_torus(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate every 2D slice of ``x`` by the per-position angle.

    The rotation matrix at position ``p``, slice ``j`` is

        R = [[cos θ, -sin θ],
             [sin θ,  cos θ]]

    applied to ``(x_{2j}, x_{2j+1})``. We use the "interleaved-pair"
    convention (``[a, b, c, d, …]`` → pairs ``(a,b), (c,d), …``) which
    matches the RoFormer reference implementation.

    Args:
        x:   ``(B, T, D)`` or ``(B, H, T, D)`` where ``D`` is even.
        cos: ``(T, D/2)`` cosine table from :func:`build_torus_cos_sin`.
        sin: ``(T, D/2)`` sine   table from :func:`build_torus_cos_sin`.

    Returns:
        A tensor with the same shape and dtype as ``x``.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            f"last dim must be even (got {x.shape[-1]}); RoPE rotates pairs"
        )
    T = x.shape[-2]
    if cos.shape[0] != T or sin.shape[0] != T:
        raise ValueError(
            f"cos/sin tables have seq_len={cos.shape[0]} but x has "
            f"seq_len={T}; rebuild with the right seq_len"
        )
    # Reshape last dim into (n_pairs, 2): (..., T, D) → (..., T, n_pairs, 2)
    n_pairs = x.shape[-1] // 2
    x_pairs = x.reshape(*x.shape[:-1], n_pairs, 2)
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]

    # Broadcast cos/sin from (T, n_pairs) to match x_pairs[..., 0]/[..., 1].
    # We need shape ending in (T, n_pairs); insert head dims as needed.
    while cos.dim() < x0.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)

    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos

    y = torch.stack([y0, y1], dim=-1)
    return y.reshape(*x.shape)


class RoPETorus(nn.Module):
    """Module wrapper for the torus-rotated positional encoding.

    Acts as an ``edge`` endpoint in the BRIAN feature DSL: its forward
    consumes ``(B, T, D)`` and returns ``(B, T, D)`` with each token's
    embedding rotated according to its position. The cos/sin tables
    are registered as buffers (no parameters → no parameter count
    bump), with an option to make the periods learnable.

    Args:
        d_model: model embedding dimension (must be even).
        max_seq_len: longest sequence the tables will be precomputed
            for. Sequences shorter than this just index a prefix.
        base: RoPE base scale (default 10000, matches RoFormer).
        schedule: ``"geometric"`` | ``"linear"`` | ``"harmonic"``.
        learnable_periods: if ``True``, expose the period vector as an
            ``nn.Parameter`` so the optimiser can shape the schedule.
            The tables are then recomputed every forward (cheap: a
            single ``cos`` + ``sin`` call on ``(T, n_pairs)`` floats).
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 2048,
        *,
        base: float = 10000.0,
        schedule: str = "geometric",
        learnable_periods: bool = False,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even (got {d_model})")
        self.d_model = d_model
        self.n_pairs = d_model // 2
        self.max_seq_len = max_seq_len
        self.base = float(base)
        self.schedule = schedule
        self.learnable_periods = bool(learnable_periods)

        periods = build_torus_periods(
            self.n_pairs, base=self.base, schedule=self.schedule
        )
        if self.learnable_periods:
            # Optimise log-periods to keep them positive without a clamp.
            self.log_periods = nn.Parameter(torch.log(periods))
            # No precomputed table — rebuilt every forward.
        else:
            self.register_buffer("periods", periods, persistent=False)
            cos, sin = build_torus_cos_sin(max_seq_len, periods)
            self.register_buffer("cos_cached", cos, persistent=False)
            self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-2]
        if self.learnable_periods:
            periods = torch.exp(self.log_periods).clamp_min(_PERIOD_EPS)
            cos, sin = build_torus_cos_sin(
                T, periods, device=x.device, dtype=x.dtype
            )
        else:
            if T > self.max_seq_len:
                # Re-extend on the fly if the call asks for more than
                # we cached. Cheap and removes a footgun.
                periods = self.periods  # type: ignore[has-type]
                cos, sin = build_torus_cos_sin(
                    T, periods, device=x.device, dtype=x.dtype
                )
            else:
                cos = self.cos_cached[:T]  # type: ignore[has-type]
                sin = self.sin_cached[:T]  # type: ignore[has-type]
        return apply_rope_torus(x, cos, sin)
