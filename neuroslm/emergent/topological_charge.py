# -*- coding: utf-8 -*-
"""C4 — Topological charge Q(t) along the token axis.

The core of RCC-Bowtie v2. We treat the per-token hidden states of the
trunk as a unit-norm vector field h_t/‖h_t‖ ∈ S^{d-1}. A fixed
seed-stable skew-symmetric R ∈ so(d) gives an in-plane / out-of-plane
decomposition of every adjacent pair (h_t, h_{t+1}); the running
winding number of (Re,Im) = (⟨h_t,h_{t+1}⟩, ⟨h_t,Rh_{t+1}⟩) is the
**discrete topological charge**:

    Q_total = (1/2π) · Σ_t arg( ⟨h_t,h_{t+1}⟩ + i·⟨h_t,Rh_{t+1}⟩ )

This is the sequence analog of the magnetic skyrmion charge:

- Plateaus of Q correspond to discourse-coherent spans.
- Sign flips of ⟨h_t,h_{t+1}⟩ are *domain walls* — semantic boundaries.
- Q cannot be smoothly destroyed by GD without crossing a wall, so it
  provides discrete perturbation-robust long-range structure.

Observation-only: Q is computed from the trunk's existing residual
stream (no parameter cost, no forward-pass change). The prediction we
test in Phase 8+ is that Q-walls align with paragraph boundaries on a
WikiText-103 eval prompt — if so, an auxiliary loss that rewards
non-trivial Q (clamped to avoid blow-up) is justified.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional

import torch


def _seed_skew(dim: int, seed: int = 1729) -> torch.Tensor:
    """Generate a deterministic skew-symmetric matrix R ∈ so(d).

    R = A - A.T with A drawn from N(0,1) under a fixed generator;
    normalised so the operator norm is 1 (otherwise the "imaginary
    part" of the winding swamps the real part on wide models).
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    A = torch.randn(dim, dim, generator=g, dtype=torch.float32)
    R = A - A.T
    # Spectral norm of a skew-symmetric matrix = max |imag-eigenvalue|.
    # Use the Frobenius norm / √(2d) as a cheap proxy normaliser.
    fn = R.pow(2).sum().sqrt().item()
    if fn > 1e-8:
        R = R / fn * math.sqrt(2.0 * dim)   # operator scale ≈ √(2d)
    return R


class TopologicalChargeProbe:
    """Per-batch winding-number probe over a sequence of hidden states.

    Use one probe per layer-readout point you want to monitor (typically
    just the final block's output, where the residual stream is the
    deepest). Stateless across calls — each `step()` consumes a fresh
    (B, T, D) tensor and returns the per-batch mean of Q-related stats.

    Parameters
    ----------
    dim : int
        Last-axis dimensionality of the activations being analysed.
    seed : int
        RNG seed for the fixed skew operator R.
    """

    def __init__(self, dim: int, seed: int = 1729):
        if dim <= 1:
            raise ValueError("dim must be >= 2 for skew-symmetric R")
        self.dim = int(dim)
        self.seed = int(seed)
        self._R = _seed_skew(self.dim, self.seed)
        # Stats accumulator
        self._last: Dict[str, float] = {
            "Q_total":       0.0,
            "Q_walls":       0.0,
            "Q_plateau_len": 0.0,
            "Q_abs":         0.0,
        }

    # ── Core math ────────────────────────────────────────────────────

    def _winding(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute per-token complex coordinate and wall mask.

        h shape: (B, T, D).
        Returns dict with:
            re   (B, T-1)   = ⟨ĥ_t, ĥ_{t+1}⟩
            im   (B, T-1)   = ⟨ĥ_t, R ĥ_{t+1}⟩
            wall (B, T-1)   = boolean mask where re < 0
            dphi (B, T-1)   = atan2(im, re) — per-step phase increment
            Q    (B,)       = (1/2π) Σ dphi
        """
        B, T, D = h.shape
        if D != self.dim:
            raise ValueError(
                f"TopologicalChargeProbe(dim={self.dim}) got tensor with "
                f"D={D}"
            )
        if T < 2:
            zero = torch.zeros(B, max(0, T - 1), device=h.device)
            return {
                "re": zero, "im": zero,
                "wall": torch.zeros_like(zero, dtype=torch.bool),
                "dphi": zero, "Q": torch.zeros(B, device=h.device),
            }

        hf = h.float()
        # Normalise to the unit sphere.
        norm = hf.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        hn = hf / norm

        h_t = hn[:, :-1, :]
        h_tp = hn[:, 1:, :]
        R = self._R.to(hn.device, dtype=hn.dtype)
        R_h_tp = h_tp @ R.T           # apply R on the right (since R is sk-sym, ok)

        re = (h_t * h_tp).sum(dim=-1)           # (B, T-1)
        im = (h_t * R_h_tp).sum(dim=-1)         # (B, T-1)
        dphi = torch.atan2(im, re)
        Q = dphi.sum(dim=-1) / (2.0 * math.pi)
        wall = re < 0
        return {"re": re, "im": im, "wall": wall, "dphi": dphi, "Q": Q}

    # ── Step ─────────────────────────────────────────────────────────

    def step(self, h: Optional[torch.Tensor]) -> Dict[str, float]:
        """Consume one (B, T, D) tensor; return scalar stats.

        Stats are batch-means, except `Q_walls` which is the per-batch
        mean wall count (so it's interpretable as "walls per sequence").
        """
        if h is None or h.numel() == 0:
            return dict(self._last)
        with torch.no_grad():
            w = self._winding(h.detach())
        # Q: signed sum, then per-batch mean.
        Q_total = float(w["Q"].mean().item())
        Q_abs = float(w["Q"].abs().mean().item())
        walls_per_seq = w["wall"].float().sum(dim=-1)        # (B,)
        Q_walls = float(walls_per_seq.mean().item())
        # Plateau length: mean run-length between walls.
        # walls per seq + 1 = plateau count (intervals between walls,
        # plus the leading run before the first wall).
        T_minus_1 = float(w["wall"].shape[-1])
        if T_minus_1 > 0:
            plateau_len = float(
                (T_minus_1 / (walls_per_seq + 1.0)).mean().item()
            )
        else:
            plateau_len = 0.0

        self._last = {
            "Q_total":       Q_total,
            "Q_walls":       Q_walls,
            "Q_plateau_len": plateau_len,
            "Q_abs":         Q_abs,
        }
        return dict(self._last)

    # ── Diagnostics for tests ────────────────────────────────────────

    def raw_winding(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Expose the winding tensors directly (for tests / analysis)."""
        with torch.no_grad():
            return self._winding(h.detach())
