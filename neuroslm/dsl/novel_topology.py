# -*- coding: utf-8 -*-
"""Novel-topology mechanisms (H15 / H16 / H19).

Three composable mechanisms designed to expose computational assets a
flat-transformer-at-scale cannot reproduce. Each defaults to OFF so the
baseline DSL cortex is bit-identical when none are enabled, and each is
toggled independently via `TrainingConfig` fields parsed from the DSL's
`training { ... }` block.

Hypotheses (see docs/findings.md):
  H15 — Episodic kNN at the cortex output (Complementary Learning
        Systems made architectural; Wu et al. Memorizing Transformers
        2022; Borgeaud et al. RETRO 2022).
  H16 — Multi-scale grid-cell positional bias (Fyhn et al. 2008; Sargolini
        et al. 2006). Provable length-OOD extrapolation via incommensurate
        sinusoidal scales.
  H19 — Surprise head + write gate (Mismatch Negativity / Lisman & Grace
        2005). Local-context LM head exposes per-token surprise; used as
        write-gate for H15 so only informative tokens get stored.

Composition. Surprise (H19) is a prerequisite for the "surprise" write
gate of episodic memory (H15). Grid positions (H16) is orthogonal and
operates on the residual stream pre-attention, so it does not interact
with the other two.

Init discipline. Every parameter that mutates the forward pass is
zero-init so the first forward AFTER attaching the mechanism reproduces
the baseline bit-identically. Behaviour only emerges as training pushes
those gates off zero — same discipline as ReZero / PCT-trunk elsewhere
in this codebase.

References:
  - Memorizing Transformers — https://arxiv.org/abs/2203.08913
  - RETRO — https://arxiv.org/abs/2112.04426
  - Grid cell coding — Sargolini et al. 2006 (Science)
  - MMN / hippocampal novelty gate — Lisman & Grace 2005 (Neuron)
"""
from __future__ import annotations

import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Config normalization helpers ──────────────────────────────────────

def _as_dict(spec) -> Dict[str, Any]:
    """Accept either `True` (defaults) or a dict; return a dict.

    Lets callers write either `grid_positions=True` (defaults) or
    `grid_positions={"n_scales": 6}` (override one field).
    """
    if spec is True:
        return {}
    if spec is False or spec is None:
        return {"enabled": False}
    if isinstance(spec, dict):
        return dict(spec)
    raise TypeError(f"expected bool|dict|None, got {type(spec).__name__}")


def _enabled(spec) -> bool:
    d = _as_dict(spec)
    return d.get("enabled", True) and (spec is not False and spec is not None)


# ──────────────────────────────────────────────────────────────────────
# H16 — Grid-cell positional bias
# ──────────────────────────────────────────────────────────────────────

class GridCellPositions(nn.Module):
    """Multi-scale sinusoidal grid-cell position bias.

    For K scales `τ_k = scale_ratio^k`, produces a per-position code
    of pairs `(cos(t/τ_k), sin(t/τ_k))`. Stacked across K scales gives
    `2K` raw features per position; a learnable `proj` maps `2K → d_model`.

    `proj` is zero-init so the first forward is identity-on-residual,
    preserving baseline parity. As `proj` learns, the position bias
    starts shaping the residual.

    Extrapolation guarantee: the code is purely a function of position
    `t` and the analytic scales `τ_k`, so it is defined for any `t >= 0`
    — including `t > max_ctx` seen during training. The flat baseline's
    learned positional embedding cannot do this without OOB lookup.
    """
    def __init__(self, d_model: int,
                 n_scales: int = 8,
                 scale_ratio: float = 1.6180339887498949,
                 base_period: float = 16.0):
        super().__init__()
        self.d_model = d_model
        self.n_scales = int(n_scales)
        self.scale_ratio = float(scale_ratio)
        self.base_period = float(base_period)
        # 2 features per scale (cos, sin) → 2K raw features
        self.proj = nn.Linear(2 * self.n_scales, d_model, bias=False)
        # Zero-init: residual identity start. Behaviour emerges as proj
        # learns to read from the multi-scale code.
        nn.init.zeros_(self.proj.weight)

    def forward(self, seq_len: int, device=None, dtype=None) -> torch.Tensor:
        """Compute per-position residual bias of shape (seq_len, d_model).

        The bias is added to the embedding before the first block, so
        the cortex's attention sees position-aware tokens. Cheap: O(L·K).
        """
        if device is None:
            device = self.proj.weight.device
        if dtype is None:
            dtype = self.proj.weight.dtype
        t = torch.arange(seq_len, device=device, dtype=dtype)        # (L,)
        # Periods τ_k = base_period * scale_ratio^k. Use 2π so the
        # raw arg is `2π t / τ_k`.
        ks = torch.arange(self.n_scales, device=device, dtype=dtype)
        taus = self.base_period * (self.scale_ratio ** ks)            # (K,)
        # (L, K) angles in radians
        ang = 2.0 * math.pi * t.unsqueeze(1) / taus.unsqueeze(0)
        # (L, 2K) [cos, sin] interleaved-by-scale
        code = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)
        return self.proj(code)                                        # (L, d_model)


# ──────────────────────────────────────────────────────────────────────
# H15 — Episodic kNN memory
# ──────────────────────────────────────────────────────────────────────

class EpisodicMemory(nn.Module):
    """Non-parametric circular buffer of (key, value) pairs.

    Writes (training only): every `T` token receives its hidden state as
    the value, projected to the key space via `key_proj`. By default
    ALL tokens are written each step until the buffer fills, then the
    write pointer wraps. When `write_gate="surprise"`, only the top
    `(1 - write_quantile)` fraction of surprising tokens (per H19) are
    written.

    Reads (always): cosine-similarity retrieval of top-k keys; values
    are blended into the residual via a ReZero-gated projection
    (`alpha` scalar, init 0 → first-forward identity).

    Gradient policy. The buffer (`_keys`, `_values`) is registered as a
    non-persistent buffer (NOT a parameter) and is updated with `.detach()`
    so no LM gradient flows into stored episodes. The buffer's parameters
    `key_proj`, `value_proj`, `alpha` are trainable.
    """
    def __init__(self, d_model: int,
                 slots: int = 4096,
                 k: int = 32,
                 alpha_init: float = 0.0,
                 write_gate: str = "all",
                 write_quantile: float = 0.8,
                 key_dim: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.slots = int(slots)
        self.k = int(k)
        self.write_gate = write_gate
        self.write_quantile = float(write_quantile)
        self.key_dim = int(key_dim) if key_dim is not None else d_model

        # Projections — trainable. Value path is zero-init so blended
        # value contributes nothing at step 0 (alpha=0 also enforces
        # this; we keep both belt-and-suspenders).
        self.key_proj = nn.Linear(d_model, self.key_dim, bias=False)
        nn.init.normal_(self.key_proj.weight, std=0.02)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.value_proj.weight)
        # ReZero gate
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        # Non-parametric buffers (not in .parameters()). Registered as
        # buffers so .to(device) / .state_dict() handles them, but with
        # persistent=False (no save) and requires_grad=False (no leaf).
        self.register_buffer("_keys",
                             torch.zeros(self.slots, self.key_dim),
                             persistent=False)
        self.register_buffer("_values",
                             torch.zeros(self.slots, d_model),
                             persistent=False)
        # Plain Python state (not a buffer): write head, fill count.
        self._write_head: int = 0
        self._n_written: int = 0
        # Diagnostic: last retrieval (B, T, D)
        self.last_retrieved: Optional[torch.Tensor] = None

    def size(self) -> int:
        return self._n_written

    def _write(self, hidden: torch.Tensor,
               surprise: Optional[torch.Tensor] = None) -> None:
        """Detach + project + push (B, T, D) into the circular buffer.

        If `surprise` is provided and write_gate=="surprise", only the
        top `(1 - write_quantile)` surprising tokens are written.
        """
        with torch.no_grad():
            B, T, D = hidden.shape
            flat = hidden.detach().reshape(B * T, D)
            if self.write_gate == "surprise" and surprise is not None:
                s_flat = surprise.detach().reshape(B * T)
                # Keep tokens with surprise above the quantile threshold
                q = torch.quantile(s_flat, self.write_quantile)
                mask = s_flat > q
                if mask.sum().item() == 0:
                    return
                flat = flat[mask]
            n_new = flat.shape[0]
            if n_new == 0:
                return
            keys = self.key_proj(flat).detach()
            # Circular write — wrap-around via modulo arithmetic.
            idx = (torch.arange(n_new, device=flat.device) + self._write_head) \
                  % self.slots
            self._keys[idx] = keys.to(self._keys.dtype)
            self._values[idx] = flat.to(self._values.dtype)
            self._write_head = int((self._write_head + n_new) % self.slots)
            self._n_written = int(min(self._n_written + n_new, self.slots))

    def _read(self, hidden: torch.Tensor) -> torch.Tensor:
        """Cosine-sim top-k retrieval. Returns blended value tensor (B,T,D).

        When the buffer is empty, returns zeros (gate would zero this
        out anyway, but explicit avoids spurious NaN from cosine on
        zero rows).
        """
        B, T, D = hidden.shape
        if self._n_written == 0:
            return torch.zeros_like(hidden)
        q = self.key_proj(hidden).reshape(B * T, self.key_dim)            # (N, K)
        # Only search the populated slots
        keys = self._keys[: self._n_written]                              # (M, K)
        vals = self._values[: self._n_written]                            # (M, D)
        # Cosine sim
        q_n = F.normalize(q, dim=-1, eps=1e-6)
        k_n = F.normalize(keys, dim=-1, eps=1e-6)
        sims = q_n @ k_n.T                                                 # (N, M)
        topk_v, topk_i = sims.topk(min(self.k, self._n_written), dim=-1)   # (N, k)
        # Soft attention over top-k via softmax-weighted value blend
        attn = F.softmax(topk_v, dim=-1)                                   # (N, k)
        gathered = vals[topk_i]                                            # (N, k, D)
        blended = (attn.unsqueeze(-1) * gathered).sum(dim=1)               # (N, D)
        out = self.value_proj(blended).reshape(B, T, D)
        return out

    def forward(self, hidden: torch.Tensor,
                surprise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Read-then-write. Returns the residual delta to add to hidden.

        Read uses the current buffer (so retrieval reflects PRIOR steps
        only); the write then appends the current step. Order matters:
        reading after writing would let the model trivially memorise
        the current token.
        """
        delta = self._read(hidden) * self.alpha
        self.last_retrieved = delta
        if self.training:
            self._write(hidden, surprise=surprise)
        return delta


# ──────────────────────────────────────────────────────────────────────
# H19 — Surprise head
# ──────────────────────────────────────────────────────────────────────

class SurpriseHead(nn.Module):
    """Local-context LM head that exposes per-token surprise.

    Operates on the trunk's final hidden state. A tiny 1-D causal conv
    (kernel `local_window`) sees only the last `local_window` tokens.
    Its output is projected to vocab; cross-entropy against the labels
    gives the "local NLL". The trunk's logits give the "global NLL".

        surprise[t] = nll_local[t] - nll_global[t]

    In expectation `surprise > 0` because the local head is weaker. We
    return surprise per-token (B, T). The harness can then use it for
    loss reweighting or episodic write-gating.

    The labels are accessed via a setter `set_labels(ids_shifted)` so
    the forward signature of `DSLLanguageCortex.forward(ids)` stays
    backward-compatible.
    """
    def __init__(self, d_model: int, vocab: int,
                 dim: int = 128, local_window: int = 64):
        super().__init__()
        self.local_window = int(local_window)
        self.dim = int(dim)
        # 1-D causal conv (kernel = local_window) — only sees the past.
        self.local_conv = nn.Conv1d(
            in_channels=d_model, out_channels=dim,
            kernel_size=self.local_window, padding=0, bias=True)
        nn.init.normal_(self.local_conv.weight, std=0.02)
        nn.init.zeros_(self.local_conv.bias)
        self.head = nn.Linear(dim, vocab, bias=False)
        nn.init.normal_(self.head.weight, std=0.02)

        self._labels: Optional[torch.Tensor] = None

    def set_labels(self, labels: Optional[torch.Tensor]) -> None:
        """Install the next-token labels (B, T) used to compute the
        local NLL on the next forward. Pass None to disable surprise
        for the next step (eval / OOD passes don't always have labels)."""
        self._labels = labels

    def forward(self, hidden: torch.Tensor,
                global_logits: torch.Tensor) -> Optional[torch.Tensor]:
        """Returns per-token surprise (B, T) or None if no labels set."""
        if self._labels is None:
            return None
        B, T, D = hidden.shape
        # Causal conv: left-pad with (local_window - 1) zeros so output
        # has the same T dimension and position t sees only [0..t].
        x = hidden.transpose(1, 2)                                # (B, D, T)
        x = F.pad(x, (self.local_window - 1, 0))                  # (B, D, T+w-1)
        z = self.local_conv(x)                                    # (B, dim, T)
        z = F.silu(z).transpose(1, 2)                             # (B, T, dim)
        local_logits = self.head(z)                               # (B, T, V)
        # NLL per token. self._labels is (B, T) of int64 next-token ids.
        nll_local = F.cross_entropy(
            local_logits.reshape(B * T, -1),
            self._labels.reshape(B * T), reduction="none").reshape(B, T)
        # nll_global: chunked + no_grad to avoid a single (B*T, V) fp32
        # softmax alloc (~6 GiB at B=16, T=2048, V=50257 on CUDA).
        _CHUNK = 512
        labels_flat = self._labels.reshape(B * T)
        global_flat = global_logits.detach().reshape(B * T, -1)
        with torch.no_grad():
            nll_global = torch.cat([
                F.cross_entropy(global_flat[i:i + _CHUNK],
                                labels_flat[i:i + _CHUNK], reduction="none")
                for i in range(0, B * T, _CHUNK)
            ]).reshape(B, T)
        return (nll_local - nll_global).detach()


# ──────────────────────────────────────────────────────────────────────
# Factories — turn DSL spec into module instance (or None when off)
# ──────────────────────────────────────────────────────────────────────

def make_grid_positions(spec, d_model: int) -> Optional[GridCellPositions]:
    if not _enabled(spec):
        return None
    d = _as_dict(spec)
    return GridCellPositions(
        d_model=d_model,
        n_scales=int(d.get("n_scales", 8)),
        scale_ratio=float(d.get("scale_ratio", 1.6180339887498949)),
        base_period=float(d.get("base_period", 16.0)),
    )


def make_episodic_memory(spec, d_model: int) -> Optional[EpisodicMemory]:
    if not _enabled(spec):
        return None
    d = _as_dict(spec)
    return EpisodicMemory(
        d_model=d_model,
        slots=int(d.get("slots", 4096)),
        k=int(d.get("k", 32)),
        alpha_init=float(d.get("alpha_init", 0.0)),
        write_gate=str(d.get("write_gate", "all")),
        write_quantile=float(d.get("write_quantile", 0.8)),
        key_dim=d.get("key_dim", None),
    )


def make_surprise_head(spec, d_model: int, vocab: int) -> Optional[SurpriseHead]:
    if not _enabled(spec):
        return None
    d = _as_dict(spec)
    return SurpriseHead(
        d_model=d_model, vocab=vocab,
        dim=int(d.get("dim", 128)),
        local_window=int(d.get("local_window", 64)),
    )


def make_nfo(spec, d_model: int):
    """Build a Neural Field Oscillator (H015..H018) from a DSL spec.

    Imported lazily so the existing ``novel_topology`` import surface
    does not gain a hard dependency on the NFO module. Returns ``None``
    when disabled, exactly like the H15/H16/H19 factories above.
    """
    if not _enabled(spec):
        return None
    # Lazy import — keeps the legacy `novel_topology` test cost flat
    # when NFO is not configured.
    from neuroslm.modules.neural_field_oscillator import make_nfo as _make
    return _make(spec, d_model)

