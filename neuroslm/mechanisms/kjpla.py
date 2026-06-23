# -*- coding: utf-8 -*-
"""KJPLA-v2: Kuramoto-Josephson Phase Lattice Attention (Phase 3, THSD program).

Each attention head carries a per-(head, layer, token) phase φ that evolves
via two coupled dynamics:

  Intra-layer  Kuramoto sync:
      φ₁_h = φ₀_h + η · (1/H) Σ_j sin(φ₀_j − φ₀_h)

  Inter-layer  Josephson coupling (consumed by harness):
      L_J = − (1/L) Σ_ℓ K̄_h · R_ℓ
      R_ℓ = |⟨e^{i(φ_ℓ − φ_{ℓ−1} − Δ_h)}⟩_t|   (order parameter)

The phase-gated attention logit replaces the vanilla (q·k)/√d:
      logit_h(t,s) = (q_t · k_s) / √d  +  β_h · cos(φ₁_h(t) − φ₁_h(s))

All phase scalars (η, β_h, K_h) are Parameters initialised to **zero**
(ReZero convention from brain.py:147-149).  At zero init the mechanism is
structurally inert: φ₀ = 0 everywhere (since w_h = 0), all cos(Δφ) = 1 but
β_h = 0, so the logit collapses back to the vanilla (q·k)/√d path and the
forward pass is bit-identical to CausalSelfAttention's manual-Hebbian branch
(forced by KJPLA always using the manual-softmax code path).

CLAUDE.md §14 contracts:
  - Zero-init test: torch.equal(kjpla_output, vanilla_output) at init
  - phi stash is bfloat16 (memory discipline — closes review FIX 9)
  - delta_h is a non-persistent buffer (not saved in state_dict)
  - Josephson order param R = 1 when phi_ℓ − phi_{ℓ−1} ≡ Δ_h exactly
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.modules.common import (
    NeuromodulatedScale,
    HebbianTrace,
    build_rope_cache,
    apply_rope,
)


class KJPLAttention(nn.Module):
    """Phase-lattice attention head with Kuramoto sync + Josephson coupling.

    Drop-in replacement for CausalSelfAttention.  Matches the q_proj + fused
    kv_proj layout so state-dicts are compatible.  Matches F.normalize pre-RoPE.

    Args:
        dim: total residual dimension.
        n_heads: number of query heads.
        max_ctx: maximum sequence length (for RoPE cache).
        n_kv_heads: number of KV heads (GQA).  Defaults to n_heads (MHA).
        n_nt: neuromodulator count.
        hebbian_rank: Hebbian fast-weight rank (0 = disabled).
        dropout: residual dropout.
        layer_index: 0-based index of this layer in the stack.
        n_layers: total number of layers (for deterministic delta_h).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        max_ctx: int,
        n_kv_heads: Optional[int] = None,
        n_nt: int = 0,
        hebbian_rank: int = 0,
        dropout: float = 0.0,
        layer_index: int = 0,
        n_layers: int = 1,
    ):
        super().__init__()
        assert dim % n_heads == 0, f"dim={dim} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0
        self.n_groups = n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads
        self.dropout = float(dropout)
        self.resid_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ── Same projections as CausalSelfAttention (state-dict compat) ──
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        cos, sin = build_rope_cache(max_ctx, self.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.nt_scale = NeuromodulatedScale(n_nt, n_heads) if n_nt > 0 else None
        self.hebbian = HebbianTrace(self.head_dim, rank=hebbian_rank) if hebbian_rank > 0 else None

        # ── KJPLA phase machinery ────────────────────────────────────────
        # w_h: content carrier for φ₀. Init zero → φ₀ = 0 → inert gate.
        self.w_h = nn.Parameter(torch.zeros(n_heads, self.head_dim))

        # delta_h: target inter-layer phase stride. Non-persistent so it is
        # reconstructed on load (deterministic from n_heads * n_layers).
        delta = torch.tensor(
            [2.0 * math.pi * h / (n_heads * max(1, n_layers))
             for h in range(n_heads)],
            dtype=torch.float32,
        )
        self.register_buffer("delta_h", delta, persistent=False)

        # ReZero-style: all init 0 so at step 0 the mechanism is structurally
        # off and the training loss starts identical to vanilla.
        self.eta = nn.Parameter(torch.zeros(1))          # Kuramoto coupling
        self.beta_h = nn.Parameter(torch.zeros(n_heads)) # phase logit scale
        self.K_h = nn.Parameter(torch.zeros(n_heads))    # Josephson coupling

        self.layer_index = int(layer_index)
        self.n_layers = int(n_layers)
        self._dim = int(dim)

    # ── Phase sub-routines ────────────────────────────────────────────────

    def _phi0(self, q: torch.Tensor) -> torch.Tensor:
        """Content-carrier phase: φ₀[b,h,t] = Σ_d w_h[d] · q[b,h,t,d]."""
        # q: (B, H, T, head_dim) — already normalize + RoPE applied
        return torch.einsum("bhtd,hd->bht", q, self.w_h)

    def _kuramoto_step(self, phi0: torch.Tensor) -> torch.Tensor:
        """Intra-layer Kuramoto sync.
        φ₁[b,h,t] = φ₀[b,h,t] + η · mean_j sin(φ₀[b,j,t] − φ₀[b,h,t])
        """
        # phi0: (B, H, T)
        # diff[b, j, h, t] = phi0_j - phi0_h
        diff = phi0.unsqueeze(2) - phi0.unsqueeze(1)   # (B, H, H, T)
        sync = self.eta * torch.sin(diff).mean(dim=1)  # (B, H, T)
        return phi0 + sync

    def _phase_gated_logits(
        self, phi1: torch.Tensor, qk_scaled: torch.Tensor
    ) -> torch.Tensor:
        """Add phase bias: logit[b,h,t,s] += β_h · cos(φ₁[b,h,t] − φ₁[b,h,s])."""
        delta_phi = phi1.unsqueeze(-1) - phi1.unsqueeze(-2)  # (B, H, T, T)
        beta = self.beta_h.view(1, -1, 1, 1)
        return qk_scaled + beta * torch.cos(delta_phi)

    def _josephson_order_param(
        self, phi1: torch.Tensor, phi_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Josephson inter-layer order parameter R and loss contribution.

        R_ℓ[h] = |⟨e^{i(φ₁ − φ_prev − Δ_h)}⟩_t|   (B, H) → scalar mean.
        L contribution = −K̄_h · R_mean   (returned; summed across layers
        in josephson_loss() below, or in _kjpla_aux_step in the harness).
        """
        # phi1, phi_prev: (B, H, T) — phi_prev in bfloat16, cast to float
        delta_h = self.delta_h.view(1, -1, 1).to(phi1.dtype)
        phase_diff = phi1 - phi_prev.to(phi1.dtype).detach() - delta_h
        cos_m = torch.cos(phase_diff).mean(dim=-1)  # (B, H)
        sin_m = torch.sin(phase_diff).mean(dim=-1)  # (B, H)
        R = (cos_m ** 2 + sin_m ** 2).sqrt()        # (B, H)
        R_mean = R.mean()
        K_mean = self.K_h.mean()
        loss = -K_mean * R_mean
        return R_mean.detach(), loss

    # ── Inert-gate check ──────────────────────────────────────────────────

    def _is_inert(self) -> bool:
        """True iff all phase parameters are zero → vanilla path."""
        return (
            self.w_h.abs().max().item() == 0.0
            and self.eta.item() == 0.0
            and self.beta_h.abs().max().item() == 0.0
            and self.K_h.abs().max().item() == 0.0
        )

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        nt: Optional[torch.Tensor] = None,
        phi_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Args:
            x:        (B, T, dim) residual stream.
            nt:       (B, n_nt) neuromodulator signal (optional).
            phi_prev: (B, H, T) phase from previous layer, bfloat16 (optional).

        Returns:
            y       : (B, T, dim) attention output.
            phi1_bf16: (B, H, T) phase stash in bfloat16 for next layer.
            aux     : dict with optional "josephson_loss" / "R_mean".
        """
        B, T, C = x.shape

        # ── Projections (identical to CausalSelfAttention) ────────────
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        if self.cos.size(0) < T:
            cos, sin = build_rope_cache(T, self.head_dim, device=x.device, dtype=x.dtype)
        else:
            cos, sin = self.cos.to(x.dtype), self.sin.to(x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_groups > 1:
            k = k[:, :, None, :, :].expand(
                -1, -1, self.n_groups, -1, -1
            ).reshape(B, self.n_heads, T, self.head_dim)
            v = v[:, :, None, :, :].expand(
                -1, -1, self.n_groups, -1, -1
            ).reshape(B, self.n_heads, T, self.head_dim)

        if self.nt_scale is not None and nt is not None:
            q = q * self.nt_scale(nt)

        # ── Causal mask (always manual — needed to inject phase logits) ──
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        # ── Inert gate (bit-identity with vanilla manual-Hebbian branch) ──
        if self._is_inert():
            if self.hebbian is not None:
                attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
                attn = attn + self.hebbian(q, k)
            else:
                attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn = attn.masked_fill(causal_mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            if self.dropout > 0 and self.training:
                attn = F.dropout(attn, p=self.dropout)
            y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
            return self.resid_drop(self.out(y)), None, {}

        # ── Phase computation ─────────────────────────────────────────
        phi0 = self._phi0(q)         # (B, H, T)
        phi1 = self._kuramoto_step(phi0)  # (B, H, T)

        # ── Josephson cross-layer coupling ────────────────────────────
        aux: dict = {}
        if phi_prev is not None and self.K_h.abs().max().item() != 0.0:
            R_mean, j_loss = self._josephson_order_param(phi1, phi_prev)
            aux["josephson_loss"] = j_loss
            aux["R_mean"] = R_mean

        # ── Phase-gated attention logits ──────────────────────────────
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = self._phase_gated_logits(phi1, attn)

        if self.hebbian is not None:
            attn = attn + self.hebbian(q, k)

        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)

        # Stash phi in bfloat16 (memory discipline, closes FIX 9).
        phi1_bf16 = phi1.detach().to(torch.bfloat16)

        return self.resid_drop(self.out(y)), phi1_bf16, aux


# ── Standalone Josephson loss function ───────────────────────────────────


def josephson_loss(
    phi_list: list,
    K_h_list: list,
    delta_h_list: list,
) -> torch.Tensor:
    """Josephson inter-layer coupling loss across a sequence of phases.

    L_J = -(1/L) Σ_{ℓ=1}^{L} mean_h(K_h[ℓ]) · R_ℓ

    Args:
        phi_list:    List[Tensor(B, H, T) bfloat16] — one per layer.
        K_h_list:    List[Tensor(H)] — K_h Parameter per layer.
        delta_h_list: List[Tensor(H)] — delta_h buffer per layer.

    Returns:
        Scalar loss. Zero if fewer than 2 phases.
    """
    if len(phi_list) < 2:
        device = phi_list[0].device if phi_list else torch.device("cpu")
        return torch.zeros((), device=device)

    total = torch.zeros((), device=K_h_list[0].device)
    n_pairs = 0
    for i in range(1, len(phi_list)):
        phi_prev = phi_list[i - 1].float()
        phi_curr = phi_list[i].float()
        delta_h = delta_h_list[i].view(1, -1, 1)
        K_h = K_h_list[i]

        phase_diff = phi_curr - phi_prev.detach() - delta_h
        cos_m = torch.cos(phase_diff).mean(dim=-1)   # (B, H)
        sin_m = torch.sin(phase_diff).mean(dim=-1)
        R = (cos_m ** 2 + sin_m ** 2).sqrt().mean()  # scalar
        total = total + (-K_h.mean() * R)
        n_pairs += 1

    return total / max(1, n_pairs)
