"""Differential Attention — noise-cancelling dual-softmax attention.

Paper: "Differential Transformer" (Ye et al., Microsoft Research, 2024)
Key insight: standard softmax attention assigns non-zero weight to irrelevant
context tokens (attention noise). Differential attention computes TWO
attention maps and subtracts them, cancelling the noise component.

    DiffAttn(X) = (softmax(Q₁K₁ᵀ/√d) - λ·softmax(Q₂K₂ᵀ/√d)) V

where λ is a learnable scalar per head. This provably reduces attention
noise and improves signal-to-noise ratio.

Benefits over standard attention:
  - Hallucination reduction (noise tokens get ~0 weight instead of small positive)
  - Better long-context retrieval (signal doesn't drown in noise)
  - Outperforms standard attention at all scales on LM benchmarks

Integration with NeuroSLM:
  - NT-modulated λ: DA increases λ (more aggressive noise cancellation when focused)
  - Hebbian traces apply to the differential attention map
  - Genome controls the initial λ and modulation sensitivity
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import build_rope_cache, apply_rope, RMSNorm


class DifferentialAttention(nn.Module):
    """Multi-head differential attention with GQA + NT modulation.

    Each head splits Q and K into two halves: (Q1, Q2) and (K1, K2).
    The attention output is: (softmax(Q1K1ᵀ) - λ·softmax(Q2K2ᵀ)) @ V
    """

    def __init__(self, dim: int, n_heads: int, max_ctx: int,
                 n_kv_heads: int | None = None,
                 n_nt: int = 0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0
        self.n_groups = n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads
        # Half-dim for each of the two attention sub-maps
        self.half_dim = self.head_dim // 2

        # Q projects to 2× head_dim (split into Q1, Q2)
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        # KV: standard GQA projection
        self.kv_proj = nn.Linear(dim, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        # Learnable λ per head — controls noise cancellation strength
        # Initialize to ~0.5 so differential effect is moderate at start
        self.lambda_init = nn.Parameter(torch.full((n_heads,), 0.0))

        # NT modulation of λ: DA → stronger noise cancellation
        if n_nt > 0:
            self.lambda_nt = nn.Linear(n_nt, n_heads, bias=False)
            nn.init.zeros_(self.lambda_nt.weight)
        else:
            self.lambda_nt = None

        # RoPE cache
        cos, sin = build_rope_cache(max_ctx, self.half_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Sub-layer norm after differential attention (stabilizes training)
        self.sub_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor,
                nt: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape

        # Q: (B, T, n_heads, head_dim) → split into Q1, Q2 each (B, H, T, half_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q1, q2 = q[..., :self.half_dim], q[..., self.half_dim:]

        # KV
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        k1, k2 = k[..., :self.half_dim], k[..., self.half_dim:]

        # Apply RoPE to each half separately
        q1 = apply_rope(q1, self.cos.to(q1.dtype), self.sin.to(q1.dtype))
        q2 = apply_rope(q2, self.cos.to(q2.dtype), self.sin.to(q2.dtype))
        k1 = apply_rope(k1, self.cos.to(k1.dtype), self.sin.to(k1.dtype))
        k2 = apply_rope(k2, self.cos.to(k2.dtype), self.sin.to(k2.dtype))

        # GQA expansion
        if self.n_groups > 1:
            k1 = k1[:, :, None].expand(-1, -1, self.n_groups, -1, -1).reshape(B, self.n_heads, T, self.half_dim)
            k2 = k2[:, :, None].expand(-1, -1, self.n_groups, -1, -1).reshape(B, self.n_heads, T, self.half_dim)
            v = v[:, :, None].expand(-1, -1, self.n_groups, -1, -1).reshape(B, self.n_heads, T, self.head_dim)

        # Compute λ: base + NT modulation
        lam = torch.sigmoid(self.lambda_init)  # (H,)
        if self.lambda_nt is not None and nt is not None:
            nt_mod = torch.sigmoid(self.lambda_nt(nt))  # (B, H)
            lam = lam.unsqueeze(0) * (0.5 + nt_mod)  # NT scales λ
        else:
            lam = lam.unsqueeze(0).expand(B, -1)
        lam = lam.unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

        # Two attention maps
        scale = self.half_dim ** -0.5
        attn1 = (q1 @ k1.transpose(-2, -1)) * scale
        attn2 = (q2 @ k2.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn1 = attn1.masked_fill(causal_mask, float('-inf'))
        attn2 = attn2.masked_fill(causal_mask, float('-inf'))

        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)

        # Differential: cancel noise
        diff_attn = attn1 - lam * attn2  # (B, H, T, T)

        # Apply to values
        y = diff_attn @ v  # (B, H, T, head_dim)

        # Sub-layer norm for stability
        y = self.sub_norm(y)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class DiffTransformerBlock(nn.Module):
    """Transformer block with differential attention."""

    def __init__(self, dim: int, n_heads: int, max_ctx: int,
                 n_kv_heads: int | None = None,
                 n_nt: int = 0):
        super().__init__()
        from .common import SwiGLU
        self.n1 = RMSNorm(dim)
        self.attn = DifferentialAttention(dim, n_heads, max_ctx, n_kv_heads, n_nt)
        self.n2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim)

    def forward(self, x: torch.Tensor,
                nt: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.n1(x), nt=nt)
        x = x + self.mlp(self.n2(x))
        return x
