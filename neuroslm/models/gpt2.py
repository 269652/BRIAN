# -*- coding: utf-8 -*-
"""GPT-2 exact implementation with HF weight loading.

Architecture matches openai-community/gpt2 exactly so that loading HF weights
via `hf_to_model_state_dict` produces bit-identical logits.

Key differences from CausalSelfAttention (BRIAN):
  - Fused QKV via c_attn (single Linear dim → 3*dim)
  - Learned position embedding (pos_embed)
  - Standard LayerNorm, not RMSNorm
  - No F.normalize pre-attention
  - No RoPE
  - HF uses Conv1D (weights stored transposed); we use nn.Linear and transpose on load
  - Bias everywhere (bias=True)
  - GELU (not SwiGLU)

THSD framing: GPT-2 is a cellular sheaf with trivial H¹ (no conservation laws).
Every self-attention layer is a coboundary operator δ: C⁰(F) → C¹(F) mapping
token stalks to interaction co-chains; the MLP is a local sheaf morphism.
"""
from __future__ import annotations
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.dsl.model_spec import ModelSpec


class GPT2Attention(nn.Module):
    """GPT-2 self-attention with fused QKV projection and no positional bias."""

    def __init__(self, dim: int, n_heads: int, max_ctx: int, bias: bool = True):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        # Fused QKV — matches HF GPT2Attention.c_attn (Conv1D → nn.Linear)
        self.c_attn = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)                             # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=-1)                  # each (B, T, C)
        # Reshape to (B, H, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # Scaled dot-product attention with causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class GPT2MLP(nn.Module):
    """GPT-2 feed-forward: Linear + GELU + Linear."""

    def __init__(self, dim: int, ff_dim: int, bias: bool = True):
        super().__init__()
        self.c_fc = nn.Linear(dim, ff_dim, bias=bias)
        self.c_proj = nn.Linear(ff_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class GPT2Block(nn.Module):
    """GPT-2 transformer block: pre-LN → attention + pre-LN → MLP."""

    def __init__(self, dim: int, n_heads: int, ff_dim: int, max_ctx: int,
                 norm_eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = GPT2Attention(dim, n_heads, max_ctx, bias=bias)
        self.ln_2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = GPT2MLP(dim, ff_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    """Complete GPT-2 model.

    `forward(input_ids) -> logits` matches HF GPT2LMHeadModel output
    after `hf_to_model_state_dict` weight loading.

    Parameter naming uses our canonical scheme so the mapping function
    below translates HF names → ours.
    """

    def __init__(self, spec: ModelSpec):
        super().__init__()
        s = spec.sheaf
        self._spec = spec
        ff_dim = int(s.dim * s.ff_mult)

        self.embed_tokens = nn.Embedding(s.vocab, s.dim)
        self.pos_embed = nn.Embedding(s.context, s.dim)

        self.blocks = nn.ModuleList([
            GPT2Block(s.dim, s.heads, ff_dim, s.context,
                      norm_eps=s.norm_eps, bias=s.bias)
            for _ in range(s.depth)
        ])
        self.norm = nn.LayerNorm(s.dim, eps=s.norm_eps)
        self.lm_head = nn.Linear(s.dim, s.vocab, bias=False)

        if s.tie_embed:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=std)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed_tokens(input_ids) + self.pos_embed(positions)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


# ── HF weight loading ─────────────────────────────────────────────────────────

def hf_to_model_state_dict(hf_sd: Dict[str, torch.Tensor],
                           spec: ModelSpec) -> Dict[str, torch.Tensor]:
    """Map HF GPT2LMHeadModel state_dict → our GPT2Model state_dict.

    HF GPT-2 uses Conv1D layers whose weights are stored transposed
    relative to nn.Linear: Conv1D.weight is (in_dim, out_dim) whereas
    nn.Linear.weight is (out_dim, in_dim).  We transpose those here.
    """
    out: Dict[str, torch.Tensor] = {}
    s = spec.sheaf

    # Token + position embeddings
    out["embed_tokens.weight"] = hf_sd["transformer.wte.weight"]
    out["pos_embed.weight"] = hf_sd["transformer.wpe.weight"]

    # Per-layer blocks
    for i in range(s.depth):
        prefix_hf = f"transformer.h.{i}"
        prefix_us = f"blocks.{i}"

        # Layer norms
        out[f"{prefix_us}.ln_1.weight"] = hf_sd[f"{prefix_hf}.ln_1.weight"]
        out[f"{prefix_us}.ln_1.bias"] = hf_sd[f"{prefix_hf}.ln_1.bias"]
        out[f"{prefix_us}.ln_2.weight"] = hf_sd[f"{prefix_hf}.ln_2.weight"]
        out[f"{prefix_us}.ln_2.bias"] = hf_sd[f"{prefix_hf}.ln_2.bias"]

        # Attention: c_attn and c_proj are Conv1D in HF (weights transposed)
        out[f"{prefix_us}.attn.c_attn.weight"] = hf_sd[f"{prefix_hf}.attn.c_attn.weight"].t()
        out[f"{prefix_us}.attn.c_attn.bias"] = hf_sd[f"{prefix_hf}.attn.c_attn.bias"]
        out[f"{prefix_us}.attn.c_proj.weight"] = hf_sd[f"{prefix_hf}.attn.c_proj.weight"].t()
        out[f"{prefix_us}.attn.c_proj.bias"] = hf_sd[f"{prefix_hf}.attn.c_proj.bias"]

        # MLP: c_fc and c_proj are also Conv1D
        out[f"{prefix_us}.mlp.c_fc.weight"] = hf_sd[f"{prefix_hf}.mlp.c_fc.weight"].t()
        out[f"{prefix_us}.mlp.c_fc.bias"] = hf_sd[f"{prefix_hf}.mlp.c_fc.bias"]
        out[f"{prefix_us}.mlp.c_proj.weight"] = hf_sd[f"{prefix_hf}.mlp.c_proj.weight"].t()
        out[f"{prefix_us}.mlp.c_proj.bias"] = hf_sd[f"{prefix_hf}.mlp.c_proj.bias"]

    # Final norm
    out["norm.weight"] = hf_sd["transformer.ln_f.weight"]
    out["norm.bias"] = hf_sd["transformer.ln_f.bias"]

    # LM head — may be tied to embed_tokens in HF too
    if "lm_head.weight" in hf_sd:
        out["lm_head.weight"] = hf_sd["lm_head.weight"]
    else:
        # Tied: lm_head.weight is wte.weight
        out["lm_head.weight"] = hf_sd["transformer.wte.weight"]

    return out
