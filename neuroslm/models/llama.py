# -*- coding: utf-8 -*-
"""LLaMA-family model implementation (SmolLM2, Qwen2.5, Mistral style).

Architecture matches HuggingFaceTB/SmolLM2-135M and Qwen/Qwen2.5-* exactly
so that `hf_to_model_state_dict` produces bit-identical logits to the HF
AutoModelForCausalLM.

Key characteristics shared by LLaMA / SmolLM2 / Qwen2:
  - Token embedding only (no learned position embedding)
  - Pre-RMSNorm (no bias in norms)
  - Separate Q, K, V projections (no fused c_attn)
  - Grouped Query Attention (kv_heads <= n_heads)
  - RoPE (rotary position encoding)
  - SwiGLU feed-forward: gate_proj, up_proj, down_proj
  - No bias in any linear layer
  - Weight-tied lm_head = embed_tokens (many variants)

Qwen2.5 adds:
  - Bias in q_proj, k_proj (not in v_proj, o_proj)
  - Very large rope_base (1_000_000)
  These are handled by the `bias` flag and `rope_base` in the spec.

THSD framing: same trivial H¹ sheaf as GPT-2, but with RoPE providing
a phase-structured coboundary that encodes positional information into
the C¹ interaction space.
"""
from __future__ import annotations
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.dsl.model_spec import ModelSpec
from neuroslm.modules.common import RMSNorm


# ── RoPE ─────────────────────────────────────────────────────────────────────

def _build_rope(seq_len: int, head_dim: int, base: float = 10000.0,
                device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                 / head_dim)
    )
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


# ── Attention ─────────────────────────────────────────────────────────────────

class LlamaAttention(nn.Module):
    """GQA + RoPE attention matching HF LlamaAttention / Qwen2Attention."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 max_ctx: int, rope_base: float = 10000.0, bias: bool = False,
                 qkv_bias: bool = False):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        self._rope_base = float(rope_base)
        self._max_ctx = max_ctx

        # qkv_bias: bias in q/k/v projections (Qwen2 uses this; separate from o_proj)
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # Precompute RoPE buffers (non-persistent — reconstructed on load)
        cos, sin = _build_rope(max_ctx, self.head_dim, base=rope_base)
        self.register_buffer("_rope_cos", cos, persistent=False)
        self.register_buffer("_rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self._rope_cos.size(0) < T:
            cos, sin = _build_rope(T, self.head_dim, base=self._rope_base,
                                   device=x.device)
        else:
            cos, sin = self._rope_cos[:T], self._rope_sin[:T]
        q, k = _apply_rope(q, k, cos.to(q.dtype), sin.to(q.dtype))

        # Expand KV for GQA
        if self.n_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1).reshape(
                B, self.n_heads, T, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1).reshape(
                B, self.n_heads, T, self.head_dim)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


# ── Feed-forward ──────────────────────────────────────────────────────────────

class LlamaMLP(nn.Module):
    """SwiGLU feed-forward matching HF LlamaMLP / Qwen2MLP.

    output = down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, ff_dim, bias=False)
        self.up_proj = nn.Linear(dim, ff_dim, bias=False)
        self.down_proj = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Block ─────────────────────────────────────────────────────────────────────

class LlamaBlock(nn.Module):
    """Pre-RMSNorm transformer block."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, ff_dim: int,
                 max_ctx: int, rope_base: float = 10000.0,
                 norm_eps: float = 1e-5, bias: bool = False,
                 qkv_bias: bool = False):
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps=norm_eps)
        self.attn = LlamaAttention(dim, n_heads, n_kv_heads, max_ctx,
                                   rope_base=rope_base, bias=bias,
                                   qkv_bias=qkv_bias)
        self.post_attention_layernorm = RMSNorm(dim, eps=norm_eps)
        self.mlp = LlamaMLP(dim, ff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ── Full model ────────────────────────────────────────────────────────────────

class LlamaModel(nn.Module):
    """Complete LLaMA-family model (SmolLM2 / Qwen2.5 / Mistral).

    `forward(input_ids) -> logits` matches HF AutoModelForCausalLM output
    after `hf_to_model_state_dict` weight loading.
    """

    def __init__(self, spec: ModelSpec):
        super().__init__()
        s = spec.sheaf
        self._spec = spec
        ff_dim = int(s.dim * s.ff_mult)

        self.embed_tokens = nn.Embedding(s.vocab, s.dim)

        self.blocks = nn.ModuleList([
            LlamaBlock(
                dim=s.dim, n_heads=s.heads, n_kv_heads=s.kv_heads,
                ff_dim=ff_dim, max_ctx=s.context,
                rope_base=float(s.rope_base), norm_eps=s.norm_eps,
                bias=s.bias, qkv_bias=s.qkv_bias,
            )
            for _ in range(s.depth)
        ])

        self.norm = RMSNorm(s.dim, eps=s.norm_eps)
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


# ── HF weight loading ─────────────────────────────────────────────────────────

def hf_to_model_state_dict(hf_sd: Dict[str, torch.Tensor],
                           spec: ModelSpec) -> Dict[str, torch.Tensor]:
    """Map HF AutoModelForCausalLM state_dict → our LlamaModel state_dict.

    Works for:
      - LLaMA / SmolLM2 (model.* prefix)
      - Qwen2 / Qwen2.5 (model.* prefix, identical layout)
      - Mistral (model.* prefix)
    """
    out: Dict[str, torch.Tensor] = {}
    s = spec.sheaf

    # Embedding
    out["embed_tokens.weight"] = hf_sd["model.embed_tokens.weight"]

    # Per-layer blocks
    for i in range(s.depth):
        ph = f"model.layers.{i}"
        pu = f"blocks.{i}"

        # Norms
        out[f"{pu}.input_layernorm.weight"] = hf_sd[f"{ph}.input_layernorm.weight"]
        out[f"{pu}.post_attention_layernorm.weight"] = hf_sd[
            f"{ph}.post_attention_layernorm.weight"]

        # Attention projections
        out[f"{pu}.attn.q_proj.weight"] = hf_sd[f"{ph}.self_attn.q_proj.weight"]
        out[f"{pu}.attn.k_proj.weight"] = hf_sd[f"{ph}.self_attn.k_proj.weight"]
        out[f"{pu}.attn.v_proj.weight"] = hf_sd[f"{ph}.self_attn.v_proj.weight"]
        out[f"{pu}.attn.o_proj.weight"] = hf_sd[f"{ph}.self_attn.o_proj.weight"]

        # QKV biases: only when spec has qkv_bias=True (Qwen2/Qwen2.5 pattern)
        if spec.sheaf.qkv_bias:
            for proj in ("q_proj", "k_proj", "v_proj"):
                bias_key = f"{ph}.self_attn.{proj}.bias"
                our_key = f"{pu}.attn.{proj}.bias"
                if bias_key in hf_sd:
                    out[our_key] = hf_sd[bias_key]

        # MLP
        out[f"{pu}.mlp.gate_proj.weight"] = hf_sd[f"{ph}.mlp.gate_proj.weight"]
        out[f"{pu}.mlp.up_proj.weight"] = hf_sd[f"{ph}.mlp.up_proj.weight"]
        out[f"{pu}.mlp.down_proj.weight"] = hf_sd[f"{ph}.mlp.down_proj.weight"]

    # Final norm
    out["norm.weight"] = hf_sd["model.norm.weight"]

    # LM head
    if "lm_head.weight" in hf_sd:
        out["lm_head.weight"] = hf_sd["lm_head.weight"]
    else:
        out["lm_head.weight"] = hf_sd["model.embed_tokens.weight"]

    return out
