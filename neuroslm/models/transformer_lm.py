# -*- coding: utf-8 -*-
"""Generic transformer LM — built entirely from ModelSpec DSL.

The DSL sub-blocks are the mechanic definitions:
  coboundary { type: mha/gqa/swa/kjpla, ... }  → attention operator
  transition  { type: mlp/swiglu/geglu, ... }   → FFN operator
  norm        { type: layernorm/rmsnorm, ... }   → normalizer
  embed       { position: learned/none, ... }    → positional encoding
  output      { tie_embed: bool, bias: bool }    → LM head

No enum dispatch. The spec IS the mechanic. Change a field in arch.neuro
and the model reflects it immediately.

HF weight loading: `hf_to_model_state_dict(hf_sd, spec)` auto-detects the
HF format from state_dict key prefixes (transformer.* for GPT-2, model.* for
LLaMA family) and maps to our canonical internal names.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.dsl.model_spec import CoboundaryConfig, ModelSpec, TransitionConfig


# ── RoPE ─────────────────────────────────────────────────────────────────────

def _build_rope(seq_len: int, head_dim: int, base: float = 10_000.0,
                device=None) -> Tuple[torch.Tensor, torch.Tensor]:
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
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


# ── Attention — built from coboundary spec ────────────────────────────────────

class Attention(nn.Module):
    """Self-attention configured by CoboundaryConfig.

    coboundary.type controls the projection layout:
      mha   — fused QKV (c_attn / c_proj, GPT-2 Conv1D style)
      gqa   — separate Q/K/V + GQA expansion + RoPE
      swa   — like gqa + sliding-window causal mask (Mistral)
      kjpla — BRIAN Kuramoto-Josephson phase-lattice attention
              (implemented in neuroslm/mechanisms/kjpla.py)
    """

    def __init__(self, spec: ModelSpec):
        super().__init__()
        s = spec.sheaf
        cb = s.coboundary
        self._cb_type = cb.type
        self._n_heads = s.heads
        self._n_kv = s.kv_heads
        self._n_groups = s.heads // s.kv_heads
        self._head_dim = cb.head_dim if cb.head_dim is not None else s.dim // s.heads
        self._window = cb.window_size

        dim = s.dim
        hd = self._head_dim
        H = s.heads
        Hkv = s.kv_heads

        if cb.type == "mha":
            # Fused QKV — matches HF GPT2Attention.c_attn
            self.c_attn = nn.Linear(dim, 3 * dim, bias=cb.bias)
            self.c_proj = nn.Linear(dim, dim, bias=cb.bias)

        elif cb.type in ("gqa", "swa", "linear"):
            self.q_proj = nn.Linear(dim, H * hd,   bias=cb.qkv_bias)
            self.k_proj = nn.Linear(dim, Hkv * hd, bias=cb.qkv_bias)
            self.v_proj = nn.Linear(dim, Hkv * hd, bias=cb.qkv_bias)
            self.o_proj = nn.Linear(H * hd, dim,   bias=cb.bias)
            # RoPE cache
            rope_base = float(cb.rope.base) if cb.rope is not None else 10_000.0
            self._rope_base = rope_base
            cos, sin = _build_rope(s.context, hd, base=rope_base)
            self.register_buffer("_rope_cos", cos, persistent=False)
            self.register_buffer("_rope_sin", sin, persistent=False)

        elif cb.type == "kjpla":
            from neuroslm.mechanisms.kjpla import KJPLAttention
            self._kjpla = KJPLAttention(
                dim=dim, n_heads=H, max_ctx=s.context,
                n_kv_heads=Hkv,
                josephson_strength=cb.josephson_strength,
                entropy_eps=cb.entropy_eps,
            )

        elif cb.type == "mla":
            from neuroslm.mechanisms.kjpla import MultiHeadLatentAttention  # if implemented
            raise NotImplementedError("MLA not yet wired — implement in mechanisms/")

        else:
            raise ValueError(f"Unknown coboundary.type: {cb.type!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        if self._cb_type == "mha":
            q, k, v = self.c_attn(x).split(C, dim=-1)
            q = q.view(B, T, self._n_heads, self._head_dim).transpose(1, 2)
            k = k.view(B, T, self._n_heads, self._head_dim).transpose(1, 2)
            v = v.view(B, T, self._n_heads, self._head_dim).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.c_proj(y)

        elif self._cb_type in ("gqa", "swa", "linear"):
            q = self.q_proj(x).view(B, T, self._n_heads, self._head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self._n_kv,   self._head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self._n_kv,   self._head_dim).transpose(1, 2)

            # RoPE
            if self._rope_cos.size(0) < T:
                cos, sin = _build_rope(T, self._head_dim, base=self._rope_base,
                                       device=x.device)
            else:
                cos, sin = self._rope_cos[:T], self._rope_sin[:T]
            q, k = _apply_rope(q, k, cos.to(q.dtype), sin.to(q.dtype))

            # GQA expansion
            if self._n_groups > 1:
                k = k.unsqueeze(2).expand(-1, -1, self._n_groups, -1, -1).reshape(
                    B, self._n_heads, T, self._head_dim)
                v = v.unsqueeze(2).expand(-1, -1, self._n_groups, -1, -1).reshape(
                    B, self._n_heads, T, self._head_dim)

            if self._window is not None:
                # Sliding-window causal mask
                i = torch.arange(T, device=x.device).unsqueeze(1)
                j = torch.arange(T, device=x.device).unsqueeze(0)
                mask = (j > i) | (i - j >= self._window)
                scale = (self._head_dim ** -0.5)
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.masked_fill(mask, float("-inf"))
                y = F.softmax(attn, dim=-1) @ v
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

            y = y.transpose(1, 2).contiguous().view(B, T, self._n_heads * self._head_dim)
            return self.o_proj(y)

        elif self._cb_type == "kjpla":
            y, _, _ = self._kjpla(x)
            return y

        raise ValueError(f"Unknown coboundary.type: {self._cb_type!r}")


# ── FFN — built from transition spec ─────────────────────────────────────────

class FFN(nn.Module):
    """Feed-forward network configured by TransitionConfig.

    transition.type controls the variant:
      mlp     — Linear + activation + Linear (GPT-2 style; default gelu)
      swiglu  — silu(gate_proj(x)) * up_proj(x) → down_proj(x)
      geglu   — gelu(gate_proj(x)) * up_proj(x) → down_proj(x)
      moe     — mixture of experts (stub; full impl in mechanisms/)
      liouville_symplectic — BRIAN Liouville-symplectic residual
    """

    def __init__(self, spec: ModelSpec):
        super().__init__()
        s = spec.sheaf
        tr = s.transition
        self._tr_type = tr.type
        dim = s.dim
        ff_dim = int(dim * tr.ff_mult)
        self._activation = tr.activation

        if tr.type == "mlp":
            self.c_fc   = nn.Linear(dim, ff_dim, bias=tr.bias)
            self.c_proj = nn.Linear(ff_dim, dim, bias=tr.bias)

        elif tr.type in ("swiglu", "geglu"):
            self.gate_proj = nn.Linear(dim, ff_dim, bias=False)
            self.up_proj   = nn.Linear(dim, ff_dim, bias=False)
            self.down_proj = nn.Linear(ff_dim, dim, bias=False)

        elif tr.type == "liouville_symplectic":
            from neuroslm.mechanisms.liouville_symplectic import LiouvilleSymplecticBlock
            self._ls = LiouvilleSymplecticBlock(dim, ff_dim,
                                                noether_strength=tr.noether_strength)

        elif tr.type == "moe":
            raise NotImplementedError("MoE FFN not yet wired — implement in mechanisms/")

        else:
            raise ValueError(f"Unknown transition.type: {tr.type!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._tr_type == "mlp":
            act = F.gelu(self.c_fc(x), approximate="tanh") \
                if self._activation == "gelu" else F.relu(self.c_fc(x))
            return self.c_proj(act)

        elif self._tr_type == "swiglu":
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

        elif self._tr_type == "geglu":
            return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))

        elif self._tr_type == "liouville_symplectic":
            return self._ls(x)

        raise ValueError(f"Unknown transition.type: {self._tr_type!r}")


# ── Norm factory ──────────────────────────────────────────────────────────────

def _make_norm(spec: ModelSpec) -> nn.Module:
    n = spec.sheaf.norm
    dim = spec.sheaf.dim
    if n.type == "layernorm":
        return nn.LayerNorm(dim, eps=n.eps)
    elif n.type == "rmsnorm":
        from neuroslm.modules.common import RMSNorm
        return RMSNorm(dim, eps=n.eps)
    raise ValueError(f"Unknown norm.type: {n.type!r}")


# ── Transformer block ─────────────────────────────────────────────────────────

class Block(nn.Module):
    """One transformer block — pre-norm → attn → pre-norm → ffn.

    Norm names match HF checkpoints:
      LayerNorm model family  → ln_1 / ln_2
      RMSNorm model family    → input_layernorm / post_attention_layernorm
    """

    def __init__(self, spec: ModelSpec):
        super().__init__()
        n_type = spec.sheaf.norm.type

        if n_type == "layernorm":
            self.ln_1 = _make_norm(spec)
            self.ln_2 = _make_norm(spec)
        else:
            self.input_layernorm = _make_norm(spec)
            self.post_attention_layernorm = _make_norm(spec)

        self._norm_is_ln = (n_type == "layernorm")
        self.attn = Attention(spec)
        self.mlp  = FFN(spec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._norm_is_ln:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            x = x + self.attn(self.input_layernorm(x))
            x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ── Full model ────────────────────────────────────────────────────────────────

class TransformerLM(nn.Module):
    """Generic transformer LM — fully driven by ModelSpec DSL.

    Covers GPT-2 (mha + mlp + layernorm + learned-pos),
    LLaMA / SmolLM2 (gqa + swiglu + rmsnorm + no-pos),
    Qwen2.5 (gqa + qkv_bias + swiglu + rmsnorm + no-pos),
    and BRIAN hybrids (kjpla + liouville_symplectic + ...).

    No enum dispatch — change arch.neuro, the model follows.
    """

    def __init__(self, spec: ModelSpec):
        super().__init__()
        s = spec.sheaf
        self._spec = spec
        self._has_pos = (s.embed.position == "learned")

        self.embed_tokens = nn.Embedding(s.vocab, s.dim)
        if self._has_pos:
            self.pos_embed = nn.Embedding(s.context, s.dim)

        self.blocks = nn.ModuleList([Block(spec) for _ in range(s.depth)])
        self.norm   = _make_norm(spec)
        self.lm_head = nn.Linear(s.dim, s.vocab, bias=s.output.bias)

        if s.output.tie_embed:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights(spec)

    def _init_weights(self, spec: ModelSpec):
        std = spec.sheaf.init.std
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=std)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        if self._has_pos:
            T = input_ids.shape[1]
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


# ── HF weight loading ─────────────────────────────────────────────────────────

def hf_to_model_state_dict(hf_sd: Dict[str, torch.Tensor],
                           spec: ModelSpec) -> Dict[str, torch.Tensor]:
    """Map any HF AutoModelForCausalLM state_dict → our TransformerLM state_dict.

    Auto-detects the HF format:
      transformer.wte.*   → GPT-2 (Conv1D weights stored transposed)
      model.embed_tokens.* → LLaMA / SmolLM2 / Qwen2.5

    Internal names depend on what Attention / FFN / Block built from the spec,
    so this function mirrors those choices exactly.
    """
    cb = spec.sheaf.coboundary
    tr = spec.sheaf.transition
    s  = spec.sheaf
    out: Dict[str, torch.Tensor] = {}

    # Detect HF format
    if "transformer.wte.weight" in hf_sd:
        return _load_gpt2_format(hf_sd, spec)
    else:
        return _load_llama_format(hf_sd, spec)


def _load_gpt2_format(hf_sd: Dict[str, torch.Tensor],
                      spec: ModelSpec) -> Dict[str, torch.Tensor]:
    """GPT-2 HF format (Conv1D weights are stored transposed)."""
    s = spec.sheaf
    out: Dict[str, torch.Tensor] = {}

    out["embed_tokens.weight"] = hf_sd["transformer.wte.weight"]
    out["pos_embed.weight"]    = hf_sd["transformer.wpe.weight"]

    for i in range(s.depth):
        ph = f"transformer.h.{i}"
        pu = f"blocks.{i}"

        out[f"{pu}.ln_1.weight"] = hf_sd[f"{ph}.ln_1.weight"]
        out[f"{pu}.ln_1.bias"]   = hf_sd[f"{ph}.ln_1.bias"]
        out[f"{pu}.ln_2.weight"] = hf_sd[f"{ph}.ln_2.weight"]
        out[f"{pu}.ln_2.bias"]   = hf_sd[f"{ph}.ln_2.bias"]

        # Conv1D → Linear: transpose weight
        out[f"{pu}.attn.c_attn.weight"] = hf_sd[f"{ph}.attn.c_attn.weight"].t()
        out[f"{pu}.attn.c_attn.bias"]   = hf_sd[f"{ph}.attn.c_attn.bias"]
        out[f"{pu}.attn.c_proj.weight"] = hf_sd[f"{ph}.attn.c_proj.weight"].t()
        out[f"{pu}.attn.c_proj.bias"]   = hf_sd[f"{ph}.attn.c_proj.bias"]

        out[f"{pu}.mlp.c_fc.weight"]   = hf_sd[f"{ph}.mlp.c_fc.weight"].t()
        out[f"{pu}.mlp.c_fc.bias"]     = hf_sd[f"{ph}.mlp.c_fc.bias"]
        out[f"{pu}.mlp.c_proj.weight"] = hf_sd[f"{ph}.mlp.c_proj.weight"].t()
        out[f"{pu}.mlp.c_proj.bias"]   = hf_sd[f"{ph}.mlp.c_proj.bias"]

    out["norm.weight"] = hf_sd["transformer.ln_f.weight"]
    out["norm.bias"]   = hf_sd["transformer.ln_f.bias"]

    if "lm_head.weight" in hf_sd:
        out["lm_head.weight"] = hf_sd["lm_head.weight"]
    else:
        out["lm_head.weight"] = hf_sd["transformer.wte.weight"]

    return out


def _load_llama_format(hf_sd: Dict[str, torch.Tensor],
                       spec: ModelSpec) -> Dict[str, torch.Tensor]:
    """LLaMA / SmolLM2 / Qwen2.5 HF format (model.* prefix, separate QKV)."""
    s  = spec.sheaf
    cb = s.coboundary
    out: Dict[str, torch.Tensor] = {}

    out["embed_tokens.weight"] = hf_sd["model.embed_tokens.weight"]

    for i in range(s.depth):
        ph = f"model.layers.{i}"
        pu = f"blocks.{i}"

        out[f"{pu}.input_layernorm.weight"]          = hf_sd[f"{ph}.input_layernorm.weight"]
        out[f"{pu}.post_attention_layernorm.weight"] = hf_sd[f"{ph}.post_attention_layernorm.weight"]

        out[f"{pu}.attn.q_proj.weight"] = hf_sd[f"{ph}.self_attn.q_proj.weight"]
        out[f"{pu}.attn.k_proj.weight"] = hf_sd[f"{ph}.self_attn.k_proj.weight"]
        out[f"{pu}.attn.v_proj.weight"] = hf_sd[f"{ph}.self_attn.v_proj.weight"]
        out[f"{pu}.attn.o_proj.weight"] = hf_sd[f"{ph}.self_attn.o_proj.weight"]

        if cb.qkv_bias:
            for proj in ("q_proj", "k_proj", "v_proj"):
                bk = f"{ph}.self_attn.{proj}.bias"
                if bk in hf_sd:
                    out[f"{pu}.attn.{proj}.bias"] = hf_sd[bk]

        out[f"{pu}.mlp.gate_proj.weight"] = hf_sd[f"{ph}.mlp.gate_proj.weight"]
        out[f"{pu}.mlp.up_proj.weight"]   = hf_sd[f"{ph}.mlp.up_proj.weight"]
        out[f"{pu}.mlp.down_proj.weight"] = hf_sd[f"{ph}.mlp.down_proj.weight"]

    out["norm.weight"] = hf_sd["model.norm.weight"]

    if "lm_head.weight" in hf_sd:
        out["lm_head.weight"] = hf_sd["lm_head.weight"]
    else:
        out["lm_head.weight"] = hf_sd["model.embed_tokens.weight"]

    return out
