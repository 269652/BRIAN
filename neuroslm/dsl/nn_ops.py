# -*- coding: utf-8 -*-
"""NeuroTensor op atoms — the lowering targets for the NN DSL (Phase N1).

Each function is the *exact* torch computation the hand-written `Brain`
performs, so a DSL model composed of these ops can be bit-identical to
the reference. The DSL codegen emits calls into this module; the
exact-match tests in tests/dsl/test_nn_ops_equivalence.py pin each op to
its `neuroslm.modules.common` counterpart.

Keep these pure (no nn.Module state): parameters are passed in explicitly
so the same op works for both eager reference comparison and generated
code. Parameter *allocation* + init lives in the codegen layer (Phase N3).
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F


# ── Linear / embedding ─────────────────────────────────────────────────

def linear(x: torch.Tensor, weight: torch.Tensor,
           bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """y = x @ weight.T (+ bias) — matches nn.Linear.

    weight is stored (out, in) like nn.Linear, so we use F.linear.
    Optional bias supports DSL layers like NeuralGeometryAdapter.
    """
    return F.linear(x, weight, bias)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched matrix multiply — torch.matmul's broadcasting semantics.
    Used in DSL blocks (e.g. NeuralGeometryAdapter's low-rank kernel
    chain `z @ kern_a @ kern_b`)."""
    return torch.matmul(a, b)


def embedding(ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """table[ids] — matches nn.Embedding (table is (vocab, dim))."""
    return F.embedding(ids, table)


# ── Normalization ──────────────────────────────────────────────────────

def rmsnorm(x: torch.Tensor, weight: torch.Tensor,
            eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm — matches neuroslm.modules.common.RMSNorm exactly:

        norm = mean(x^2, -1, keepdim).add(eps).rsqrt()
        return x * norm * weight
    """
    norm = x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return x * norm * weight


def layernorm(x: torch.Tensor, weight: torch.Tensor,
              bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Standard layer norm over the last dim — matches nn.LayerNorm."""
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


# ── SwiGLU MLP ─────────────────────────────────────────────────────────

def swiglu_hidden_dim(dim: int) -> int:
    """Hidden width SwiGLU uses: round_up_to_8(dim * 8 / 3).

    Matches neuroslm.modules.common.SwiGLU's default sizing so DSL-
    allocated params line up with the reference.
    """
    hidden = int(dim * 8 / 3)
    return (hidden + 7) // 8 * 8


def swiglu(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
           w3: torch.Tensor) -> torch.Tensor:
    """SwiGLU — matches common.SwiGLU: w3(silu(x@w1) * (x@w2)).

    w1, w2: (hidden, dim)   w3: (dim, hidden)   (nn.Linear layout)
    """
    return F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w2), w3)


# ── Activations ────────────────────────────────────────────────────────

def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


def relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.softmax(x, dim=dim)


# ── RoPE ───────────────────────────────────────────────────────────────

def rope_cache(seq_len: int, head_dim: int, base: float = 10000.0,
               device=None, dtype: torch.dtype = torch.float32):
    """Build (cos, sin) caches — matches common.build_rope_cache."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device,
                                            dtype=dtype) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding — matches common.apply_rope.

    x: (B, H, T, D). Interleaved even/odd rotate-half.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, : x.size(-2), :]
    sin = sin[None, None, : x.size(-2), :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)


# ── Causal self-attention (the keystone op) ────────────────────────────

def causal_self_attention(x: torch.Tensor,
                          q_weight: torch.Tensor,
                          kv_weight: torch.Tensor,
                          out_weight: torch.Tensor,
                          n_heads: int,
                          n_kv_heads: int,
                          max_ctx: int,
                          rope_base: float = 10000.0) -> torch.Tensor:
    """Causal self-attention — matches common.CausalSelfAttention base path.

    Reproduces, in order: GQA Q/KV projections, QK unit-normalisation,
    RoPE, GQA head expansion, causal scaled-dot-product attention, output
    projection. NT-modulation and Hebbian traces (the n_nt>0 /
    hebbian_rank>0 branches) are intentionally excluded — those are
    optional and handled in a later phase.

    Weights follow nn.Linear layout (out, in):
        q_weight:  (n_heads * head_dim, dim)
        kv_weight: (2 * n_kv_heads * head_dim, dim)
        out_weight:(dim, dim)
    """
    B, T, C = x.shape
    head_dim = C // n_heads
    n_groups = n_heads // n_kv_heads

    q = F.linear(x, q_weight).view(B, T, n_heads, head_dim).transpose(1, 2)
    kv = F.linear(x, kv_weight).view(B, T, 2, n_kv_heads, head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]

    # QK unit-sphere normalisation (entropy-collapse guard) — before RoPE.
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    cos, sin = rope_cache(max_ctx, head_dim, base=rope_base,
                          device=x.device, dtype=x.dtype)
    q = rope(q, cos.to(q.dtype), sin.to(q.dtype))
    k = rope(k, cos.to(k.dtype), sin.to(k.dtype))

    # GQA: expand KV heads to match Q heads
    if n_groups > 1:
        k = k[:, :, None, :, :].expand(-1, -1, n_groups, -1, -1).reshape(B, n_heads, T, head_dim)
        v = v[:, :, None, :, :].expand(-1, -1, n_groups, -1, -1).reshape(B, n_heads, T, head_dim)

    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return F.linear(y, out_weight)


# ── N7: cognitive attention subsystems ─────────────────────────────────

def neuromod_scale(nt: torch.Tensor, proj_weight: torch.Tensor,
                   proj_bias: torch.Tensor) -> torch.Tensor:
    """NT vector → per-head attention temperature scale.

    Matches neuro_attention.NeuromodulatedScale exactly:
        scale = softplus(proj(nt) + 0.5413)   # softplus(0.5413) ≈ 1.0
        → (B, n_heads, 1, 1)
    DA sharpens (higher scale), NE broadens (lower). Zero-init proj gives
    scale ≈ 1.0 (attention unmodified).
    """
    raw = F.linear(nt, proj_weight, proj_bias)      # (B, n_heads)
    scale = F.softplus(raw + 0.5413)
    return scale.unsqueeze(-1).unsqueeze(-1)         # (B, H, 1, 1)


def hebbian_trace(q: torch.Tensor, k: torch.Tensor,
                  query_proj_w: torch.Tensor, key_proj_w: torch.Tensor,
                  log_decay: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Low-rank Hebbian fast-weight trace bias for attention logits.

    Matches neuro_attention.HebbianTrace exactly: project q,k to rank-R,
    build a causal exponential-moving-average of the key projections, then
    bias[i,j] = q_r[i] · ema[j] (causal-masked), scaled.

    q, k: (B, H, T, D) → (B, H, T, T) additive bias.
    """
    B, H, T, D = q.shape
    decay = torch.sigmoid(log_decay)

    q_r = F.linear(q, query_proj_w)   # (B, H, T, R)
    k_r = F.linear(k, key_proj_w)     # (B, H, T, R)

    # Causal decayed EMA of k_r: ema[t] = decay*ema[t-1] + k_r[t]
    ema = torch.zeros_like(k_r[:, :, :1, :])
    ema_list = []
    for t in range(T):
        ema = decay * ema + k_r[:, :, t:t + 1, :]
        ema_list.append(ema)
    ema_all = torch.cat(ema_list, dim=2)   # (B, H, T, R)

    trace_bias = torch.einsum('bhir,bhjr->bhij', q_r, ema_all)
    causal_mask = torch.triu(
        torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    trace_bias = trace_bias.masked_fill(causal_mask, 0.0)
    return trace_bias * scale
