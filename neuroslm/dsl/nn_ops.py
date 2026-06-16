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
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ── GIF-5: Attention head diversity stash ──────────────────────────────
# When set to a list, causal_self_attention appends per-head Q tensors
# (with grad) for post-hoc diversity loss computation. Set to None to
# disable. Thread-unsafe; fine for single-GPU training.
from typing import List as _List
_DIVERSITY_STASH: Optional[_List[torch.Tensor]] = None


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


# ── GIF-6: Cosine LM Head ─────────────────────────────────────────────

def cosine_lm_head(h: torch.Tensor, weight: torch.Tensor,
                   temperature: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity LM head — eliminates magnitude as a DoF.

    z_i = τ · (h̄ · w̄_i)    where h̄ = h/‖h‖, w̄_i = w_i/‖w_i‖

    Args:
        h:           (B, T, D) hidden states
        weight:      (V, D) token embeddings (same layout as nn.Linear)
        temperature: scalar learnable τ (init √d_model)

    Returns:
        (B, T, V) logits bounded in [-τ, +τ]

    Gradients flow through h, weight, and temperature. Numerical
    stability: F.normalize adds eps=1e-12 to avoid div-by-zero.
    """
    h_norm = F.normalize(h, dim=-1)       # (B, T, D)
    w_norm = F.normalize(weight, dim=-1)  # (V, D)
    return temperature * F.linear(h_norm, w_norm)


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

    # GIF-5: stash post-RoPE Q for attention head diversity loss.
    # q shape: (B, n_heads, T, head_dim) — with gradient.
    if _DIVERSITY_STASH is not None:
        _DIVERSITY_STASH.append(q)

    # GQA: expand KV heads to match Q heads
    if n_groups > 1:
        k = k[:, :, None, :, :].expand(-1, -1, n_groups, -1, -1).reshape(B, n_heads, T, head_dim)
        v = v[:, :, None, :, :].expand(-1, -1, n_groups, -1, -1).reshape(B, n_heads, T, head_dim)

    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return F.linear(y, out_weight)


# ── Tonnetz-masked attention (Stage 2 OOD push) ────────────────────────

_TONNETZ_MASK_CACHE: Dict[Tuple[int, int, int], torch.Tensor] = {}


def _tonnetz_attention_mask(T: int, period: int = 12,
                              device=None, dtype=torch.float32) -> torch.Tensor:
    """Toroidal (Tonnetz) attention mask.

    Each query at position q can attend to keys at positions k where
    the *circular distance* `min(|q-k| mod period, period - |q-k| mod period)`
    falls within a small bandwidth. This embeds a periodic harmonic
    topology into the attention pattern: distant tokens that "rhyme"
    modulo the period are reachable, but arbitrary leaps are suppressed.

    The mask returned is additive (-inf for blocked positions, 0 for
    allowed) so it composes with standard scaled-dot-product attention.

    Args:
        T:      sequence length
        period: torus period (default 12 — musical octave)
        device, dtype: tensor placement

    Returns:
        (T, T) additive attention mask. Combined with `is_causal=True`
        at the call site to keep the autoregressive constraint.
    """
    key = (T, period, hash(device))
    if key in _TONNETZ_MASK_CACHE:
        cached = _TONNETZ_MASK_CACHE[key]
        if cached.device == torch.device(device) if device else cached.device:
            return cached.to(dtype)

    idx = torch.arange(T, device=device)
    # circular distance modulo period
    diff = (idx[:, None] - idx[None, :]).abs() % period
    circ = torch.minimum(diff, period - diff)
    # Allow positions within bandwidth = period // 4 (e.g. 3 of 12)
    bandwidth = max(1, period // 4)
    allowed = circ <= bandwidth
    # Also allow the "anchor" half of recent context regardless of torus
    # (keeps short-range continuity intact — without this the mask is
    # too restrictive on token-level dependencies).
    local_window = max(8, period)
    local = (idx[:, None] - idx[None, :]).abs() <= local_window
    allowed = allowed | local
    mask = torch.where(allowed, 0.0, float("-inf")).to(dtype)
    _TONNETZ_MASK_CACHE[key] = mask
    return mask


def causal_self_attention_tonnetz(x: torch.Tensor,
                                    q_weight: torch.Tensor,
                                    kv_weight: torch.Tensor,
                                    out_weight: torch.Tensor,
                                    n_heads: int,
                                    n_kv_heads: int,
                                    max_ctx: int,
                                    tonnetz_period: int = 12,
                                    rope_base: float = 10000.0) -> torch.Tensor:
    """Causal self-attention with a Tonnetz toroidal mask.

    Identical to `causal_self_attention` except an additive Tonnetz
    mask is composed with the causal constraint. The mask exponentially
    suppresses attention to positions outside the torus bandwidth +
    local-window, bounding the "convex hull volume" of attention mass
    (cited as a hallucination biomarker).
    """
    B, T, C = x.shape
    head_dim = C // n_heads
    n_groups = n_heads // n_kv_heads

    q = F.linear(x, q_weight).view(B, T, n_heads, head_dim).transpose(1, 2)
    kv = F.linear(x, kv_weight).view(B, T, 2, n_kv_heads, head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    cos, sin = rope_cache(max_ctx, head_dim, base=rope_base,
                          device=x.device, dtype=x.dtype)
    q = rope(q, cos.to(q.dtype), sin.to(q.dtype))
    k = rope(k, cos.to(k.dtype), sin.to(k.dtype))
    if n_groups > 1:
        k = k[:, :, None, :, :].expand(-1, -1, n_groups, -1, -1).reshape(B, n_heads, T, head_dim)
        v = v[:, :, None, :, :].expand(-1, -1, n_groups, -1, -1).reshape(B, n_heads, T, head_dim)

    # Build the additive attention mask: causal AND tonnetz.
    causal = torch.triu(
        torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype),
        diagonal=1)
    tonnetz = _tonnetz_attention_mask(T, period=tonnetz_period,
                                       device=x.device, dtype=x.dtype)
    attn_mask = causal + tonnetz
    # F.scaled_dot_product_attention accepts an additive mask via
    # attn_mask=...; is_causal must be False when a custom mask is given.
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                        is_causal=False, dropout_p=0.0)
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return F.linear(y, out_weight)


# ── Stage 5 OOD push: Fisher-Rao retrieval metric ─────────────────────

def fisher_rao_distance(query: torch.Tensor, keys: torch.Tensor,
                         variance: torch.Tensor,
                         eps: float = 1e-6) -> torch.Tensor:
    """Fisher-Rao distance between a query and a bank of keys, weighted
    by per-dimension variance (precision = 1/variance).

    Replaces cosine similarity for hippocampal-style retrieval. Each
    dimension's contribution is weighted by its statistical precision
    so noisy / high-variance dimensions are down-weighted in the metric.
    Result: as memory size scales, the retrieval ignores uncertain
    dimensions automatically (no manual feature selection).

    Args:
        query:    (B, D)   batched query vectors
        keys:     (M, D)   memory bank (M items, D features)
        variance: (D,)     per-dimension running variance (≥ eps)

    Returns:
        (B, M)   distance matrix; smaller = more similar
    """
    precision = 1.0 / (variance.clamp(min=eps))  # (D,)
    # Weighted squared diff: (B, M, D) → sum over D → (B, M)
    diff = query.unsqueeze(1) - keys.unsqueeze(0)            # (B, M, D)
    weighted_sq = (diff ** 2) * precision.view(1, 1, -1)
    return weighted_sq.sum(dim=-1).sqrt()


def fisher_rao_topk(query: torch.Tensor, keys: torch.Tensor,
                     variance: torch.Tensor, k: int = 1
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k nearest keys to each query under the Fisher-Rao metric.

    Returns (distances, indices) each of shape (B, k).
    """
    d = fisher_rao_distance(query, keys, variance)
    return d.topk(k, dim=-1, largest=False)


def predictive_coding_head(h_current: torch.Tensor, h_next: torch.Tensor,
                            w1: torch.Tensor, b1: torch.Tensor,
                            w2: torch.Tensor) -> torch.Tensor:
    """Scalar predictive-coding loss between consecutive layer states.

    Bit-identical to neuro_attention.PredictiveCodingHead: residual
    predictor (linear → SiLU → linear), prediction = layer_norm(h + δ),
    target = layer_norm(h_next.detach()); loss = cos_err + 0.1 · l2_err.
    """
    with torch.no_grad():
        target = F.layer_norm(h_next.detach(), [h_next.size(-1)])
    delta = F.linear(F.silu(F.linear(h_current, w1, b1)), w2)
    pred = F.layer_norm(h_current + delta, [h_current.size(-1)])
    cosine_err = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    l2_err = (pred - target).pow(2).mean()
    return cosine_err + 0.1 * l2_err


def mod_block(x: torch.Tensor,
              # router MLP (2-layer with SiLU)
              router_w1: torch.Tensor, router_b1: torch.Tensor,
              router_w2: torch.Tensor, router_b2: torch.Tensor,
              # differential attention block
              gamma1: torch.Tensor,
              q_weight: torch.Tensor, kv_weight: torch.Tensor,
              out_weight: torch.Tensor,
              lambda_init: torch.Tensor, sub_norm_weight: torch.Tensor,
              # post-attn norm + SwiGLU
              gamma2: torch.Tensor,
              w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor,
              # config
              n_heads: int, n_kv_heads: int, max_ctx: int,
              capacity_ratio: float = 0.5) -> torch.Tensor:
    """Mixture-of-Depths block (use_diff_attn=True, maturity=1.0, no NT) —
    bit-identical to modules.mixture_of_depths.MoDBlock.

    Routes only the top-C tokens (per a 2-layer MLP score) through diff-
    attn + SwiGLU; unrouted tokens get the identity residual. Scatters
    the processed outputs back into the original positions.
    """
    B, T, D = x.shape

    # 2-layer SiLU MLP router → per-token score
    h = F.linear(x, router_w1, router_b1)
    h = F.silu(h)
    router_logits = F.linear(h, router_w2, router_b2)        # (B, T, 1)

    C = max(1, int(T * capacity_ratio))
    if C == T:
        a = differential_attention(rmsnorm(x, gamma1), q_weight, kv_weight,
                                    out_weight, lambda_init, sub_norm_weight,
                                    n_heads, n_kv_heads, max_ctx)
        x = x + a
        m = swiglu(rmsnorm(x, gamma2), w1, w2, w3)
        return x + m

    scores = router_logits.squeeze(-1)                       # (B, T)
    _, topk_idx = scores.topk(C, dim=-1, sorted=False)
    topk_idx, _ = topk_idx.sort(dim=-1)                      # causal order
    selected_x = x.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, D))

    h = differential_attention(rmsnorm(selected_x, gamma1),
                               q_weight, kv_weight, out_weight,
                               lambda_init, sub_norm_weight,
                               n_heads, n_kv_heads, max_ctx)
    h = h + swiglu(rmsnorm(selected_x + h, gamma2), w1, w2, w3)

    out = x.clone()
    out.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, D), selected_x + h)
    return out


def differential_attention(x: torch.Tensor,
                           q_weight: torch.Tensor,
                           kv_weight: torch.Tensor,
                           out_weight: torch.Tensor,
                           lambda_init: torch.Tensor,
                           sub_norm_weight: torch.Tensor,
                           n_heads: int,
                           n_kv_heads: int,
                           max_ctx: int,
                           rope_base: float = 10000.0) -> torch.Tensor:
    """Differential attention — matches modules.differential_attention.
    DifferentialAttention(n_nt=0) base path bit-for-bit.

    Each head splits Q/K into halves; output is
        (softmax(Q1 K1ᵀ) − λ·softmax(Q2 K2ᵀ)) @ V
    with a head-wise λ = sigmoid(lambda_init) for noise cancellation,
    plus per-half F.normalize, RoPE on half_dim, GQA expansion, manual
    causal mask, and a head-dim RMSNorm before the output projection.
    """
    B, T, C = x.shape
    head_dim = C // n_heads
    half_dim = head_dim // 2
    n_groups = n_heads // n_kv_heads

    q = F.linear(x, q_weight).view(B, T, n_heads, head_dim).transpose(1, 2)
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    kv = F.linear(x, kv_weight).view(B, T, 2, n_kv_heads, head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]

    q1, q2 = F.normalize(q1, dim=-1), F.normalize(q2, dim=-1)
    k1, k2 = F.normalize(k1, dim=-1), F.normalize(k2, dim=-1)

    cos, sin = rope_cache(max_ctx, half_dim, base=rope_base,
                          device=x.device, dtype=x.dtype)
    q1 = rope(q1, cos.to(q1.dtype), sin.to(q1.dtype))
    q2 = rope(q2, cos.to(q2.dtype), sin.to(q2.dtype))
    k1 = rope(k1, cos.to(k1.dtype), sin.to(k1.dtype))
    k2 = rope(k2, cos.to(k2.dtype), sin.to(k2.dtype))

    if n_groups > 1:
        k1 = k1[:, :, None].expand(-1, -1, n_groups, -1, -1).reshape(B, n_heads, T, half_dim)
        k2 = k2[:, :, None].expand(-1, -1, n_groups, -1, -1).reshape(B, n_heads, T, half_dim)
        v = v[:, :, None].expand(-1, -1, n_groups, -1, -1).reshape(B, n_heads, T, head_dim)

    lam = torch.sigmoid(lambda_init).unsqueeze(0).expand(B, -1).unsqueeze(-1).unsqueeze(-1)

    scale = half_dim ** -0.5
    causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
    attn1 = (q1 @ k1.transpose(-2, -1)) * scale
    attn2 = (q2 @ k2.transpose(-2, -1)) * scale
    attn1 = F.softmax(attn1.masked_fill(causal_mask, float("-inf")), dim=-1)
    attn2 = F.softmax(attn2.masked_fill(causal_mask, float("-inf")), dim=-1)
    diff_attn = attn1 - lam * attn2

    y = diff_attn @ v                                  # (B, H, T, head_dim)
    y = rmsnorm(y, sub_norm_weight)                    # per-head-dim RMSNorm
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
