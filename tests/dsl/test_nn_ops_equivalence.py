# -*- coding: utf-8 -*-
"""Phase N1 — NeuroTensor op atoms, exact-match vs PyTorch reference.

Every DSL op must lower to the *exact* torch computation the hand-written
`Brain` uses, so a DSL-built model can be bit-identical to the reference.
These tests pin that: each op is compared with `torch.allclose(atol=1e-6)`
against the corresponding implementation in `neuroslm.modules.common`.

If any of these drift, a DSL model cannot match Brain — so they are the
foundation gate for the whole NeuroTensor redesign.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.dsl import nn_ops
from neuroslm.modules.common import (
    RMSNorm, SwiGLU, build_rope_cache, apply_rope,
)


# ── rmsnorm ────────────────────────────────────────────────────────────

class TestRMSNorm:
    def test_matches_reference(self):
        dim = 64
        ref = RMSNorm(dim)
        with torch.no_grad():
            ref.weight.copy_(torch.randn(dim))   # non-trivial weight

        x = torch.randn(2, 16, dim)
        ref_out = ref(x)
        dsl_out = nn_ops.rmsnorm(x, ref.weight, eps=ref.eps)
        assert torch.allclose(dsl_out, ref_out, atol=1e-6)

    def test_default_eps_matches(self):
        dim = 32
        ref = RMSNorm(dim)  # default eps 1e-6
        x = torch.randn(4, 8, dim)
        assert torch.allclose(nn_ops.rmsnorm(x, ref.weight), ref(x), atol=1e-6)


# ── swiglu ─────────────────────────────────────────────────────────────

class TestSwiGLU:
    def test_matches_reference(self):
        dim = 64
        ref = SwiGLU(dim)
        x = torch.randn(2, 16, dim)
        ref_out = ref(x)
        dsl_out = nn_ops.swiglu(x, ref.w1.weight, ref.w2.weight, ref.w3.weight)
        assert torch.allclose(dsl_out, ref_out, atol=1e-6)

    def test_hidden_dim_convention(self):
        # SwiGLU hidden = round_to_8(dim * 8/3). The DSL helper must agree.
        for dim in (64, 128, 256):
            ref = SwiGLU(dim)
            assert nn_ops.swiglu_hidden_dim(dim) == ref.w1.weight.shape[0]


# ── linear ─────────────────────────────────────────────────────────────

class TestLinear:
    def test_matches_nn_linear_no_bias(self):
        ref = nn.Linear(64, 128, bias=False)
        x = torch.randn(2, 16, 64)
        assert torch.allclose(nn_ops.linear(x, ref.weight), ref(x), atol=1e-6)


# ── embedding ──────────────────────────────────────────────────────────

class TestEmbedding:
    def test_matches_nn_embedding(self):
        ref = nn.Embedding(100, 64)
        ids = torch.randint(0, 100, (2, 16))
        assert torch.allclose(nn_ops.embedding(ids, ref.weight), ref(ids), atol=1e-6)


# ── silu / gelu / relu ─────────────────────────────────────────────────

class TestActivations:
    def test_silu(self):
        x = torch.randn(4, 32)
        assert torch.allclose(nn_ops.silu(x), F.silu(x), atol=1e-6)

    def test_gelu(self):
        x = torch.randn(4, 32)
        assert torch.allclose(nn_ops.gelu(x), F.gelu(x), atol=1e-6)

    def test_relu(self):
        x = torch.randn(4, 32)
        assert torch.allclose(nn_ops.relu(x), F.relu(x), atol=1e-6)


# ── rope ───────────────────────────────────────────────────────────────

class TestRoPE:
    def test_matches_reference(self):
        B, H, T, D = 2, 4, 16, 16
        x = torch.randn(B, H, T, D)
        cos, sin = build_rope_cache(T, D)
        ref_out = apply_rope(x, cos, sin)
        dsl_out = nn_ops.rope(x, cos, sin)
        assert torch.allclose(dsl_out, ref_out, atol=1e-6)

    def test_rope_cache_matches(self):
        T, D = 32, 16
        ref_cos, ref_sin = build_rope_cache(T, D)
        dsl_cos, dsl_sin = nn_ops.rope_cache(T, D)
        assert torch.allclose(dsl_cos, ref_cos, atol=1e-6)
        assert torch.allclose(dsl_sin, ref_sin, atol=1e-6)


# ── softmax ────────────────────────────────────────────────────────────

class TestSoftmax:
    def test_matches_reference(self):
        x = torch.randn(2, 8, 16)
        assert torch.allclose(nn_ops.softmax(x, dim=-1), F.softmax(x, dim=-1), atol=1e-6)


# ── causal_self_attention (T vs max_ctx length extrapolation) ──────────

class TestCausalSelfAttentionLengthHandling:
    """Regression: GIF OOD probe passes T>max_ctx to the DSL attention path.

    The original implementation built ``rope_cache(max_ctx, ...)`` and then
    ``rope()`` did ``cos[..., :T, :]``. When T > max_ctx, the slice
    returned only ``max_ctx`` entries and the next ``x1 * cos`` blew up
    with::

        RuntimeError: The size of tensor a (358) must match the size
        of tensor b (128) at non-singleton dimension 2

    Hit during real Lightning AI deploy at step 80 — the GIF.OOD_probe
    evaluates length-extrapolation on cached sequences (~T=358) against
    a model trained with max_ctx=128. RoPE itself supports arbitrary
    length (it's parameter-free), so the model must be ABLE to run
    these batches without crashing.

    Fix: build cache at ``max(T, max_ctx)``.
    """

    def _weights(self, dim, n_heads, n_kv_heads):
        head_dim = dim // n_heads
        return (
            torch.randn(n_heads * head_dim, dim) * 0.02,
            torch.randn(2 * n_kv_heads * head_dim, dim) * 0.02,
            torch.randn(dim, dim) * 0.02,
        )

    def test_T_equal_max_ctx_works(self):
        """Sanity: T == max_ctx (the training path) still works."""
        B, dim, n_heads, n_kv = 2, 64, 4, 2
        T = max_ctx = 16
        x = torch.randn(B, T, dim)
        qw, kvw, ow = self._weights(dim, n_heads, n_kv)
        y = nn_ops.causal_self_attention(x, qw, kvw, ow, n_heads, n_kv, max_ctx)
        assert y.shape == (B, T, dim)

    def test_T_less_than_max_ctx_works(self):
        """T < max_ctx (e.g. mid-batch trim) — same shape contract."""
        B, dim, n_heads, n_kv = 2, 64, 4, 2
        T, max_ctx = 12, 16
        x = torch.randn(B, T, dim)
        qw, kvw, ow = self._weights(dim, n_heads, n_kv)
        y = nn_ops.causal_self_attention(x, qw, kvw, ow, n_heads, n_kv, max_ctx)
        assert y.shape == (B, T, dim)

    def test_T_greater_than_max_ctx_no_shape_mismatch(self):
        """REGRESSION: T > max_ctx must not crash (length extrapolation)."""
        B, dim, n_heads, n_kv = 2, 64, 4, 2
        # Mirror the production failure: T=358, max_ctx=128 (≈3× larger)
        max_ctx, T = 16, 48
        x = torch.randn(B, T, dim)
        qw, kvw, ow = self._weights(dim, n_heads, n_kv)
        y = nn_ops.causal_self_attention(x, qw, kvw, ow, n_heads, n_kv, max_ctx)
        # Output must preserve the input sequence length, not get
        # silently truncated to max_ctx.
        assert y.shape == (B, T, dim), (
            f"expected (B={B}, T={T}, dim={dim}), got {tuple(y.shape)} — "
            "RoPE cache was sliced to max_ctx and lost positions"
        )

    def test_T_three_times_max_ctx_matches_gif_probe_scenario(self):
        """The actual production failure ratio (T=358 / max_ctx=128 ≈ 2.8)."""
        B, dim, n_heads, n_kv = 1, 64, 4, 2
        max_ctx = 16
        T = 48  # 3× max_ctx
        x = torch.randn(B, T, dim)
        qw, kvw, ow = self._weights(dim, n_heads, n_kv)
        # Should not raise.
        y = nn_ops.causal_self_attention(x, qw, kvw, ow, n_heads, n_kv, max_ctx)
        assert y.shape == (B, T, dim)
        # Sanity: output is finite (RoPE didn't explode at the long
        # positions; the inv_freq values are well-behaved).
        assert torch.isfinite(y).all()


class TestDifferentialAttentionLengthHandling:
    """Regression: differential_attention has the SAME RoPE bug pattern.

    The first deploy (commit b6bb7ed) only fixed ``causal_self_attention``
    but the SmolLM arch uses ``DiffBlock`` for half its layers, so the
    GIF OOD probe crashed at the same step 80 with the same shape error
    in ``differential_attention`` instead. Fix MUST mirror the
    causal-self-attention pattern across every RoPE call site.
    """

    def _weights(self, dim, n_heads, n_kv):
        head_dim = dim // n_heads
        return (
            torch.randn(n_heads * head_dim, dim) * 0.02,
            torch.randn(2 * n_kv * head_dim, dim) * 0.02,
            torch.randn(dim, dim) * 0.02,
            torch.zeros(n_heads),                 # lambda_init
            torch.ones(head_dim),                 # sub_norm_weight (per-head_dim RMSNorm γ)
        )

    def test_T_equal_max_ctx_works(self):
        B, dim, n_heads, n_kv = 2, 64, 4, 2
        T = max_ctx = 16
        x = torch.randn(B, T, dim)
        qw, kvw, ow, lam, sn = self._weights(dim, n_heads, n_kv)
        y = nn_ops.differential_attention(
            x, qw, kvw, ow, lam, sn, n_heads, n_kv, max_ctx)
        assert y.shape == (B, T, dim)

    def test_T_greater_than_max_ctx_no_shape_mismatch(self):
        """REGRESSION: T > max_ctx in the differential path."""
        B, dim, n_heads, n_kv = 2, 64, 4, 2
        max_ctx, T = 16, 48
        x = torch.randn(B, T, dim)
        qw, kvw, ow, lam, sn = self._weights(dim, n_heads, n_kv)
        y = nn_ops.differential_attention(
            x, qw, kvw, ow, lam, sn, n_heads, n_kv, max_ctx)
        assert y.shape == (B, T, dim), (
            f"differential_attention sliced RoPE cache to max_ctx — "
            f"expected (B={B}, T={T}, dim={dim}), got {tuple(y.shape)}"
        )
        assert torch.isfinite(y).all()


class TestTonnetzAttentionLengthHandling:
    """Same regression for the Tonnetz-masked variant
    (``causal_self_attention_tonnetz``).
    """

    def _weights(self, dim, n_heads, n_kv):
        head_dim = dim // n_heads
        return (
            torch.randn(n_heads * head_dim, dim) * 0.02,
            torch.randn(2 * n_kv * head_dim, dim) * 0.02,
            torch.randn(dim, dim) * 0.02,
        )

    def test_T_greater_than_max_ctx_no_shape_mismatch(self):
        B, dim, n_heads, n_kv = 1, 64, 4, 2
        max_ctx, T = 16, 48
        x = torch.randn(B, T, dim)
        qw, kvw, ow = self._weights(dim, n_heads, n_kv)
        y = nn_ops.causal_self_attention_tonnetz(
            x, qw, kvw, ow, n_heads, n_kv, max_ctx,
            tonnetz_period=12)
        assert y.shape == (B, T, dim)
        assert torch.isfinite(y).all()


class TestLegacyNNModulesLengthHandling:
    """The legacy ``modules.common.CausalSelfAttention`` and
    ``modules.differential_attention.DifferentialAttention`` register
    RoPE buffers at ``__init__`` — must rebuild on-the-fly when T
    exceeds the cached size, OR training crashes the same way.
    """

    def test_causal_self_attention_T_greater_than_max_ctx(self):
        from neuroslm.modules.common import CausalSelfAttention
        attn = CausalSelfAttention(dim=64, n_heads=4, max_ctx=16,
                                   n_kv_heads=2)
        attn.eval()
        x = torch.randn(2, 48, 64)  # T=48 > max_ctx=16
        with torch.no_grad():
            y = attn(x)
        assert y.shape == (2, 48, 64)
        assert torch.isfinite(y).all()

    def test_differential_attention_T_greater_than_max_ctx(self):
        from neuroslm.modules.differential_attention import \
            DifferentialAttention
        attn = DifferentialAttention(
            dim=64, n_heads=4, max_ctx=16, n_kv_heads=2)
        attn.eval()
        x = torch.randn(2, 48, 64)
        with torch.no_grad():
            y = attn(x)
        assert y.shape == (2, 48, 64)
        assert torch.isfinite(y).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
