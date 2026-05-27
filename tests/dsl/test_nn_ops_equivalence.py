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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
