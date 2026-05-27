# -*- coding: utf-8 -*-
"""Phase N3 — NeuroTensor language: layer/model parser → nn.Module.

Lets a layer be *written* in DSL text and compiled to a runnable
nn.Module whose forward calls the N1/N2 op atoms. Exact-match tested by
composing a layer from atoms and comparing to the hand-written reference
with synced weights.
"""
import pytest
import torch
import torch.nn as nn

from neuroslm.dsl.nn_lang import compile_layer
from neuroslm.modules.common import RMSNorm, SwiGLU, CausalSelfAttention


def _sync(dsl: nn.Module, ref: nn.Module):
    """Copy reference params into the DSL module by position (same order)."""
    dsl_params = list(dsl.parameters())
    ref_params = list(ref.parameters())
    assert len(dsl_params) == len(ref_params), \
        f"param count: dsl={len(dsl_params)} ref={len(ref_params)}"
    with torch.no_grad():
        for d, r in zip(dsl_params, ref_params):
            assert d.shape == r.shape, f"shape {d.shape} vs {r.shape}"
            d.copy_(r)


# ── Parse basics ───────────────────────────────────────────────────────

class TestParse:
    def test_compiles_to_nn_module_class(self):
        src = '''
        layer Norm(D) {
            param gamma: (D,) init=ones
            forward(x) {
                return rmsnorm(x, gamma)
            }
        }
        '''
        Cls = compile_layer(src)
        assert isinstance(Cls, type)
        assert issubclass(Cls, nn.Module)

    def test_param_allocated_with_correct_shape(self):
        src = '''
        layer Norm(D) {
            param gamma: (D,) init=ones
            forward(x) { return rmsnorm(x, gamma) }
        }
        '''
        m = compile_layer(src)(D=64)
        assert m.gamma.shape == (64,)


# ── RMSNorm layer — exact match ────────────────────────────────────────

class TestRMSNormLayer:
    def test_matches_reference(self):
        src = '''
        layer Norm(D) {
            param gamma: (D,) init=ones
            forward(x) { return rmsnorm(x, gamma) }
        }
        '''
        m = compile_layer(src)(D=64)
        ref = RMSNorm(64)
        _sync(m, ref)
        x = torch.randn(2, 16, 64)
        assert torch.allclose(m(x), ref(x), atol=1e-6)


# ── SwiGLU layer — exact match ─────────────────────────────────────────

class TestSwiGLULayer:
    def test_matches_reference(self):
        # SwiGLU hidden = round_to_8(D*8/3). Express via the helper the op
        # exposes — the DSL evaluates `swiglu_hidden(D)` for the shape.
        src = '''
        layer Mlp(D, H) {
            param w1: (H, D) init=xavier
            param w2: (H, D) init=xavier
            param w3: (D, H) init=xavier
            forward(x) { return swiglu(x, w1, w2, w3) }
        }
        '''
        from neuroslm.dsl.nn_ops import swiglu_hidden_dim
        D = 64
        H = swiglu_hidden_dim(D)
        m = compile_layer(src)(D=D, H=H)
        ref = SwiGLU(D)
        _sync(m, ref)
        x = torch.randn(2, 16, D)
        assert torch.allclose(m(x), ref(x), atol=1e-6)


# ── TransformerBlock — the real target ─────────────────────────────────

TRANSFORMER_BLOCK_SRC = '''
layer TransformerBlock(D, n_heads, n_kv_heads, max_ctx, H) {
    param gamma1: (D,) init=ones
    param Wq: (D, D) init=xavier
    param Wkv: (Dkv, D) init=xavier
    param Wo: (D, D) init=xavier
    param gamma2: (D,) init=ones
    param w1: (H, D) init=xavier
    param w2: (H, D) init=xavier
    param w3: (D, H) init=xavier

    forward(x) {
        a = causal_self_attention(rmsnorm(x, gamma1), Wq, Wkv, Wo, n_heads, n_kv_heads, max_ctx)
        x = x + a
        m = swiglu(rmsnorm(x, gamma2), w1, w2, w3)
        return x + m
    }
}
'''


class TestTransformerBlock:
    def test_matches_reference_block(self):
        from neuroslm.modules.common import TransformerBlock as RefBlock
        from neuroslm.dsl.nn_ops import swiglu_hidden_dim

        D, n_heads, max_ctx = 64, 8, 128
        H = swiglu_hidden_dim(D)
        Dkv = 2 * D  # n_kv_heads == n_heads → kv_proj is (2*D, D)

        ref = RefBlock(D, n_heads, max_ctx, n_nt=0, hebbian_rank=0, dropout=0.0)
        ref.eval()

        Cls = compile_layer(TRANSFORMER_BLOCK_SRC)
        m = Cls(D=D, n_heads=n_heads, n_kv_heads=n_heads, max_ctx=max_ctx, H=H, Dkv=Dkv)

        # Sync by name: map DSL params → reference params
        with torch.no_grad():
            m.gamma1.copy_(ref.n1.weight)
            m.Wq.copy_(ref.attn.q_proj.weight)
            m.Wkv.copy_(ref.attn.kv_proj.weight)
            m.Wo.copy_(ref.attn.out.weight)
            m.gamma2.copy_(ref.n2.weight)
            m.w1.copy_(ref.mlp.w1.weight)
            m.w2.copy_(ref.mlp.w2.weight)
            m.w3.copy_(ref.mlp.w3.weight)

        x = torch.randn(2, 16, D)
        with torch.no_grad():
            ref_out = ref(x)
            dsl_out = m(x)
        assert torch.allclose(dsl_out, ref_out, atol=1e-5), \
            f"max diff {(dsl_out - ref_out).abs().max().item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
