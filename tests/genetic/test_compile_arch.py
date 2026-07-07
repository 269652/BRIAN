# -*- coding: utf-8 -*-
"""Compile a DSL architecture (nn_lang forward graph) into an NGL program.

The contract is byte-equivalence: the NGL program, given the *same* parameters,
must produce the same forward output as the compiled nn_lang module. This is the
bridge that lets discovery / simplification run on the actual architecture, not
just toy optimizers.
"""
import torch

from neuroslm.dsl.nn_lang import compile_layer
from neuroslm.genetic.language import REGISTRY
from neuroslm.genetic.compile_arch import (
    compile_layer_to_ngl,
    run_compiled,
)


FFN_DSL = """
layer FFN(D, H) {
    param gamma: (D,) init=ones
    param w1: (H, D) init=xavier
    param w2: (H, D) init=xavier
    param w3: (D, H) init=xavier
    forward(x) {
        h = rmsnorm(x, gamma)
        m = swiglu(h, w1, w2, w3)
        return x + m
    }
}
"""

NORM_LINEAR_DSL = """
layer NormProj(D, O) {
    param gamma: (D,) init=ones
    param Wo: (O, D) init=xavier
    forward(x) {
        h = rmsnorm(x, gamma)
        return linear(h, Wo)
    }
}
"""


class TestCompositeOpsRegistered:
    def test_nn_ops_are_in_registry(self):
        for op in ("linear", "rmsnorm", "layernorm", "swiglu", "gelu"):
            assert op in REGISTRY, f"{op} missing from NGL registry"


class TestArchToNGLEquivalence:
    def test_ffn_block_matches_reference(self):
        Cls = compile_layer(FFN_DSL)
        ref = Cls(D=16, H=32)
        ref.eval()
        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            expected = ref(x)

        compiled = compile_layer_to_ngl(FFN_DSL)
        params = {name: p.detach() for name, p in ref.named_parameters()}
        got = run_compiled(compiled, params, {"x": x})
        assert got.shape == expected.shape
        assert torch.allclose(got, expected, atol=1e-6), (got - expected).abs().max()

    def test_norm_linear_matches_reference(self):
        Cls = compile_layer(NORM_LINEAR_DSL)
        ref = Cls(D=12, O=7)
        ref.eval()
        x = torch.randn(3, 4, 12)
        with torch.no_grad():
            expected = ref(x)

        compiled = compile_layer_to_ngl(NORM_LINEAR_DSL)
        params = {name: p.detach() for name, p in ref.named_parameters()}
        got = run_compiled(compiled, params, {"x": x})
        assert torch.allclose(got, expected, atol=1e-6), (got - expected).abs().max()

    def test_compiled_program_lists_params_and_inputs(self):
        compiled = compile_layer_to_ngl(FFN_DSL)
        assert set(compiled.param_regs) == {"gamma", "w1", "w2", "w3"}
        assert set(compiled.input_regs) == {"x"}
        # the program is a real NGL Program with instructions
        assert len(compiled.program.instructions) >= 3


class TestUnsupportedLoweringIsHonest:
    def test_attention_scalar_config_raises_clearly(self):
        # causal_self_attention mixes scalar config (n_heads, ...) with tensors;
        # the current lowering doesn't support that and must say so, not silently
        # miscompile.
        import pytest
        from neuroslm.dsl.nn_lang import TRANSFORMER_BLOCK_DSL
        from neuroslm.genetic.compile_arch import UnsupportedLowering

        with pytest.raises(UnsupportedLowering):
            compile_layer_to_ngl(TRANSFORMER_BLOCK_DSL)
