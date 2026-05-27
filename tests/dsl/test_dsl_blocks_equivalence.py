# -*- coding: utf-8 -*-
"""N8 keystone — Brain's cognitive blocks written in pure .neuro DSL,
bit-identical to their Python reference.

Per the user directive: 'DSLLM should support implementing these blocks
within the DSL itself ... declaratively or using ODE / algebra'. Each
block becomes a layer in pure DSL text; the exact-match test compiles
the DSL layer, syncs weights from Brain's reference module, and asserts
torch.allclose on the forward.

Starts with NeuralGeometryAdapter (the per-block adapter in
LanguageCortex). DiffTransformerBlock + MoDBlock follow once this proves
the DSL is expressive enough for these constructions.
"""
import pytest
import torch
import torch.nn as nn

from neuroslm.dsl.nn_lang import compile_layer
from neuroslm.dsl.nn_ops import swiglu_hidden_dim
from neuroslm.modules.language import NeuralGeometryAdapter


NEURAL_GEOMETRY_ADAPTER_DSL = '''
layer NeuralGeometryAdapter(D, Dhyper, R) {
    param gamma:   (D,) init=ones
    param Wup:     (Dhyper, D) init=xavier
    param kern_a:  (Dhyper, R) init=normal(0.01)
    param kern_b:  (R, Dhyper) init=normal(0.01)
    param Wgate:   (Dhyper, Dhyper) init=xavier
    param bgate:   (Dhyper,) init=constant(-2.0)
    param Wdown:   (D, Dhyper) init=zeros

    forward(x) {
        h     = rmsnorm(x, gamma)
        z     = linear(h, Wup)
        k     = matmul(matmul(z, kern_a), kern_b)
        g     = sigmoid(linear(z, Wgate, bgate))
        z_new = silu(k) * g
        out   = linear(z_new, Wdown)
        return x + out
    }
}
'''


class TestNeuralGeometryAdapterDSL:
    def test_dsl_matches_reference(self):
        d_hidden = 32
        ref = NeuralGeometryAdapter(d_hidden, expansion=2.0)
        ref.eval()

        Cls = compile_layer(NEURAL_GEOMETRY_ADAPTER_DSL)
        dsl = Cls(D=d_hidden, Dhyper=ref.d_hyper, R=ref.rank)
        dsl.eval()

        # Sync params: DSL → reference
        with torch.no_grad():
            dsl.gamma.copy_(ref.norm.weight)
            dsl.Wup.copy_(ref.up.weight)
            dsl.kern_a.copy_(ref.kern_a)
            dsl.kern_b.copy_(ref.kern_b)
            dsl.Wgate.copy_(ref.gate.weight)
            dsl.bgate.copy_(ref.gate.bias)
            dsl.Wdown.copy_(ref.down.weight)

        x = torch.randn(2, 16, d_hidden)
        with torch.no_grad():
            ref_out = ref(x)
            dsl_out = dsl(x)
        diff = (ref_out - dsl_out).abs().max().item()
        assert diff < 1e-5, f"DSL adapter diverged from reference (max diff {diff})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
