# -*- coding: utf-8 -*-
"""Full TransformerBlock (incl. attention) compiles to NGL, byte-equivalent.

Attention mixes tensor args (x, Wq, Wkv, Wo) with scalar config (n_heads, …).
NGL instructions carry an optional ``config`` tuple so the whole mechanic lowers
to a single opaque node — which means the block is compilable, simplifiable
(residuals around the opaque op) and evolvable, not just the FFN.
"""
import torch

from neuroslm.dsl.nn_lang import compile_layer, TRANSFORMER_BLOCK_DSL
from neuroslm.genetic.compile_arch import compile_layer_to_ngl, run_compiled
from neuroslm.genetic.simplify import simplify
from neuroslm.genetic.language import REGISTRY


_BINDINGS = dict(D=16, n_heads=4, n_kv_heads=2, max_ctx=16, H=32, Dkv=16, rope_base=10000.0)


class TestConfigOpRegistered:
    def test_attention_is_a_config_op(self):
        assert "causal_self_attention" in REGISTRY
        assert REGISTRY["causal_self_attention"].uses_config


class TestFullBlockEquivalence:
    def test_transformer_block_matches_reference(self):
        Cls = compile_layer(TRANSFORMER_BLOCK_DSL)
        ref = Cls(**_BINDINGS)
        ref.eval()
        x = torch.randn(2, 8, 16)
        with torch.no_grad():
            expected = ref(x)

        compiled = compile_layer_to_ngl(TRANSFORMER_BLOCK_DSL, bindings=_BINDINGS)
        params = {name: p.detach() for name, p in ref.named_parameters()}
        got = run_compiled(compiled, params, {"x": x})
        assert got.shape == expected.shape
        assert torch.allclose(got, expected, atol=1e-5), (got - expected).abs().max()

    def test_block_config_is_captured(self):
        compiled = compile_layer_to_ngl(TRANSFORMER_BLOCK_DSL, bindings=_BINDINGS)
        attn = [i for i in compiled.program.instructions if i.op == "causal_self_attention"]
        assert len(attn) == 1
        cfg = dict(attn[0].config)
        assert cfg["n_heads"] == 4
        assert cfg["n_kv_heads"] == 2
        assert cfg["max_ctx"] == 16


class TestFullBlockSimplifiable:
    def test_simplify_preserves_block_on_real_probes(self):
        # verify simplification with SHAPE-CORRECT random params/inputs, not the
        # degenerate all-zero generic probes
        from neuroslm.genetic.compile_arch import make_probes
        compiled = compile_layer_to_ngl(TRANSFORMER_BLOCK_DSL, bindings=_BINDINGS)
        probes = make_probes(compiled, _BINDINGS, batch=2, seq=8, n=3, seed=0)
        slim = simplify(compiled.program, n_probes=3, seed=0, probes=probes)
        # attention op must survive (opaque, not deletable)
        assert any(i.op == "causal_self_attention" for i in slim.instructions)
        # behaviour preserved on the real probes
        from neuroslm.genetic.simplify import programs_equivalent
        assert programs_equivalent(compiled.program, slim, probes=probes)
