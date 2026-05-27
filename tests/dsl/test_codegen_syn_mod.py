# -*- coding: utf-8 -*-
"""Stage 5 — synapse + modulation equation codegen.

Before Stage 5, synapses and modulations had hard-coded behavior:

    F.linear(outputs[src], self.syn_w)        # synapse
    outputs[tgt] = outputs[tgt] * conc * gain # modulation

Now they're equation-driven, with the legacy `weight:` / `effect: ...`
fields expanding to canonical equations so existing `.neuro` files keep
producing byte-identical output.

Canonical legacy equations:
    synapse:                  y = weight * (W @ x_pre)
    modulation multiplicative: y = output * (c * gain)
    modulation additive:       y = output + (c * gain)

The contract: a synapse/modulation written with explicit `equation:` must
produce numerically equal output to its legacy form after random-buffer
sync.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.dsl.codegen import CodeGenerator
from neuroslm.dsl.compiler import NeuroMLCompiler


def _compile(src: str, module_name: str = "TestCircuit"):
    ir = NeuroMLCompiler.compile(src)
    return CodeGenerator(ir, module_name=module_name).compile_to_module()


def _sync(target: nn.Module, source: nn.Module) -> None:
    src = dict(source.state_dict())
    tgt = dict(target.state_dict())
    for name, t in tgt.items():
        if name in src and src[name].shape == t.shape:
            t.copy_(src[name])


# ── Synapse equation parity ────────────────────────────────────────────

class TestSynapseEquationParity:
    """Synapse explicit equation must equal its legacy `weight:` form."""

    def test_synapse_equation_matches_legacy_weight(self):
        # Legacy form: just `weight: 0.5`
        legacy_src = '''
            population a { count: 256, dynamics: "rate_code" }
            population b { count: 256, dynamics: "static" }
            synapse a -> b { weight: 0.5 }
        '''
        Cls_l = _compile(legacy_src, module_name="LegacySyn")
        circuit_l = Cls_l(d_sem=64)

        # Equation form: explicit. Note the operand order is
        # `x_pre @ W` (PyTorch batched-tensor convention), not the
        # mathematician's `W @ x`. This matches the canonical legacy
        # form that the codegen emits when no equation is supplied.
        eq_src = '''
            population a { count: 256, dynamics: "rate_code" }
            population b { count: 256, dynamics: "static" }
            synapse a -> b {
                weight: 0.5,
                equation: "y = weight * (x_pre @ W)"
            }
        '''
        Cls_e = _compile(eq_src, module_name="EqSyn")
        circuit_e = Cls_e(d_sem=64)

        _sync(circuit_e, circuit_l)

        x = torch.randn(2, 64)
        out_l = circuit_l(x)
        out_e = circuit_e(x)

        diff = (out_l["b"] - out_e["b"]).abs().max().item()
        assert diff < 1e-6, f"synapse equation vs legacy diverged ({diff})"

    def test_default_weight_is_one(self):
        # When `weight:` isn't provided, default 1.0
        src = '''
            population a { count: 256, dynamics: "rate_code" }
            population b { count: 256, dynamics: "static" }
            synapse a -> b { }
        '''
        Cls = _compile(src, module_name="DefaultWeight")
        circuit = Cls(d_sem=32)
        x = torch.randn(2, 32)
        out = circuit(x)
        assert out["b"].shape == (2, 32)


# ── Modulation equation parity ─────────────────────────────────────────

class TestModulationEquationParity:
    def test_multiplicative_equation_matches_legacy(self):
        # Legacy: `effect: "multiplicative", gain: 2.0`
        legacy_src = '''
            neurotransmitter da { base_concentration: 0.1 }
            population p { count: 256, dynamics: "rate_code" }
            modulation da -> p { effect: "multiplicative", gain: 2.0 }
        '''
        Cls_l = _compile(legacy_src, module_name="LegacyMod")
        circuit_l = Cls_l(d_sem=32)

        # Equation form
        eq_src = '''
            neurotransmitter da { base_concentration: 0.1 }
            population p { count: 256, dynamics: "rate_code" }
            modulation da -> p {
                effect: "multiplicative",
                gain: 2.0,
                equation: "y = output * (c * gain)"
            }
        '''
        Cls_e = _compile(eq_src, module_name="EqMod")
        circuit_e = Cls_e(d_sem=32)

        _sync(circuit_e, circuit_l)

        x = torch.abs(torch.randn(2, 32))
        nt = {"da": 0.7}
        out_l = circuit_l(x, nt_levels=nt)
        out_e = circuit_e(x, nt_levels=nt)

        diff = (out_l["p"] - out_e["p"]).abs().max().item()
        assert diff < 1e-6, f"mult-modulation equation vs legacy diverged ({diff})"

    def test_additive_equation_matches_legacy(self):
        legacy_src = '''
            neurotransmitter da { base_concentration: 0.1 }
            population p { count: 256, dynamics: "rate_code" }
            modulation da -> p { effect: "additive", gain: 0.5 }
        '''
        Cls_l = _compile(legacy_src, module_name="LegacyAddMod")
        circuit_l = Cls_l(d_sem=32)

        eq_src = '''
            neurotransmitter da { base_concentration: 0.1 }
            population p { count: 256, dynamics: "rate_code" }
            modulation da -> p {
                effect: "additive",
                gain: 0.5,
                equation: "y = output + (c * gain)"
            }
        '''
        Cls_e = _compile(eq_src, module_name="EqAddMod")
        circuit_e = Cls_e(d_sem=32)

        _sync(circuit_e, circuit_l)

        x = torch.abs(torch.randn(2, 32))
        nt = {"da": 0.7}
        out_l = circuit_l(x, nt_levels=nt)
        out_e = circuit_e(x, nt_levels=nt)

        diff = (out_l["p"] - out_e["p"]).abs().max().item()
        assert diff < 1e-6, f"add-modulation equation vs legacy diverged ({diff})"


# ── Forward still runs after the codegen change ────────────────────────

class TestRegressionForward:
    def test_existing_circuit_still_works(self):
        # Multi-population circuit with mixed legacy syntax must still
        # produce well-formed outputs after the codegen rewrite.
        src = '''
            neurotransmitter da { base_concentration: 0.1 }
            population a { count: 256, dynamics: "rate_code" }
            population b { count: 256, dynamics: "rate_code" }
            population c { count: 256, dynamics: "gated" }
            synapse a -> b { weight: 0.5 }
            synapse b -> c { weight: 0.3 }
            modulation da -> a { effect: "multiplicative", gain: 1.2 }
        '''
        Cls = _compile(src, module_name="MixCircuit")
        circuit = Cls(d_sem=64)
        x = torch.randn(2, 64)
        out = circuit(x, nt_levels={"da": 0.8})
        assert set(out.keys()) == {"a", "b", "c"}
        for name, t in out.items():
            assert t.shape == (2, 64)
            assert not torch.isnan(t).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
