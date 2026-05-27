# -*- coding: utf-8 -*-
"""Tests for DSL → PyTorch code generation (Phase 5 + Phase 7 fused).

The codegen turns a compiled `.neuro` DSL program into an `nn.Module`
whose forward pass implements every population's algebraic equation.
The same circuit must produce identical numerical output whether the
populations are specified via:

  * Enum macro          dynamics: "rate_code"
  * Explicit equation   equation: "y = ReLU(x)"
  * Reference impl      hand-written PyTorch module

This file pins down that three-way equivalence with `torch.allclose`.
Each dynamics type gets its own equivalence test before any compositional
test runs.
"""
import ast
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.dsl.codegen import CodeGenerator
from neuroslm.dsl.compiler import NeuroMLCompiler


# ── Reference implementations ──────────────────────────────────────────
#
# These are the "ground truth" — what each dynamics should compute. The
# generated module must match them within float tolerance. When Stage 2
# adds ODE dynamics, integrate_and_fire will get its reference here too.

class RefRateCode(nn.Module):
    def __init__(self, d_sem):
        super().__init__()
        self.d_sem = d_sem
    def forward(self, x):
        if x is None:
            x = torch.zeros(1, self.d_sem)
        return F.relu(x)


class RefWinnerTakeAll(nn.Module):
    def __init__(self, d_sem):
        super().__init__()
        self.d_sem = d_sem
    def forward(self, x):
        if x is None:
            x = torch.zeros(1, self.d_sem)
        return F.softmax(x / 0.1, dim=-1) * self.d_sem


class RefGated(nn.Module):
    def __init__(self, d_sem):
        super().__init__()
        self.d_sem = d_sem
        self.gate = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        if x is None:
            x = torch.zeros(1, self.d_sem)
        return F.relu(x) * torch.sigmoid(self.gate)


class RefAttractor(nn.Module):
    def __init__(self, d_sem):
        super().__init__()
        self.d_sem = d_sem
        self.register_buffer("s", torch.zeros(1, d_sem))
    def forward(self, x):
        if x is None:
            x = torch.zeros(1, self.d_sem)
        alpha = 0.1
        return (1 - alpha) * self.s + alpha * F.relu(x)


class RefAttentionPool(nn.Module):
    def __init__(self, d_sem):
        super().__init__()
        self.d_sem = d_sem
    def forward(self, x):
        if x is None:
            x = torch.zeros(1, self.d_sem)
        return F.softmax(x, dim=-1) * F.relu(x)


class RefStatic(nn.Module):
    def __init__(self, d_sem):
        super().__init__()
        self.d_sem = d_sem
    def forward(self, x):
        if x is None:
            x = torch.zeros(1, self.d_sem)
        return x


REFERENCES = {
    "rate_code":          RefRateCode,
    "winner_take_all":    RefWinnerTakeAll,
    "gated":              RefGated,
    "attractor_network":  RefAttractor,
    "attention_pool":     RefAttentionPool,
    "static":             RefStatic,
}


# ── Helpers ────────────────────────────────────────────────────────────

def _one_pop_circuit(name: str, dynamics: str = None, equation: str = None) -> str:
    """Single-population DSL source for isolation tests."""
    fields = ["count: 256"]
    if dynamics:
        fields.append(f'dynamics: "{dynamics}"')
    if equation:
        fields.append(f'equation: "{equation}"')
    body = ",\n    ".join(fields)
    return f"population {name} {{\n    {body}\n}}"


def _compile_circuit(source: str, module_name: str = "TestCircuit"):
    """DSL string → compiled nn.Module class."""
    ir = NeuroMLCompiler.compile(source)
    gen = CodeGenerator(ir, module_name=module_name)
    return gen.compile_to_module(), gen


# ── Code generation basics ─────────────────────────────────────────────

class TestCodeGenSyntax:
    def test_generates_valid_python(self):
        ir = NeuroMLCompiler.compile(_one_pop_circuit("test_pop", "rate_code"))
        gen = CodeGenerator(ir, module_name="SmokeCircuit")
        src = gen.generate()
        ast.parse(src)  # must not raise

    def test_includes_class_definition(self):
        ir = NeuroMLCompiler.compile(_one_pop_circuit("test_pop", "rate_code"))
        src = CodeGenerator(ir, module_name="SmokeCircuit").generate()
        assert "class SmokeCircuit" in src
        assert "def __init__" in src
        assert "def forward" in src

    def test_compile_returns_nn_module_class(self):
        Cls, _ = _compile_circuit(_one_pop_circuit("test_pop", "rate_code"),
                                  module_name="CompileTest")
        assert isinstance(Cls, type)
        assert issubclass(Cls, nn.Module)


# ── Per-dynamics semantic equivalence (macro path) ────────────────────
#
# Parametrized over every algebraic dynamics: for each, generate the
# circuit from the enum, instantiate, copy any learnable params from the
# reference module, and assert equal output.

class TestMacroSemanticEquivalence:
    @pytest.mark.parametrize("dynamics", list(REFERENCES.keys()))
    def test_macro_matches_reference(self, dynamics):
        Cls, _ = _compile_circuit(_one_pop_circuit("p", dynamics))
        gen_circuit = Cls(d_sem=256)
        ref = REFERENCES[dynamics](d_sem=256)

        # Sync learnable params and state buffers so values match
        _sync_state(gen_circuit.p, ref)

        torch.manual_seed(0)
        x = torch.randn(2, 256)
        gen_out = gen_circuit.p(x)
        ref_out = ref(x)
        assert torch.allclose(gen_out, ref_out, atol=1e-6), \
            f"{dynamics}: gen vs ref mismatch (max diff {(gen_out - ref_out).abs().max()})"


# ── Per-dynamics explicit-equation parity ─────────────────────────────
#
# For each macro, also assert that writing the canonical equation
# explicitly produces the same module as using the enum. This is the
# core "math-first DSL" guarantee.

class TestEquationVsMacroParity:
    from neuroslm.dsl.equations import DYNAMICS_EQUATIONS

    @pytest.mark.parametrize("dynamics,equation",
                             [(d, e) for d, e in DYNAMICS_EQUATIONS.items() if e is not None])
    def test_explicit_equation_matches_macro(self, dynamics, equation):
        # Macro form
        Cls_m, _ = _compile_circuit(_one_pop_circuit("p", dynamics=dynamics))
        m_circuit = Cls_m(d_sem=256)

        # Explicit equation form
        Cls_e, _ = _compile_circuit(_one_pop_circuit("p", equation=equation))
        e_circuit = Cls_e(d_sem=256)

        # Both should declare the same params/state — sync to make sure
        _sync_state(e_circuit.p, m_circuit.p)

        torch.manual_seed(0)
        x = torch.randn(2, 256)
        m_out = m_circuit.p(x)
        e_out = e_circuit.p(x)
        assert torch.allclose(m_out, e_out, atol=1e-6), \
            f"{dynamics}: macro vs explicit-equation mismatch"


# ── Forward pass (compositional) ──────────────────────────────────────

class TestCircuitForward:
    def test_forward_returns_dict_of_outputs(self):
        src = '''
            population a { count: 256, dynamics: "rate_code" }
            population b { count: 256, dynamics: "static" }
        '''
        Cls, _ = _compile_circuit(src, module_name="TwoPop")
        circuit = Cls(d_sem=256)
        x = torch.randn(2, 256)
        out = circuit(x)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"a", "b"}

    def test_forward_output_shapes(self):
        src = '''
            population p { count: 256, dynamics: "rate_code" }
        '''
        Cls, _ = _compile_circuit(src, module_name="ShapeCheck")
        circuit = Cls(d_sem=256)
        x = torch.randn(4, 256)
        out = circuit(x)
        assert out["p"].shape == (4, 256)

    def test_gradient_flow(self):
        src = '''
            population p { count: 256, dynamics: "rate_code" }
        '''
        Cls, _ = _compile_circuit(src, module_name="GradCheck")
        circuit = Cls(d_sem=256)
        x = torch.randn(2, 256, requires_grad=True)
        out = circuit(x)
        out["p"].sum().backward()
        assert x.grad is not None


# ── Synapse routing ───────────────────────────────────────────────────

class TestSynapseRouting:
    def test_synapse_passes_activation(self):
        # b receives a's output via a synapse — should be non-zero for nonneg input
        src = '''
            population a { count: 256, dynamics: "rate_code" }
            population b { count: 256, dynamics: "rate_code" }
            synapse a -> b { weight: 1.0 }
        '''
        Cls, _ = _compile_circuit(src, module_name="SynRoute")
        circuit = Cls(d_sem=256)
        x = torch.abs(torch.randn(2, 256))  # all positive
        out = circuit(x)
        # b's output should differ from a "pure relu(0)" — synapse delivered signal
        assert out["b"].abs().sum() > 0


# ── NT modulation ─────────────────────────────────────────────────────

class TestNTModulation:
    def test_modulation_changes_output(self):
        src = '''
            neurotransmitter dopamine { base_concentration: 0.1 }
            population p { count: 256, dynamics: "rate_code" }
            modulation dopamine -> p { effect: "multiplicative", gain: 2.0 }
        '''
        Cls, _ = _compile_circuit(src, module_name="ModCircuit")
        circuit = Cls(d_sem=256)
        x = torch.abs(torch.randn(2, 256))  # nonneg
        unmod = circuit(x)
        mod = circuit(x, nt_levels={"dopamine": 0.5})
        # Should differ (multiplied by 0.5 * 2.0 = 1.0 → actually equal! pick gain 3.0)
        # Try with clearly different scaling
        src2 = src.replace("gain: 2.0", "gain: 5.0")
        Cls2, _ = _compile_circuit(src2, module_name="ModCircuit2")
        circuit2 = Cls2(d_sem=256)
        unmod2 = circuit2(x)
        mod2 = circuit2(x, nt_levels={"dopamine": 1.0})
        assert not torch.allclose(unmod2["p"], mod2["p"])


# ── Reference parity helper ────────────────────────────────────────────

def _sync_state(target: nn.Module, source: nn.Module):
    """Copy parameters and buffers from `source` to `target`, by name."""
    src_state = dict(source.state_dict())
    tgt_state = dict(target.state_dict())
    for name in tgt_state:
        if name in src_state and tgt_state[name].shape == src_state[name].shape:
            tgt_state[name].copy_(src_state[name])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
