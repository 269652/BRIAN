# -*- coding: utf-8 -*-
"""Compile .neuro equations to Wolfram Alpha / Mathematica syntax.

The DSL equations (algebraic + ODE) lower to symbolic SymPy expressions;
this compiler emits them as Wolfram-language code that can be pasted into
Wolfram Alpha and solved / DSolved against any variable. Nonlinearities
map to Wolfram builtins: ReLU→Max[0,x], sigmoid→1/(1+E^-x), tanh→Tanh,
softmax→Exp[x]/Total[Exp[x]].
"""
import pytest

from neuroslm.dsl import wolfram as W


# ── Algebraic equations ────────────────────────────────────────────────

class TestEquationToWolfram:
    def test_relu(self):
        out = W.equation_to_wolfram("y = ReLU(x)")
        assert out == "y == Max[0, x]"

    def test_linear_affine(self):
        out = W.equation_to_wolfram("y = W * x + b")
        # Wolfram uses no '*' between symbols by default; accept either.
        assert "==" in out and "b" in out and "x" in out

    def test_sigmoid_closed_form(self):
        out = W.equation_to_wolfram("y = sigmoid(x)")
        # 1/(1 + E^(-x)) in Wolfram syntax
        assert "E^" in out or "Exp" in out
        assert "==" in out

    def test_tanh(self):
        out = W.equation_to_wolfram("y = tanh(x)")
        assert "Tanh[" in out


# ── ODEs ───────────────────────────────────────────────────────────────

class TestODEToWolfram:
    def test_leaky_integrator(self):
        out = W.ode_to_wolfram("dV/dt = -V + x")
        # State var becomes V[t], derivative V'[t]
        assert "V'[t]" in out
        assert "V[t]" in out
        assert "==" in out

    def test_with_coefficient_normalized(self):
        out = W.ode_to_wolfram("tau * dV/dt = -V + x")
        # Normalized form divides by tau
        assert "V'[t]" in out
        assert "tau" in out

    def test_dsolve_wrapper(self):
        out = W.ode_to_wolfram("dV/dt = -V + x", dsolve=True)
        assert out.startswith("DSolve[")
        assert "V[t]" in out and ", t]" in out


# ── Fixed-point solving ────────────────────────────────────────────────

class TestFixedPointWolfram:
    def test_algebraic_fixed_point(self):
        # x = f(x): solve f(x) - x == 0
        out = W.solve_fixed_point_wolfram("y = 0.5 - x", input_symbol="x")
        assert out.startswith("Solve[")
        assert "x]" in out

    def test_ode_steady_state(self):
        # dV/dt = 0 → Solve[rhs == 0, V]
        out = W.solve_fixed_point_wolfram("dV/dt = -V + I", is_ode=True)
        assert out.startswith("Solve[")
        assert "V]" in out


# ── Whole-architecture system ──────────────────────────────────────────

class TestArchitectureToWolfram:
    def test_emits_system_from_folder(self, tmp_path):
        # Minimal architecture folder (arch.neuro imports the populations,
        # matching the real rcc_bowtie module convention).
        (tmp_path / "arch.neuro").write_text(
            'architecture x { d_sem: 4 }\n'
            'import { a, b } from "@/p"', encoding="utf-8")
        (tmp_path / "p.neuro").write_text(
            'export population a { count: 4, equation: "y = ReLU(x)" }\n'
            'export population b { count: 4, equation: "y = tanh(x)" }',
            encoding="utf-8")

        sys_code = W.architecture_to_wolfram(tmp_path)
        # A Wolfram list of equations, one per population with an equation
        assert "Max[0, x]" in sys_code
        assert "Tanh[" in sys_code
        # Wrapped as a Wolfram list {...}
        assert sys_code.strip().startswith("{") and sys_code.strip().endswith("}")


class TestArchitectureFullIITGrade:
    """IIT-grade Wolfram emission: populations + synapses + modulations
    + NT homeostatic ODEs as one Wolfram Association — every causal
    element of the architecture in a single CAS-analyzable system."""

    def test_emits_all_four_sections_for_rcc_bowtie(self):
        import os
        arch_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "architectures", "master")
        if not os.path.isdir(arch_root):
            pytest.skip("master arch not present in this checkout")
        code = W.architecture_to_wolfram_full(arch_root)
        for sec in ("Populations", "Synapses", "Modulations",
                    "NeurotransmitterDynamics"):
            assert f'"{sec}"' in code, f"missing section {sec}"
        assert code.startswith("<|") and code.endswith("|>")

    def test_section_flags_slice_output(self):
        import os
        arch_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "architectures", "master")
        if not os.path.isdir(arch_root):
            pytest.skip("master arch not present")
        code = W.architecture_to_wolfram_full(
            arch_root, include_populations=False, include_synapses=True,
            include_modulations=False, include_nt_dynamics=False)
        assert '"Synapses"' in code
        assert '"Populations"' not in code
        assert '"NeurotransmitterDynamics"' not in code

    def test_nt_dynamics_ode_form(self):
        """NT entries emitted as proper Wolfram ODEs: `c_<nt>'[t] == ...`."""
        import os
        arch_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "architectures", "master")
        if not os.path.isdir(arch_root):
            pytest.skip("master arch not present")
        code = W.architecture_to_wolfram_full(
            arch_root, include_populations=False, include_synapses=False,
            include_modulations=False, include_nt_dynamics=True)
        for nt in ("dopamine", "norepinephrine", "serotonin",
                   "acetylcholine", "endocannabinoid", "glutamate", "gaba"):
            assert f"c_{nt}'[t]" in code, f"NT ODE for {nt} missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
