# -*- coding: utf-8 -*-
"""Tests for ODE-form dynamics (Phase 7 Stage 2).

An ODE dynamics declares the *rate of change* of a state variable rather
than a closed-form output:

    population p {
        ode: "tau * dV/dt = -V + x",
        params: { tau: 0.05 }       # optional — defaults come from decl
    }

Validates:
    1. Parser splits `coef * dvar/dt = rhs` and isolates dvar/dt
    2. ODE IR exposes state_var, normalized rhs, free symbols
    3. Codegen emits an Euler-step forward that matches reference math
    4. `integrate_and_fire` macro maps to a leaky-integrator ODE
    5. Fixed-point solver: dV/dt = 0 reduces to an algebraic root-find
    6. Jacobian-based stability classifier
"""
import pytest
import torch
import torch.nn as nn

from neuroslm.dsl.equations import (
    parse_ode,
    ODEExpr,
    DYNAMICS_DECLS,
    ode_fixed_point,
    ode_stable_at,
)
from neuroslm.dsl.compiler import NeuroMLCompiler
from neuroslm.dsl.codegen import CodeGenerator


# ── Parser ─────────────────────────────────────────────────────────────

class TestODEParser:
    def test_basic_form(self):
        ode = parse_ode("dy/dt = -y + x")
        assert ode.state_var == "y"
        # rhs should symbolically include -y + x
        s = str(ode.rhs)
        assert "y" in s and "x" in s

    def test_with_coefficient(self):
        # tau * dV/dt = -V + I   ⇒   dV/dt = (-V + I) / tau
        ode = parse_ode("tau * dV/dt = -V + I")
        assert ode.state_var == "V"
        # rhs should reference 1/tau (or tau in a denominator)
        s = str(ode.rhs)
        assert "tau" in s

    def test_state_var_extraction(self):
        for src, expected in [
            ("dV/dt = -V",     "V"),
            ("dy/dt = -y + x", "y"),
            ("ds/dt = s * x",  "s"),
        ]:
            assert parse_ode(src).state_var == expected

    def test_rejects_missing_derivative(self):
        with pytest.raises(ValueError):
            parse_ode("y = -y + x")   # algebraic, not ODE

    def test_rejects_unsupported_derivative(self):
        with pytest.raises(ValueError):
            parse_ode("d2y/dt2 = -y")  # second-order not yet supported

    def test_free_symbols(self):
        ode = parse_ode("tau * dV/dt = -(V - V_rest) + R * I")
        assert {"V", "V_rest", "R", "I", "tau"} <= set(ode.free_symbols)


# ── Macro: integrate_and_fire ──────────────────────────────────────────

class TestIntegrateAndFireMacro:
    def test_decl_now_populated(self):
        decl = DYNAMICS_DECLS["integrate_and_fire"]
        assert decl is not None
        # Should be an ODE form (uses dV/dt or similar)
        assert decl.is_ode
        assert "dt" in decl.ode

    def test_decl_declares_state_and_constants(self):
        decl = DYNAMICS_DECLS["integrate_and_fire"]
        # state must include the membrane potential
        assert "V" in decl.state
        # tau and dt should be in constants for the Euler step
        assert "tau" in decl.constants
        assert "dt" in decl.constants


# ── Numerical integration via codegen ─────────────────────────────────

class TestODECodegen:
    def _single_pop(self, name, ode):
        return f'''
            population {name} {{
                count: 256,
                ode: "{ode}"
            }}
        '''

    def test_codegen_compiles(self):
        src = self._single_pop("p", "dV/dt = (-V + x) / 0.05")
        ir = NeuroMLCompiler.compile(src)
        Cls = CodeGenerator(ir, module_name="ODECircuit").compile_to_module()
        circuit = Cls(d_sem=64)
        # forward should run without crashing
        x = torch.randn(2, 64)
        out = circuit(x)
        assert out["p"].shape == (2, 64)

    def test_euler_step_matches_reference(self):
        # Leaky integrator: dV/dt = (-V + x) / tau
        # Euler:  V_new = V + dt * (-V + x) / tau
        tau, dt = 0.05, 0.01
        src = self._single_pop("p", f"dV/dt = (-V + x) / {tau}")
        ir = NeuroMLCompiler.compile(src)
        Cls = CodeGenerator(ir, module_name="LeakyCircuit").compile_to_module()
        circuit = Cls(d_sem=32)

        # Reference: hand-written Euler step
        torch.manual_seed(0)
        x = torch.randn(2, 32)
        V = circuit.p.V.clone()  # initial state — should be zeros
        V_ref = V + dt * (-V + x) / tau

        out = circuit(x)
        assert torch.allclose(out["p"], V_ref, atol=1e-6)

    def test_integrate_and_fire_macro(self):
        # The integrate_and_fire enum should produce a working module.
        src = '''
            population neuron {
                count: 64,
                dynamics: "integrate_and_fire"
            }
        '''
        ir = NeuroMLCompiler.compile(src)
        Cls = CodeGenerator(ir, module_name="LIFCircuit").compile_to_module()
        circuit = Cls(d_sem=64)
        x = torch.randn(2, 64)
        out = circuit(x)
        assert out["neuron"].shape == (2, 64)
        assert not torch.isnan(out["neuron"]).any()


# ── Fixed-point & stability analysis ───────────────────────────────────

class TestODEAnalysis:
    def test_leaky_fixed_point(self):
        # dV/dt = -V + I  ⇒  V* = I  (when I treated as constant)
        ode = parse_ode("dV/dt = -V + I")
        fp = ode_fixed_point(ode, param_bindings={"I": 1.5})
        assert fp is not None
        assert abs(fp - 1.5) < 1e-6

    def test_leaky_is_stable(self):
        # dV/dt = -V at V=0 has Jacobian -1 → stable (|λ| < 1)
        ode = parse_ode("dV/dt = -V")
        stable = ode_stable_at(ode, point=0.0)
        assert stable is True

    def test_amplifier_is_unstable(self):
        # dV/dt = V at V=0 has Jacobian +1 → marginally unstable
        # dV/dt = 2V → eigenvalue 2 → strictly unstable
        ode = parse_ode("dV/dt = 2 * V")
        stable = ode_stable_at(ode, point=0.0)
        assert stable is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
