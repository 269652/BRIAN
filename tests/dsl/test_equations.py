# -*- coding: utf-8 -*-
"""Tests for the algebraic-equation DSL layer (Phase 7 Stage 1).

Validates:
    1. Parser turns equation strings into SymPy IR
    2. Enum dynamics expand to canonical equations
    3. Lowering produces valid torch-op Python that matches reference math
    4. Fixed-point solver finds roots when they exist
    5. Compiler picks up `equation:` field from .neuro source
"""
import pytest
import torch
import torch.nn.functional as F

from neuroslm.dsl.equations import (
    parse_equation,
    expand_dynamics_macro,
    equation_for_population,
    lower_to_torch,
    find_fixed_point,
    jacobian_at,
    DYNAMICS_EQUATIONS,
)
from neuroslm.dsl.compiler import NeuroMLCompiler


# ── Parser ─────────────────────────────────────────────────────────────

class TestParser:
    def test_basic_equation(self):
        e = parse_equation("y = ReLU(x)")
        assert e.lhs.name == "y"
        assert "x" in e.free_symbols
        assert "ReLU" in str(e.rhs)

    def test_matmul_rewrite(self):
        e = parse_equation("y = W @ x + b")
        # After rewrite, the SymPy expression should reference matmul
        assert "matmul" in str(e.rhs)
        assert {"W", "x", "b"} <= e.free_symbols

    def test_nested_matmul(self):
        e = parse_equation("y = U @ V @ x")
        # left-assoc: matmul(matmul(U, V), x)
        assert str(e.rhs).count("matmul") == 2

    def test_complex_expression(self):
        e = parse_equation("y = ReLU(W @ x + b) * sigmoid(g)")
        assert "ReLU" in str(e.rhs)
        assert "sigmoid" in str(e.rhs)
        assert {"W", "x", "b", "g"} <= e.free_symbols

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="must contain"):
            parse_equation("y ReLU(x)")

    def test_bad_lhs_raises(self):
        with pytest.raises(ValueError, match="LHS"):
            parse_equation("y + z = x")

    def test_max_with_comma(self):
        # Critical: SymPy uses `Max` (capital), and parens-aware parser must
        # not eat the comma. Lowercase `max` also routes to sp.Max via locals.
        e = parse_equation("y = Max(0, x)")
        assert {"x"} <= e.free_symbols


# ── Macro expansion ────────────────────────────────────────────────────

class TestMacroExpansion:
    def test_rate_code_macro(self):
        assert expand_dynamics_macro("rate_code") == "y = ReLU(x)"

    def test_unknown_dynamics(self):
        assert expand_dynamics_macro("does_not_exist") is None

    def test_stage2_placeholder_returns_none(self):
        # integrate_and_fire is an ODE — Stage 2, not Stage 1
        assert expand_dynamics_macro("integrate_and_fire") is None

    def test_equation_for_population_uses_explicit(self):
        # explicit equation wins over dynamics enum
        e = equation_for_population(dynamics="rate_code", equation="y = 2 * x")
        assert e is not None
        assert "2" in str(e.rhs)
        # ReLU should NOT appear — explicit took precedence
        assert "ReLU" not in str(e.rhs)

    def test_equation_for_population_falls_back_to_dynamics(self):
        e = equation_for_population(dynamics="rate_code", equation=None)
        assert e is not None
        assert "ReLU" in str(e.rhs)

    def test_equation_for_population_returns_none_when_unsupported(self):
        e = equation_for_population(dynamics="integrate_and_fire", equation=None)
        assert e is None

    def test_all_stage1_macros_parse(self):
        # Every non-None canonical equation must parse without error
        for name, eq in DYNAMICS_EQUATIONS.items():
            if eq is None:
                continue
            parsed = parse_equation(eq)
            assert parsed is not None, f"macro {name!r} failed to parse: {eq!r}"


# ── Lowering ───────────────────────────────────────────────────────────

class TestLowering:
    def _eval(self, src: str, scope: dict) -> torch.Tensor:
        """Compile + eval the lowered torch expression in a scope."""
        return eval(src, {"torch": torch, "F": F}, scope)

    def test_relu_lowering(self):
        e = parse_equation("y = ReLU(x)")
        src = lower_to_torch(e)
        x = torch.tensor([-1.0, 0.0, 2.5])
        y = self._eval(src, {"x": x})
        assert torch.allclose(y, F.relu(x))

    def test_matmul_lowering(self):
        e = parse_equation("y = W @ x + b")
        src = lower_to_torch(e)
        W = torch.randn(3, 4)
        x = torch.randn(4)
        b = torch.randn(3)
        y = self._eval(src, {"W": W, "x": x, "b": b})
        assert torch.allclose(y, W @ x + b)

    def test_sigmoid_lowering(self):
        e = parse_equation("y = sigmoid(x)")
        src = lower_to_torch(e)
        x = torch.tensor([-2.0, 0.0, 2.0])
        y = self._eval(src, {"x": x})
        assert torch.allclose(y, torch.sigmoid(x))

    def test_softmax_lowering(self):
        e = parse_equation("y = softmax(x)")
        src = lower_to_torch(e)
        x = torch.tensor([1.0, 2.0, 3.0])
        y = self._eval(src, {"x": x})
        assert torch.allclose(y, F.softmax(x, dim=-1))

    def test_composite_matches_manual(self):
        # y = ReLU(W @ x + b) * sigmoid(g)
        e = parse_equation("y = ReLU(W @ x + b) * sigmoid(g)")
        src = lower_to_torch(e)
        W = torch.randn(3, 4); x = torch.randn(4)
        b = torch.randn(3);    g = torch.randn(3)
        y = self._eval(src, {"W": W, "x": x, "b": b, "g": g})
        expected = F.relu(W @ x + b) * torch.sigmoid(g)
        assert torch.allclose(y, expected)

    def test_rate_code_macro_matches_template(self):
        # The canonical rate_code equation `y = ReLU(x)` must produce
        # the same output as the existing Phase-5 template (`F.relu(x)`).
        e = equation_for_population(dynamics="rate_code", equation=None)
        src = lower_to_torch(e)
        x = torch.randn(1, 256)
        y = self._eval(src, {"x": x})
        assert torch.allclose(y, F.relu(x))


# ── Fixed-point solver ─────────────────────────────────────────────────

class TestFixedPoint:
    def test_linear_fixpoint(self):
        # y = -x + 0.5  ⇒  x* = 0.25
        e = parse_equation("y = 0.5 - x")
        fp = find_fixed_point(e, input_symbol="x", guess=0.0)
        assert fp is not None
        assert abs(fp - 0.25) < 1e-6

    def test_sigmoid_fixpoint(self):
        # y = sigmoid(x); the unique fixed point ≈ 0.6590
        e = parse_equation("y = sigmoid(x)")
        fp = find_fixed_point(e, input_symbol="x", guess=0.5)
        assert fp is not None
        assert 0.65 < fp < 0.66

    def test_no_unique_fixpoint(self):
        # y = 2*x has only x* = 0; from guess 0 SymPy returns 0.
        e = parse_equation("y = 2 * x")
        fp = find_fixed_point(e, input_symbol="x", guess=0.0)
        # Either it returns 0 (the trivial fixed point) or None — both
        # outcomes are acceptable; the test is that it doesn't crash and
        # doesn't return a garbage non-zero value.
        if fp is not None:
            assert abs(fp) < 1e-6


# ── Stability via Jacobian ─────────────────────────────────────────────

class TestStability:
    def test_sigmoid_locally_stable_at_origin(self):
        # d/dx sigmoid(x) at x=0 is 0.25 — recurrence x ← sigmoid(x) is
        # locally contracting (|0.25| < 1) so the fixed point is stable.
        e = parse_equation("y = sigmoid(x)")
        slope = jacobian_at(e, "x", point=0.0)
        assert slope is not None
        assert abs(slope - 0.25) < 1e-6

    def test_linear_amplifier_unstable(self):
        # y = 2*x has |df/dx| = 2 > 1 — unstable.
        e = parse_equation("y = 2 * x")
        slope = jacobian_at(e, "x", point=0.0)
        assert slope == 2.0


# ── Compiler integration ───────────────────────────────────────────────

class TestCompilerIntegration:
    def test_population_with_equation_field(self):
        src = '''
            population test_pop {
                count: 128,
                dynamics: "rate_code",
                equation: "y = ReLU(W @ x + b)"
            }
        '''
        ir = NeuroMLCompiler.compile(src)
        assert len(ir.populations) == 1
        pop = ir.populations[0]
        assert pop.name == "test_pop"
        assert pop.equation == "y = ReLU(W @ x + b)"

    def test_population_without_equation(self):
        src = '''
            population legacy {
                count: 256,
                dynamics: "rate_code"
            }
        '''
        ir = NeuroMLCompiler.compile(src)
        pop = ir.populations[0]
        assert pop.equation is None
        assert pop.dynamics == "rate_code"

    def test_equation_with_comma_survives(self):
        # `Max(0, x)` has a comma — the property parser must not split on it.
        src = '''
            population gated {
                count: 64,
                equation: "y = Max(0, x)"
            }
        '''
        ir = NeuroMLCompiler.compile(src)
        pop = ir.populations[0]
        assert pop.equation == "y = Max(0, x)"
        # And it must still be parseable downstream
        parsed = parse_equation(pop.equation)
        assert "x" in parsed.free_symbols

    def test_synapse_equation(self):
        src = '''
            population a { count: 32 }
            population b { count: 32 }
            synapse a -> b {
                weight: 0.5,
                equation: "y = sigmoid(W @ x_pre)"
            }
        '''
        ir = NeuroMLCompiler.compile(src)
        assert len(ir.synapses) == 1
        assert ir.synapses[0].equation == "y = sigmoid(W @ x_pre)"

    def test_modulation_equation(self):
        src = '''
            neurotransmitter dopamine { base_concentration: 0.1 }
            population pfc { count: 64 }
            modulation dopamine -> pfc {
                effect: "multiplicative",
                gain: 1.5,
                equation: "gain = 1 + 0.5 * c"
            }
        '''
        ir = NeuroMLCompiler.compile(src)
        assert len(ir.modulations) == 1
        assert ir.modulations[0].equation == "gain = 1 + 0.5 * c"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
