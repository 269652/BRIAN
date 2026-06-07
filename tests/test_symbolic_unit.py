# -*- coding: utf-8 -*-
"""TDD acceptance suite — `SymbolicHyperNeuron`.

Phase C of the Multi-Objective-Fitness work order:
    "C → A → B, volle Implementierung mit strikter TDD"

Headline goal (paraphrased from the work order):
    "Spezialisierte Hyper-Neuronen drücken ihre interne Logik in Form
     von expliziten mathematischen Gleichungen aus."

Each `SymbolicHyperNeuron` is a learnable layer that internally
chooses — via Gumbel-Softmax over a small operator bank — exactly
one binary operator `op_i` and two input features `(a_i, b_i)` per
output unit. At low temperature this hardens to a one-hot selection,
making the unit's computation extractable as a printable formula such
as ``"phi * surprise"`` or ``"exp(metabolic_demand)"``.

This is the **mathematical-invention primitive**: when the LM-loss
gradient pulls the operator distribution toward a particular formula,
that formula has been *discovered* by training, not hard-coded.

Test taxonomy:
    OperatorBank                — the closed-form binary ops + apply_all
    Construction                — shape contract, parameter registration
    Forward                     — shape, autograd, determinism in eval
    ExpressionExtraction        — readable strings honour feature names
    SparsityRegularization      — entropy-based loss for one-hot pressure
    TemperatureAnnealing        — Gumbel-Softmax tau control
"""
from __future__ import annotations

import pytest
import torch

from neuroslm.modules.symbolic_unit import (
    OperatorBank,
    SymbolicHyperNeuron,
)


# ──────────────────────────────────────────────────────────────────────
# Operator bank — the closed-form binary-op alphabet
# ──────────────────────────────────────────────────────────────────────

class TestOperatorBank:
    """The bank is the alphabet from which expressions are composed.
    It must support a small, numerically-stable, differentiable set."""

    def test_default_bank_contains_minimum_ops(self):
        bank = OperatorBank.default()
        # The minimal set every symbolic-regression library uses.
        for required in ("identity", "add", "sub", "mul"):
            assert required in bank.names, \
                f"default bank missing required op {required!r}"

    def test_default_bank_includes_nonlinear_op(self):
        """Without at least one non-linear op, the symbolic unit
        collapses to a linear projection and provides no novelty."""
        bank = OperatorBank.default()
        nonlinear = {"exp", "sin", "tanh", "cos"}
        assert nonlinear & set(bank.names), \
            f"default bank lacks any non-linear op (need one of {nonlinear})"

    def test_n_ops_matches_names_length(self):
        bank = OperatorBank.default()
        assert bank.n_ops == len(bank.names)

    def test_apply_all_output_shape(self):
        """`apply_all(x_a, x_b)` evaluates every op at every position
        and stacks along a new last axis."""
        bank = OperatorBank.default()
        x_a = torch.randn(3, 5)
        x_b = torch.randn(3, 5)
        out = bank.apply_all(x_a, x_b)
        assert out.shape == (3, 5, bank.n_ops)

    def test_add_operator_is_correct(self):
        bank = OperatorBank.default()
        x_a = torch.tensor([1.0, 2.0, 3.0])
        x_b = torch.tensor([4.0, 5.0, 6.0])
        out = bank.apply_all(x_a, x_b)
        idx = bank.names.index("add")
        assert torch.allclose(out[..., idx], torch.tensor([5.0, 7.0, 9.0]))

    def test_mul_operator_is_correct(self):
        bank = OperatorBank.default()
        x_a = torch.tensor([2.0, 3.0])
        x_b = torch.tensor([4.0, 5.0])
        out = bank.apply_all(x_a, x_b)
        idx = bank.names.index("mul")
        assert torch.allclose(out[..., idx], torch.tensor([8.0, 15.0]))

    def test_identity_returns_first_operand(self):
        bank = OperatorBank.default()
        x_a = torch.tensor([7.0, 8.0])
        x_b = torch.tensor([99.0, 99.0])  # must be ignored
        out = bank.apply_all(x_a, x_b)
        idx = bank.names.index("identity")
        assert torch.allclose(out[..., idx], x_a)

    def test_no_nan_or_inf_on_extreme_inputs(self):
        """Numerical stability is non-negotiable — symbolic units
        sit inside the autograd graph and a single NaN poisons the
        entire training step."""
        bank = OperatorBank.default()
        x_a = torch.tensor([1e6, -1e6, 0.0])
        x_b = torch.tensor([1e6, -1e6, 0.0])
        out = bank.apply_all(x_a, x_b)
        assert torch.isfinite(out).all(), \
            f"NaN/Inf in op outputs: {out}"

    def test_apply_all_is_differentiable(self):
        bank = OperatorBank.default()
        x_a = torch.randn(4, requires_grad=True)
        x_b = torch.randn(4, requires_grad=True)
        out = bank.apply_all(x_a, x_b).sum()
        out.backward()
        assert x_a.grad is not None
        assert x_b.grad is not None


# ──────────────────────────────────────────────────────────────────────
# Construction & parameter contract
# ──────────────────────────────────────────────────────────────────────

class TestSymbolicHyperNeuronConstruction:
    """Shape + parameter-registration contract."""

    def test_construct_with_minimum_args(self):
        unit = SymbolicHyperNeuron(n_units=4, n_features=8)
        assert unit.n_units == 4
        assert unit.n_features == 8

    def test_owns_an_operator_bank(self):
        unit = SymbolicHyperNeuron(n_units=2, n_features=4)
        assert isinstance(unit.operator_bank, OperatorBank)
        assert unit.operator_bank.n_ops > 0

    def test_registers_three_logit_tensors_per_unit(self):
        """Each unit needs: (a) input-A selection, (b) input-B selection,
        (c) operator selection. With n_units=4, n_features=8, default
        bank, the total parameter count must be ≥ 4*(8+8+n_ops)."""
        unit = SymbolicHyperNeuron(n_units=4, n_features=8)
        n_ops = unit.operator_bank.n_ops
        total = sum(p.numel() for p in unit.parameters())
        expected_min = 4 * (8 + 8 + n_ops)
        assert total >= expected_min, \
            f"expected ≥ {expected_min} params, got {total}"

    def test_logits_shapes(self):
        """Public attribute names that the rest of the codebase will
        introspect (esp. for telemetry and the expression extractor)."""
        unit = SymbolicHyperNeuron(n_units=3, n_features=5)
        n_ops = unit.operator_bank.n_ops
        assert unit.input_a_logits.shape == (3, 5)
        assert unit.input_b_logits.shape == (3, 5)
        assert unit.op_logits.shape == (3, n_ops)

    def test_custom_operator_bank(self):
        """A caller can hand-craft a minimal bank — used in unit tests
        and when the curriculum wants to restrict the search."""
        bank = OperatorBank(names=["identity", "add"],
                            ops=[lambda a, b: a, lambda a, b: a + b])
        unit = SymbolicHyperNeuron(n_units=2, n_features=3,
                                   operator_bank=bank)
        assert unit.operator_bank.n_ops == 2

    def test_construction_rejects_zero_features(self):
        with pytest.raises(ValueError, match="n_features"):
            SymbolicHyperNeuron(n_units=2, n_features=0)

    def test_construction_rejects_zero_units(self):
        with pytest.raises(ValueError, match="n_units"):
            SymbolicHyperNeuron(n_units=0, n_features=4)


# ──────────────────────────────────────────────────────────────────────
# Forward pass
# ──────────────────────────────────────────────────────────────────────

class TestSymbolicHyperNeuronForward:
    """Input `(..., n_features)` → output `(..., n_units)`."""

    def test_forward_2d_input(self):
        unit = SymbolicHyperNeuron(n_units=4, n_features=8)
        x = torch.randn(5, 8)
        y = unit(x)
        assert y.shape == (5, 4)

    def test_forward_3d_input(self):
        """The typical (B, T, D) trunk-residual shape must be preserved."""
        unit = SymbolicHyperNeuron(n_units=4, n_features=8)
        x = torch.randn(2, 16, 8)
        y = unit(x)
        assert y.shape == (2, 16, 4)

    def test_forward_output_is_finite(self):
        torch.manual_seed(0)
        unit = SymbolicHyperNeuron(n_units=4, n_features=8)
        x = torch.randn(2, 10, 8)
        y = unit(x)
        assert torch.isfinite(y).all()

    def test_forward_rejects_wrong_feature_count(self):
        unit = SymbolicHyperNeuron(n_units=4, n_features=8)
        x = torch.randn(2, 16, 7)  # wrong: 7 instead of 8
        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            unit(x)

    def test_forward_is_differentiable_through_inputs(self):
        unit = SymbolicHyperNeuron(n_units=2, n_features=4)
        x = torch.randn(1, 3, 4, requires_grad=True)
        loss = unit(x).sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_forward_is_differentiable_through_logits(self):
        """The Gumbel-Softmax must allow gradient through the operator
        and input-selection logits — that's how the *correct* formula
        gets discovered."""
        unit = SymbolicHyperNeuron(n_units=2, n_features=4)
        x = torch.randn(1, 3, 4)
        loss = unit(x).sum()
        loss.backward()
        assert unit.input_a_logits.grad is not None
        assert unit.input_b_logits.grad is not None
        assert unit.op_logits.grad is not None
        assert torch.isfinite(unit.op_logits.grad).all()

    def test_eval_mode_is_deterministic(self):
        """In eval mode, Gumbel noise is suppressed → reproducible forward."""
        torch.manual_seed(123)
        unit = SymbolicHyperNeuron(n_units=4, n_features=8, tau=0.1)
        unit.eval()
        x = torch.randn(2, 10, 8)
        y1 = unit(x)
        y2 = unit(x)
        assert torch.allclose(y1, y2), \
            "eval-mode forward must be deterministic (no Gumbel noise)"

    def test_training_mode_has_stochastic_gumbel_noise(self):
        """In train mode, Gumbel noise produces different outputs across
        calls — this is *desired* (encourages exploration of the
        operator space)."""
        torch.manual_seed(0)
        unit = SymbolicHyperNeuron(n_units=4, n_features=8, tau=1.0)
        unit.train()
        x = torch.randn(2, 10, 8)
        y1 = unit(x)
        y2 = unit(x)
        # At tau=1.0 the noise is large enough to guarantee different
        # outputs across draws.
        assert not torch.allclose(y1, y2, atol=1e-4), \
            "train-mode forward must be stochastic (Gumbel sampling)"


# ──────────────────────────────────────────────────────────────────────
# Expression extraction — the headline feature
# ──────────────────────────────────────────────────────────────────────

class TestExpressionExtraction:
    """A SymbolicHyperNeuron's value over a vanilla MLP is being able
    to PRINT what it has learned. Every method here is a contract on
    that interpretability."""

    def test_extract_returns_one_string_per_unit(self):
        unit = SymbolicHyperNeuron(n_units=3, n_features=5)
        exprs = unit.expression_strings()
        assert isinstance(exprs, list)
        assert len(exprs) == 3
        for e in exprs:
            assert isinstance(e, str)
            assert len(e) > 0

    def test_default_feature_names_are_x0_x1_etc(self):
        unit = SymbolicHyperNeuron(n_units=2, n_features=4)
        # Sharpening logits to make selection deterministic doesn't change
        # the fact that some x{0..3} name must appear.
        exprs = unit.expression_strings()
        joined = " ".join(exprs)
        assert any(f"x{i}" in joined for i in range(4)), \
            f"none of x0..x3 appeared in {exprs}"

    def test_custom_feature_names_are_honoured(self):
        names = ["phi", "metabolic_demand", "surprise", "act_ema"]
        unit = SymbolicHyperNeuron(n_units=2, n_features=4,
                                   feature_names=names)
        exprs = unit.expression_strings()
        joined = " ".join(exprs)
        assert any(n in joined for n in names), \
            f"none of {names} appeared in {exprs}"

    def test_expression_mentions_an_operator(self):
        """Each unit's printed form must reflect the operator it has
        selected — either by name (`add`, `mul`) or symbol (`+`, `*`)."""
        unit = SymbolicHyperNeuron(n_units=1, n_features=4)
        expr = unit.expression_strings()[0]
        op_evidence = ("+", "-", "*", "add", "sub", "mul", "exp",
                       "sin", "cos", "tanh", "id")
        assert any(token in expr for token in op_evidence), \
            f"no operator evidence in expression: {expr!r}"

    def test_hardened_logits_produce_stable_expression(self):
        """Once we sharpen the operator/input logits to near-one-hot,
        the extracted formula must stop changing across calls."""
        unit = SymbolicHyperNeuron(n_units=2, n_features=4)
        with torch.no_grad():
            unit.input_a_logits.data *= 100
            unit.input_b_logits.data *= 100
            unit.op_logits.data *= 100
        e1 = unit.expression_strings()
        e2 = unit.expression_strings()
        assert e1 == e2


# ──────────────────────────────────────────────────────────────────────
# Regularization losses
# ──────────────────────────────────────────────────────────────────────

class TestSparsityRegularization:
    """`sparsity_loss` is the entropy of the operator + input
    distributions. Driven to zero by training, it produces one-hot
    selections — i.e. interpretable formulas."""

    def test_sparsity_loss_is_scalar(self):
        unit = SymbolicHyperNeuron(n_units=4, n_features=8)
        loss = unit.sparsity_loss()
        assert loss.dim() == 0

    def test_sparsity_loss_is_non_negative(self):
        unit = SymbolicHyperNeuron(n_units=4, n_features=8)
        loss = unit.sparsity_loss()
        assert loss.item() >= 0.0

    def test_sparsity_loss_is_differentiable(self):
        unit = SymbolicHyperNeuron(n_units=4, n_features=8)
        loss = unit.sparsity_loss()
        loss.backward()
        assert unit.op_logits.grad is not None

    def test_sparsity_loss_is_lower_for_sharper_logits(self):
        """Sharper (closer to one-hot) logits ⇒ lower entropy ⇒ lower
        sparsity loss. This is the contract that lets the optimizer
        anneal toward discrete formulas."""
        torch.manual_seed(0)
        u_uniform = SymbolicHyperNeuron(n_units=4, n_features=8)
        u_sharp   = SymbolicHyperNeuron(n_units=4, n_features=8)
        with torch.no_grad():
            for p in u_sharp.parameters():
                p.data = p.data * 50.0
        assert u_uniform.sparsity_loss().item() \
            > u_sharp.sparsity_loss().item()


# ──────────────────────────────────────────────────────────────────────
# Temperature annealing — Gumbel-Softmax control
# ──────────────────────────────────────────────────────────────────────

class TestTemperatureAnnealing:
    def test_initial_tau_is_set(self):
        unit = SymbolicHyperNeuron(n_units=2, n_features=4, tau=0.5)
        assert unit.tau == 0.5

    def test_set_tau_updates_value(self):
        unit = SymbolicHyperNeuron(n_units=2, n_features=4, tau=1.0)
        unit.set_tau(0.05)
        assert unit.tau == 0.05

    def test_set_tau_rejects_non_positive(self):
        unit = SymbolicHyperNeuron(n_units=2, n_features=4)
        with pytest.raises(ValueError, match="tau"):
            unit.set_tau(0.0)
        with pytest.raises(ValueError, match="tau"):
            unit.set_tau(-1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
