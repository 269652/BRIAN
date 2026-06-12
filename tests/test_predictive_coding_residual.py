"""Math contracts for predictive-coding residuals (Rao-Ballard 1999).

Pins:
  * The functional update implements x ← x - α·(x - x̂).
  * Residual is exactly x - x̂.
  * Step size in (0, 1] is enforced; bad inputs raise.
  * Module forward preserves shape, exposes residual norm, supports
    iterative mode.
  * Repeated iteration drives the residual toward zero (convergence).
  * All parameters receive gradients.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuroslm.modules.predictive_coding_residual import (
    PredictiveCodingResidual,
    predictive_residual,
)


# ──────────────────────────────────────────────────────────────────────
# 1. Functional update — algebraic contract
# ──────────────────────────────────────────────────────────────────────


class TestFunctionalUpdate:
    def test_residual_equals_x_minus_prediction(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4)
        predictor = nn.Linear(4, 4)
        x_above = torch.randn(2, 4)
        with torch.no_grad():
            x_hat = predictor(x_above)
            expected_residual = x - x_hat
        _, residual = predictive_residual(x, x_above, predictor, step_size=0.1)
        assert torch.allclose(residual, expected_residual, atol=1e-6)

    def test_state_update_is_gradient_step(self):
        """x' = x - α·residual (one step of GD on ½ε²)."""
        torch.manual_seed(1)
        x = torch.randn(3, 8)
        predictor = nn.Linear(8, 8)
        x_above = torch.randn(3, 8)
        x_new, residual = predictive_residual(
            x, x_above, predictor, step_size=0.25
        )
        expected = x - 0.25 * residual
        assert torch.allclose(x_new, expected, atol=1e-6)

    def test_zero_residual_when_prediction_matches(self):
        """If x̂ == x, the residual is zero and the state is unchanged."""
        torch.manual_seed(2)
        x = torch.randn(2, 5)
        # Use an identity predictor and pass x as x_above too.
        predictor = nn.Linear(5, 5, bias=False)
        with torch.no_grad():
            nn.init.eye_(predictor.weight)
        x_new, residual = predictive_residual(
            x, x, predictor, step_size=0.5
        )
        assert torch.allclose(residual, torch.zeros_like(x), atol=1e-6)
        assert torch.allclose(x_new, x, atol=1e-6)

    def test_step_size_out_of_range_raises(self):
        x = torch.randn(2, 4)
        p = nn.Linear(4, 4)
        with pytest.raises(ValueError, match="step_size"):
            predictive_residual(x, x, p, step_size=0.0)
        with pytest.raises(ValueError, match="step_size"):
            predictive_residual(x, x, p, step_size=-0.1)
        with pytest.raises(ValueError, match="step_size"):
            predictive_residual(x, x, p, step_size=1.5)

    def test_shape_mismatch_raises(self):
        x = torch.randn(2, 4)
        x_above = torch.randn(2, 8)
        predictor = nn.Linear(8, 6)  # produces wrong shape
        with pytest.raises(ValueError, match="must match"):
            predictive_residual(x, x_above, predictor)


# ──────────────────────────────────────────────────────────────────────
# 2. Module — shape + behaviour contracts
# ──────────────────────────────────────────────────────────────────────


class TestPredictiveCodingResidualModule:
    def test_forward_preserves_shape(self):
        mod = PredictiveCodingResidual(d_model=16)
        x = torch.randn(2, 5, 16)
        y = mod(x)
        assert y.shape == x.shape

    def test_forward_exposes_residual_norm(self):
        mod = PredictiveCodingResidual(d_model=16)
        x = torch.randn(2, 5, 16)
        _ = mod(x)
        # The buffer must be populated, non-negative, finite.
        norm = mod.last_residual_norm.item()
        assert norm >= 0.0
        assert torch.isfinite(mod.last_residual_norm).all()

    def test_iterative_mode_decreases_residual(self):
        """Repeated PC iteration must reduce the residual norm —
        that's the whole point of the gradient step."""
        torch.manual_seed(3)
        mod1 = PredictiveCodingResidual(
            d_model=32, mode="iterative", n_iterations=1, step_size=0.1
        )
        # Reuse the SAME predictor weights so the comparison is fair.
        mod5 = PredictiveCodingResidual(
            d_model=32, mode="iterative", n_iterations=10, step_size=0.1
        )
        with torch.no_grad():
            mod5.predictor.weight.copy_(mod1.predictor.weight)
            if mod5.predictor.bias is not None:
                mod5.predictor.bias.copy_(mod1.predictor.bias)
        x = torch.randn(1, 4, 32)
        _ = mod1(x)
        r1 = mod1.last_residual_norm.item()
        _ = mod5(x)
        r5 = mod5.last_residual_norm.item()
        assert r5 < r1, (
            f"iterative PC did not reduce residual: 1-iter norm={r1:.4f}, "
            f"10-iter norm={r5:.4f}"
        )

    def test_wrong_input_dim_raises(self):
        mod = PredictiveCodingResidual(d_model=8)
        with pytest.raises(ValueError, match=r"\(B, T, D\)"):
            mod(torch.randn(2, 8))  # missing T axis

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            PredictiveCodingResidual(d_model=8, mode="batch")  # type: ignore

    def test_bad_n_iterations_raises(self):
        with pytest.raises(ValueError, match="n_iterations"):
            PredictiveCodingResidual(d_model=8, n_iterations=0)

    def test_parameters_receive_gradients(self):
        torch.manual_seed(4)
        mod = PredictiveCodingResidual(d_model=8)
        x = torch.randn(1, 3, 8, requires_grad=True)
        loss = mod(x).pow(2).sum()
        loss.backward()
        for name, p in mod.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
            assert p.grad.abs().sum() > 0, f"{name} grad is zero"

    def test_custom_predictor_is_used(self):
        custom = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            nn.init.eye_(custom.weight)
        mod = PredictiveCodingResidual(d_model=8, predictor=custom)
        assert mod.predictor is custom


# ──────────────────────────────────────────────────────────────────────
# 3. End-to-end: identity predictor → near-zero residual
# ──────────────────────────────────────────────────────────────────────


class TestPredictiveCodingEndToEnd:
    def test_identity_predictor_yields_zero_residual_norm(self):
        """If we lock the predictor to identity-no-noise, residual on
        self-input is exactly zero and the state is unchanged.
        """
        torch.manual_seed(5)
        # Identity predictor with NO additive noise.
        custom = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            nn.init.eye_(custom.weight)
        mod = PredictiveCodingResidual(d_model=8, predictor=custom)
        x = torch.randn(1, 3, 8)
        y = mod(x)
        assert mod.last_residual_norm.item() < 1e-5
        assert torch.allclose(y, x, atol=1e-5)
