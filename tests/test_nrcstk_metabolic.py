# -*- coding: utf-8 -*-
"""TDD acceptance suite — `NRCSTKController` metabolic budget enforcer.

Phase B of the C → A → B implementation plan.

The NRCSTK ("Neuronal-Resource-Constrained Selection Through Killing")
controller implements the metabolic-market selection pressure from the
work order:

    "Neuronen, die ihre Oszillationen nicht effizient mit den
     Eingangsmustern synchronisieren, 'verhungern' und werden gelöscht.
     Dies erzwingt die Entdeckung von Least-Action-Prinzipien und
     extrem sparsamen Topologien."

Mathematical contract
---------------------
For a target layer producing activations ``a in R^{B,T,D}``:

  1. Demand:     d_i(t) = mean_BT |a_i|
                 d_ema_i = alpha * d_i + (1 - alpha) * d_ema_i        (EMA)

  2. Budget loss: L_met = relu(mean_i d_ema_i - B)^2  with B in [0,1]

  3. Pruning mask: m_i = 1 if d_ema_i > tau_prune else 0

  4. Live count:  N_live = sum_i m_i  (telemetry)

The metabolic loss is the gradient signal that pushes weakly-used
neurons toward zero output → their EMA collapses → they get pruned.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuroslm.modules.nrcstk import NRCSTKController


# ──────────────────────────────────────────────────────────────────────
# Construction & defaults
# ──────────────────────────────────────────────────────────────────────

class TestConstruction:
    def test_construct_with_minimum_args(self):
        ctrl = NRCSTKController(target_dim=16)
        assert ctrl.target_dim == 16

    def test_default_budget_and_threshold_are_sensible(self):
        ctrl = NRCSTKController(target_dim=16)
        # Defaults match FitnessConfig defaults so a DSL-spawned
        # controller has identical behaviour to a code-spawned one.
        assert 0.0 < ctrl.budget <= 1.0
        assert 0.0 <= ctrl.prune_threshold < 0.5

    def test_initial_demand_ema_is_zero(self):
        """No observations yet ⇒ no demand recorded."""
        ctrl = NRCSTKController(target_dim=8)
        assert ctrl.demand_ema.shape == (8,)
        assert torch.allclose(ctrl.demand_ema, torch.zeros(8))

    def test_initial_pruning_mask_is_all_ones(self):
        """Before any pruning runs, every neuron is alive."""
        ctrl = NRCSTKController(target_dim=8)
        mask = ctrl.pruning_mask()
        assert mask.shape == (8,)
        assert (mask == 1.0).all()

    def test_initial_n_live_neurons_equals_target_dim(self):
        ctrl = NRCSTKController(target_dim=12)
        assert ctrl.n_live_neurons() == 12

    def test_construction_rejects_invalid_budget(self):
        with pytest.raises(ValueError, match="budget"):
            NRCSTKController(target_dim=8, budget=1.5)
        with pytest.raises(ValueError, match="budget"):
            NRCSTKController(target_dim=8, budget=-0.1)

    def test_construction_rejects_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            NRCSTKController(target_dim=8, ema_alpha=1.5)


# ──────────────────────────────────────────────────────────────────────
# Demand observation — the EMA dynamics
# ──────────────────────────────────────────────────────────────────────

class TestDemandObservation:
    """`observe(activations)` must update the per-neuron demand EMA
    in proportion to |activations|."""

    def test_observe_high_activity_increases_ema(self):
        ctrl = NRCSTKController(target_dim=4, ema_alpha=0.5)
        x = torch.ones(2, 8, 4) * 10.0  # large constant activations
        ctrl.observe(x)
        # EMA should be far from zero now.
        assert (ctrl.demand_ema > 1.0).all(), \
            f"EMA stayed near zero after observing |x|=10: {ctrl.demand_ema}"

    def test_observe_zero_activity_keeps_ema_zero(self):
        ctrl = NRCSTKController(target_dim=4, ema_alpha=0.5)
        x = torch.zeros(2, 8, 4)
        ctrl.observe(x)
        assert torch.allclose(ctrl.demand_ema, torch.zeros(4))

    def test_observe_per_neuron_demand(self):
        """Each neuron's EMA must track only its own activation magnitude
        — not the mean across neurons. This is the contract that lets
        selective pruning work."""
        ctrl = NRCSTKController(target_dim=3, ema_alpha=1.0)  # full step
        # Neuron 0: zero. Neuron 1: small. Neuron 2: large.
        x = torch.stack([
            torch.zeros(8),
            torch.ones(8) * 0.5,
            torch.ones(8) * 5.0,
        ], dim=-1).unsqueeze(0)  # (1, 8, 3)
        ctrl.observe(x)
        # demand_ema must be ordered [0, 0.5, 5.0] (up to small fp noise)
        assert ctrl.demand_ema[0].item() == pytest.approx(0.0)
        assert ctrl.demand_ema[1].item() == pytest.approx(0.5)
        assert ctrl.demand_ema[2].item() == pytest.approx(5.0)

    def test_observe_ema_alpha_controls_smoothing(self):
        """Low alpha ⇒ slow EMA; high alpha ⇒ fast."""
        ctrl_slow = NRCSTKController(target_dim=2, ema_alpha=0.01)
        ctrl_fast = NRCSTKController(target_dim=2, ema_alpha=0.99)
        x = torch.ones(1, 4, 2) * 10.0
        ctrl_slow.observe(x)
        ctrl_fast.observe(x)
        # Fast EMA jumped much further on the first observation.
        assert ctrl_fast.demand_ema[0] > ctrl_slow.demand_ema[0] * 10

    def test_observe_does_not_require_grad(self):
        """The EMA is a buffer, not a parameter — no autograd tracking."""
        ctrl = NRCSTKController(target_dim=4)
        x = torch.ones(2, 8, 4) * 5.0
        ctrl.observe(x)
        assert ctrl.demand_ema.requires_grad is False

    def test_observe_rejects_wrong_dim(self):
        ctrl = NRCSTKController(target_dim=4)
        bad = torch.randn(2, 8, 5)  # last dim 5 ≠ 4
        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            ctrl.observe(bad)


# ──────────────────────────────────────────────────────────────────────
# Metabolic loss — the gradient signal that drives selection
# ──────────────────────────────────────────────────────────────────────

class TestMetabolicLoss:
    """`metabolic_loss(activations)` returns relu(mean(|a|) - budget)^2.

    This is the loss that pushes the optimiser to shrink weakly-used
    neurons' outputs to zero, after which their EMA decays and they
    get pruned by `prune_step()`.
    """

    def test_under_budget_returns_zero(self):
        ctrl = NRCSTKController(target_dim=4, budget=0.7)
        x = torch.zeros(2, 8, 4)
        loss = ctrl.metabolic_loss(x)
        assert loss.item() == pytest.approx(0.0)

    def test_at_budget_returns_zero(self):
        ctrl = NRCSTKController(target_dim=4, budget=1.0)
        x = torch.ones(2, 8, 4) * 1.0  # mean |x| = 1.0 == budget
        loss = ctrl.metabolic_loss(x)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_over_budget_returns_positive(self):
        ctrl = NRCSTKController(target_dim=4, budget=0.5)
        x = torch.ones(2, 8, 4) * 2.0  # well over budget
        loss = ctrl.metabolic_loss(x)
        assert loss.item() > 0.0

    def test_metabolic_loss_grows_with_overshoot(self):
        ctrl = NRCSTKController(target_dim=4, budget=0.5)
        x_small = torch.ones(2, 8, 4) * 1.0
        x_large = torch.ones(2, 8, 4) * 3.0
        assert ctrl.metabolic_loss(x_large).item() \
            > ctrl.metabolic_loss(x_small).item()

    def test_metabolic_loss_is_differentiable(self):
        """The loss must allow gradient flow back to the activations —
        without this, the optimiser cannot push down weak neurons."""
        ctrl = NRCSTKController(target_dim=4, budget=0.5)
        # NB: ``torch.ones(..., requires_grad=True) * 2.0`` would create
        # a *non-leaf* tensor and ``.grad`` would never populate.  Use a
        # leaf tensor by setting requires_grad on the final value.
        x = (torch.ones(2, 8, 4) * 2.0).requires_grad_()
        loss = ctrl.metabolic_loss(x)
        loss.backward()
        assert x.grad is not None
        assert (x.grad != 0).any(), "no gradient propagated"


# ──────────────────────────────────────────────────────────────────────
# Pruning mask — kills neurons whose EMA fell below threshold
# ──────────────────────────────────────────────────────────────────────

class TestPruningMask:
    def test_mask_zeros_low_ema_neurons(self):
        """After observing zero activity on a neuron, its EMA decays
        toward zero. Below prune_threshold ⇒ mask entry is zero."""
        ctrl = NRCSTKController(target_dim=3, ema_alpha=1.0,
                                prune_threshold=0.1)
        # Neuron 0: dead. Neuron 1: alive. Neuron 2: marginal.
        x = torch.stack([
            torch.zeros(8),
            torch.ones(8) * 5.0,
            torch.ones(8) * 0.05,  # below threshold
        ], dim=-1).unsqueeze(0)  # (1, 8, 3)
        ctrl.observe(x)
        mask = ctrl.pruning_mask()
        assert mask[0].item() == 0.0
        assert mask[1].item() == 1.0
        assert mask[2].item() == 0.0

    def test_mask_preserves_high_ema_neurons(self):
        ctrl = NRCSTKController(target_dim=4, ema_alpha=1.0,
                                prune_threshold=0.1)
        x = torch.ones(2, 8, 4) * 5.0
        ctrl.observe(x)
        mask = ctrl.pruning_mask()
        assert (mask == 1.0).all()

    def test_n_live_neurons_reflects_mask(self):
        ctrl = NRCSTKController(target_dim=4, ema_alpha=1.0,
                                prune_threshold=0.1)
        x = torch.stack([torch.zeros(8), torch.zeros(8),
                         torch.ones(8) * 5.0, torch.ones(8) * 5.0],
                        dim=-1).unsqueeze(0)
        ctrl.observe(x)
        assert ctrl.n_live_neurons() == 2

    def test_apply_mask_zeros_corresponding_features(self):
        """`apply_mask(x)` is the canonical way to actually prune the
        forward path: x_pruned = x * mask."""
        ctrl = NRCSTKController(target_dim=3, ema_alpha=1.0,
                                prune_threshold=0.1)
        x = torch.stack([
            torch.zeros(8),
            torch.ones(8) * 5.0,
            torch.zeros(8),
        ], dim=-1).unsqueeze(0)
        ctrl.observe(x)
        # New input with full activity everywhere
        x_new = torch.ones(1, 8, 3) * 2.0
        x_pruned = ctrl.apply_mask(x_new)
        assert (x_pruned[..., 0] == 0).all()       # was pruned
        assert (x_pruned[..., 1] == 2.0).all()     # alive
        assert (x_pruned[..., 2] == 0).all()       # was pruned


# ──────────────────────────────────────────────────────────────────────
# Integration with FitnessComposer
# ──────────────────────────────────────────────────────────────────────

class TestComposerIntegration:
    """When `fitness.objectives["metabolic"].enabled`, the harness
    should be able to wire an NRCSTKController into the FitnessComposer
    so the metabolic loss flows through the standard LossBundle path."""

    def test_controller_can_be_built_from_fitness_config(self):
        from neuroslm.dsl.training_config import FitnessConfig
        cfg = FitnessConfig(enabled=True)
        cfg.metabolic_budget = 0.6
        cfg.metabolic_prune_threshold = 0.08
        ctrl = NRCSTKController.from_fitness_config(target_dim=16, config=cfg)
        assert ctrl.target_dim == 16
        assert ctrl.budget == pytest.approx(0.6)
        assert ctrl.prune_threshold == pytest.approx(0.08)

    def test_controller_state_round_trip(self):
        """A controller's EMA + mask state must survive serialisation
        (state_dict round-trip) so checkpoints are reproducible."""
        ctrl_a = NRCSTKController(target_dim=4, ema_alpha=1.0)
        x = torch.stack([torch.zeros(8), torch.ones(8) * 5.0,
                         torch.zeros(8), torch.ones(8) * 2.0],
                        dim=-1).unsqueeze(0)
        ctrl_a.observe(x)
        ema_before = ctrl_a.demand_ema.clone()

        ctrl_b = NRCSTKController(target_dim=4, ema_alpha=1.0)
        ctrl_b.load_state_dict(ctrl_a.state_dict())
        assert torch.allclose(ctrl_b.demand_ema, ema_before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
