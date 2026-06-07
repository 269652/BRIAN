# -*- coding: utf-8 -*-
"""TDD acceptance suite — `FitnessComposer`.

Phase A/F2 of the C → A → B implementation plan.

`FitnessComposer` is the runtime counterpart of the `FitnessConfig`
DSL block: given a config and a set of per-objective loss sources, it
returns

    (total_loss : torch.Tensor,   # scalar autograd-traced
     telemetry  : Dict[str, float])  # per-objective contribution

so the harness can replace its hard-coded `total_loss_config` formula
with a single declarative pipeline.

Contracts under test:
  Construction          — instantiates from FitnessConfig, registers
                          its own SymbolicHyperNeuron if needed.
  ObjectiveContribution — every objective produces a scalar; disabled
                          objectives contribute zero.
  Aggregation           — total = Σ weight_i * loss_i; telemetry sums.
  Schedule              — "gated" multiplies by maturity phase gate.
  Symbolic integration  — when symbolic objective is enabled, the
                          composer instantiates a SymbolicHyperNeuron
                          and consumes both its forward output and
                          sparsity_loss.
"""
from __future__ import annotations

import pytest
import torch

from neuroslm.dsl.training_config import (
    FitnessConfig,
    FitnessObjective,
)
from neuroslm.fitness import (
    FitnessComposer,
    LossBundle,
)


# ──────────────────────────────────────────────────────────────────────
# Helper: build a config for tests
# ──────────────────────────────────────────────────────────────────────

def _make_lm_only_config() -> FitnessConfig:
    """Legacy reproducibility — only `lm` enabled, weight 1.0."""
    cfg = FitnessConfig(enabled=True)
    cfg.objectives = {
        "lm": FitnessObjective(enabled=True, weight=1.0, schedule="constant"),
    }
    return cfg


def _make_multi_objective_config() -> FitnessConfig:
    cfg = FitnessConfig(enabled=True)
    cfg.objectives = {
        "lm":      FitnessObjective(enabled=True, weight=1.0),
        "phi":     FitnessObjective(enabled=True, weight=0.02, schedule="gated"),
        "symbolic":FitnessObjective(enabled=True, weight=0.05),
    }
    return cfg


# ──────────────────────────────────────────────────────────────────────
# LossBundle — the dataclass the composer consumes
# ──────────────────────────────────────────────────────────────────────

class TestLossBundle:
    """A typed bag of per-objective scalar losses produced by the
    harness on each step. Missing entries default to None, which
    composer treats as 0-contribution."""

    def test_construct_with_only_lm(self):
        b = LossBundle(lm=torch.tensor(2.5))
        assert b.lm.item() == pytest.approx(2.5)
        assert b.phi is None
        assert b.symbolic is None

    def test_construct_with_all_fields(self):
        b = LossBundle(
            lm=torch.tensor(2.5),
            phi=torch.tensor(0.1),
            symbolic=torch.tensor(0.3),
            piso=torch.tensor(0.2),
            metabolic=torch.tensor(0.05),
            nis_plus=torch.tensor(0.4),
        )
        assert b.lm.item() == pytest.approx(2.5)
        assert b.phi.item() == pytest.approx(0.1)


# ──────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────

class TestComposerConstruction:
    def test_construct_with_minimum_config(self):
        c = FitnessComposer(_make_lm_only_config())
        assert c.config.enabled is True

    def test_disabled_config_is_legitimate(self):
        """A disabled config must still construct — the composer
        returns zero-contribution loss when called."""
        cfg = FitnessConfig()  # enabled=False
        c = FitnessComposer(cfg)
        assert c.config.enabled is False

    def test_symbolic_unit_is_built_when_objective_enabled(self):
        """If the `symbolic` objective is enabled the composer must own
        a SymbolicHyperNeuron sized per the symbolic.n_units / n_features
        fields. This wires Phase C into the multi-objective machinery."""
        from neuroslm.modules.symbolic_unit import SymbolicHyperNeuron
        cfg = FitnessConfig(enabled=True)
        cfg.objectives = {
            "symbolic": FitnessObjective(enabled=True, weight=0.05),
        }
        cfg.symbolic_n_units = 4
        cfg.symbolic_n_features = 6
        c = FitnessComposer(cfg)
        assert isinstance(c.symbolic_unit, SymbolicHyperNeuron)
        assert c.symbolic_unit.n_units == 4
        assert c.symbolic_unit.n_features == 6

    def test_symbolic_unit_is_none_when_objective_disabled(self):
        cfg = FitnessConfig(enabled=True)
        # symbolic NOT in objectives → no unit built.
        c = FitnessComposer(cfg)
        assert c.symbolic_unit is None


# ──────────────────────────────────────────────────────────────────────
# Aggregation — the core math
# ──────────────────────────────────────────────────────────────────────

class TestComposerAggregation:
    """`compose(bundle, maturity)` must produce:

        total = Σ_i enabled[i] * schedule(weight[i], maturity) * loss[i]
        telemetry = {name: weighted_contribution}
    """

    def test_lm_only_recovers_lm_loss_exactly(self):
        c = FitnessComposer(_make_lm_only_config())
        bundle = LossBundle(lm=torch.tensor(2.5))
        total, telemetry = c.compose(bundle, maturity=0.5)
        assert total.item() == pytest.approx(2.5)
        assert telemetry == pytest.approx({"lm": 2.5})

    def test_two_objectives_sum_correctly(self):
        cfg = FitnessConfig(enabled=True)
        cfg.objectives = {
            "lm":  FitnessObjective(enabled=True, weight=1.0),
            "phi": FitnessObjective(enabled=True, weight=0.5),
        }
        c = FitnessComposer(cfg)
        bundle = LossBundle(lm=torch.tensor(3.0), phi=torch.tensor(0.4))
        total, telemetry = c.compose(bundle, maturity=0.5)
        # constant schedule ⇒ weight is applied as-is
        # total = 1.0*3.0 + 0.5*0.4 = 3.2
        assert total.item() == pytest.approx(3.2)
        assert telemetry["lm"] == pytest.approx(3.0)
        assert telemetry["phi"] == pytest.approx(0.2)

    def test_disabled_objective_contributes_zero(self):
        cfg = FitnessConfig(enabled=True)
        cfg.objectives = {
            "lm":  FitnessObjective(enabled=True,  weight=1.0),
            "phi": FitnessObjective(enabled=False, weight=0.5),  # off
        }
        c = FitnessComposer(cfg)
        bundle = LossBundle(lm=torch.tensor(2.0), phi=torch.tensor(99.0))
        total, telemetry = c.compose(bundle, maturity=0.5)
        assert total.item() == pytest.approx(2.0)
        assert "phi" not in telemetry  # disabled ⇒ not in telemetry

    def test_missing_bundle_value_contributes_zero(self):
        """Objective enabled but harness didn't compute its loss this
        step ⇒ silently zero (allows step-staggered metrics)."""
        cfg = FitnessConfig(enabled=True)
        cfg.objectives = {
            "lm":  FitnessObjective(enabled=True, weight=1.0),
            "phi": FitnessObjective(enabled=True, weight=0.5),
        }
        c = FitnessComposer(cfg)
        bundle = LossBundle(lm=torch.tensor(2.0), phi=None)
        total, telemetry = c.compose(bundle, maturity=0.5)
        assert total.item() == pytest.approx(2.0)
        # phi appears with weight 0 (visible: "configured but missing")
        assert telemetry.get("phi", 0.0) == pytest.approx(0.0)

    def test_total_loss_is_differentiable(self):
        """The composed total must remain in the autograd graph."""
        c = FitnessComposer(_make_lm_only_config())
        lm = torch.tensor(2.5, requires_grad=True)
        total, _ = c.compose(LossBundle(lm=lm), maturity=0.5)
        total.backward()
        assert lm.grad is not None
        assert lm.grad.item() == pytest.approx(1.0)

    def test_disabled_master_switch_returns_lm_only_passthrough(self):
        """When the master switch is OFF the composer behaves as a
        no-op pass-through of `bundle.lm` — guarantees legacy
        reproducibility."""
        cfg = FitnessConfig(enabled=False)
        c = FitnessComposer(cfg)
        bundle = LossBundle(
            lm=torch.tensor(2.5),
            phi=torch.tensor(0.1),     # must be ignored
        )
        total, telemetry = c.compose(bundle, maturity=0.5)
        assert total.item() == pytest.approx(2.5)
        assert telemetry == {"lm": 2.5}


# ──────────────────────────────────────────────────────────────────────
# Schedule — gated / linear / constant
# ──────────────────────────────────────────────────────────────────────

class TestSchedule:
    """Verify the three schedule kinds modulate the weight correctly."""

    def test_constant_schedule_passes_weight_unmodified(self):
        cfg = FitnessConfig(enabled=True)
        cfg.objectives = {
            "lm": FitnessObjective(enabled=True, weight=0.7,
                                   schedule="constant"),
        }
        c = FitnessComposer(cfg)
        bundle = LossBundle(lm=torch.tensor(1.0))
        # Constant: weight is 0.7 regardless of maturity.
        total_low, _  = c.compose(bundle, maturity=0.0)
        total_high, _ = c.compose(bundle, maturity=1.0)
        assert total_low.item() == pytest.approx(0.7)
        assert total_high.item() == pytest.approx(0.7)

    def test_gated_schedule_is_zero_outside_window(self):
        """Gated weight ≈ 0 for maturity far from the gate center."""
        cfg = FitnessConfig(enabled=True)
        cfg.objectives = {
            "phi": FitnessObjective(enabled=True, weight=0.5,
                                    schedule="gated"),
        }
        c = FitnessComposer(cfg)
        bundle = LossBundle(phi=torch.tensor(1.0))
        # Far below gate (maturity 0.0) ⇒ near-zero contribution
        total_low, _ = c.compose(bundle, maturity=0.0)
        # Near gate center (maturity 0.5, default gate) ⇒ near full
        total_mid, _ = c.compose(bundle, maturity=0.5)
        assert total_low.item() < 0.05, \
            "gated schedule must be ~0 outside window"
        assert total_mid.item() > total_low.item()

    def test_linear_schedule_ramps_with_maturity(self):
        cfg = FitnessConfig(enabled=True)
        cfg.objectives = {
            "lm": FitnessObjective(enabled=True, weight=1.0,
                                   schedule="linear"),
        }
        c = FitnessComposer(cfg)
        bundle = LossBundle(lm=torch.tensor(1.0))
        # Linear: weight = weight * maturity (clamped to [0, weight]).
        total_0,  _ = c.compose(bundle, maturity=0.0)
        total_h,  _ = c.compose(bundle, maturity=0.5)
        total_1,  _ = c.compose(bundle, maturity=1.0)
        assert total_0.item() == pytest.approx(0.0)
        assert total_h.item() == pytest.approx(0.5)
        assert total_1.item() == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────
# Symbolic integration — wires Phase C into the composer
# ──────────────────────────────────────────────────────────────────────

class TestSymbolicIntegration:
    """When the `symbolic` objective is enabled the composer owns a
    SymbolicHyperNeuron and exposes:

      * `composer.compute_symbolic_loss(features)` — wrapper that
        runs the unit and returns sparsity_loss * sparsity_weight.
      * the sparsity_loss flows through `bundle.symbolic` and gets
        the composer's per-objective weight applied on top.
    """

    def test_compute_symbolic_loss_returns_scalar(self):
        cfg = FitnessConfig(enabled=True)
        cfg.objectives = {
            "symbolic": FitnessObjective(enabled=True, weight=0.05),
        }
        cfg.symbolic_n_features = 4
        cfg.symbolic_n_units = 2
        cfg.symbolic_sparsity_weight = 0.01
        c = FitnessComposer(cfg)
        features = torch.randn(8, 4)
        loss = c.compute_symbolic_loss(features)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0.0

    def test_compute_symbolic_loss_returns_zero_when_disabled(self):
        cfg = FitnessConfig(enabled=True)
        c = FitnessComposer(cfg)
        features = torch.randn(8, 4)
        loss = c.compute_symbolic_loss(features)
        assert loss.item() == pytest.approx(0.0)

    def test_compose_uses_symbolic_loss_correctly(self):
        cfg = FitnessConfig(enabled=True)
        cfg.objectives = {
            "lm":       FitnessObjective(enabled=True, weight=1.0),
            "symbolic": FitnessObjective(enabled=True, weight=0.1),
        }
        c = FitnessComposer(cfg)
        bundle = LossBundle(
            lm=torch.tensor(2.0),
            symbolic=torch.tensor(0.5),
        )
        total, telemetry = c.compose(bundle, maturity=1.0)
        # total = 1.0*2.0 + 0.1*0.5 = 2.05
        assert total.item() == pytest.approx(2.05)
        assert telemetry["symbolic"] == pytest.approx(0.05)

    def test_symbolic_unit_anneals_tau_via_set_tau(self):
        """The composer must expose tau control so the harness can
        anneal during training (tau_init → tau_final over warmup steps)."""
        cfg = FitnessConfig(enabled=True)
        cfg.objectives = {
            "symbolic": FitnessObjective(enabled=True, weight=0.05),
        }
        cfg.symbolic_tau_init = 1.0
        cfg.symbolic_tau_final = 0.05
        c = FitnessComposer(cfg)
        # Initial tau = tau_init
        assert c.symbolic_unit.tau == pytest.approx(1.0)
        # Anneal to final
        c.set_symbolic_tau(0.05)
        assert c.symbolic_unit.tau == pytest.approx(0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
