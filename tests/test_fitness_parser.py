# -*- coding: utf-8 -*-
"""TDD acceptance suite — `training { fitness { ... } }` block parser.

Phase A/F1 of the C → A → B implementation plan.

The `fitness` block is the **central switchboard** for Multi-Objective
training: it declares which objectives contribute to the total loss,
their relative weights, and per-objective configuration (symbolic
hyper-neuron count, metabolic budget, etc.).

Design contract:
    fitness {
        enabled: true,
        objectives: {
            lm:        { weight: 1.0,  enabled: true },
            phi:       { weight: 0.02, enabled: true, schedule: "gated" },
            symbolic:  { weight: 0.05, enabled: true },
            piso:      { weight: 0.10, enabled: false },
            metabolic: { weight: 0.30, enabled: false }
        },
        symbolic: {
            n_units: 8,
            n_features: 16,
            tau_init: 1.0,
            tau_final: 0.05,
            sparsity_weight: 0.01
        },
        metabolic: {
            budget: 0.7,
            prune_threshold: 0.05
        }
    }

Forward-compat: missing `fitness` block ⇒ all objectives disabled and
the existing `total_loss_config` (Brain bit-identical) takes over.
"""
from __future__ import annotations

import pytest

from neuroslm.dsl.training_config import (
    FitnessConfig,
    FitnessObjective,
    TrainingConfig,
    parse_training_config,
)


# ──────────────────────────────────────────────────────────────────────
# Default-construction contract
# ──────────────────────────────────────────────────────────────────────

class TestFitnessConfigDefaults:
    """A freshly constructed config must be DISABLED so legacy archs
    (no fitness block) train identically to before."""

    def test_master_switch_off_by_default(self):
        cfg = FitnessConfig()
        assert cfg.enabled is False

    def test_default_objectives_dict_is_empty_or_lm_only(self):
        """The defaults must NOT activate any new aux losses — only
        the legacy lm loss may be present, with weight 1.0."""
        cfg = FitnessConfig()
        for name, spec in cfg.objectives.items():
            if name == "lm":
                # lm with weight 1.0 IS the legacy single-objective.
                continue
            assert spec.enabled is False, \
                f"objective {name!r} must be disabled by default"

    def test_symbolic_defaults_match_module(self):
        """The defaults must produce a SymbolicHyperNeuron with sensible
        hyperparameters (sparsity_weight=0.01, tau_init=1.0)."""
        cfg = FitnessConfig()
        assert cfg.symbolic_n_units == 8
        assert cfg.symbolic_n_features == 16
        assert cfg.symbolic_tau_init == 1.0
        assert cfg.symbolic_tau_final == 0.05
        assert cfg.symbolic_sparsity_weight == 0.01

    def test_metabolic_defaults_are_safe(self):
        cfg = FitnessConfig()
        assert cfg.metabolic_budget == 0.7
        assert 0.0 < cfg.metabolic_prune_threshold < 0.5

    def test_training_config_carries_default_fitness(self):
        """`TrainingConfig()` (no args) must own a default-disabled
        FitnessConfig so harness wiring code can safely read it."""
        tc = TrainingConfig()
        assert isinstance(tc.fitness, FitnessConfig)
        assert tc.fitness.enabled is False


# ──────────────────────────────────────────────────────────────────────
# FitnessObjective dataclass
# ──────────────────────────────────────────────────────────────────────

class TestFitnessObjective:
    def test_construct_with_defaults(self):
        o = FitnessObjective()
        assert o.enabled is False
        assert o.weight == 0.0
        assert o.schedule == "constant"

    def test_construct_with_explicit_values(self):
        o = FitnessObjective(enabled=True, weight=0.5, schedule="gated")
        assert o.enabled is True
        assert o.weight == 0.5
        assert o.schedule == "gated"


# ──────────────────────────────────────────────────────────────────────
# Parser contract — headline TDD requirement
# ──────────────────────────────────────────────────────────────────────

class TestParseFitnessBlock:
    """`parse_training_config` must read every documented fitness
    field from an arch.neuro snippet."""

    def test_parses_minimal_enabled_block(self):
        body = """
            fitness: {
                enabled: true
            }
        """
        cfg = parse_training_config(body)
        assert cfg.fitness.enabled is True
        # All other fields keep their defaults
        assert cfg.fitness.symbolic_n_units == 8

    def test_parses_single_objective(self):
        body = """
            fitness: {
                enabled: true,
                objectives: {
                    lm: { weight: 1.0, enabled: true }
                }
            }
        """
        cfg = parse_training_config(body)
        assert cfg.fitness.objectives["lm"].weight == pytest.approx(1.0)
        assert cfg.fitness.objectives["lm"].enabled is True

    def test_parses_multiple_objectives_with_schedule(self):
        body = """
            fitness: {
                enabled: true,
                objectives: {
                    lm:       { weight: 1.0,  enabled: true },
                    phi:      { weight: 0.02, enabled: true, schedule: "gated" },
                    symbolic: { weight: 0.05, enabled: true }
                }
            }
        """
        cfg = parse_training_config(body)
        objs = cfg.fitness.objectives
        assert objs["lm"].weight == pytest.approx(1.0)
        assert objs["phi"].weight == pytest.approx(0.02)
        assert objs["phi"].schedule == "gated"
        assert objs["symbolic"].enabled is True

    def test_parses_symbolic_subblock(self):
        body = """
            fitness: {
                enabled: true,
                symbolic: {
                    n_units: 16,
                    n_features: 32,
                    tau_init: 2.0,
                    tau_final: 0.01,
                    sparsity_weight: 0.05
                }
            }
        """
        cfg = parse_training_config(body)
        assert cfg.fitness.symbolic_n_units == 16
        assert cfg.fitness.symbolic_n_features == 32
        assert cfg.fitness.symbolic_tau_init == pytest.approx(2.0)
        assert cfg.fitness.symbolic_tau_final == pytest.approx(0.01)
        assert cfg.fitness.symbolic_sparsity_weight == pytest.approx(0.05)

    def test_parses_metabolic_subblock(self):
        body = """
            fitness: {
                enabled: true,
                metabolic: {
                    budget: 0.5,
                    prune_threshold: 0.1
                }
            }
        """
        cfg = parse_training_config(body)
        assert cfg.fitness.metabolic_budget == pytest.approx(0.5)
        assert cfg.fitness.metabolic_prune_threshold == pytest.approx(0.1)

    def test_parses_full_production_block(self):
        """The block that will go into arch.neuro."""
        body = """
            fitness: {
                enabled: true,
                objectives: {
                    lm:        { weight: 1.0,  enabled: true },
                    phi:       { weight: 0.02, enabled: true, schedule: "gated" },
                    symbolic:  { weight: 0.05, enabled: true },
                    piso:      { weight: 0.10, enabled: false },
                    metabolic: { weight: 0.30, enabled: false }
                },
                symbolic: {
                    n_units: 8,
                    n_features: 16,
                    tau_init: 1.0,
                    tau_final: 0.05,
                    sparsity_weight: 0.01
                },
                metabolic: {
                    budget: 0.7,
                    prune_threshold: 0.05
                }
            }
        """
        cfg = parse_training_config(body)
        f = cfg.fitness
        assert f.enabled is True
        assert len(f.objectives) == 5
        assert f.objectives["symbolic"].enabled is True
        assert f.objectives["piso"].enabled is False
        assert f.symbolic_n_units == 8
        assert f.metabolic_budget == pytest.approx(0.7)


# ──────────────────────────────────────────────────────────────────────
# Validation — fail loudly on bad configs
# ──────────────────────────────────────────────────────────────────────

class TestParseFitnessValidation:
    """Bad configs must raise at parse time, not silently at runtime."""

    def test_unknown_objective_name_raises(self):
        """Typos like `fhi` instead of `phi` produce silent zero-weight
        bugs — surface them at parse time."""
        body = """
            fitness: {
                enabled: true,
                objectives: {
                    fhi: { weight: 0.02, enabled: true }
                }
            }
        """
        with pytest.raises(ValueError, match="objective"):
            parse_training_config(body)

    def test_negative_weight_raises(self):
        body = """
            fitness: {
                enabled: true,
                objectives: {
                    lm: { weight: -1.0, enabled: true }
                }
            }
        """
        with pytest.raises(ValueError, match="weight"):
            parse_training_config(body)

    def test_unknown_schedule_raises(self):
        body = """
            fitness: {
                enabled: true,
                objectives: {
                    phi: { weight: 0.02, enabled: true, schedule: "exponential" }
                }
            }
        """
        with pytest.raises(ValueError, match="schedule"):
            parse_training_config(body)

    def test_symbolic_n_units_must_be_positive(self):
        body = """
            fitness: {
                enabled: true,
                symbolic: { n_units: 0 }
            }
        """
        with pytest.raises(ValueError, match="n_units"):
            parse_training_config(body)

    def test_metabolic_budget_must_be_in_unit_interval(self):
        body = """
            fitness: {
                enabled: true,
                metabolic: { budget: 1.5 }
            }
        """
        with pytest.raises(ValueError, match="budget"):
            parse_training_config(body)


# ──────────────────────────────────────────────────────────────────────
# Forward-compat: absence ⇒ disabled, no crash on legacy arch.neuro
# ──────────────────────────────────────────────────────────────────────

class TestParseLegacyArchsNoFitness:
    """All existing arch.neuro files without a fitness block must
    continue to parse and yield `fitness.enabled = False`."""

    def test_empty_training_body_is_disabled(self):
        cfg = parse_training_config("")
        assert cfg.fitness.enabled is False

    def test_unrelated_training_keys_dont_affect_fitness(self):
        body = """
            optimizer: "adamw",
            learning_rate: 0.0003,
            batch_size: 16
        """
        cfg = parse_training_config(body)
        assert cfg.fitness.enabled is False
        assert cfg.optimizer == "adamw"
        assert cfg.batch_size == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
