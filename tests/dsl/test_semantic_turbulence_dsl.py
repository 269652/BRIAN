# -*- coding: utf-8 -*-
"""TDD: DSL parser for semantic_turbulence { ... } block.

Contracts verified here:
  - SemanticTurbulenceConfig exists with correct defaults
  - parse_training_config accepts semantic_turbulence { ... }
  - All fields parse correctly (booleans, ints, floats)
  - Unknown sub-fields are ignored (forward-compat)
  - enabled=false produces disabled config

Run:  brian test tests/dsl/test_semantic_turbulence_dsl.py
"""
from __future__ import annotations

import pytest


# ── helpers ───────────────────────────────────────────────────────────────


def _parse(body: str):
    """Wrap body in training { } and parse."""
    from neuroslm.dsl.training_config import parse_training_config
    return parse_training_config(f"semantic_turbulence: {body}")


# ── SemanticTurbulenceConfig dataclass ────────────────────────────────────


class TestSemanticTurbulenceConfigDefaults:
    """Config dataclass must exist with documented defaults."""

    def test_config_is_importable(self):
        from neuroslm.dsl.training_config import SemanticTurbulenceConfig
        assert SemanticTurbulenceConfig is not None

    def test_enabled_defaults_false(self):
        from neuroslm.dsl.training_config import SemanticTurbulenceConfig
        cfg = SemanticTurbulenceConfig()
        assert cfg.enabled is False

    def test_n_rg_groups_defaults_3(self):
        from neuroslm.dsl.training_config import SemanticTurbulenceConfig
        cfg = SemanticTurbulenceConfig()
        assert cfg.n_rg_groups == 3

    def test_kolmogorov_init_defaults_true(self):
        from neuroslm.dsl.training_config import SemanticTurbulenceConfig
        cfg = SemanticTurbulenceConfig()
        assert cfg.kolmogorov_init is True

    def test_gpe_steps_defaults_4(self):
        from neuroslm.dsl.training_config import SemanticTurbulenceConfig
        cfg = SemanticTurbulenceConfig()
        assert cfg.gpe_steps == 4

    def test_gpe_coupling_init_defaults_0_01(self):
        from neuroslm.dsl.training_config import SemanticTurbulenceConfig
        cfg = SemanticTurbulenceConfig()
        assert abs(cfg.gpe_coupling_init - 0.01) < 1e-9

    def test_gpe_dt_defaults_0_01(self):
        from neuroslm.dsl.training_config import SemanticTurbulenceConfig
        cfg = SemanticTurbulenceConfig()
        assert abs(cfg.gpe_dt - 0.01) < 1e-9

    def test_criticality_target_defaults_1_0(self):
        from neuroslm.dsl.training_config import SemanticTurbulenceConfig
        cfg = SemanticTurbulenceConfig()
        assert abs(cfg.criticality_target - 1.0) < 1e-9

    def test_criticality_weight_defaults_0_01(self):
        from neuroslm.dsl.training_config import SemanticTurbulenceConfig
        cfg = SemanticTurbulenceConfig()
        assert abs(cfg.criticality_weight - 0.01) < 1e-9

    def test_criticality_da_reward_defaults_0_1(self):
        from neuroslm.dsl.training_config import SemanticTurbulenceConfig
        cfg = SemanticTurbulenceConfig()
        assert abs(cfg.criticality_da_reward - 0.1) < 1e-9

    def test_criticality_ema_alpha_defaults_0_05(self):
        from neuroslm.dsl.training_config import SemanticTurbulenceConfig
        cfg = SemanticTurbulenceConfig()
        assert abs(cfg.criticality_ema_alpha - 0.05) < 1e-9

    def test_rho_gate_enabled_defaults_true(self):
        """ρ (phase coherence) should gate the P3 context gate by default."""
        from neuroslm.dsl.training_config import SemanticTurbulenceConfig
        cfg = SemanticTurbulenceConfig()
        assert cfg.rho_gate_enabled is True


# ── TrainingConfig integration ────────────────────────────────────────────


class TestSemanticTurbulenceInTrainingConfig:
    """TrainingConfig must carry semantic_turbulence field."""

    def test_training_config_has_semantic_turbulence(self):
        from neuroslm.dsl.training_config import TrainingConfig, SemanticTurbulenceConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, "semantic_turbulence")
        assert isinstance(cfg.semantic_turbulence, SemanticTurbulenceConfig)

    def test_default_is_disabled(self):
        from neuroslm.dsl.training_config import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.semantic_turbulence.enabled is False


# ── DSL parser ────────────────────────────────────────────────────────────


class TestSemanticTurbulenceDSLParsing:
    """parse_training_config must read semantic_turbulence { } correctly."""

    def test_parses_enabled_true(self):
        cfg = _parse("{ enabled: true }")
        assert cfg.semantic_turbulence.enabled is True

    def test_parses_enabled_false(self):
        cfg = _parse("{ enabled: false }")
        assert cfg.semantic_turbulence.enabled is False

    def test_parses_n_rg_groups(self):
        cfg = _parse("{ enabled: true, n_rg_groups: 4 }")
        assert cfg.semantic_turbulence.n_rg_groups == 4

    def test_parses_kolmogorov_init(self):
        cfg = _parse("{ enabled: true, kolmogorov_init: false }")
        assert cfg.semantic_turbulence.kolmogorov_init is False

    def test_parses_gpe_steps(self):
        cfg = _parse("{ enabled: true, gpe_steps: 8 }")
        assert cfg.semantic_turbulence.gpe_steps == 8

    def test_parses_gpe_coupling_init(self):
        cfg = _parse("{ enabled: true, gpe_coupling_init: 0.05 }")
        assert abs(cfg.semantic_turbulence.gpe_coupling_init - 0.05) < 1e-9

    def test_parses_gpe_dt(self):
        cfg = _parse("{ enabled: true, gpe_dt: 0.005 }")
        assert abs(cfg.semantic_turbulence.gpe_dt - 0.005) < 1e-9

    def test_parses_criticality_target(self):
        cfg = _parse("{ enabled: true, criticality_target: 0.9 }")
        assert abs(cfg.semantic_turbulence.criticality_target - 0.9) < 1e-9

    def test_parses_criticality_weight(self):
        cfg = _parse("{ enabled: true, criticality_weight: 0.05 }")
        assert abs(cfg.semantic_turbulence.criticality_weight - 0.05) < 1e-9

    def test_parses_criticality_da_reward(self):
        cfg = _parse("{ enabled: true, criticality_da_reward: 0.2 }")
        assert abs(cfg.semantic_turbulence.criticality_da_reward - 0.2) < 1e-9

    def test_parses_criticality_ema_alpha(self):
        cfg = _parse("{ enabled: true, criticality_ema_alpha: 0.1 }")
        assert abs(cfg.semantic_turbulence.criticality_ema_alpha - 0.1) < 1e-9

    def test_parses_rho_gate_enabled(self):
        cfg = _parse("{ enabled: true, rho_gate_enabled: false }")
        assert cfg.semantic_turbulence.rho_gate_enabled is False

    def test_unknown_fields_ignored(self):
        """Forward-compat: unknown sub-fields must not raise."""
        cfg = _parse("{ enabled: true, future_field_xyz: 42 }")
        assert cfg.semantic_turbulence.enabled is True

    def test_empty_block_gives_defaults(self):
        cfg = _parse("{}")
        ste = cfg.semantic_turbulence
        assert ste.enabled is False
        assert ste.n_rg_groups == 3

    def test_full_block(self):
        """Canonical production block must parse without error."""
        cfg = _parse("""{
            enabled:               true,
            n_rg_groups:           3,
            kolmogorov_init:       true,
            gpe_steps:             4,
            gpe_coupling_init:     0.01,
            gpe_dt:                0.01,
            criticality_target:    1.0,
            criticality_weight:    0.01,
            criticality_da_reward: 0.1,
            criticality_ema_alpha: 0.05,
            rho_gate_enabled:      true
        }""")
        ste = cfg.semantic_turbulence
        assert ste.enabled is True
        assert ste.n_rg_groups == 3
        assert ste.kolmogorov_init is True
        assert ste.gpe_steps == 4
        assert abs(ste.gpe_coupling_init - 0.01) < 1e-9
        assert abs(ste.criticality_target - 1.0) < 1e-9
        assert ste.rho_gate_enabled is True
