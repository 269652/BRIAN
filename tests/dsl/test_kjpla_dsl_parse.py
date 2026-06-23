# -*- coding: utf-8 -*-
"""RED-first DSL parse tests for KJPLAPhaseLatticeConfig (Phase 3).

These tests verify that `regularization { kjpla_phase_lattice { ... } }`
in a .neuro file parses correctly to KJPLAPhaseLatticeConfig.

All tests will RED until KJPLAPhaseLatticeConfig is added to
neuroslm/dsl/regularization.py.
"""
from __future__ import annotations

import pytest
from neuroslm.dsl.regularization import (
    parse_regularization_block,
    KJPLAPhaseLatticeConfig,
    RegularizationConfig,
)


class TestKJPLAPhaseLatticeDefaults:
    def test_default_disabled(self):
        cfg = KJPLAPhaseLatticeConfig()
        assert cfg.enabled is False

    def test_default_josephson_strength_zero(self):
        cfg = KJPLAPhaseLatticeConfig()
        assert cfg.josephson_strength == 0.0

    def test_default_entropy_strength_zero(self):
        cfg = KJPLAPhaseLatticeConfig()
        assert cfg.entropy_strength == 0.0

    def test_default_eps_H_positive(self):
        """eps_H (entropy floor in nats) must be positive by default."""
        cfg = KJPLAPhaseLatticeConfig()
        assert cfg.eps_H > 0.0

    def test_instance_of_dataclass(self):
        cfg = KJPLAPhaseLatticeConfig()
        import dataclasses
        assert dataclasses.is_dataclass(cfg)


class TestKJPLAFieldOnRegularizationConfig:
    def test_field_exists(self):
        cfg = RegularizationConfig()
        assert hasattr(cfg, "kjpla_phase_lattice")

    def test_field_is_kjpla_config(self):
        cfg = RegularizationConfig()
        assert isinstance(cfg.kjpla_phase_lattice, KJPLAPhaseLatticeConfig)

    def test_kjpla_in_any_enabled_false_by_default(self):
        cfg = RegularizationConfig()
        # Default is disabled; any_enabled should not flip to True for kjpla alone.
        cfg2 = RegularizationConfig()
        cfg2.kjpla_phase_lattice.enabled = False
        assert not cfg2.any_enabled()

    def test_kjpla_in_any_enabled_true_when_enabled(self):
        cfg = RegularizationConfig()
        cfg.kjpla_phase_lattice.enabled = True
        assert cfg.any_enabled()


class TestKJPLAParseFromDSL:
    def _parse(self, body: str) -> RegularizationConfig:
        return parse_regularization_block(body)

    def test_parse_empty_gives_disabled_kjpla(self):
        cfg = self._parse("")
        assert not cfg.kjpla_phase_lattice.enabled

    def test_parse_kjpla_enabled(self):
        raw = "kjpla_phase_lattice: { enabled: true }"
        cfg = self._parse(raw)
        assert cfg.kjpla_phase_lattice.enabled is True

    def test_parse_josephson_strength(self):
        raw = "kjpla_phase_lattice: { enabled: true, josephson_strength: 0.05 }"
        cfg = self._parse(raw)
        assert abs(cfg.kjpla_phase_lattice.josephson_strength - 0.05) < 1e-9

    def test_parse_entropy_strength(self):
        raw = "kjpla_phase_lattice: { enabled: true, entropy_strength: 0.01 }"
        cfg = self._parse(raw)
        assert abs(cfg.kjpla_phase_lattice.entropy_strength - 0.01) < 1e-9

    def test_parse_eps_H(self):
        raw = "kjpla_phase_lattice: { eps_H: 0.5 }"
        cfg = self._parse(raw)
        assert abs(cfg.kjpla_phase_lattice.eps_H - 0.5) < 1e-9

    def test_parse_josephson_strength_negative_raises(self):
        """Negative josephson_strength is invalid."""
        raw = "kjpla_phase_lattice: { josephson_strength: -1.0 }"
        with pytest.raises(ValueError, match="josephson_strength"):
            self._parse(raw)

    def test_parse_entropy_strength_negative_raises(self):
        """Negative entropy_strength is invalid."""
        raw = "kjpla_phase_lattice: { entropy_strength: -0.1 }"
        with pytest.raises(ValueError, match="entropy_strength"):
            self._parse(raw)

    def test_parse_eps_H_zero_raises(self):
        """eps_H <= 0 is invalid."""
        raw = "kjpla_phase_lattice: { eps_H: 0.0 }"
        with pytest.raises(ValueError, match="eps_H"):
            self._parse(raw)

    def test_full_block(self):
        raw = (
            "kjpla_phase_lattice: { enabled: true, "
            "josephson_strength: 0.1, entropy_strength: 0.05, eps_H: 0.3 }"
        )
        cfg = self._parse(raw)
        assert cfg.kjpla_phase_lattice.enabled
        assert abs(cfg.kjpla_phase_lattice.josephson_strength - 0.1) < 1e-9
        assert abs(cfg.kjpla_phase_lattice.entropy_strength - 0.05) < 1e-9
        assert abs(cfg.kjpla_phase_lattice.eps_H - 0.3) < 1e-9
