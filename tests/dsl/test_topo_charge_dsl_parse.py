# -*- coding: utf-8 -*-
"""DSL parser contract for the pontryagin_topo_charge regularization block.

Pins that the .neuro grammar accepts the block and produces a typed
PontryaginTopoChargeConfig with the documented defaults and bounds.
Companion to tests/dsl/test_topo_charge.py (which pins the underlying
math) and tests/dsl/test_topo_charge_stub_audit.py (which pins the
contract strength).
"""
from __future__ import annotations

import pytest

from neuroslm.dsl.regularization import (
    RegularizationConfig,
    PontryaginTopoChargeConfig,
    parse_regularization_block,
)


def _parse(body: str) -> RegularizationConfig:
    return parse_regularization_block(body)


class TestPontryaginTopoChargeDefaults:
    """Defaults match the dataclass spec and any_enabled() reflects them."""

    def test_dataclass_defaults(self):
        cfg = PontryaginTopoChargeConfig()
        assert cfg.enabled is False
        assert cfg.alpha == 0.0
        assert cfg.gamma == 0.0
        assert cfg.Q_target == 0.0
        assert cfg.weight_init_std == 0.02

    def test_disabled_by_default_in_regularization_config(self):
        cfg = RegularizationConfig()
        assert cfg.pontryagin_topo_charge.enabled is False
        assert cfg.any_enabled() is False

    def test_enabling_topo_charge_flips_any_enabled(self):
        cfg = RegularizationConfig()
        cfg.pontryagin_topo_charge.enabled = True
        assert cfg.any_enabled() is True


class TestPontryaginTopoChargeParse:
    """End-to-end DSL parse for the new block."""

    def test_parses_minimal_enable_only(self):
        body = "pontryagin_topo_charge: { enabled: true }"
        cfg = _parse(body)
        tc = cfg.pontryagin_topo_charge
        assert tc.enabled is True
        # All other fields stay at dataclass defaults.
        assert tc.alpha == 0.0
        assert tc.gamma == 0.0
        assert tc.Q_target == 0.0
        assert tc.weight_init_std == 0.02

    def test_parses_full_block(self):
        body = ("pontryagin_topo_charge: { enabled: true, alpha: 0.05, "
                "gamma: 0.01, Q_target: 0.25, weight_init_std: 0.05 }")
        cfg = _parse(body)
        tc = cfg.pontryagin_topo_charge
        assert tc.enabled is True
        assert tc.alpha == pytest.approx(0.05)
        assert tc.gamma == pytest.approx(0.01)
        assert tc.Q_target == pytest.approx(0.25)
        assert tc.weight_init_std == pytest.approx(0.05)

    def test_empty_block_keeps_defaults(self):
        cfg = _parse("pontryagin_topo_charge: { }")
        tc = cfg.pontryagin_topo_charge
        assert tc.enabled is False
        assert tc.alpha == 0.0

    def test_rejects_negative_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            _parse(
                "pontryagin_topo_charge: "
                "{ enabled: true, alpha: -0.1 }"
            )

    def test_rejects_negative_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            _parse(
                "pontryagin_topo_charge: "
                "{ enabled: true, gamma: -0.001 }"
            )

    def test_rejects_zero_weight_init_std(self):
        """weight_init_std == 0 would re-introduce the section 14
        banned decorative-mechanism failure mode (Q_h becomes input-
        independent). The parser must reject it."""
        with pytest.raises(ValueError, match="weight_init_std"):
            _parse(
                "pontryagin_topo_charge: "
                "{ enabled: true, weight_init_std: 0.0 }"
            )

    def test_block_coexists_with_other_regs(self):
        """Sanity: enabling topo_charge does not perturb the other 7
        interventions."""
        body = (
            "warmup_steps: 1000, "
            "isotropy: { enabled: true, weight: 0.05 }, "
            "pontryagin_topo_charge: { enabled: true, alpha: 0.01 }"
        )
        cfg = _parse(body)
        assert cfg.warmup_steps == 1000
        assert cfg.isotropy.enabled is True
        assert cfg.isotropy.weight == pytest.approx(0.05)
        assert cfg.pontryagin_topo_charge.enabled is True
        assert cfg.pontryagin_topo_charge.alpha == pytest.approx(0.01)
        # Other 6 stay default-disabled.
        assert cfg.dar.enabled is False
        assert cfg.pcc.enabled is False
        assert cfg.cmd.enabled is False
        assert cfg.adaptive_mixture.enabled is False
        assert cfg.freq_balance.enabled is False
        assert cfg.cdga.enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
