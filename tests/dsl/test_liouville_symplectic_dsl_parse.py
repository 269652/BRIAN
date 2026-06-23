# -*- coding: utf-8 -*-
"""DSL parser contract for the liouville_symplectic regularization block.

Pins that the .neuro grammar accepts the block and produces a typed
LiouvilleSymplecticConfig with the documented defaults and bounds.
Companion to tests/dsl/test_liouville_symplectic.py (which pins the
underlying math).
"""
from __future__ import annotations

import pytest

from neuroslm.dsl.regularization import (
    RegularizationConfig,
    LiouvilleSymplecticConfig,
    parse_regularization_block,
)


def _parse(body: str) -> RegularizationConfig:
    return parse_regularization_block(body)


class TestLiouvilleSymplecticDefaults:
    def test_dataclass_defaults(self):
        cfg = LiouvilleSymplecticConfig()
        assert cfg.enabled is False
        assert cfg.noether_strength == 0.0
        assert cfg.dtau_init == pytest.approx(0.1)
        assert cfg.potential_kind == "quadratic"
        assert cfg.w_rank == 4

    def test_disabled_by_default_in_regularization_config(self):
        cfg = RegularizationConfig()
        assert cfg.liouville_symplectic.enabled is False

    def test_enabling_flips_any_enabled(self):
        cfg = RegularizationConfig()
        cfg.liouville_symplectic.enabled = True
        assert cfg.any_enabled() is True


class TestLiouvilleSymplecticParse:
    def test_parses_minimal_enable_only(self):
        body = "liouville_symplectic: { enabled: true }"
        cfg = _parse(body)
        ls = cfg.liouville_symplectic
        assert ls.enabled is True
        # All other fields stay at defaults.
        assert ls.noether_strength == 0.0
        assert ls.dtau_init == pytest.approx(0.1)
        assert ls.potential_kind == "quadratic"
        assert ls.w_rank == 4

    def test_parses_full_block(self):
        body = (
            "liouville_symplectic: { enabled: true, noether_strength: 0.01, "
            "dtau_init: 0.05, potential_kind: swiglu, w_rank: 8 }"
        )
        cfg = _parse(body)
        ls = cfg.liouville_symplectic
        assert ls.enabled is True
        assert ls.noether_strength == pytest.approx(0.01)
        assert ls.dtau_init == pytest.approx(0.05)
        assert ls.potential_kind == "swiglu"
        assert ls.w_rank == 8

    def test_empty_block_keeps_defaults(self):
        cfg = _parse("liouville_symplectic: { }")
        ls = cfg.liouville_symplectic
        assert ls.enabled is False
        assert ls.noether_strength == 0.0

    def test_rejects_negative_noether_strength(self):
        with pytest.raises(ValueError, match="noether_strength"):
            _parse("liouville_symplectic: { enabled: true, noether_strength: -0.01 }")

    def test_rejects_negative_dtau_init(self):
        with pytest.raises(ValueError, match="dtau_init"):
            _parse("liouville_symplectic: { enabled: true, dtau_init: -0.1 }")

    def test_block_coexists_with_other_regs(self):
        body = (
            "warmup_steps: 500, "
            "isotropy: { enabled: true, weight: 0.05 }, "
            "liouville_symplectic: { enabled: true, noether_strength: 0.01 }"
        )
        cfg = _parse(body)
        assert cfg.warmup_steps == 500
        assert cfg.isotropy.enabled is True
        assert cfg.liouville_symplectic.enabled is True
        assert cfg.liouville_symplectic.noether_strength == pytest.approx(0.01)
        # Other regs stay default.
        assert cfg.dar.enabled is False
        assert cfg.pontryagin_topo_charge.enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
