"""Tests for the NFO DSL parsing — ``training { nfo: { ... } }`` round-trips
to ``TrainingConfig.nfo`` and feeds the factory."""
from __future__ import annotations

import pytest

from neuroslm.dsl.novel_topology import make_nfo
from neuroslm.dsl.training_config import parse_training_config


# NOTE: ``parse_training_config`` expects the *body* of a ``training { ... }``
# block (without the wrapping braces) — see the function's own docstring. The
# constants below therefore look like the inside of a training block.
_NFO_ON = """
    optimizer:     "adamw"
    learning_rate: 0.0003
    nfo: {
        enabled: true,
        n_osc: 16,
        n_steps: 2,
        kappa_init: 0.1,
        alpha_init: 0.0,
        mu_init: 0.5,
        a_star_init: 1.0,
        expose_phi_lower_bound: true
    }
"""

_NFO_OFF = """
    optimizer:     "adamw"
    learning_rate: 0.0003
    nfo: { enabled: false }
"""

_NFO_ABSENT = """
    optimizer:     "adamw"
    learning_rate: 0.0003
"""


class TestNFOConfigParse:

    def test_nfo_block_parses_to_dict(self):
        cfg = parse_training_config(_NFO_ON)
        assert cfg.nfo is not None
        assert cfg.nfo["enabled"] is True
        assert cfg.nfo["n_osc"] == 16
        assert cfg.nfo["n_steps"] == 2
        assert cfg.nfo["kappa_init"] == pytest.approx(0.1)
        assert cfg.nfo["alpha_init"] == pytest.approx(0.0)
        assert cfg.nfo["mu_init"] == pytest.approx(0.5)
        assert cfg.nfo["a_star_init"] == pytest.approx(1.0)
        assert cfg.nfo["expose_phi_lower_bound"] is True

    def test_nfo_disabled_round_trips(self):
        cfg = parse_training_config(_NFO_OFF)
        assert cfg.nfo is not None
        assert cfg.nfo.get("enabled") is False

    def test_nfo_absent_is_none(self):
        cfg = parse_training_config(_NFO_ABSENT)
        assert cfg.nfo is None

    def test_factory_builds_module_from_parsed_dict(self):
        cfg = parse_training_config(_NFO_ON)
        blk = make_nfo(cfg.nfo, d_model=64)
        assert blk is not None
        assert blk.cfg.n_osc == 16
        assert blk.cfg.n_steps == 2

    def test_factory_returns_none_on_disabled(self):
        cfg = parse_training_config(_NFO_OFF)
        assert make_nfo(cfg.nfo, d_model=64) is None

    def test_factory_returns_none_when_absent(self):
        cfg = parse_training_config(_NFO_ABSENT)
        assert make_nfo(cfg.nfo, d_model=64) is None
