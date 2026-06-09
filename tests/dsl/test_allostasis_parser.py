# -*- coding: utf-8 -*-
"""TDD: the `allostasis { ... }` DSL block parses into AllostasisConfig.

The `arch.neuro` training block grows a new sibling alongside
`multi_cortex`, `mechanisms`, `genetics`, etc.::

    training {
        ...
        allostasis: {
            enabled:            true,
            load_ema_alpha:     0.10,
            cort_ema_alpha:     0.01,
            w_ne:               0.30,
            w_gaba:             0.20,
            w_loss:             0.30,
            w_grad:             0.20,
            ne_baseline:        0.25,
            gaba_baseline:      0.20,
            grad_norm_ceiling:  5.0,
            suppress_ne:        true,
            suppress_trophic:   true,
            suppress_lr:        true,
            gamma_ne:           0.7,
            gamma_trophic:      1.0,
            gamma_lr:           0.5
        }
    }

Contracts pinned by this suite:
  * Missing block ⇒ defaults (disabled).
  * Each declared key round-trips into the config dataclass.
  * Unknown keys are silently ignored (forward-compat).
"""
from __future__ import annotations

import pytest


def _parse(body: str):
    """Wrap a body in a training { ... } parser call."""
    from neuroslm.dsl.training_config import parse_training_config
    return parse_training_config(body)


# ───────────────────────────────────────────────────────────────────────
# Defaults
# ───────────────────────────────────────────────────────────────────────
class TestDefaults:

    def test_missing_block_yields_default_disabled_config(self):
        cfg = _parse("")
        from neuroslm.dsl.training_config import AllostasisConfig
        assert isinstance(cfg.allostasis, AllostasisConfig)
        assert cfg.allostasis.enabled is False, (
            "When arch.neuro declares no `allostasis` block the controller "
            "must remain disabled (back-compat with every existing arch)."
        )

    def test_empty_block_yields_disabled_config_but_dataclass_present(self):
        cfg = _parse("allostasis: { }")
        assert cfg.allostasis.enabled is False


# ───────────────────────────────────────────────────────────────────────
# Field round-trip
# ───────────────────────────────────────────────────────────────────────
class TestFieldRoundTrip:

    def test_enabled_true_parsed(self):
        cfg = _parse("allostasis: { enabled: true }")
        assert cfg.allostasis.enabled is True

    def test_ema_alphas_parsed(self):
        body = """
        allostasis: {
            enabled: true,
            load_ema_alpha: 0.20,
            cort_ema_alpha: 0.02
        }
        """
        cfg = _parse(body)
        assert cfg.allostasis.load_ema_alpha == 0.20
        assert cfg.allostasis.cort_ema_alpha == 0.02

    def test_stress_weights_parsed(self):
        body = """
        allostasis: {
            enabled: true,
            w_ne:   0.4,
            w_gaba: 0.1,
            w_loss: 0.3,
            w_grad: 0.2
        }
        """
        cfg = _parse(body)
        assert cfg.allostasis.w_ne   == 0.4
        assert cfg.allostasis.w_gaba == 0.1
        assert cfg.allostasis.w_loss == 0.3
        assert cfg.allostasis.w_grad == 0.2

    def test_baselines_parsed(self):
        body = """
        allostasis: {
            enabled: true,
            ne_baseline:       0.30,
            gaba_baseline:     0.25,
            grad_norm_ceiling: 8.0
        }
        """
        cfg = _parse(body)
        assert cfg.allostasis.ne_baseline       == 0.30
        assert cfg.allostasis.gaba_baseline     == 0.25
        assert cfg.allostasis.grad_norm_ceiling == 8.0

    def test_suppress_switches_parsed(self):
        body = """
        allostasis: {
            enabled: true,
            suppress_ne:      true,
            suppress_trophic: false,
            suppress_lr:      true
        }
        """
        cfg = _parse(body)
        assert cfg.allostasis.suppress_ne      is True
        assert cfg.allostasis.suppress_trophic is False
        assert cfg.allostasis.suppress_lr      is True

    def test_gammas_parsed(self):
        body = """
        allostasis: {
            enabled: true,
            gamma_ne:      0.6,
            gamma_trophic: 0.9,
            gamma_lr:      0.4
        }
        """
        cfg = _parse(body)
        assert cfg.allostasis.gamma_ne      == 0.6
        assert cfg.allostasis.gamma_trophic == 0.9
        assert cfg.allostasis.gamma_lr      == 0.4


# ───────────────────────────────────────────────────────────────────────
# Forward-compat: unknown keys ignored, not raised
# ───────────────────────────────────────────────────────────────────────
class TestForwardCompat:
    """Unknown keys inside an allostasis block must not crash the parser
    — same forward-compat contract as every other DSL sub-block."""

    def test_unknown_key_is_silently_ignored(self):
        body = """
        allostasis: {
            enabled: true,
            quack_factor: 1.5,
            load_ema_alpha: 0.15
        }
        """
        cfg = _parse(body)  # must not raise
        assert cfg.allostasis.enabled is True
        assert cfg.allostasis.load_ema_alpha == 0.15
