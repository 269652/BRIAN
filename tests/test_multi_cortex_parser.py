# -*- coding: utf-8 -*-
"""TDD acceptance suite — `training { multi_cortex { ... } }` block parser.

Step 1 of the 4-step Multi-Trunk-V2 ⇒ HuggingFace wiring sequence:

  1. PARSER       — this file: assert `parse_training_config` recognises
                    the multi_cortex block and populates a dataclass.
  2. requirements — transformers>=4.30 added to requirements.txt.
  3. harness inj  — `BRIANHarness.from_multi_cortex_ensemble(...)` builds
                    a GPT-2 ensemble from the parsed config.
  4. integration  — end-to-end test: arch.neuro ⇒ harness ⇒ stub ensemble.

Why a dedicated parser test: the existing `tests/test_ensemble_routing.py`
covers the Python side only and uses `StubSubCortex` directly — it never
touches the DSL. Without this test, a typo in arch.neuro's
`multi_cortex` block (or a missed key in the parser) silently produces
a vanilla LM run that never invokes the ensemble — exactly the failure
mode this step closes.
"""
from __future__ import annotations

import pytest

from neuroslm.dsl.training_config import (
    MultiCortexConfig,
    TrainingConfig,
    parse_training_config,
)


# ──────────────────────────────────────────────────────────────────────
# Default-construction contract
# ──────────────────────────────────────────────────────────────────────

class TestMultiCortexConfigDefaults:
    """A freshly constructed config must be disabled with safe defaults
    so legacy archs (no multi_cortex block) train identically to before."""

    def test_defaults_are_disabled(self):
        cfg = MultiCortexConfig()
        assert cfg.enabled is False
        assert cfg.n_cortices == 4
        assert cfg.weights == "stub"           # safe default = no HF download
        assert cfg.freeze_weights is True
        assert cfg.lexical_bias_weight == 2.0
        assert cfg.bema_tau == 0.5
        assert cfg.router_d_model == 256
        assert cfg.domains == ["math", "code", "chat", "general"]

    def test_training_config_carries_default_multi_cortex(self):
        """A TrainingConfig built with no args must own a default-disabled
        MultiCortexConfig so harness wiring code can read it safely."""
        tc = TrainingConfig()
        assert isinstance(tc.multi_cortex, MultiCortexConfig)
        assert tc.multi_cortex.enabled is False


# ──────────────────────────────────────────────────────────────────────
# Parser contract — the headline TDD requirement
# ──────────────────────────────────────────────────────────────────────

class TestParseMultiCortexBlock:
    """`parse_training_config` must read every documented multi_cortex
    field from an arch.neuro snippet."""

    def test_parses_minimal_enabled_block(self):
        body = """
            multi_cortex: {
                enabled: true
            }
        """
        cfg = parse_training_config(body)
        assert cfg.multi_cortex.enabled is True
        # All other fields keep their defaults
        assert cfg.multi_cortex.weights == "stub"
        assert cfg.multi_cortex.n_cortices == 4

    def test_parses_full_gpt2_production_block(self):
        """The exact block currently shipped in arch.neuro Section 5.7."""
        body = """
            multi_cortex: {
                enabled:             true,
                n_cortices:          4,
                domains:             ["math", "code", "chat", "general"],
                weights:             "gpt2",
                freeze_weights:      true,
                lexical_bias_weight: 2.0,
                bema_tau:            0.5,
                router_d_model:      256
            }
        """
        cfg = parse_training_config(body)
        mc = cfg.multi_cortex
        assert mc.enabled is True
        assert mc.n_cortices == 4
        assert mc.domains == ["math", "code", "chat", "general"]
        assert mc.weights == "gpt2"
        assert mc.freeze_weights is True
        assert mc.lexical_bias_weight == pytest.approx(2.0)
        assert mc.bema_tau == pytest.approx(0.5)
        assert mc.router_d_model == 256

    def test_parses_custom_domains_and_two_cortices(self):
        body = """
            multi_cortex: {
                enabled: true,
                n_cortices: 2,
                domains: ["specialist", "generalist"],
                weights: "stub",
                bema_tau: 0.0
            }
        """
        cfg = parse_training_config(body)
        assert cfg.multi_cortex.n_cortices == 2
        assert cfg.multi_cortex.domains == ["specialist", "generalist"]
        assert cfg.multi_cortex.bema_tau == 0.0

    def test_freeze_weights_false_round_trip(self):
        body = """
            multi_cortex: {
                enabled: true,
                freeze_weights: false
            }
        """
        cfg = parse_training_config(body)
        assert cfg.multi_cortex.freeze_weights is False

    def test_unknown_weights_value_raises(self):
        """The parser must reject typos like `weights: "gtp2"` so a bad
        arch.neuro fails LOUDLY at compile time, not silently at run
        time after a 1.6 GB HF download."""
        body = """
            multi_cortex: {
                enabled: true,
                weights: "deepseek"
            }
        """
        with pytest.raises(ValueError, match="weights"):
            parse_training_config(body)

    def test_n_cortices_must_match_domains_length(self):
        """A mismatch between `n_cortices` and `len(domains)` is a
        guaranteed run-time crash inside ThalamicRouter — surface it
        at parse time instead."""
        body = """
            multi_cortex: {
                enabled: true,
                n_cortices: 3,
                domains: ["a", "b"]
            }
        """
        with pytest.raises(ValueError, match="n_cortices"):
            parse_training_config(body)

    def test_bema_tau_must_be_in_unit_interval_half_open(self):
        body = """
            multi_cortex: {
                enabled: true,
                bema_tau: 1.5
            }
        """
        with pytest.raises(ValueError, match="bema_tau"):
            parse_training_config(body)

    def test_lexical_bias_weight_must_be_non_negative(self):
        body = """
            multi_cortex: {
                enabled: true,
                lexical_bias_weight: -1.0
            }
        """
        with pytest.raises(ValueError, match="lexical_bias_weight"):
            parse_training_config(body)


# ──────────────────────────────────────────────────────────────────────
# Forward-compat: absence ⇒ disabled, no crash on legacy arch.neuro
# ──────────────────────────────────────────────────────────────────────

class TestParseLegacyArchsNoMultiCortex:
    """All existing arch.neuro files without a multi_cortex block must
    continue to parse and yield `multi_cortex.enabled = False`."""

    def test_empty_training_body_is_disabled(self):
        cfg = parse_training_config("")
        assert cfg.multi_cortex.enabled is False

    def test_unrelated_training_keys_dont_affect_multi_cortex(self):
        body = """
            optimizer: "adamw"
            learning_rate: 0.0003
            batch_size: 16
        """
        cfg = parse_training_config(body)
        assert cfg.multi_cortex.enabled is False
        # And the unrelated fields still parsed
        assert cfg.optimizer == "adamw"
        assert cfg.batch_size == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
