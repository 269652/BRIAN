"""Contracts for the ``experts: [...]`` block inside ``multi_cortex { ... }``.

The expert-roster DSL replaces the legacy ``weights: "gpt2"`` shorthand with
an explicit, per-expert spec:

    multi_cortex: {
        enabled: true,
        experts: [
            { id: "gpt2-medium",                domain: "general",   freeze: true },
            { id: "microsoft/CodeGPT-small-py", domain: "code",      freeze: true },
            { id: "Qwen/Qwen2.5-0.5B",          domain: "reasoning", freeze: true },
        ],
        # ... other fields unchanged ...
    }

Validations:
  * each row has ``id`` (HF model id, str) and ``domain`` (str)
  * optional ``freeze`` (bool, default True)
  * optional ``weight`` (float ≥ 0, default 1.0) — multiplicative prior on
    the router output for this expert (orthogonal to the lexical bias)
  * if the legacy ``weights`` and the new ``experts`` are BOTH present,
    ``experts`` wins and a deprecation warning is emitted
  * if ``experts`` is present, ``n_cortices`` and ``domains`` are
    auto-derived from the roster (overriding any explicit values)

Back-compat: omitting ``experts`` leaves all other behaviour unchanged.
"""
from __future__ import annotations

import warnings

import pytest

from neuroslm.dsl.training_config import (
    ExpertSpec,
    MultiCortexConfig,
    _parse_multi_cortex,
)


# ──────────────────────────────────────────────────────────────────────
# Construction defaults
# ──────────────────────────────────────────────────────────────────────


class TestExpertSpecDefaults:
    def test_minimal_spec(self):
        e = ExpertSpec(id="gpt2", domain="general")
        assert e.id == "gpt2"
        assert e.domain == "general"
        assert e.freeze is True
        assert e.weight == 1.0

    def test_full_spec(self):
        e = ExpertSpec(
            id="Qwen/Qwen2.5-0.5B",
            domain="reasoning",
            freeze=False,
            weight=2.5,
        )
        assert e.id == "Qwen/Qwen2.5-0.5B"
        assert e.freeze is False
        assert e.weight == 2.5

    def test_rejects_negative_weight(self):
        with pytest.raises(ValueError, match="weight.*>= 0"):
            ExpertSpec(id="x", domain="y", weight=-0.5)

    def test_rejects_empty_id(self):
        with pytest.raises(ValueError, match="id.*non-empty"):
            ExpertSpec(id="", domain="y")

    def test_rejects_empty_domain(self):
        with pytest.raises(ValueError, match="domain.*non-empty"):
            ExpertSpec(id="x", domain="")


class TestMultiCortexConfigExpertsField:
    def test_default_is_none(self):
        cfg = MultiCortexConfig()
        assert cfg.experts is None

    def test_can_be_set(self):
        cfg = MultiCortexConfig(experts=[ExpertSpec(id="gpt2", domain="general")])
        assert cfg.experts is not None
        assert len(cfg.experts) == 1


# ──────────────────────────────────────────────────────────────────────
# Parser contracts
# ──────────────────────────────────────────────────────────────────────


class TestExpertsBlockParser:
    def test_parser_handles_empty_roster_omitted(self):
        body = "{ enabled: true, n_cortices: 4, domains: [\"a\", \"b\", \"c\", \"d\"] }"
        cfg = _parse_multi_cortex(body)
        assert cfg.experts is None

    def test_parser_handles_single_expert(self):
        body = """{
            enabled: true,
            experts: [
                { id: "gpt2", domain: "general" }
            ]
        }"""
        cfg = _parse_multi_cortex(body)
        assert cfg.experts is not None
        assert len(cfg.experts) == 1
        assert cfg.experts[0].id == "gpt2"
        assert cfg.experts[0].domain == "general"
        assert cfg.experts[0].freeze is True

    def test_parser_handles_multiple_experts(self):
        body = """{
            enabled: true,
            experts: [
                { id: "gpt2-medium",                domain: "general",   freeze: true },
                { id: "microsoft/CodeGPT-small-py", domain: "code",      freeze: true },
                { id: "Qwen/Qwen2.5-0.5B",          domain: "reasoning", freeze: true }
            ]
        }"""
        cfg = _parse_multi_cortex(body)
        assert cfg.experts is not None
        assert len(cfg.experts) == 3
        ids = [e.id for e in cfg.experts]
        assert ids == [
            "gpt2-medium",
            "microsoft/CodeGPT-small-py",
            "Qwen/Qwen2.5-0.5B",
        ]
        domains = [e.domain for e in cfg.experts]
        assert domains == ["general", "code", "reasoning"]

    def test_parser_propagates_freeze_false(self):
        body = """{
            enabled: true,
            experts: [
                { id: "gpt2", domain: "general", freeze: false }
            ]
        }"""
        cfg = _parse_multi_cortex(body)
        assert cfg.experts[0].freeze is False

    def test_parser_propagates_custom_weight(self):
        body = """{
            enabled: true,
            experts: [
                { id: "gpt2", domain: "general", weight: 1.5 }
            ]
        }"""
        cfg = _parse_multi_cortex(body)
        assert cfg.experts[0].weight == 1.5

    def test_parser_auto_derives_domains_from_roster(self):
        """When ``experts`` is present, ``domains`` and ``n_cortices``
        are auto-derived so the operator can't get them out of sync."""
        body = """{
            enabled: true,
            experts: [
                { id: "a", domain: "alpha" },
                { id: "b", domain: "beta" }
            ]
        }"""
        cfg = _parse_multi_cortex(body)
        assert cfg.domains == ["alpha", "beta"]
        assert cfg.n_cortices == 2

    def test_parser_rejects_missing_id(self):
        body = """{
            enabled: true,
            experts: [
                { domain: "general" }
            ]
        }"""
        with pytest.raises(ValueError, match="missing.*id"):
            _parse_multi_cortex(body)

    def test_parser_rejects_missing_domain(self):
        body = """{
            enabled: true,
            experts: [
                { id: "gpt2" }
            ]
        }"""
        with pytest.raises(ValueError, match="missing.*domain"):
            _parse_multi_cortex(body)

    def test_parser_rejects_negative_weight(self):
        body = """{
            enabled: true,
            experts: [
                { id: "gpt2", domain: "general", weight: -0.5 }
            ]
        }"""
        with pytest.raises(ValueError, match="weight.*>= 0"):
            _parse_multi_cortex(body)

    def test_parser_rejects_duplicate_domains(self):
        """Two experts can't claim the same domain — the router uses
        domain as the key. This is a guaranteed crash inside the
        ThalamicRouter so we catch it at parse time."""
        body = """{
            enabled: true,
            experts: [
                { id: "gpt2",        domain: "general" },
                { id: "gpt2-medium", domain: "general" }
            ]
        }"""
        with pytest.raises(ValueError, match="duplicate domain"):
            _parse_multi_cortex(body)


class TestLegacyShorthand:
    """The legacy ``weights: "gpt2"`` shorthand still works (back-compat),
    but emits a deprecation warning when ``experts: [...]`` is also set."""

    def test_legacy_only_unchanged(self):
        body = """{
            enabled: true,
            n_cortices: 4,
            domains: ["math", "code", "chat", "general"],
            weights: "gpt2"
        }"""
        cfg = _parse_multi_cortex(body)
        assert cfg.weights == "gpt2"
        assert cfg.experts is None

    def test_experts_wins_over_legacy_with_warning(self):
        body = """{
            enabled: true,
            weights: "gpt2",
            experts: [
                { id: "gpt2", domain: "general" }
            ]
        }"""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = _parse_multi_cortex(body)
        # Expert spec takes precedence
        assert cfg.experts is not None
        assert len(cfg.experts) == 1
        # Deprecation warning was emitted
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, DeprecationWarning)]
        assert any("experts" in m and "weights" in m for m in msgs), (
            f"expected deprecation warning, got: {msgs}"
        )


# ──────────────────────────────────────────────────────────────────────
# Cross-field validation
# ──────────────────────────────────────────────────────────────────────


class TestCrossFieldValidation:
    def test_empty_experts_list_rejected(self):
        body = """{
            enabled: true,
            experts: []
        }"""
        with pytest.raises(ValueError, match="experts.*at least one"):
            _parse_multi_cortex(body)

    def test_unknown_field_ignored_for_forward_compat(self):
        """Unknown per-expert fields are silently ignored so we can add
        new flags (e.g. ``bridge_mode``) without breaking older arch.neuro."""
        body = """{
            enabled: true,
            experts: [
                { id: "gpt2", domain: "general", future_flag: true }
            ]
        }"""
        cfg = _parse_multi_cortex(body)  # should not raise
        assert cfg.experts[0].id == "gpt2"
