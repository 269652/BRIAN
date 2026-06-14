# -*- coding: utf-8 -*-
"""TDD contract — Item 5: expert model alias registry.

Goal
====
Make every HuggingFace causal-LM plug-and-play in the DSL roster without
code changes. The DSL writes ``id: "smollm2_360m"`` (a short, stable
alias) OR a full HF model id (``"HuggingFaceTB/SmolLM2-360M"``) OR a
``hf://owner/repo`` URL. The alias registry resolves any of these to a
canonical HF id and ``LMExpert`` loads it generically.

Contract
========
* ``resolve_expert_alias("gpt2")``                   → ``"gpt2"``
* ``resolve_expert_alias("smollm2_360m")``           → ``"HuggingFaceTB/SmolLM2-360M"``
* ``resolve_expert_alias("smollm2_135m")``           → ``"HuggingFaceTB/SmolLM2-135M"``
* ``resolve_expert_alias("HuggingFaceTB/SmolLM2-360M")``
                                                      → identity (already canonical)
* ``resolve_expert_alias("hf://Qwen/Qwen2.5-0.5B")`` → ``"Qwen/Qwen2.5-0.5B"``
* ``resolve_expert_alias("microsoft/CodeGPT-small-py")``
                                                      → identity
* ``register_expert_alias("foo", "owner/foo-1b")`` then
  ``resolve_expert_alias("foo") == "owner/foo-1b"``
* Unknown bare-name alias (no ``/``, not registered) raises
  ``ValueError`` with the list of known aliases — never silently
  passes a typo to the HF hub.

The shipped registry must include at minimum:
  * gpt2, gpt2-medium, gpt2-large, distilgpt2  (back-compat)
  * smollm2_135m, smollm2_360m, smollm2_1_7b   (new SmolLM2 family)
  * qwen2_5_0_5b, qwen2_5_1_5b                 (Qwen recommended set)
  * codegpt_py                                 (current code expert)

Integration
===========
* ``LMExpert(model_id="smollm2_360m", ...)`` must resolve the alias
  internally before calling the HF loader.
* ``ExpertSpec(id="smollm2_360m", domain=...)`` round-trips through the
  parser: parsing then materialising must produce a spec whose ``.id``
  is the **alias** (unchanged), so the DSL stays human-readable. The
  resolution happens at LMExpert construction time only.
"""
from __future__ import annotations

import pytest


# ── Pure-function contracts (no HF I/O) ─────────────────────────────────


class TestResolveExpertAlias:
    """The alias resolver is the single source of truth — every
    HF-loading path in the codebase must go through it."""

    def test_module_exports_resolve_and_register(self):
        from neuroslm.experts import resolve_expert_alias, register_expert_alias
        assert callable(resolve_expert_alias)
        assert callable(register_expert_alias)

    def test_canonical_hf_id_is_identity(self):
        from neuroslm.experts import resolve_expert_alias
        # `owner/repo` form is always already canonical.
        for hf_id in [
            "HuggingFaceTB/SmolLM2-360M",
            "Qwen/Qwen2.5-0.5B",
            "microsoft/CodeGPT-small-py",
            "openai-community/gpt2",
        ]:
            assert resolve_expert_alias(hf_id) == hf_id, \
                f"canonical id {hf_id!r} must resolve to itself"

    def test_hf_url_scheme_is_stripped(self):
        from neuroslm.experts import resolve_expert_alias
        assert resolve_expert_alias("hf://Qwen/Qwen2.5-0.5B") \
            == "Qwen/Qwen2.5-0.5B"
        assert resolve_expert_alias("hf://gpt2") == "gpt2"

    def test_legacy_gpt2_family_aliases(self):
        """The gpt2 names ship without an owner prefix because HF
        accepts them as canonical model ids (back-compat with every
        existing arch.neuro)."""
        from neuroslm.experts import resolve_expert_alias
        for short in ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"]:
            assert resolve_expert_alias(short) == short

    def test_smollm2_aliases_resolve_to_huggingfacetb(self):
        from neuroslm.experts import resolve_expert_alias
        assert resolve_expert_alias("smollm2_135m") \
            == "HuggingFaceTB/SmolLM2-135M"
        assert resolve_expert_alias("smollm2_360m") \
            == "HuggingFaceTB/SmolLM2-360M"
        assert resolve_expert_alias("smollm2_1_7b") \
            == "HuggingFaceTB/SmolLM2-1.7B"

    def test_qwen_aliases_resolve_to_qwen_org(self):
        from neuroslm.experts import resolve_expert_alias
        assert resolve_expert_alias("qwen2_5_0_5b") == "Qwen/Qwen2.5-0.5B"
        assert resolve_expert_alias("qwen2_5_1_5b") == "Qwen/Qwen2.5-1.5B"

    def test_codegpt_alias(self):
        from neuroslm.experts import resolve_expert_alias
        assert resolve_expert_alias("codegpt_py") \
            == "microsoft/CodeGPT-small-py"

    def test_unknown_bare_alias_raises_with_hint(self):
        from neuroslm.experts import resolve_expert_alias
        with pytest.raises(ValueError) as ei:
            resolve_expert_alias("not_a_real_alias_xyz")
        msg = str(ei.value)
        assert "not_a_real_alias_xyz" in msg, "error message must echo the bad alias"
        # Must list at least one known alias so the user can see what's available.
        assert "smollm2_360m" in msg or "gpt2" in msg, (
            "error message must list known aliases as a hint"
        )

    def test_unknown_owner_repo_passes_through(self):
        """A novel ``owner/repo`` form is assumed to be a real HF model
        and passed through. Catching typos here would require a network
        round-trip; we trust the user when an owner prefix is present."""
        from neuroslm.experts import resolve_expert_alias
        assert resolve_expert_alias("acme/awesome-llm-9b") \
            == "acme/awesome-llm-9b"

    def test_register_alias_round_trips(self):
        from neuroslm.experts import resolve_expert_alias, register_expert_alias
        register_expert_alias("test_xyz_alias_42", "test-org/test-repo")
        try:
            assert resolve_expert_alias("test_xyz_alias_42") \
                == "test-org/test-repo"
        finally:
            # Cleanup so other tests don't see this alias.
            from neuroslm.experts import _EXPERT_ALIAS_REGISTRY
            _EXPERT_ALIAS_REGISTRY.pop("test_xyz_alias_42", None)

    def test_register_alias_rejects_bad_canonical(self):
        """``register_expert_alias("foo", "bar")`` must reject ``bar`` —
        a bare name without owner cannot be a canonical id (except for
        the legacy gpt2 family which is pre-registered)."""
        from neuroslm.experts import register_expert_alias
        with pytest.raises(ValueError):
            register_expert_alias("xyz", "no_slash")


# ── DSL round-trip contract ────────────────────────────────────────────


class TestDSLRosterAcceptsAliases:
    """An arch.neuro author writes the alias; ``ExpertSpec.id`` keeps
    the alias text. Resolution happens at LMExpert construction time."""

    def _parse_block(self, body: str):
        from neuroslm.dsl.training_config import _parse_multi_cortex
        return _parse_multi_cortex(body)

    def test_alias_survives_parse(self):
        mc = self._parse_block("""{
            enabled: true,
            experts: [
                { id: "smollm2_360m", domain: "general", freeze: true },
                { id: "codegpt_py",   domain: "code",    freeze: true }
            ],
            trunk_tokenizer: "gpt2"
        }""")
        assert mc.experts is not None
        ids = [e.id for e in mc.experts]
        assert ids == ["smollm2_360m", "codegpt_py"], (
            f"DSL alias text must be preserved through parsing, got {ids}"
        )

    def test_canonical_id_also_survives_parse(self):
        mc = self._parse_block("""{
            enabled: true,
            experts: [
                { id: "HuggingFaceTB/SmolLM2-360M", domain: "general" }
            ],
            trunk_tokenizer: "gpt2"
        }""")
        assert mc.experts is not None
        assert mc.experts[0].id == "HuggingFaceTB/SmolLM2-360M"


# ── LMExpert wiring (no HF download — uses cache mocking) ──────────────


class TestLMExpertResolvesAlias:
    """LMExpert must resolve the alias BEFORE calling the HF loader so
    the alias and canonical paths share one cache entry."""

    def test_alias_resolution_target_matches_canonical(self):
        """The HF id LMExpert would load for an alias must equal the
        HF id it would load for the alias's canonical target.
        Tested via the public resolver to avoid hitting HF.
        """
        from neuroslm.experts import resolve_expert_alias
        alias = "smollm2_360m"
        canonical = "HuggingFaceTB/SmolLM2-360M"
        # Both must resolve to the same target.
        assert resolve_expert_alias(alias) == resolve_expert_alias(canonical)
        assert resolve_expert_alias(alias) == canonical
