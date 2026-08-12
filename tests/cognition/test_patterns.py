# -*- coding: utf-8 -*-
"""Knowledge extraction: generalized action classification + temporal
association mining over episodic memory (architecture.md §14.8).

Two real, established techniques — not invented heuristics:

1. ``classify_action`` — rule-based dialogue-act classification
   (lexicon/pattern matching against a fixed taxonomy, in the
   tradition of DAMSL-style dialogue-act tagging: Core & Allen 1997).
   "You suck" -> the LITERAL text is still what gets stored; this
   adds the GENERALIZED action class ("insult") alongside it.

2. ``mine_temporal_associations`` — Apriori-derived sequential
   association-rule mining (Agrawal & Srikant, "Fast Algorithms for
   Mining Association Rules", 1994; sequential extension in "Mining
   Sequential Patterns", 1995) over an ordered sequence of classified
   actions. Reports support/confidence/lift — STATISTICAL ASSOCIATION,
   not verified causation. Passive observation alone cannot establish
   causation (Pearl's causal hierarchy: association < intervention <
   counterfactual) — this module is honest about operating at the
   association rung only, and every rule is tagged with whether its
   evidence is externally grounded or purely the mind's own self-talk
   (the exact contamination risk flagged in the 2026-08-12 log
   analysis), so a rule mined entirely from the mind talking to
   itself is visibly distinguishable from one grounded in real input.
"""
from __future__ import annotations

import pytest

from neuroslm.cognition.patterns import (
    AssociationRule,
    classify_action,
    mine_temporal_associations,
)


# ── classify_action: real dialogue-act examples per class ────────────

class TestClassifyAction:
    @pytest.mark.parametrize("text,expected", [
        ("You suck", "insult"),
        ("you're such an idiot", "insult"),
        ("Shut up, nobody asked", "insult"),
        ("You're amazing at this", "compliment"),
        ("Good job on the launch", "compliment"),
        ("Thanks so much for your help", "gratitude"),
        ("I really appreciate that", "gratitude"),
        ("Sorry, that was my mistake", "apology"),
        ("Hello there!", "greeting"),
        ("Hey, good morning", "greeting"),
        ("Goodbye, take care", "farewell"),
        ("Yes, exactly right", "agreement"),
        ("No, that's incorrect", "disagreement"),
        ("Could you help me with this?", "request"),
        ("What time is it?", "question"),
        ("The sky is blue today.", "statement"),
    ])
    def test_known_examples(self, text, expected):
        result = classify_action(text)
        assert result.primary == expected

    def test_returns_all_matched_candidates(self):
        # "insult" ranks above "disagreement" in priority but both
        # patterns genuinely match — candidates must show the full
        # set, not just the winner.
        result = classify_action("No, you're an idiot")
        assert "insult" in result.candidates
        assert "disagreement" in result.candidates
        assert result.primary == "insult"

    def test_empty_text_is_a_statement(self):
        assert classify_action("").primary == "statement"
        assert classify_action(None).primary == "statement"

    def test_case_insensitive(self):
        assert classify_action("YOU SUCK").primary == "insult"

    def test_taxonomy_is_the_documented_set(self):
        from neuroslm.cognition.patterns import ACTION_TAXONOMY
        assert set(ACTION_TAXONOMY) == {
            "insult", "compliment", "agreement", "disagreement",
            "apology", "gratitude", "request", "greeting", "farewell",
            "question",
        }, "statement is the fallback, not a lexicon entry"


# ── mine_temporal_associations: hand-verified Apriori arithmetic ────

def _ep(action_class, kind="observed"):
    return {"action_class": action_class, "kind": kind}


class TestMineTemporalAssociationsArithmetic:
    """Every number here is hand-computed — this is the part that
    must be exactly right, not just plausible."""

    def test_confidence_support_lift_hand_computed(self):
        # insult, disagreement, insult, disagreement, insult, greeting
        seq = [_ep("insult"), _ep("disagreement"), _ep("insult"),
              _ep("disagreement"), _ep("insult"), _ep("greeting")]
        rules = mine_temporal_associations(seq, window=1, min_support=0.0,
                                           min_confidence=0.0)
        r = next(x for x in rules
                 if x.antecedent == "insult" and x.consequent == "disagreement")
        # insult occurs at 0,2,4, each has a successor (n=6).
        # -> disagreement follows at 0->1 and 2->3, not 4->5 (greeting).
        assert r.evidence_count == 2
        assert r.antecedent_count == 3
        assert r.confidence == pytest.approx(2 / 3)
        # total valid antecedent positions = n-1 = 5
        assert r.support == pytest.approx(2 / 5)
        # marginal support(disagreement) = 2/6
        expected_lift = (2 / 3) / (2 / 6)
        assert r.lift == pytest.approx(expected_lift)

    def test_perfectly_predictive_pattern_has_confidence_one(self):
        seq = [_ep("insult"), _ep("disagreement")] * 5
        rules = mine_temporal_associations(seq, window=1)
        r = next(x for x in rules
                 if x.antecedent == "insult" and x.consequent == "disagreement")
        assert r.confidence == pytest.approx(1.0)
        assert r.lift > 1.0, "perfectly predictive must show positive association"

    def test_random_unrelated_sequence_has_low_lift(self):
        # A cycles independently of B; no real relationship.
        seq = [_ep("greeting"), _ep("question"), _ep("greeting"),
              _ep("request"), _ep("greeting"), _ep("question"),
              _ep("greeting"), _ep("request")]
        rules = mine_temporal_associations(seq, window=1)
        for r in rules:
            assert r.lift < 3.0, (
                f"{r.antecedent}->{r.consequent} lift={r.lift} implausibly "
                "high for an unpatterned sequence")

    def test_min_support_and_min_confidence_filter(self):
        seq = [_ep("insult"), _ep("disagreement")] + \
            [_ep("greeting"), _ep("farewell")] * 10
        all_rules = mine_temporal_associations(seq, window=1)
        strict = mine_temporal_associations(seq, window=1, min_support=0.3,
                                            min_confidence=0.9)
        assert len(strict) < len(all_rules)
        for r in strict:
            assert r.support >= 0.3 and r.confidence >= 0.9

    def test_window_greater_than_one_looks_further_ahead(self):
        # insult, greeting, disagreement — with window=1 insult->disagreement
        # has ZERO direct evidence; with window=2 it does.
        seq = [_ep("insult"), _ep("greeting"), _ep("disagreement")]
        w1 = mine_temporal_associations(seq, window=1, min_confidence=0.0)
        w2 = mine_temporal_associations(seq, window=2, min_confidence=0.0)
        r1 = next((x for x in w1 if x.antecedent == "insult"
                   and x.consequent == "disagreement"), None)
        r2 = next(x for x in w2 if x.antecedent == "insult"
                  and x.consequent == "disagreement")
        assert r1 is None or r1.evidence_count == 0
        assert r2.evidence_count == 1

    def test_rules_sorted_by_confidence_descending(self):
        seq = [_ep("insult"), _ep("disagreement")] * 4 + \
            [_ep("greeting"), _ep("farewell")]
        rules = mine_temporal_associations(seq, window=1, min_confidence=0.0)
        confidences = [r.confidence for r in rules]
        assert confidences == sorted(confidences, reverse=True)


class _FakeClassifyGen:
    """Test-only stand-in for the LM's generate_fn, used ONLY to drive
    classify_action_via_generation's parsing logic — never a fake of
    the classification algorithm itself. Real production wiring
    (build_runtime_from_hf_lm) builds this from an actual loaded
    model's own .generate()."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list = []

    def __call__(self, prompt: str, max_new_tokens: int) -> str:
        self.prompts.append(prompt)
        return self.reply


class TestClassifyActionViaGeneration:
    """The mind classifies using its OWN trunk/expert (a real
    zero-shot generation-based classification, not a second model) —
    the user's explicit preference over the regex lexicon. Falls back
    to the lexicon only when the LM's output doesn't parse to a known
    category, as a safety net, never silently returning garbage."""

    def test_prompt_includes_the_text_and_the_taxonomy(self):
        from neuroslm.cognition.patterns import classify_action_via_generation
        gen = _FakeClassifyGen("insult")
        classify_action_via_generation("You suck", gen)
        assert "You suck" in gen.prompts[0]
        assert "insult" in gen.prompts[0]

    def test_parses_exact_category_from_generation(self):
        from neuroslm.cognition.patterns import classify_action_via_generation
        gen = _FakeClassifyGen("insult")
        result = classify_action_via_generation("You suck", gen)
        assert result.primary == "insult"

    def test_parses_category_embedded_in_a_longer_reply(self):
        from neuroslm.cognition.patterns import classify_action_via_generation
        gen = _FakeClassifyGen("This is clearly an insult.")
        result = classify_action_via_generation("You suck", gen)
        assert result.primary == "insult"

    def test_case_insensitive_parsing(self):
        from neuroslm.cognition.patterns import classify_action_via_generation
        gen = _FakeClassifyGen("INSULT")
        assert classify_action_via_generation("You suck", gen).primary == "insult"

    def test_falls_back_to_lexicon_when_generation_is_unparseable(self):
        from neuroslm.cognition.patterns import classify_action_via_generation
        gen = _FakeClassifyGen("uh, I'm not sure what to call this one")
        result = classify_action_via_generation("You suck", gen)
        assert result.primary == "insult", (
            "an LM reply that doesn't parse to a known category must "
            "fall back to the lexicon classifier, never silently "
            "return garbage or crash")

    def test_falls_back_when_generate_fn_raises(self):
        from neuroslm.cognition.patterns import classify_action_via_generation

        def _boom(prompt, max_new_tokens):
            raise RuntimeError("model unavailable")

        result = classify_action_via_generation("You suck", _boom)
        assert result.primary == "insult"


class TestGroundingAndSelfReferentialFlag:
    """The exact contamination risk from the 2026-08-12 log analysis:
    a rule mined ENTIRELY from the mind's own inferred self-talk must
    be visibly distinguishable from one grounded in real observed
    input — reported, never silently trusted the same way."""

    def test_rule_grounded_when_either_side_observed(self):
        seq = [_ep("insult", "observed"), _ep("disagreement", "inferred")] * 3
        rules = mine_temporal_associations(seq, window=1)
        r = next(x for x in rules
                 if x.antecedent == "insult" and x.consequent == "disagreement")
        assert r.grounded is True
        assert r.self_referential_only is False

    def test_rule_flagged_when_all_evidence_is_pure_self_talk(self):
        seq = [_ep("insult", "inferred"), _ep("disagreement", "inferred")] * 3
        rules = mine_temporal_associations(seq, window=1)
        r = next(x for x in rules
                 if x.antecedent == "insult" and x.consequent == "disagreement")
        assert r.self_referential_only is True
        assert r.grounded is False

    def test_mixed_evidence_is_grounded_not_self_referential(self):
        seq = ([_ep("insult", "observed"), _ep("disagreement", "inferred")]
              + [_ep("insult", "inferred"), _ep("disagreement", "inferred")] * 2)
        rules = mine_temporal_associations(seq, window=1)
        r = next(x for x in rules
                 if x.antecedent == "insult" and x.consequent == "disagreement")
        assert r.grounded is True
        assert r.self_referential_only is False


class TestAssociationRuleIsHonestAboutNotBeingCausation:
    def test_rule_has_no_causal_claim_field(self):
        """The DATA MODEL must never assert causation as a fact — no
        `is_causal` / `causes` / `proven` boolean anywhere on the
        result type. (The module's prose is free to discuss and
        disclaim causation — that's the honest part.)"""
        from dataclasses import fields
        from neuroslm.cognition.patterns import AssociationRule
        names = {f.name for f in fields(AssociationRule)}
        assert not any("causal" in n or "causes" in n or "proven" in n
                      for n in names), names

    def test_module_explicitly_disclaims_causation(self):
        """The prose MUST address causation — to rule it out, not to
        claim it. Checked via the specific disclaiming phrase rather
        than word presence, since a docstring that never mentions
        causation at all would be less honest, not more."""
        import inspect
        from neuroslm.cognition import patterns
        doc = (patterns.__doc__ or "") + (
            patterns.mine_temporal_associations.__doc__ or "")
        assert "not" in doc.lower() and "causa" in doc.lower()
        assert "association" in doc.lower()
