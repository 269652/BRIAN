# -*- coding: utf-8 -*-
"""Contracts for the ``warmup`` DSL block + rule-evaluation engine.

The block declares **when** to cut a teacher signal (and **what** to
do when the cut fires)::

    warmup teacher_cutoff {
        target:     ensemble_funnel,     # which funnel to detach
        action:     "detach",            # "detach"|"anneal_alpha"|"gate_to_zero"
        combinator: "any",               # "any"|"all"
        rules: [
            { metric: "step",      op: ">=", value: 10000 },
            { metric: "ppl",       op: "<",  value: 50.0,  window: 100 },
            { metric: "ood_ratio", op: "<",  value: 1.5 },
            { metric: "phi",       op: ">",  value: 0.3 },
        ],
    }

Two contracts in one module
===========================

1. **Parser**: ``warmup`` block compiles to :class:`WarmupIR` with
   :class:`WarmupRule` rows. References to ``target`` (a declared
   ``funnel``) are validated at compile time.

2. **Rule engine**: :func:`should_cutoff` evaluates the rule list
   against a metric history and returns ``True`` once the combinator
   condition is met. ``window=N`` requires N consecutive observations
   where the condition held — that's what makes the PPL/OOD rules
   robust to single-step noise.

The engine is **standalone** — no torch, no autograd. The harness
feeds it a list of ``{metric: value}`` dicts (one per logged step)
and reads back a bool. This decouples the rule semantics from the
trainer's plumbing.
"""
from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────
# Contract A — parser
# ─────────────────────────────────────────────────────────────────────


class TestWarmupBlockParses:

    def test_step_only_warmup(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population lm_trunk { count: 64 }\n"
            "expert E { model: \"gpt2\", role: \"math\" }\n"
            "funnel f { inputs: [E], target: lm_trunk }\n"
            "warmup tc {\n"
            "    target: f,\n"
            "    action: \"detach\",\n"
            "    rules:  [ { metric: \"step\", op: \">=\", value: 10000 } ]\n"
            "}\n"
        )
        prog = NeuroMLCompiler.compile(src)
        assert hasattr(prog, "warmups")
        w = prog.warmups[0]
        assert w.name == "tc"
        assert w.target == "f"
        assert w.action == "detach"
        assert w.combinator == "any", "default combinator is `any`"
        assert len(w.rules) == 1
        r = w.rules[0]
        assert r.metric == "step"
        assert r.op == ">="
        assert r.value == 10000
        assert r.window == 1

    def test_multi_rule_warmup_with_windows(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population lm_trunk { count: 64 }\n"
            "expert E { model: \"gpt2\", role: \"math\" }\n"
            "funnel f { inputs: [E], target: lm_trunk }\n"
            "warmup tc {\n"
            "    target:     f,\n"
            "    action:     \"detach\",\n"
            "    combinator: \"any\",\n"
            "    rules: [\n"
            "        { metric: \"step\",      op: \">=\", value: 10000 },\n"
            "        { metric: \"ppl\",       op: \"<\",  value: 50.0, window: 100 },\n"
            "        { metric: \"ood_ratio\", op: \"<\",  value: 1.5 }\n"
            "    ]\n"
            "}\n"
        )
        prog = NeuroMLCompiler.compile(src)
        w = prog.warmups[0]
        metrics = [r.metric for r in w.rules]
        assert metrics == ["step", "ppl", "ood_ratio"]
        # Window-specific check on PPL row
        ppl_rule = next(r for r in w.rules if r.metric == "ppl")
        assert ppl_rule.window == 100


# ─────────────────────────────────────────────────────────────────────
# Contract B — reference resolution + enum validation
# ─────────────────────────────────────────────────────────────────────


class TestValidation:

    def test_unknown_target_funnel_raises(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 64 }\n"
            "warmup tc {\n"
            "    target: ghost,\n"
            "    action: \"detach\",\n"
            "    rules:  [ { metric: \"step\", op: \">=\", value: 100 } ]\n"
            "}\n"
        )
        with pytest.raises(NeuroMLError, match="ghost"):
            NeuroMLCompiler.compile(src)

    def test_unknown_action_raises(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population lm_trunk { count: 64 }\n"
            "expert E { model: \"gpt2\", role: \"math\" }\n"
            "funnel f { inputs: [E], target: lm_trunk }\n"
            "warmup tc {\n"
            "    target: f,\n"
            "    action: \"vaporize\",\n"
            "    rules:  [ { metric: \"step\", op: \">=\", value: 100 } ]\n"
            "}\n"
        )
        with pytest.raises(NeuroMLError, match=r"action|vaporize"):
            NeuroMLCompiler.compile(src)

    def test_empty_rules_list_raises(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population lm_trunk { count: 64 }\n"
            "expert E { model: \"gpt2\", role: \"math\" }\n"
            "funnel f { inputs: [E], target: lm_trunk }\n"
            "warmup tc {\n"
            "    target: f, action: \"detach\", rules: []\n"
            "}\n"
        )
        with pytest.raises(NeuroMLError, match=r"rules|empty"):
            NeuroMLCompiler.compile(src)


# ─────────────────────────────────────────────────────────────────────
# Contract C — rule engine (standalone, no compiler involvement)
# ─────────────────────────────────────────────────────────────────────


class TestRuleEngine:

    def test_step_rule_fires_at_exact_step(self):
        from neuroslm.dsl.warmup_rules import WarmupRule, evaluate_rule
        rule = WarmupRule(metric="step", op=">=", value=100)
        # Need just one observation at step >= 100.
        assert evaluate_rule(rule, [{"step": 100}]) is True
        assert evaluate_rule(rule, [{"step": 99}]) is False
        assert evaluate_rule(rule, []) is False

    def test_window_requires_consecutive_satisfied_observations(self):
        from neuroslm.dsl.warmup_rules import WarmupRule, evaluate_rule
        rule = WarmupRule(metric="ppl", op="<", value=50.0, window=3)
        # Only 2 consecutive below — doesn't fire.
        hist = [{"ppl": 100}, {"ppl": 49}, {"ppl": 48}]
        assert evaluate_rule(rule, hist) is False
        # Now 3 in a row — fires.
        hist = [{"ppl": 100}, {"ppl": 49}, {"ppl": 48}, {"ppl": 47}]
        assert evaluate_rule(rule, hist) is True
        # 3 in the middle but the latest broke the streak — doesn't fire
        # (the window must include the most recent observation).
        hist = [{"ppl": 49}, {"ppl": 48}, {"ppl": 47}, {"ppl": 100}]
        assert evaluate_rule(rule, hist) is False

    def test_missing_metric_in_history_does_not_fire(self):
        """An observation that doesn't carry the metric key is treated
        as 'not satisfied' (conservative; never accidentally cuts a
        teacher because a metric was missing from a partial log row)."""
        from neuroslm.dsl.warmup_rules import WarmupRule, evaluate_rule
        rule = WarmupRule(metric="ood_ratio", op="<", value=1.5)
        assert evaluate_rule(rule, [{"step": 100}]) is False  # no ood_ratio

    def test_any_combinator_fires_when_one_rule_fires(self):
        from neuroslm.dsl.warmup_rules import WarmupRule, should_cutoff
        rules = [
            WarmupRule(metric="step", op=">=", value=10000),
            WarmupRule(metric="ppl",  op="<",  value=50.0, window=100),
        ]
        hist = [{"step": 10000, "ppl": 100}]
        assert should_cutoff(rules, hist, combinator="any") is True
        # Neither rule fires
        hist = [{"step": 50, "ppl": 100}]
        assert should_cutoff(rules, hist, combinator="any") is False

    def test_all_combinator_requires_every_rule(self):
        from neuroslm.dsl.warmup_rules import WarmupRule, should_cutoff
        rules = [
            WarmupRule(metric="step", op=">=", value=100),
            WarmupRule(metric="ppl",  op="<",  value=50.0),
        ]
        # Both fire on the final observation
        hist = [{"step": 100, "ppl": 49}]
        assert should_cutoff(rules, hist, combinator="all") is True
        # Only step fires
        hist = [{"step": 100, "ppl": 100}]
        assert should_cutoff(rules, hist, combinator="all") is False

    def test_op_eq_and_neq_supported(self):
        from neuroslm.dsl.warmup_rules import WarmupRule, evaluate_rule
        eq_rule = WarmupRule(metric="phase", op="==", value=2)
        ne_rule = WarmupRule(metric="phase", op="!=", value=2)
        assert evaluate_rule(eq_rule, [{"phase": 2}]) is True
        assert evaluate_rule(eq_rule, [{"phase": 3}]) is False
        assert evaluate_rule(ne_rule, [{"phase": 3}]) is True
        assert evaluate_rule(ne_rule, [{"phase": 2}]) is False
