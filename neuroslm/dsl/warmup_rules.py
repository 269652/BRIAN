# -*- coding: utf-8 -*-
"""Rule-based teacher-cutoff engine for the ``warmup`` DSL block.

The trainer feeds this engine a list of per-step metric dicts (one
row per logged step). The engine evaluates the warmup's rule list
against the history and returns ``True`` once the cutoff condition
is met. The decoupling lets us:

* keep the engine torch-free (only stdlib)
* unit-test the rule semantics without spinning up a model
* hot-swap the cutoff predicate without touching the trainer's
  hot path (the trainer just stores the latest history slice).

Pinned by ``tests/dsl/test_warmup_block.py::TestRuleEngine``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence


__all__ = [
    "WarmupRule",
    "evaluate_rule",
    "should_cutoff",
    "ALLOWED_OPS",
    "ALLOWED_METRICS",
    "ALLOWED_ACTIONS",
    "ALLOWED_COMBINATORS",
]


# Operator + enum tables — also imported by the compiler for parse-
# time validation. Keeping them here means there is exactly one
# source of truth for "what does the DSL accept".
ALLOWED_OPS = {">=", ">", "<=", "<", "==", "!="}
ALLOWED_METRICS = {
    "step", "ppl", "lm_loss", "ood_ratio", "phi", "mat_phase",
    "distill_kl", "distill_lambda",
}
ALLOWED_ACTIONS = {"detach", "anneal_alpha", "gate_to_zero"}
ALLOWED_COMBINATORS = {"any", "all"}


@dataclass
class WarmupRule:
    """One condition row of a ``warmup`` block.

    Attributes:
      metric: scalar key the trainer publishes each step (``step``,
        ``ppl``, ``ood_ratio``, ``phi``, ``mat_phase``, …).
      op: comparison op ∈ :data:`ALLOWED_OPS`.
      value: RHS of the comparison.
      window: number of consecutive most-recent observations that
        must satisfy the comparison. Default 1 — fire on a single
        observation. ``window=100`` makes the rule robust to a
        single-step PPL spike.
    """
    metric: str
    op: str
    value: float
    window: int = 1


def _cmp(lhs: float, op: str, rhs: float) -> bool:
    """Pure scalar comparison. Encapsulated so the operator table is
    not scattered across the module."""
    if op == ">=":
        return lhs >= rhs
    if op == ">":
        return lhs > rhs
    if op == "<=":
        return lhs <= rhs
    if op == "<":
        return lhs < rhs
    if op == "==":
        return lhs == rhs
    if op == "!=":
        return lhs != rhs
    raise ValueError(f"unknown op {op!r}; must be one of {ALLOWED_OPS}")


def evaluate_rule(
    rule: WarmupRule,
    history: Sequence[Mapping[str, float]],
) -> bool:
    """Return True iff ``rule.metric`` satisfied ``rule.op rule.value``
    over the ``rule.window`` most-recent observations.

    Semantics:
      * Empty ``history`` → ``False`` (the rule can't have fired yet).
      * Observation missing the metric key → treated as "not
        satisfied" for that step (conservative; never cut a teacher
        because a partial log row dropped the metric).
      * ``window=N`` requires the LAST N observations all to
        satisfy. A single failing observation in the middle of a
        long satisfied run is fine, but the most-recent ones must
        be a contiguous satisfied streak of length ≥ N.
    """
    if not history:
        return False
    window = max(1, int(rule.window))
    if len(history) < window:
        return False
    tail = history[-window:]
    for obs in tail:
        if rule.metric not in obs:
            return False
        try:
            v = float(obs[rule.metric])
        except (TypeError, ValueError):
            return False
        if not _cmp(v, rule.op, float(rule.value)):
            return False
    return True


def should_cutoff(
    rules: Sequence[WarmupRule],
    history: Sequence[Mapping[str, float]],
    *,
    combinator: str = "any",
) -> bool:
    """Aggregate the rule list under ``combinator``.

    Args:
      rules: list of :class:`WarmupRule` from a parsed ``warmup`` block.
      history: latest-most-recent-last list of metric dicts.
      combinator: ``"any"`` (default) or ``"all"``.

    Returns:
      ``True`` once the cutoff condition is met — the trainer should
      detach / anneal / zero the teacher per the warmup's ``action``.
    """
    if combinator not in ALLOWED_COMBINATORS:
        raise ValueError(
            f"unknown combinator {combinator!r}; "
            f"must be one of {ALLOWED_COMBINATORS}"
        )
    fired: List[bool] = [evaluate_rule(r, history) for r in rules]
    if not fired:
        return False
    if combinator == "any":
        return any(fired)
    return all(fired)
