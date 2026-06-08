# -*- coding: utf-8 -*-
"""ImprovementGate — admit mutations only when they statistically
improve a target metric.

This is the formal admission criterion the architecture's evolutionary
loop applies *after* the :class:`TripleGuard` invariant gate has
cleared a candidate. Where ``TripleGuard`` asks *"would this mutation
destroy a structural invariant?"*, ``ImprovementGate`` asks
*"does this mutation measurably make the model better at what we
care about?"* — quantified by:

  1. **Direction**: ``effect = mean(after) - mean(before)`` has the
     sign the caller asked for (``"decrease"`` for ppl / OOD-gap /
     loss, ``"increase"`` for Φ / intelligence-density / accuracy).
  2. **Statistical significance**: Welch's one-sided *t*-test on the
     per-batch samples gives ``p < alpha`` (default ``0.05``). This
     is the formal *"not noise"* test.
  3. **Practical significance**: ``|relative effect|`` exceeds
     ``min_effect`` (default ``0.01`` = 1 %). Microscopic effects
     that are statistically significant only because of large *n*
     are still rejected — they're not worth persisting into the DNA.

Why a statistical gate and not (yet) a Lean proof
-------------------------------------------------
A Lean proof can establish, for an *algebraic* mechanism, that a
particular quantity is monotone or bounded. But what the user's
ask requires here is empirical: *"this mutation made ppl drop on
held-out data"*. That is an *experimental* claim about an opaque
PyTorch model and a finite sample; the only thing that can prove
it formally is a statistical hypothesis test on the measurements.

The Lean route enters in :phase:`4` as a second backend behind the
same :class:`ImprovementGate` interface: if the mutation's source
form admits a formal proof of monotonicity (e.g. *"the new term is
non-negative ⇒ Φ cannot decrease"*), the Lean subprocess returns a
verified proof object, and the gate admits *without needing the
empirical pass*. Until then, the empirical Welch's gate **is** the
formal admission criterion — and that is what this module ships.

References
----------
* Welch (1947), "The generalization of Student's problem when several
  different population variances are involved", Biometrika 34(1/2).
* CLAUDE.md §1 — TDD: see ``tests/verification/test_improvement_gate.py``.
* docs/formal_framework.md §9 (added in this commit).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple, Union


# ──────────────────────────────────────────────────────────────────
# Verdict — auditable record of every admission decision
# ──────────────────────────────────────────────────────────────────


@dataclass
class ImprovementVerdict:
    """Structured outcome of a single :py:meth:`ImprovementGate.admit`
    call. Round-trips through JSON via :py:meth:`to_dict`."""

    admitted: bool
    effect: float                 # mean(after) - mean(before)
    p_value: float                # one-sided Welch's t-test p
    reasons: List[str] = field(default_factory=list)
    metric_before: float = 0.0
    metric_after: float = 0.0
    direction: str = ""
    alpha: float = 0.05
    min_effect: float = 0.01
    n_before: int = 0
    n_after: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "admitted": bool(self.admitted),
            "effect": float(self.effect),
            "p_value": float(self.p_value),
            "reasons": list(self.reasons),
            "metric_before": float(self.metric_before),
            "metric_after": float(self.metric_after),
            "direction": str(self.direction),
            "alpha": float(self.alpha),
            "min_effect": float(self.min_effect),
            "n_before": int(self.n_before),
            "n_after": int(self.n_after),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


# ──────────────────────────────────────────────────────────────────
# Welch's t-test — pure-Python implementation
# ──────────────────────────────────────────────────────────────────
#
# We don't pull in scipy just for this; the formula is short and
# closed-form. Implementing it inline (a) avoids a heavy dependency
# in the verification subpackage, (b) makes the formal contract
# auditable from this single file, and (c) keeps the gate usable
# from cold-start cli contexts where scipy isn't imported.


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _var_unbiased(xs: Sequence[float], mean: float) -> float:
    n = len(xs)
    if n < 2:
        raise ValueError("variance requires at least 2 samples")
    return sum((x - mean) ** 2 for x in xs) / (n - 1)


def _welch_t_one_sided(
    before: Sequence[float], after: Sequence[float], *,
    less: bool,
) -> Tuple[float, float]:
    """Welch's *t* statistic and the one-sided *p*-value.

    ``less=True`` tests H₁: mean(after) < mean(before)
    ``less=False`` tests H₁: mean(after) > mean(before)
    """
    n_b, n_a = len(before), len(after)
    m_b, m_a = _mean(before), _mean(after)
    v_b = _var_unbiased(before, m_b)
    v_a = _var_unbiased(after, m_a)

    # Welch's t statistic
    se2 = v_b / n_b + v_a / n_a
    if se2 <= 0.0:
        # Both samples are constant. Degenerate: t = ±inf if means
        # differ, NaN if they don't. We map the directional case to
        # p = 0 (perfectly significant) when the effect matches the
        # alternative, and p = 1 otherwise — degeneracy reported via
        # the reason list, not the p-value.
        if m_a == m_b:
            return 0.0, 1.0
        right_dir = (m_a < m_b) if less else (m_a > m_b)
        return (float("-inf") if less and right_dir
                else float("inf") if not less and right_dir
                else 0.0), (0.0 if right_dir else 1.0)
    se = math.sqrt(se2)
    t = (m_a - m_b) / se

    # Welch–Satterthwaite degrees of freedom
    num = (v_b / n_b + v_a / n_a) ** 2
    den = ((v_b / n_b) ** 2) / (n_b - 1) + ((v_a / n_a) ** 2) / (n_a - 1)
    df = num / den if den > 0 else float(n_b + n_a - 2)

    # One-sided p-value from the t distribution CDF.
    # Use the regularized incomplete beta function via mpmath-free
    # closed form: for a t statistic with df degrees of freedom,
    #   P(T <= t) = 1 - 0.5 * I_x(df/2, 0.5)
    # where x = df / (df + t^2) when t > 0, and symmetric otherwise.
    p_t_le = _t_cdf(t, df)

    if less:
        # H1: mean_after < mean_before  ⇒  t < 0  ⇒  small p
        p = p_t_le
    else:
        # H1: mean_after > mean_before  ⇒  t > 0  ⇒  small p
        p = 1.0 - p_t_le
    # Clamp for numerical safety
    return t, max(0.0, min(1.0, p))


def _t_cdf(t: float, df: float) -> float:
    """CDF of the Student-t distribution with ``df`` degrees of freedom.

    Closed form via the regularised incomplete beta function:
        P(T <= t) = 1 - 0.5 * I_{df/(df+t²)}(df/2, 1/2)   for t >= 0
        P(T <= t) =     0.5 * I_{df/(df+t²)}(df/2, 1/2)   for t <  0
    """
    if not math.isfinite(t):
        return 1.0 if t > 0 else 0.0
    x = df / (df + t * t)
    ix = _reg_incomplete_beta(x, df / 2.0, 0.5)
    return 1.0 - 0.5 * ix if t >= 0 else 0.5 * ix


def _reg_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta function I_x(a, b) by Lentz's
    continued-fraction algorithm (Numerical Recipes §6.4)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Symmetry: I_x(a,b) = 1 - I_{1-x}(b,a) for faster convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _reg_incomplete_beta(1.0 - x, b, a)
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    # Lentz's algorithm
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d_ = 1.0 - qab * x / qap
    if abs(d_) < fpmin:
        d_ = fpmin
    d_ = 1.0 / d_
    h = d_
    for m in range(1, 1001):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d_ = 1.0 + aa * d_
        if abs(d_) < fpmin:
            d_ = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d_ = 1.0 / d_
        h *= d_ * c
        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d_ = 1.0 + aa * d_
        if abs(d_) < fpmin:
            d_ = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d_ = 1.0 / d_
        delta = d_ * c
        h *= delta
        if abs(delta - 1.0) < 3e-7:
            break
    return front * h / a


# ──────────────────────────────────────────────────────────────────
# ImprovementGate
# ──────────────────────────────────────────────────────────────────


_VALID_DIRECTIONS = {"decrease", "increase"}


@dataclass
class ImprovementGate:
    """Statistical-significance + practical-significance gate.

    Parameters
    ----------
    alpha : float
        One-sided significance threshold for Welch's t-test. Default
        ``0.05``. Mutations whose ``p_value > alpha`` are rejected
        as "not statistically distinguishable from noise".
    min_effect : float
        Minimum *relative* effect size required.
        ``|mean(after) - mean(before)| / max(|mean(before)|, 1e-12)``
        must exceed this. Default ``0.01`` (1 %). Mutations below
        this threshold are rejected as "real but microscopic".
    """

    alpha: float = 0.05
    min_effect: float = 0.01

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(
                f"alpha must be in (0, 1), got {self.alpha!r}")
        if self.min_effect < 0.0:
            raise ValueError(
                f"min_effect must be >= 0, got {self.min_effect!r}")

    def admit(
        self,
        before: Sequence[float],
        after: Sequence[float],
        *,
        direction: str,
    ) -> ImprovementVerdict:
        """Decide admission for one before/after evidence pair.

        Parameters
        ----------
        before, after : sequence of float
            Per-batch (or per-eval) measurements of the same metric
            on the architecture *before* and *after* the candidate
            mutation. Must be non-empty, ≥ 2 elements each, all finite.
        direction : ``"decrease"`` | ``"increase"``
            Whether a *better* value is *smaller* (ppl, loss, OOD-gap)
            or *larger* (Φ, accuracy, intelligence density).
        """
        # ── input validation (defensive) ─────────────────────────
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {_VALID_DIRECTIONS}, "
                f"got {direction!r}"
            )
        for name, xs in (("before", before), ("after", after)):
            if len(xs) == 0:
                raise ValueError(f"{name} is empty — at least 2 samples required")
            if len(xs) < 2:
                raise ValueError(
                    f"{name} has {len(xs)} sample(s); at least 2 required "
                    f"for Welch's t-test"
                )
            for x in xs:
                if not math.isfinite(x):
                    raise ValueError(
                        f"{name} contains a non-finite sample (NaN/Inf): {x}"
                    )

        mean_b, mean_a = _mean(before), _mean(after)
        effect = mean_a - mean_b

        # ── direction check (cheap, do first) ────────────────────
        wants_less = direction == "decrease"
        right_sign = (effect < 0) if wants_less else (effect > 0)

        # ── statistical significance (Welch's one-sided t) ───────
        _t, p_value = _welch_t_one_sided(before, after, less=wants_less)

        # ── practical significance (relative effect size) ────────
        denom = max(abs(mean_b), 1e-12)
        rel_effect = abs(effect) / denom

        reasons: List[str] = []
        if not right_sign:
            reasons.append(
                f"wrong direction: requested {direction!r} but "
                f"mean moved from {mean_b:.6g} to {mean_a:.6g} "
                f"(effect={effect:+.6g})"
            )
        if p_value > self.alpha:
            reasons.append(
                f"not statistically significant: p={p_value:.4g} > "
                f"alpha={self.alpha:.4g}"
            )
        if rel_effect < self.min_effect:
            reasons.append(
                f"effect size below threshold: |{effect:+.6g}|/|{mean_b:.6g}| "
                f"= {rel_effect:.4g} < min_effect={self.min_effect:.4g}"
            )

        return ImprovementVerdict(
            admitted=(len(reasons) == 0),
            effect=float(effect),
            p_value=float(p_value),
            reasons=reasons,
            metric_before=float(mean_b),
            metric_after=float(mean_a),
            direction=direction,
            alpha=float(self.alpha),
            min_effect=float(self.min_effect),
            n_before=len(before),
            n_after=len(after),
        )


# ──────────────────────────────────────────────────────────────────
# CompositeGate — chain multiple gates with AND semantics
# ──────────────────────────────────────────────────────────────────


# Type alias for one evidence tuple. Currently only "improvement"
# evidence is recognised; future gates (e.g. TripleGuard) can be
# wired in via the ``kind`` discriminator without breaking callers.
EvidenceTuple = Tuple[str, Sequence[float], Sequence[float], str]


@dataclass
class CompositeGate:
    """Chain N admission gates with AND semantics: admit iff *every*
    constituent admits.

    Each ``admit(*evidence_tuples)`` call passes the i-th evidence
    tuple to the i-th gate. The composite verdict aggregates failure
    reasons across all rejecting gates so the auditor sees every
    obstacle the mutation hit, not just the first.

    Currently only :class:`ImprovementGate` is supported in the
    composite (evidence kind ``"improvement"``); future kinds
    (``"triple_guard"``, ``"lean_proof"``) plug in here with their
    own evidence tuple format.
    """

    gates: List[ImprovementGate] = field(default_factory=list)

    def admit(self, *evidences: EvidenceTuple) -> ImprovementVerdict:
        if len(evidences) != len(self.gates):
            raise ValueError(
                f"CompositeGate expected {len(self.gates)} evidence "
                f"tuples, got {len(evidences)}"
            )

        verdicts: List[ImprovementVerdict] = []
        for gate, evid in zip(self.gates, evidences):
            kind = evid[0]
            if kind != "improvement":
                raise ValueError(
                    f"unknown evidence kind {kind!r}; supported: "
                    f"{{'improvement'}}"
                )
            _kind, before, after, direction = evid
            verdicts.append(gate.admit(before, after, direction=direction))

        all_admitted = all(v.admitted for v in verdicts)
        reasons: List[str] = []
        for i, v in enumerate(verdicts):
            for r in v.reasons:
                reasons.append(f"[gate {i}] {r}")
        # Use the LAST verdict's numeric fields as canonical (they
        # describe the same evidence in the common single-evidence
        # case; for multi-evidence composites the caller is expected
        # to inspect per-gate diagnostics via the reasons list).
        last = verdicts[-1]
        return ImprovementVerdict(
            admitted=all_admitted,
            effect=last.effect,
            p_value=last.p_value,
            reasons=reasons,
            metric_before=last.metric_before,
            metric_after=last.metric_after,
            direction=last.direction,
            alpha=last.alpha,
            min_effect=last.min_effect,
            n_before=last.n_before,
            n_after=last.n_after,
        )
