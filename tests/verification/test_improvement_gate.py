# -*- coding: utf-8 -*-
"""TDD: ``ImprovementGate`` — admit mutations only if they produce a
**statistically significant improvement** in a target metric.

This is the formal admission criterion the user's prompt names
"Lean validation that gates persisting discovered mechanics … only if
it can be formally proven and validated that the discovered mechanic
mathematically improves e.g. language modelling, ppl, intelligence
density, OOD gap ratio".

Semantically, what "formally proven to improve" means is:

  1. ``metric_after - metric_before`` has the **right sign**
     (decrease for ppl / OOD-gap, increase for Φ / intelligence-density).
  2. The improvement is **statistically significant** at the configured
     α (default 0.05) under a one-sided Welch's t-test on the per-batch
     samples — i.e. it is not noise.
  3. The improvement is **practically significant** at the configured
     ``min_effect`` (default: 1% relative for ratios, configurable for
     absolute metrics) — i.e. it is not microscopic.

Lean is one possible backend for step 1+2+3 (formally check the
inequality in the IEEE-754 model); the gate here ships the statistical
backend. Phase 4 swaps in a ``LeanProofBackend`` behind the same
``ImprovementGate.admit()`` interface.

Contract pinned by this suite:

  * ``ImprovementGate.admit(before, after, *, direction)`` returns a
    ``ImprovementVerdict`` with ``admitted``, ``effect``, ``p_value``,
    ``reasons``.
  * ``direction = "decrease"`` admits iff ``after`` is significantly
    smaller than ``before``; ``direction = "increase"`` is the mirror.
  * Equal samples (no change) ⇒ rejected.
  * Wrong-sign change ⇒ rejected with a wrong-direction reason.
  * Right-sign but not significant (p > α) ⇒ rejected with a noise reason.
  * Below the ``min_effect`` threshold ⇒ rejected with an effect-size reason.
  * Above all thresholds AND right sign AND significant ⇒ admitted.

This is a *composable* gate: ``CompositeGate([TripleGuard(...),
ImprovementGate(...)])`` admits iff **all** sub-gates admit.
"""
from __future__ import annotations

import pytest

from neuroslm.verification.improvement_gate import (
    ImprovementGate,
    ImprovementVerdict,
    CompositeGate,
)


# ──────────────────────────────────────────────────────────────────
# Contract 1 — basic admission semantics on synthetic samples
# ──────────────────────────────────────────────────────────────────

class TestImprovementGateDirection:
    """Direction sensitivity: "decrease" wants metric down, "increase"
    wants metric up. Wrong direction always rejected."""

    def test_decrease_admits_clear_improvement(self):
        gate = ImprovementGate(alpha=0.05, min_effect=0.01)
        # ppl-like metric: 32 batches, before mean ~100, after mean ~80
        # — clear, big, consistent improvement
        before = [100.0 + (i % 5) * 0.5 for i in range(32)]   # ~100 ± 1
        after = [80.0 + (i % 5) * 0.5 for i in range(32)]     # ~80 ± 1
        verdict = gate.admit(before, after, direction="decrease")
        assert verdict.admitted, (
            f"clear 20% ppl decrease must be admitted; got reasons="
            f"{verdict.reasons}, p={verdict.p_value:.4g}, "
            f"effect={verdict.effect:.4g}"
        )
        assert verdict.effect < 0, "decrease ⇒ negative effect"
        assert verdict.p_value < 0.05

    def test_increase_admits_clear_improvement(self):
        gate = ImprovementGate(alpha=0.05, min_effect=0.01)
        # Φ-like metric: before mean ~0.3, after mean ~0.5 — clear gain
        before = [0.30 + (i % 5) * 0.005 for i in range(32)]
        after = [0.50 + (i % 5) * 0.005 for i in range(32)]
        verdict = gate.admit(before, after, direction="increase")
        assert verdict.admitted
        assert verdict.effect > 0

    def test_wrong_direction_rejected_even_if_significant(self):
        """A clear *increase* in a metric the user wanted to *decrease*
        must be rejected with the wrong-direction reason — not silently
        flipped."""
        gate = ImprovementGate(alpha=0.05, min_effect=0.01)
        before = [100.0 + (i % 5) * 0.5 for i in range(32)]
        after = [120.0 + (i % 5) * 0.5 for i in range(32)]   # GOT WORSE
        verdict = gate.admit(before, after, direction="decrease")
        assert not verdict.admitted
        assert any("wrong" in r.lower() or "direction" in r.lower()
                   for r in verdict.reasons), (
            f"expected a wrong-direction reason in {verdict.reasons}"
        )


# ──────────────────────────────────────────────────────────────────
# Contract 2 — statistical significance gate
# ──────────────────────────────────────────────────────────────────

class TestImprovementGateSignificance:
    """No-improvement and noisy-improvement cases must be rejected."""

    def test_zero_change_rejected(self):
        gate = ImprovementGate(alpha=0.05, min_effect=0.01)
        samples = [50.0 + (i % 7) * 0.1 for i in range(32)]
        verdict = gate.admit(samples, samples, direction="decrease")
        assert not verdict.admitted
        # Either "no change" or "below min_effect" — both are legitimate
        # rejection reasons; we don't pin which one to keep the impl free.
        assert verdict.reasons, "zero change must produce a rejection reason"

    def test_tiny_change_below_min_effect_rejected(self):
        """0.1% improvement is real but below the 1% practical threshold."""
        gate = ImprovementGate(alpha=0.05, min_effect=0.01)
        before = [100.0 + (i % 5) * 0.05 for i in range(64)]
        after = [99.9 + (i % 5) * 0.05 for i in range(64)]   # 0.1% better
        verdict = gate.admit(before, after, direction="decrease")
        assert not verdict.admitted
        assert any("effect" in r.lower() for r in verdict.reasons), (
            f"expected effect-size reason in {verdict.reasons}"
        )

    def test_noisy_improvement_above_min_effect_but_not_significant_rejected(self):
        """5 batches, 3% mean drop swamped by σ ≈ 10% — not significant."""
        import random
        rng = random.Random(0)
        before = [100.0 + rng.gauss(0, 10) for _ in range(5)]
        after = [97.0 + rng.gauss(0, 10) for _ in range(5)]
        gate = ImprovementGate(alpha=0.05, min_effect=0.01)
        verdict = gate.admit(before, after, direction="decrease")
        assert not verdict.admitted, (
            f"5-sample noisy run should be rejected as not significant; "
            f"got p={verdict.p_value:.4g}"
        )

    def test_significance_threshold_respected(self):
        """A change borderline at α=0.05 must be admitted at α=0.10
        and rejected at α=0.01 — pins the α parameter is actually used."""
        # Use a moderate effect that lands somewhere in the 0.01–0.10
        # p-value range; deterministic-ish via fixed seed.
        import random
        rng = random.Random(7)
        before = [100.0 + rng.gauss(0, 3) for _ in range(20)]
        after = [97.5 + rng.gauss(0, 3) for _ in range(20)]

        loose = ImprovementGate(alpha=0.20, min_effect=0.005)
        strict = ImprovementGate(alpha=0.001, min_effect=0.005)

        v_loose = loose.admit(before, after, direction="decrease")
        v_strict = strict.admit(before, after, direction="decrease")

        # The loose gate should admit, strict should not — that
        # difference IS the α parameter doing its job.
        assert v_loose.admitted, (
            f"α=0.20 must admit borderline case; p={v_loose.p_value:.4g}"
        )
        assert not v_strict.admitted, (
            f"α=0.001 must reject borderline case; p={v_strict.p_value:.4g}"
        )


# ──────────────────────────────────────────────────────────────────
# Contract 3 — Verdict shape (auditable record)
# ──────────────────────────────────────────────────────────────────

class TestImprovementVerdictShape:
    """The verdict must carry enough info to audit every decision."""

    def test_verdict_has_all_required_fields(self):
        gate = ImprovementGate(alpha=0.05, min_effect=0.01)
        before = [100.0] * 16
        after = [80.0] * 16
        v = gate.admit(before, after, direction="decrease")
        # admitted, effect, p_value, reasons, metric_before, metric_after
        assert isinstance(v.admitted, bool)
        assert isinstance(v.effect, float)
        assert isinstance(v.p_value, float)
        assert isinstance(v.reasons, list)
        assert isinstance(v.metric_before, float)
        assert isinstance(v.metric_after, float)

    def test_verdict_serializes_to_dict(self):
        """Audit record must round-trip through JSON."""
        import json
        gate = ImprovementGate(alpha=0.05, min_effect=0.01)
        before = [10.0] * 8
        after = [8.0] * 8
        v = gate.admit(before, after, direction="decrease")
        d = v.to_dict()
        # Round-trip through JSON to prove serialisability
        round_tripped = json.loads(json.dumps(d))
        assert round_tripped["admitted"] == v.admitted
        assert round_tripped["effect"] == pytest.approx(v.effect)
        assert round_tripped["p_value"] == pytest.approx(v.p_value)


# ──────────────────────────────────────────────────────────────────
# Contract 4 — input validation (defensive contract)
# ──────────────────────────────────────────────────────────────────

class TestImprovementGateInputValidation:
    """Bad inputs must raise predictable errors, not silently admit."""

    def test_empty_samples_raises(self):
        gate = ImprovementGate()
        with pytest.raises(ValueError, match="empty"):
            gate.admit([], [1.0, 2.0], direction="decrease")

    def test_single_sample_raises(self):
        """Welch's t-test undefined for n=1; must reject *loudly*."""
        gate = ImprovementGate()
        with pytest.raises(ValueError, match=r"at least 2"):
            gate.admit([1.0], [0.5], direction="decrease")

    def test_unknown_direction_raises(self):
        gate = ImprovementGate()
        with pytest.raises(ValueError, match="direction"):
            gate.admit([1.0, 2.0], [0.5, 1.5], direction="sideways")

    def test_non_finite_sample_raises(self):
        gate = ImprovementGate()
        with pytest.raises(ValueError, match="finite|NaN|Inf"):
            gate.admit(
                [1.0, float("nan"), 2.0], [0.5, 0.6, 0.7],
                direction="decrease",
            )


# ──────────────────────────────────────────────────────────────────
# Contract 5 — CompositeGate chains
# ──────────────────────────────────────────────────────────────────

class TestCompositeGate:
    """A composite admits iff EVERY sub-gate admits. Order-independent."""

    def test_composite_with_only_admitting_gates_admits(self):
        gate = ImprovementGate(alpha=0.05, min_effect=0.01)
        composite = CompositeGate([gate])
        before = [100.0] * 16
        after = [80.0] * 16
        v = composite.admit(
            ("improvement", before, after, "decrease"),
        )
        assert v.admitted

    def test_composite_with_any_rejecting_gate_rejects(self):
        gate_ok = ImprovementGate(alpha=0.05, min_effect=0.01)
        gate_strict = ImprovementGate(alpha=0.001, min_effect=0.5)  # impossible
        composite = CompositeGate([gate_ok, gate_strict])
        before = [100.0] * 16
        after = [80.0] * 16
        # Both gates run on same evidence; second one has unreachable
        # thresholds, so composite must reject.
        v = composite.admit(
            ("improvement", before, after, "decrease"),
            ("improvement", before, after, "decrease"),
        )
        assert not v.admitted
        assert any("strict" in r.lower() or "effect" in r.lower()
                   or "significance" in r.lower()
                   for r in v.reasons)

    def test_composite_collects_all_failure_reasons(self):
        """When two gates reject, the composite verdict carries reasons
        from BOTH (auditable)."""
        gate_a = ImprovementGate(alpha=0.001, min_effect=0.5)
        gate_b = ImprovementGate(alpha=0.001, min_effect=0.5)
        composite = CompositeGate([gate_a, gate_b])
        before = [100.0] * 8
        after = [99.9] * 8   # microscopic change
        v = composite.admit(
            ("improvement", before, after, "decrease"),
            ("improvement", before, after, "decrease"),
        )
        assert not v.admitted
        assert len(v.reasons) >= 2, (
            f"expected ≥2 reasons (one per failing gate); got {v.reasons}"
        )
