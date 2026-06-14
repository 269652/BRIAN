# -*- coding: utf-8 -*-
"""TDD contract — Item 3: NT → distillation λ coupling.

Goal
====
Serotonin (5HT) is the brain's *conservative / persistence* channel;
dopamine (DA) is the *reward / explore* channel. Couple them to the
KL-distillation strength so the trunk leans on the cortex teacher when
stressed (high 5HT, low DA) and explores on its own when rewarded
(low 5HT, high DA).

Formula (locked here, math-first)
=================================
With ``z_5HT = 2 * (5HT - 0.5)`` and ``z_DA = 2 * (DA - 0.5)`` in
``[-1, +1]``:

    nt_mult = clamp(1 + k_5HT * z_5HT - k_DA * z_DA, 0.0, 2.0)
    λ_final = λ_base * nt_mult

* ``λ_base`` is the existing gap-driven piecewise-linear ramp.
* ``k_5HT`` and ``k_DA`` are new ``distillation_5ht_gain`` and
  ``distillation_da_gain`` knobs in ``multi_cortex``.
* Both default to 0.0 → identity (back-compat).
* Lower clamp at 0 (no negative distillation), upper at 2 (cap at 2×
  base — keeps the term from running away if both NTs saturate the
  "trust the teacher" direction).

Why this shape:
* 5HT high (≈ 0.8) → ``z_5HT = 0.6`` → with k_5HT=0.5 → +0.3 to mult.
  Trunk under stress leans harder on the cortex.
* DA high (≈ 0.6) → ``z_DA = 0.2`` → with k_DA=0.5 → -0.1 to mult.
  Trunk under reward signal explores on its own.
* Both at baseline (DA≈0.15, 5HT≈0.50 from DrivenNTSystem defaults)
  → mult ≈ 1.0 — no behavioural change vs the gap-only baseline.

Contract pinned in this file
============================
A. ``BRIANHarness`` accepts and stores the new gains.
B. ``_distillation_lambda`` returns the gap-ramp value when both gains
   are 0 (back-compat — every existing test must still pass).
C. With ``k_5HT > 0`` and 5HT pinned high, λ goes UP relative to baseline.
D. With ``k_DA > 0`` and DA pinned high, λ goes DOWN relative to baseline.
E. The final λ is clamped to ``[0, 2 * lambda_max]`` — never negative.
F. The DSL ``multi_cortex { distillation_5ht_gain: 0.5,
   distillation_da_gain: 0.5 }`` parses and round-trips.
"""
from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────
# Pure unit tests on the lambda formula (no full harness construction).
# ─────────────────────────────────────────────────────────────────────


def _gap_lambda(gap: float, floor: float = 0.1, ceiling: float = 2.0,
                lam_max: float = 1.0) -> float:
    """Reference reimplementation of the existing gap ramp — kept here
    so the NT-mod tests don't depend on harness internals."""
    if gap <= floor:
        return 0.0
    if gap >= ceiling:
        return lam_max
    return lam_max * (gap - floor) / (ceiling - floor)


def _nt_multiplier(da: float, ht5: float,
                   k_da: float, k_5ht: float) -> float:
    """Reference reimplementation of the NT multiplier."""
    z_da = 2.0 * (da - 0.5)
    z_5ht = 2.0 * (ht5 - 0.5)
    raw = 1.0 + k_5ht * z_5ht - k_da * z_da
    return max(0.0, min(2.0, raw))


class TestNTMultiplierFormula:
    """The math contract (independent of where it's called)."""

    def test_centre_point_is_one(self):
        m = _nt_multiplier(da=0.5, ht5=0.5, k_da=0.5, k_5ht=0.5)
        assert m == pytest.approx(1.0)

    def test_zero_gains_is_one(self):
        m = _nt_multiplier(da=0.9, ht5=0.1, k_da=0.0, k_5ht=0.0)
        assert m == pytest.approx(1.0)

    def test_high_5ht_increases_mult(self):
        m = _nt_multiplier(da=0.5, ht5=0.9, k_da=0.0, k_5ht=0.5)
        assert m > 1.0

    def test_high_da_decreases_mult(self):
        m = _nt_multiplier(da=0.9, ht5=0.5, k_da=0.5, k_5ht=0.0)
        assert m < 1.0

    def test_clamped_to_nonneg(self):
        m = _nt_multiplier(da=1.0, ht5=0.0, k_da=10.0, k_5ht=10.0)
        # raw = 1 + 10*(-1) - 10*(+1) = -19 → clamped to 0
        assert m == 0.0

    def test_clamped_to_two(self):
        m = _nt_multiplier(da=0.0, ht5=1.0, k_da=10.0, k_5ht=10.0)
        # raw = 1 + 10*(+1) - 10*(-1) = 21 → clamped to 2
        assert m == 2.0


# ─────────────────────────────────────────────────────────────────────
# Harness-level integration: _distillation_lambda actually uses NT.
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def stub_harness():
    """Minimal harness substitute exposing only what _distillation_lambda
    needs: a `training_config.multi_cortex` config and the method itself.
    We build a real BRIANHarness if possible, otherwise instantiate the
    method on a SimpleNamespace stub — both must give identical results
    by contract.
    """
    from types import SimpleNamespace
    from neuroslm.harness import BRIANHarness

    # Construct a config that mirrors the new DSL knobs.
    mc = SimpleNamespace(
        enabled=True,
        distillation_enabled=True,
        distillation_lambda_max=1.0,
        distillation_temperature=4.0,
        distillation_gap_floor=0.1,
        distillation_gap_ceiling=2.0,
        distillation_5ht_gain=0.0,
        distillation_da_gain=0.0,
        # Other fields _distillation_lambda doesn't touch.
        inhibition_enabled=False,
        inhibition_ema_alpha=0.05,
        inhibition_temperature=1.0,
    )
    cfg = SimpleNamespace(multi_cortex=mc)

    # Build a method-only stub that calls the SAME bound implementation.
    stub = SimpleNamespace(
        training_config=cfg,
        _nt_levels_for_distill={},   # set per-test via .levels
    )
    # Bind the actual method to the stub.
    stub._distillation_lambda = BRIANHarness._distillation_lambda.__get__(stub)
    return stub


class TestDistillationLambdaNTCoupling:

    def test_zero_gains_match_gap_ramp(self, stub_harness):
        """B: with k=0 the result must equal the pure gap ramp value."""
        for gap in [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0]:
            stub_harness.training_config.multi_cortex.distillation_5ht_gain = 0.0
            stub_harness.training_config.multi_cortex.distillation_da_gain = 0.0
            expected = _gap_lambda(gap)
            got = stub_harness._distillation_lambda(gap)
            assert got == pytest.approx(expected, abs=1e-9), (
                f"zero-gain λ mismatch at gap={gap}: expected {expected}, got {got}"
            )

    def test_centre_nt_with_positive_gains_is_identity(self, stub_harness):
        """At NE=DA=5HT=0.5 the multiplier is 1 → λ identical to baseline."""
        stub_harness.training_config.multi_cortex.distillation_5ht_gain = 0.5
        stub_harness.training_config.multi_cortex.distillation_da_gain = 0.5
        stub_harness._nt_levels_for_distill = {"DA": 0.5, "5HT": 0.5}
        for gap in [0.3, 0.7, 1.5]:
            expected = _gap_lambda(gap)
            got = stub_harness._distillation_lambda(gap)
            assert got == pytest.approx(expected, abs=1e-6), (
                f"centred NT at gap={gap}: expected {expected}, got {got}"
            )

    def test_high_5ht_increases_lambda(self, stub_harness):
        """C: 5HT high → λ goes up."""
        gap = 1.0  # mid-ramp; base = 0.5
        stub_harness.training_config.multi_cortex.distillation_5ht_gain = 0.5
        stub_harness.training_config.multi_cortex.distillation_da_gain = 0.0
        stub_harness._nt_levels_for_distill = {"DA": 0.5, "5HT": 0.5}
        base = stub_harness._distillation_lambda(gap)
        stub_harness._nt_levels_for_distill = {"DA": 0.5, "5HT": 0.9}
        high = stub_harness._distillation_lambda(gap)
        assert high > base + 1e-6, (
            f"high 5HT must increase λ: base={base}, high={high}"
        )

    def test_high_da_decreases_lambda(self, stub_harness):
        """D: DA high → λ goes down."""
        gap = 1.0
        stub_harness.training_config.multi_cortex.distillation_5ht_gain = 0.0
        stub_harness.training_config.multi_cortex.distillation_da_gain = 0.5
        stub_harness._nt_levels_for_distill = {"DA": 0.5, "5HT": 0.5}
        base = stub_harness._distillation_lambda(gap)
        stub_harness._nt_levels_for_distill = {"DA": 0.9, "5HT": 0.5}
        low = stub_harness._distillation_lambda(gap)
        assert low < base - 1e-6, (
            f"high DA must decrease λ: base={base}, low={low}"
        )

    def test_clamped_nonneg(self, stub_harness):
        """E: λ never negative even with adversarial NT pinning."""
        gap = 1.0
        stub_harness.training_config.multi_cortex.distillation_5ht_gain = 10.0
        stub_harness.training_config.multi_cortex.distillation_da_gain = 10.0
        stub_harness._nt_levels_for_distill = {"DA": 1.0, "5HT": 0.0}
        got = stub_harness._distillation_lambda(gap)
        assert got >= 0.0, f"λ went negative: {got}"

    def test_clamped_upper(self, stub_harness):
        """E: λ ≤ 2 * lambda_max even when both NTs slam favourable."""
        gap = 1.0  # base = 0.5
        stub_harness.training_config.multi_cortex.distillation_5ht_gain = 10.0
        stub_harness.training_config.multi_cortex.distillation_da_gain = 10.0
        stub_harness._nt_levels_for_distill = {"DA": 0.0, "5HT": 1.0}
        got = stub_harness._distillation_lambda(gap)
        lam_max = stub_harness.training_config.multi_cortex.distillation_lambda_max
        # base * 2 = 0.5 * 2 = 1.0 -> clamped at the multiplier-2 cap
        assert got <= 2.0 * lam_max + 1e-6, f"λ exceeded 2*lam_max: {got}"


# ─────────────────────────────────────────────────────────────────────
# DSL parse round-trip.
# ─────────────────────────────────────────────────────────────────────


class TestDSLParse:

    def test_parser_reads_5ht_and_da_gains(self):
        from neuroslm.dsl.training_config import _parse_multi_cortex
        mc = _parse_multi_cortex("""{
            enabled: true,
            distillation_enabled: true,
            distillation_5ht_gain: 0.4,
            distillation_da_gain:  0.6,
            experts: [
                { id: "gpt2", domain: "general" }
            ],
            trunk_tokenizer: "gpt2"
        }""")
        assert hasattr(mc, "distillation_5ht_gain")
        assert hasattr(mc, "distillation_da_gain")
        assert mc.distillation_5ht_gain == pytest.approx(0.4)
        assert mc.distillation_da_gain == pytest.approx(0.6)

    def test_parser_defaults_are_zero(self):
        from neuroslm.dsl.training_config import _parse_multi_cortex
        mc = _parse_multi_cortex("""{
            enabled: true,
            experts: [
                { id: "gpt2", domain: "general" }
            ],
            trunk_tokenizer: "gpt2"
        }""")
        assert mc.distillation_5ht_gain == 0.0
        assert mc.distillation_da_gain == 0.0
