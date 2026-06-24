# -*- coding: utf-8 -*-
"""TDD: mid-training OOD ``train_ppl`` must be derived from the LM-only
loss, not the aux-inflated total.

Observed bug (vast.ai instance 40743889, 2026-06-12):

    step  500 | loss 14.87 | lm  9.75 | ppl 17172
    [mid-ood] step 500: wikitext ppl=24608.8 gap_ratio=0.01 (train_ppl=2878624.1)

    step 1000 | loss 15.14 | lm  7.99 | ppl  2962
    [mid-ood] step 1000: wikitext ppl=7091.3 gap_ratio=0.00 (train_ppl=3753701.3)

The printed ``ppl`` column (derived from ``lm_ema``) tracks 17172 → 2962,
but the OOD-snapshot ``train_ppl`` reports 2.8M → 3.7M — three orders
of magnitude higher because it's exponentiating the **total** loss
(LM CE + VBB + PC-reentry + MSPCC + distill aux terms), not the
LM-only cross-entropy. Result: ``gap_ratio = ood_ppl / train_ppl ≈ 0``
mechanically, which is meaningless and makes the OOD telemetry
unactionable.

Contract:
    The ``train_ppl_history`` dict consumed by ``_mid_ood_eval`` must
    be populated from ``avg_lm`` (the LM-only running mean), not
    ``avg`` (the total optimization loss).
"""
from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DSL = REPO_ROOT / "neuroslm" / "train_dsl.py"


# ─────────────────────────────────────────────────────────────────────
# Contract A — the source line uses avg_lm, not avg
# ─────────────────────────────────────────────────────────────────────


class TestTrainPplHistoryUsesLmOnlyLoss:
    """The dict feeding mid-OOD's gap_ratio computation must come from
    the LM-only loss. This is a source-level grep test because the
    bug is a one-line variable typo and we want the contract pinned
    where the bug lives, not in a 30-minute integration test."""

    def test_train_ppl_history_assignment_uses_avg_lm(self):
        """The assignment ``train_ppl_history[step] = exp(min(<X>, 20))``
        must use ``avg_lm`` for ``<X>``. ``avg`` (total loss) would
        produce the 1000× inflation observed in the wild."""
        src = TRAIN_DSL.read_text(encoding="utf-8")
        # Match the assignment regardless of formatting variations.
        # Captures whatever expression is fed to exp().
        m = re.search(
            r"train_ppl_history\[step\]\s*=\s*[_\w]*\.?exp\(\s*min\(\s*([a-zA-Z_]+)",
            src,
        )
        assert m is not None, (
            "could not find ``train_ppl_history[step] = exp(min(<var>, ...))`` "
            "in train_dsl.py — the source-level contract for this test "
            "depends on that pattern. If you refactored the structure, "
            "update this test to match the new shape.")
        var_used = m.group(1)
        # Allowed: avg_lm (no-cortex path) or _nats_for_hist (cortex path,
        # which resolves to lm_loss_ema when present, else avg_lm — both
        # LM-only). Forbidden: avg (total loss, inflates by ~1000×).
        assert var_used != "avg", (
            f"train_ppl_history must NOT use the total loss (``avg``), "
            f"but the source uses ``{var_used}``. That inflates train_ppl "
            f"by ~1000× and collapses gap_ratio to 0.")
        assert var_used in ("avg_lm", "_nats_for_hist"), (
            f"Expected ``avg_lm`` or ``_nats_for_hist`` (trunk-only CE), "
            f"got ``{var_used}``. Update this allowlist if the variable "
            f"was renamed."
        )


# ─────────────────────────────────────────────────────────────────────
# Contract B — behavioural smoke test of the gap_ratio math
# ─────────────────────────────────────────────────────────────────────


class TestGapRatioMathIsSane:
    """Given the numbers observed at step 1000 (lm CE = 7.99 → train_ppl
    ≈ 2962), gap_ratio for an OOD ppl of 7091 must land in a sane
    range (1.5 – 5), NOT collapse to ≈ 0 the way the total-loss
    version does."""

    def test_with_lm_only_loss_gap_ratio_is_meaningful(self):
        """Reproduce the math the fixed implementation should do."""
        lm_ce = 7.99
        train_ppl = math.exp(min(lm_ce, 20.0))   # 2962
        ood_ppl = 7091.3
        gap_ratio = ood_ppl / train_ppl

        # train_ppl is the LM-only ppl, so gap_ratio should be O(1).
        # Observed buggy behaviour was gap_ratio = 0.00 (because
        # ood_ppl / 3753701 ≈ 0).
        assert 1.5 < gap_ratio < 5.0, (
            f"gap_ratio computed from LM-only loss should be in (1.5, 5); "
            f"got {gap_ratio:.3f}. (For reference the buggy version with "
            f"total-loss train_ppl gave 0.00.)")

    def test_with_total_loss_gap_ratio_collapses_to_zero(self):
        """Document the bug: using total loss gives a meaningless ratio.
        This test exists to make the regression obvious if someone
        ever reverts the fix."""
        total_loss = 15.14   # observed at step 1000
        bad_train_ppl = math.exp(min(total_loss, 20.0))   # ~3.7M
        ood_ppl = 7091.3
        bad_gap_ratio = ood_ppl / bad_train_ppl
        assert bad_gap_ratio < 0.01, (
            f"total-loss train_ppl should make gap_ratio collapse to ~0; "
            f"got {bad_gap_ratio:.4f}. If this assertion fails the math "
            f"in this test is wrong, not the implementation.")


# ─────────────────────────────────────────────────────────────────────
# Contract C — OOD probe measures the STANDALONE TRUNK (cortex dropped)
# ─────────────────────────────────────────────────────────────────────


class TestOodEvalLogitsAreTrunkOnly:
    """The OOD probe must evaluate the trunk WITHOUT the cortex, so the OOD
    ppl reflects the standalone trunk (the deploy target) and is consistent
    with the trunk-only train ppl. Previously it ran the full fused forward,
    which made gap_ratio (fused OOD / trunk train_ppl) meaningless."""

    def test_uses_language_model_not_fused_forward(self):
        torch = pytest.importorskip("torch")
        from neuroslm.train_dsl import _ood_eval_logits

        class _Trunk:
            def __call__(self, ids):
                return torch.full((1, ids.shape[1], 4), 7.0)

        class _Harness:
            language_model = _Trunk()

            def __call__(self, ids):  # the FUSED forward — must NOT be used
                return torch.full((1, ids.shape[1], 4), -99.0)

        ids = torch.zeros(1, 3, dtype=torch.long)
        out = _ood_eval_logits(_Harness(), ids)
        assert (out == 7.0).all(), (
            "OOD must evaluate the trunk (language_model), not the fused "
            "cortex+trunk forward")

    def test_falls_back_to_harness_when_no_trunk(self):
        torch = pytest.importorskip("torch")
        from neuroslm.train_dsl import _ood_eval_logits

        class _Harness:
            language_model = None

            def __call__(self, ids):
                return torch.full((1, ids.shape[1], 4), -1.0)

        ids = torch.zeros(1, 3, dtype=torch.long)
        out = _ood_eval_logits(_Harness(), ids)
        assert (out == -1.0).all(), (
            "with no separable trunk, OOD falls back to the full forward")
