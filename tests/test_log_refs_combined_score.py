# -*- coding: utf-8 -*-
"""TDD: combined-score ranking — min(ppl + 4 × gap_ratio).

The user-facing best-run score is the **combined score**:

    score = train_ppl + GAP_RATIO_WEIGHT × gap_ratio        (lower = better)

with ``GAP_RATIO_WEIGHT = 4.0``.  The factor weighs OOD generalisation
~4× as heavily as raw training-set fit; runs that overfit (low train PPL,
huge gap) are properly penalised against runs that generalise.

Backward-compatibility tier rules are preserved:

* a run that has ``gap_ratio`` ALWAYS beats a run that only has ``train_ppl``
  (a measured OOD eval is more informative than no measurement at all);
* if no run has ``gap_ratio``, the fallback ranking is pure ``train_ppl``;
* when both runs have both metrics, the combined score decides — NOT raw
  ``gap_ratio`` alone as in the legacy implementation.

Contracts pinned here
---------------------
CS-1   GAP_RATIO_WEIGHT exported as a module constant equal to 4.0.
CS-2   RunScore.combined_score = train_ppl + 4 × gap_ratio.
CS-3   combined_score returns +inf when train_ppl is None.
CS-4   combined_score returns train_ppl when gap_ratio is None.
CS-5   find_best_log default metric ("combined") uses combined score.
CS-6   metric="combined" picks the lower combined-score run even when its
       raw gap_ratio is HIGHER than a competitor's.
CS-7   metric="gap_ratio" still ranks by raw gap_ratio (legacy contract).
CS-8   metric="ppl" still ranks by raw train_ppl (legacy contract).
CS-9   tier preservation: any-gap_ratio beats only-train_ppl.
CS-10  empty log dir → None (no regression).
CS-11  the .ln comment written by update_best_run_pointer reflects the
       new "combined" metric label.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


# ── Synthetic log fragments ────────────────────────────────────────────

# Run A: low gap_ratio (2.0) but high train_ppl (100.0) → combined = 108.0
_LOG_LOW_GAP_HIGH_PPL = textwrap.dedent("""\
    [train_dsl] boot
    step  1000 | loss 4.500 | lm 4.500 | ppl 100.0 | gnorm 1.0 | lr 3.00e-04 | 500 tok/s
    [mid-ood] step 1000: wikitext ppl=200.0 gap_ratio=2.00 (train_ppl=100.0) (50 seq, 6430 tok)
""")

# Run B: higher gap_ratio (5.0) but much lower train_ppl (20.0) → combined = 40.0
_LOG_HIGH_GAP_LOW_PPL = textwrap.dedent("""\
    [train_dsl] boot
    step  1000 | loss 2.996 | lm 2.996 | ppl 20.0 | gnorm 0.7 | lr 2.00e-04 | 700 tok/s
    [mid-ood] step 1000: wikitext ppl=100.0 gap_ratio=5.00 (train_ppl=20.0) (50 seq, 6430 tok)
""")

# Run C: very balanced — gap 3.0, ppl 30.0 → combined = 42.0 (slightly worse than B)
_LOG_BALANCED = textwrap.dedent("""\
    [train_dsl] boot
    step  2000 | loss 3.401 | lm 3.401 | ppl 30.0 | gnorm 0.8 | lr 1.50e-04 | 650 tok/s
    [mid-ood] step 2000: wikitext ppl=90.0 gap_ratio=3.00 (train_ppl=30.0) (50 seq, 6430 tok)
""")

# Run with only train_ppl, no OOD eval
_LOG_PPL_ONLY = textwrap.dedent("""\
    [train_dsl] boot
    step   500 | loss 5.298 | lm 5.298 | ppl 200.0 | gnorm 1.5 | lr 3.00e-04 | 500 tok/s
""")


# ── helpers ────────────────────────────────────────────────────────────

def _write_log(directory: Path, name: str, content: str) -> Path:
    p = directory / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _score_from_text(text: str):
    """Run score_log on a synthetic text fragment."""
    import tempfile
    import os
    from neuroslm.log_refs import score_log
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log",
                                     delete=False, encoding="utf-8") as f:
        f.write(text)
        name = f.name
    try:
        return score_log(Path(name))
    finally:
        os.unlink(name)


# ── CS-1: GAP_RATIO_WEIGHT constant ────────────────────────────────────

class TestGapRatioWeightConstant:

    def test_constant_exposed(self):
        from neuroslm.log_refs import GAP_RATIO_WEIGHT
        assert GAP_RATIO_WEIGHT == 4.0

    def test_constant_is_float(self):
        from neuroslm.log_refs import GAP_RATIO_WEIGHT
        assert isinstance(GAP_RATIO_WEIGHT, float)


# ── CS-2/CS-3/CS-4: RunScore.combined_score property ──────────────────

class TestRunScoreCombinedScore:

    def test_combined_score_formula(self):
        """combined_score = train_ppl + 4 × gap_ratio."""
        from neuroslm.log_refs import RunScore
        s = RunScore(log_path=Path("x.log"), gap_ratio=2.0, train_ppl=100.0)
        assert s.combined_score == pytest.approx(108.0, rel=1e-6)

    def test_combined_score_different_values(self):
        from neuroslm.log_refs import RunScore
        s = RunScore(log_path=Path("y.log"), gap_ratio=5.0, train_ppl=20.0)
        # 20 + 4*5 = 40
        assert s.combined_score == pytest.approx(40.0, rel=1e-6)

    def test_combined_score_no_train_ppl_is_inf(self):
        from neuroslm.log_refs import RunScore
        s = RunScore(log_path=Path("z.log"), gap_ratio=2.0, train_ppl=None)
        assert s.combined_score == float("inf")

    def test_combined_score_no_gap_ratio_is_train_ppl(self):
        """A run with only train_ppl scores as just train_ppl (no penalty)."""
        from neuroslm.log_refs import RunScore
        s = RunScore(log_path=Path("p.log"), gap_ratio=None, train_ppl=50.0)
        assert s.combined_score == pytest.approx(50.0, rel=1e-6)

    def test_combined_score_both_none_is_inf(self):
        from neuroslm.log_refs import RunScore
        s = RunScore(log_path=Path("q.log"), gap_ratio=None, train_ppl=None)
        assert s.combined_score == float("inf")


# ── CS-5/CS-6: find_best_log default uses combined ranking ─────────────

class TestFindBestLogCombinedMetric:

    def test_default_metric_is_combined(self, tmp_path):
        """find_best_log() with no metric arg uses combined ranking."""
        from neuroslm.log_refs import find_best_log
        # A: gap 2.0, ppl 100 → combined 108
        # B: gap 5.0, ppl 20  → combined 40  ← best
        _write_log(tmp_path, "a.log", _LOG_LOW_GAP_HIGH_PPL)
        _write_log(tmp_path, "b.log", _LOG_HIGH_GAP_LOW_PPL)
        best = find_best_log(tmp_path)
        assert best == tmp_path / "b.log", \
            "default ranking should be combined score (lower-ppl wins)"

    def test_combined_picks_lower_combined_score_winner(self, tmp_path):
        """B (combined=40) wins over C (combined=42) even though gap_ratios
        favour neither dramatically."""
        from neuroslm.log_refs import find_best_log
        _write_log(tmp_path, "b.log", _LOG_HIGH_GAP_LOW_PPL)  # combined 40
        _write_log(tmp_path, "c.log", _LOG_BALANCED)           # combined 42
        best = find_best_log(tmp_path, metric="combined")
        assert best == tmp_path / "b.log"

    def test_explicit_combined_metric_arg(self, tmp_path):
        """metric='combined' is the canonical opt-in name; works the same as default."""
        from neuroslm.log_refs import find_best_log
        _write_log(tmp_path, "a.log", _LOG_LOW_GAP_HIGH_PPL)
        _write_log(tmp_path, "b.log", _LOG_HIGH_GAP_LOW_PPL)
        assert find_best_log(tmp_path, metric="combined") == tmp_path / "b.log"


# ── CS-7: legacy metric="gap_ratio" still works ────────────────────────

class TestLegacyGapRatioMetric:

    def test_metric_gap_ratio_picks_lower_gap_winner(self, tmp_path):
        """With metric='gap_ratio', A (gap 2.0) beats B (gap 5.0) even though
        B has a much better combined score."""
        from neuroslm.log_refs import find_best_log
        _write_log(tmp_path, "a.log", _LOG_LOW_GAP_HIGH_PPL)  # gap 2.0
        _write_log(tmp_path, "b.log", _LOG_HIGH_GAP_LOW_PPL)  # gap 5.0
        best = find_best_log(tmp_path, metric="gap_ratio")
        assert best == tmp_path / "a.log", \
            "metric=gap_ratio must use raw gap_ratio ranking (legacy)"


# ── CS-8: legacy metric="ppl" still works ──────────────────────────────

class TestLegacyPplMetric:

    def test_metric_ppl_picks_lower_train_ppl_winner(self, tmp_path):
        """With metric='ppl', the run with lower train_ppl wins."""
        from neuroslm.log_refs import find_best_log
        _write_log(tmp_path, "a.log", _LOG_LOW_GAP_HIGH_PPL)  # ppl 100
        _write_log(tmp_path, "b.log", _LOG_HIGH_GAP_LOW_PPL)  # ppl 20
        best = find_best_log(tmp_path, metric="ppl")
        assert best == tmp_path / "b.log"


# ── CS-9: tier preservation ────────────────────────────────────────────

class TestTierPreservation:

    def test_any_gap_ratio_beats_only_train_ppl(self, tmp_path):
        """A run WITH gap_ratio always beats a run with only train_ppl,
        regardless of which has the lower combined score numerically."""
        from neuroslm.log_refs import find_best_log
        # ppl_only: combined = 200 (just the ppl, since no gap data)
        # with_gap_high: combined = 108 — would win by score alone
        # Yet with_gap_high SHOULD win, and we test that scenario:
        _write_log(tmp_path, "ppl_only.log", _LOG_PPL_ONLY)           # ppl 200
        _write_log(tmp_path, "with_gap.log", _LOG_LOW_GAP_HIGH_PPL)   # ppl 100 + gap 2
        best = find_best_log(tmp_path, metric="combined")
        assert best == tmp_path / "with_gap.log", \
            "tier-1 (with gap_ratio) must beat tier-2 (only train_ppl)"

    def test_only_ppl_runs_fall_back_to_train_ppl(self, tmp_path):
        """When NO run has gap_ratio, ranking is purely by train_ppl."""
        from neuroslm.log_refs import find_best_log
        _write_log(tmp_path, "a.log", _LOG_PPL_ONLY)
        log_lower_ppl = textwrap.dedent("""\
            [train_dsl] boot
            step  1000 | loss 4.110 | lm 4.110 | ppl 61.0 | gnorm 0.9 | lr 2.00e-04 | 600 tok/s
        """)
        _write_log(tmp_path, "b.log", log_lower_ppl)
        best = find_best_log(tmp_path, metric="combined")
        assert best == tmp_path / "b.log"


# ── CS-10: empty dir regression guard ──────────────────────────────────

class TestEmptyDirRegression:

    def test_no_logs_returns_none_default_metric(self, tmp_path):
        from neuroslm.log_refs import find_best_log
        assert find_best_log(tmp_path) is None

    def test_no_logs_returns_none_combined_metric(self, tmp_path):
        from neuroslm.log_refs import find_best_log
        assert find_best_log(tmp_path, metric="combined") is None


# ── CS-11: .ln comment reflects "combined" metric ──────────────────────

class TestUpdateBestRunPointerCombinedComment:

    def test_combined_default_writes_combined_in_comment(self, tmp_path):
        from neuroslm.log_refs import update_best_run_pointer, read_ref, BEST_RUN_LN
        logs = tmp_path / "logs"
        _write_log(logs, "a.log", _LOG_HIGH_GAP_LOW_PPL)
        # default metric (no arg) should be "combined"
        update_best_run_pointer(root=tmp_path, log_dir=logs)
        ref = read_ref(tmp_path / BEST_RUN_LN)
        assert "combined" in ref.comment.lower()

    def test_explicit_gap_ratio_metric_writes_gap_ratio_in_comment(self, tmp_path):
        from neuroslm.log_refs import update_best_run_pointer, read_ref, BEST_RUN_LN
        logs = tmp_path / "logs"
        _write_log(logs, "a.log", _LOG_HIGH_GAP_LOW_PPL)
        update_best_run_pointer(root=tmp_path, log_dir=logs, metric="gap_ratio")
        ref = read_ref(tmp_path / BEST_RUN_LN)
        assert "gap_ratio" in ref.comment.lower()
