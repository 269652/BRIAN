"""Tests for neuroslm.readme_metrics — README template metric collection.

TDD contract:
  - parse_layer_b_table(text) extracts best/baseline from findings.md table
  - count_tests_from_durations_cache(path) reads the duration JSON or returns 0
  - render_template(template, metrics) does safe ${PLACEHOLDER} substitution
  - build_metrics(repo_root) returns a complete str→str mapping
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from neuroslm.readme_metrics import (
    build_metrics,
    count_tests_from_durations_cache,
    parse_layer_b_table,
    render_template,
)

_SAMPLE_TABLE = textwrap.dedent(
    """\
    | Row | Branch | Ckpt step | Params | train_ppl | OOD_ppl | **gap_ratio** | verdict | artifact |
    |---|---|---|---|---|---|---|---|---|
    | **B0** **flat-transformer baseline** | `stabilize/trunk-grad-isolation` | 80000 | 106.9M | **66.0** | **404.0** | 6.12 | STRONG OVERFITTING | `results/ood_baseline.json` |
    | **B2** trunk-iso + ReZero (load-bug) | `stabilize/trunk-grad-isolation` | 7000 | 107.8M | 1169.9 | 5242.7 | 4.48 | ARTIFACT -- params zero-init at eval (see B2.fix) | `results/buggy.json` |
    | **B2.fix** trunk-iso + ReZero | `stabilize/trunk-grad-isolation` | 7000 | 107.8M | 258.8 | 1351.5 | **5.22** | STRONG OVERFITTING | `results/ood_rezero-fixed.json` |
    | **B4** abstain-fix + multi-cortex | `master` @ `a22eecc` | 2000 | **889.6M** | **102.9** | **295.9** | **2.87** | **NEW BAND** | `logs/vast/...` |
    | **B5** H21 10k rerun | `master` @ `8d7140c` | 3000 (mid-run) | 889.6M | **45.0** | **130.1** | 2.89 | COMPARABLE | `logs/vast/...` |
    | **B6** SmolLM2 upgrade | `master` @ `c19bf62` | 10000 | **1127M** | **23.6** | **155.0** | 6.55 | GAP REGRESSION | `logs/vast/...` |
    """
)


class TestParseLayerBTable:
    def test_baseline_row_extracted(self):
        r = parse_layer_b_table(_SAMPLE_TABLE)
        assert r["baseline"]["row"] == "B0"
        assert r["baseline"]["gap_ratio"] == pytest.approx(6.12)

    def test_artifact_row_excluded_from_best(self):
        # B2 (buggy) has gap_ratio 4.48 — would be the best if not excluded.
        # Best should be B4 at 2.87, not B2's 4.48.
        r = parse_layer_b_table(_SAMPLE_TABLE)
        assert r["best"]["row"] == "B4"

    def test_best_gap_ratio(self):
        r = parse_layer_b_table(_SAMPLE_TABLE)
        assert r["best"]["gap_ratio"] == pytest.approx(2.87)

    def test_best_train_ppl(self):
        r = parse_layer_b_table(_SAMPLE_TABLE)
        assert r["best"]["train_ppl"] == pytest.approx(102.9)

    def test_best_ood_ppl(self):
        r = parse_layer_b_table(_SAMPLE_TABLE)
        assert r["best"]["ood_ppl"] == pytest.approx(295.9)

    def test_improvement_pct_vs_baseline(self):
        # (6.12 - 2.87) / 6.12 * 100 ≈ 53.1 %
        r = parse_layer_b_table(_SAMPLE_TABLE)
        assert r["best"]["improvement_pct"] == pytest.approx(53.1, abs=0.5)

    def test_b2fix_is_not_artifact(self):
        # B2.fix row should appear in all_rows and not be marked as artifact
        r = parse_layer_b_table(_SAMPLE_TABLE)
        b2fix_rows = [row for row in r["all_rows"] if row["row"] == "B2.fix"]
        assert b2fix_rows, "B2.fix row missing"
        assert not b2fix_rows[0]["is_artifact"]

    def test_missing_baseline_raises(self):
        text = "| Row | gap_ratio |\n|---|---|\n| **B4** foo | 2.87 |\n"
        with pytest.raises(ValueError, match="baseline"):
            parse_layer_b_table(text)

    def test_no_table_raises(self):
        with pytest.raises(ValueError):
            parse_layer_b_table("nothing here")


class TestCountTestsFromDurationsCache:
    def test_counts_entries(self, tmp_path):
        cache = tmp_path / "test_durations.json"
        cache.write_text(json.dumps({
            "tests/test_a.py::TestFoo::test_bar": 0.12,
            "tests/test_b.py::test_baz": 0.08,
            "tests/test_c.py::test_qux": 0.05,
        }))
        assert count_tests_from_durations_cache(cache) == 3

    def test_returns_zero_for_missing_file(self, tmp_path):
        assert count_tests_from_durations_cache(tmp_path / "missing.json") == 0

    def test_returns_zero_for_corrupt_json(self, tmp_path):
        cache = tmp_path / "test_durations.json"
        cache.write_text("{not valid json")
        assert count_tests_from_durations_cache(cache) == 0

    def test_handles_empty_dict(self, tmp_path):
        cache = tmp_path / "test_durations.json"
        cache.write_text("{}")
        assert count_tests_from_durations_cache(cache) == 0


class TestRenderTemplate:
    def test_substitutes_braced_placeholders(self):
        tpl = "Count: ${LAYER_A_TEST_COUNT} Gap: ${LAYER_B_BEST_GAP_RATIO}"
        out = render_template(tpl, {"LAYER_A_TEST_COUNT": "1511", "LAYER_B_BEST_GAP_RATIO": "2.87"})
        assert out == "Count: 1511 Gap: 2.87"

    def test_unknown_placeholder_left_intact(self):
        tpl = "known: ${LAYER_A_TEST_COUNT} unknown: ${DOES_NOT_EXIST}"
        out = render_template(tpl, {"LAYER_A_TEST_COUNT": "42"})
        assert "${DOES_NOT_EXIST}" in out
        assert "42" in out

    def test_dollar_space_not_touched(self):
        tpl = "```bash\n$ pip install brian\n```"
        out = render_template(tpl, {})
        assert out == tpl

    def test_empty_metrics_leaves_template_unchanged(self):
        tpl = "Hello ${NAME}"
        out = render_template(tpl, {})
        assert out == tpl


class TestCountTestsByScanning:
    def test_counts_test_functions(self, tmp_path):
        from neuroslm.readme_metrics import count_tests_by_scanning
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_foo.py").write_text(
            "def test_bar(): pass\ndef test_baz(): pass\n", encoding="utf-8"
        )
        assert count_tests_by_scanning(tmp_path / "tests") == 2

    def test_skips_training_subdir(self, tmp_path):
        from neuroslm.readme_metrics import count_tests_by_scanning
        (tmp_path / "tests" / "training").mkdir(parents=True)
        (tmp_path / "tests" / "training" / "test_slow.py").write_text(
            "def test_slow(): pass\n", encoding="utf-8"
        )
        assert count_tests_by_scanning(tmp_path / "tests") == 0

    def test_counts_class_methods(self, tmp_path):
        from neuroslm.readme_metrics import count_tests_by_scanning
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_class.py").write_text(
            "class TestFoo:\n    def test_a(self): pass\n    def test_b(self): pass\n",
            encoding="utf-8",
        )
        assert count_tests_by_scanning(tmp_path / "tests") == 2

    def test_missing_dir_returns_zero(self, tmp_path):
        from neuroslm.readme_metrics import count_tests_by_scanning
        assert count_tests_by_scanning(tmp_path / "nonexistent") == 0


class TestBuildMetrics:
    def test_returns_str_dict(self, tmp_path):
        # Point build_metrics at a minimal fake repo root with just what it needs.
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "FINDINGS.md").write_text(_SAMPLE_TABLE, encoding="utf-8")
        (tmp_path / ".neuro").mkdir()
        (tmp_path / ".neuro" / "test_durations.json").write_text(
            json.dumps({"tests/a.py::test_one": 0.1, "tests/b.py::test_two": 0.2}),
            encoding="utf-8",
        )
        metrics = build_metrics(tmp_path)
        assert isinstance(metrics, dict)
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in metrics.items())

    def test_includes_layer_a_test_count(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "FINDINGS.md").write_text(_SAMPLE_TABLE, encoding="utf-8")
        (tmp_path / ".neuro").mkdir()
        (tmp_path / ".neuro" / "test_durations.json").write_text(
            json.dumps({"tests/a.py::test_one": 0.1}),
            encoding="utf-8",
        )
        metrics = build_metrics(tmp_path)
        assert metrics["LAYER_A_TEST_COUNT"] == "1"

    def test_includes_layer_b_best_gap_ratio(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "FINDINGS.md").write_text(_SAMPLE_TABLE, encoding="utf-8")
        (tmp_path / ".neuro").mkdir()
        (tmp_path / ".neuro" / "test_durations.json").write_text("{}", encoding="utf-8")
        metrics = build_metrics(tmp_path)
        assert metrics["LAYER_B_BEST_GAP_RATIO"] == "2.87"

    def test_layer_a_count_falls_back_when_no_cache(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "FINDINGS.md").write_text(_SAMPLE_TABLE, encoding="utf-8")
        (tmp_path / ".neuro").mkdir()
        # No test_durations.json — returns "0"
        metrics = build_metrics(tmp_path)
        count = int(metrics["LAYER_A_TEST_COUNT"])
        assert count >= 0
