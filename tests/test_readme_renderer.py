"""Tests for neuroslm.readme_renderer.

TDD contract — these must be RED before the module exists,
GREEN after it is implemented.

Run with:
    brian test tests/test_readme_renderer.py
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


# ── import under test ─────────────────────────────────────────────────
from neuroslm.readme_renderer import (
    ReadmeRenderError,
    extract_placeholders,
    load_metrics,
    render,
    render_readme,
    resolve_log_macros,
)


# ── extract_placeholders ──────────────────────────────────────────────

def test_extract_placeholders_basic():
    tmpl = "foo ${A} bar ${B_C} baz"
    assert extract_placeholders(tmpl) == {"A", "B_C"}


def test_extract_placeholders_empty():
    assert extract_placeholders("no placeholders here") == set()


def test_extract_placeholders_ignores_lowercase():
    # Only ${UPPER_CASE} placeholders are managed; lowercase/mixed are
    # treated as literal text (e.g. shell vars in code examples).
    assert extract_placeholders("${lower} ${MixedCase} ${UPPER}") == {"UPPER"}


def test_extract_placeholders_no_duplicates():
    tmpl = "${A} ${A} ${B}"
    assert extract_placeholders(tmpl) == {"A", "B"}


# ── render ────────────────────────────────────────────────────────────

def test_render_basic():
    tmpl = "train_ppl=${B4_TRAIN_PPL}, gap=${B4_GAP_RATIO}"
    metrics = {"B4_TRAIN_PPL": "102.9", "B4_GAP_RATIO": "2.87"}
    assert render(tmpl, metrics) == "train_ppl=102.9, gap=2.87"


def test_render_missing_key_raises_render_error():
    tmpl = "ppl=${B4_TRAIN_PPL} gap=${MISSING_KEY}"
    metrics = {"B4_TRAIN_PPL": "102.9"}
    with pytest.raises(ReadmeRenderError) as exc_info:
        render(tmpl, metrics)
    assert "MISSING_KEY" in exc_info.value.missing


def test_render_reports_all_missing_keys_at_once():
    tmpl = "${KEY_A} ${KEY_B} ${KEY_C}"
    with pytest.raises(ReadmeRenderError) as exc_info:
        render(tmpl, {})
    missing = exc_info.value.missing
    assert set(missing) == {"KEY_A", "KEY_B", "KEY_C"}


def test_render_leaves_non_placeholder_dollars_intact():
    # Dollar signs in code fences or prose that are NOT ${UPPER} patterns
    # must pass through unchanged.
    tmpl = "cost $1.50/hr — ${B4_TRAIN_PPL} ppl"
    metrics = {"B4_TRAIN_PPL": "102.9"}
    result = render(tmpl, metrics)
    assert "$1.50" in result
    assert "${B4_TRAIN_PPL}" not in result
    assert "102.9" in result


def test_render_preserves_content_outside_placeholders():
    tmpl = "# Title\n\n${VAL}\n\n## End"
    metrics = {"VAL": "hello"}
    assert render(tmpl, metrics) == "# Title\n\nhello\n\n## End"


# ── load_metrics ──────────────────────────────────────────────────────

def test_load_metrics_returns_flat_string_dict(tmp_path):
    toml_path = tmp_path / "metrics.toml"
    toml_path.write_text(textwrap.dedent("""\
        [meta]
        TEST_COUNT = "2870"

        [b4]
        B4_TRAIN_PPL = "102.9"
        B4_GAP_RATIO = "2.87"
    """), encoding="utf-8")
    metrics = load_metrics(toml_path)
    assert metrics["TEST_COUNT"] == "2870"
    assert metrics["B4_TRAIN_PPL"] == "102.9"
    assert metrics["B4_GAP_RATIO"] == "2.87"


def test_load_metrics_converts_numbers_to_strings(tmp_path):
    toml_path = tmp_path / "metrics.toml"
    toml_path.write_text('[b]\nB0_STEPS = 80000\nB0_TRAIN_PPL = 66.0\n',
                         encoding="utf-8")
    metrics = load_metrics(toml_path)
    assert metrics["B0_STEPS"] == "80000"
    assert metrics["B0_TRAIN_PPL"] == "66.0"


def test_load_metrics_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metrics(tmp_path / "nonexistent.toml")


# ── render_readme (integration) ───────────────────────────────────────

def _make_fixture(tmp_path: Path, template: str, metrics_toml: str,
                  existing_readme: str | None = None):
    tpl = tmp_path / "README.template.md"
    tpl.write_text(template, encoding="utf-8")
    met = tmp_path / "readme_metrics.toml"
    met.write_text(metrics_toml, encoding="utf-8")
    out = tmp_path / "README.md"
    if existing_readme is not None:
        out.write_text(existing_readme, encoding="utf-8")
    return tpl, met, out


def test_render_readme_writes_output(tmp_path):
    tmpl = "ppl=${B4_TRAIN_PPL}"
    toml = '[b]\nB4_TRAIN_PPL = "102.9"\n'
    tpl, met, out = _make_fixture(tmp_path, tmpl, toml)

    rendered, is_clean = render_readme(tpl, met, out)

    assert rendered == "ppl=102.9"
    assert out.read_text(encoding="utf-8") == "ppl=102.9"
    assert is_clean is True


def test_render_readme_check_clean(tmp_path):
    tmpl = "ppl=${B4_TRAIN_PPL}"
    toml = '[b]\nB4_TRAIN_PPL = "102.9"\n'
    tpl, met, out = _make_fixture(tmp_path, tmpl, toml,
                                  existing_readme="ppl=102.9")

    rendered, is_clean = render_readme(tpl, met, out, check=True)

    assert is_clean is True
    # check mode must not write
    assert out.read_text(encoding="utf-8") == "ppl=102.9"


def test_render_readme_check_dirty(tmp_path):
    tmpl = "ppl=${B4_TRAIN_PPL}"
    toml = '[b]\nB4_TRAIN_PPL = "102.9"\n'
    tpl, met, out = _make_fixture(tmp_path, tmpl, toml,
                                  existing_readme="ppl=STALE_VALUE")

    rendered, is_clean = render_readme(tpl, met, out, check=True)

    assert is_clean is False
    # must not overwrite in check mode
    assert out.read_text(encoding="utf-8") == "ppl=STALE_VALUE"


def test_render_readme_check_no_existing_file_is_dirty(tmp_path):
    tmpl = "ppl=${B4_TRAIN_PPL}"
    toml = '[b]\nB4_TRAIN_PPL = "102.9"\n'
    tpl, met, out = _make_fixture(tmp_path, tmpl, toml)
    # out does not exist yet

    rendered, is_clean = render_readme(tpl, met, out, check=True)

    assert is_clean is False
    assert not out.exists()


def test_render_readme_missing_key_raises(tmp_path):
    tmpl = "ppl=${MISSING}"
    toml = '[b]\nB4_TRAIN_PPL = "102.9"\n'
    tpl, met, out = _make_fixture(tmp_path, tmpl, toml)

    with pytest.raises(ReadmeRenderError) as exc_info:
        render_readme(tpl, met, out)
    assert "MISSING" in exc_info.value.missing


# ── resolve_log_macros ────────────────────────────────────────────────────────
# ${LOG_TAIL:source:N} → GitHub link + last-N-lines fenced code block
# ${LOG_LINK:source}   → GitHub link only (for table cells)
# Sources: "best" (.brian/best_run.ln), "latest" (mtime-newest), TOML key, literal path

def _write_log(path: Path, n_lines: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"log line {i}" for i in range(n_lines)), encoding="utf-8")


def _write_best_ln(root: Path, log_path: Path) -> None:
    """Create .brian/best_run.ln pointing at log_path (relative to root)."""
    ln = root / ".brian" / "best_run.ln"
    ln.parent.mkdir(parents=True, exist_ok=True)
    try:
        rel = log_path.relative_to(root)
    except ValueError:
        rel = log_path
    ln.write_text(f"{rel}\n", encoding="utf-8")


class TestResolveLogMacros:
    """${LOG_TAIL:src:N} and ${LOG_LINK:src} macro resolution contracts."""

    # ── ${LOG_TAIL:best:N} ────────────────────────────────────────────────

    def test_log_tail_best_reads_ln_file(self, tmp_path):
        log = tmp_path / "logs" / "run.log"
        _write_log(log, 10)
        _write_best_ln(tmp_path, log)
        out = resolve_log_macros("${LOG_TAIL:best:5}", {}, repo_root=tmp_path)
        assert "log line 5" in out or "log line 9" in out  # tail lines present

    def test_log_tail_best_renders_github_link(self, tmp_path):
        log = tmp_path / "logs" / "run.log"
        _write_log(log)
        _write_best_ln(tmp_path, log)
        out = resolve_log_macros("${LOG_TAIL:best:3}", {}, repo_root=tmp_path)
        # link format: [`logs/run.log`](logs/run.log)
        assert "[`logs/run.log`](logs/run.log)" in out

    def test_log_tail_best_wraps_tail_in_fenced_block(self, tmp_path):
        log = tmp_path / "logs" / "run.log"
        _write_log(log, 20)
        _write_best_ln(tmp_path, log)
        out = resolve_log_macros("${LOG_TAIL:best:5}", {}, repo_root=tmp_path)
        assert "```" in out
        # Exactly the last 5 lines (indices 15-19)
        assert "log line 15" in out
        assert "log line 19" in out
        assert "log line 14" not in out  # line before the tail must not appear

    def test_log_tail_best_missing_ln_renders_gracefully(self, tmp_path):
        out = resolve_log_macros("${LOG_TAIL:best:10}", {}, repo_root=tmp_path)
        assert "not available" in out.lower()

    # ── ${LOG_TAIL:latest:N} ─────────────────────────────────────────────

    def test_log_tail_latest_uses_most_recently_modified(self, tmp_path):
        import time
        logs = tmp_path / "logs"
        old = logs / "old.log"
        new = logs / "new.log"
        _write_log(old)
        time.sleep(0.01)
        _write_log(new, 5)
        new.write_text("newest line\n", encoding="utf-8")
        out = resolve_log_macros("${LOG_TAIL:latest:5}", {}, repo_root=tmp_path)
        assert "[`logs/new.log`](logs/new.log)" in out

    def test_log_tail_latest_missing_logs_dir_renders_gracefully(self, tmp_path):
        out = resolve_log_macros("${LOG_TAIL:latest:5}", {}, repo_root=tmp_path)
        assert "not available" in out.lower()

    # ── ${LOG_TAIL:<toml-key>:N} ─────────────────────────────────────────

    def test_log_tail_toml_key_resolves_to_path(self, tmp_path):
        log = tmp_path / "logs" / "b6.log"
        _write_log(log, 15)
        metrics = {"B6_LOG": "logs/b6.log"}
        out = resolve_log_macros("${LOG_TAIL:B6_LOG:5}", metrics, repo_root=tmp_path)
        assert "[`logs/b6.log`](logs/b6.log)" in out

    # ── ${LOG_TAIL:<literal-path>:N} ─────────────────────────────────────

    def test_log_tail_literal_path(self, tmp_path):
        log = tmp_path / "logs" / "specific.log"
        _write_log(log, 8)
        out = resolve_log_macros(
            "${LOG_TAIL:logs/specific.log:4}", {}, repo_root=tmp_path
        )
        assert "[`logs/specific.log`](logs/specific.log)" in out
        assert "log line 4" in out  # one of the last 4 lines (4-7)

    # ── ${LOG_LINK:source} ────────────────────────────────────────────────

    def test_log_link_best_renders_link_only(self, tmp_path):
        log = tmp_path / "logs" / "run.log"
        _write_log(log)
        _write_best_ln(tmp_path, log)
        out = resolve_log_macros("${LOG_LINK:best}", {}, repo_root=tmp_path)
        assert out == "[`logs/run.log`](logs/run.log)"

    def test_log_link_toml_key(self, tmp_path):
        log = tmp_path / "logs" / "b6.log"
        _write_log(log)
        metrics = {"B6_LOG": "logs/b6.log"}
        out = resolve_log_macros("${LOG_LINK:B6_LOG}", metrics, repo_root=tmp_path)
        assert out == "[`logs/b6.log`](logs/b6.log)"

    def test_log_link_missing_renders_gracefully(self, tmp_path):
        out = resolve_log_macros("${LOG_LINK:best}", {}, repo_root=tmp_path)
        assert "not available" in out.lower()

    # ── integration: log macros survive standard ${KEY} render ───────────

    def test_log_macros_not_consumed_by_standard_render(self):
        """${LOG_TAIL:best:10} is NOT matched by the standard placeholder regex
        and must be left for resolve_log_macros to handle."""
        from neuroslm.readme_renderer import extract_placeholders
        tmpl = "gap=${B4_GAP_RATIO}\n${LOG_TAIL:best:10}\n"
        placeholders = extract_placeholders(tmpl)
        assert "B4_GAP_RATIO" in placeholders
        # LOG_TAIL:best:10 must NOT be in the standard placeholder set
        assert not any("LOG_TAIL" in p for p in placeholders)

    def test_render_readme_resolves_log_macros(self, tmp_path):
        """render_readme integrates log macro resolution end-to-end."""
        log = tmp_path / "logs" / "run.log"
        _write_log(log, 10)
        _write_best_ln(tmp_path, log)

        tmpl = "ppl=${PPL}\n\n${LOG_TAIL:best:3}\n"
        toml = '[m]\nPPL = "102.9"\n'
        tpl, met, out = _make_fixture(tmp_path, tmpl, toml)

        rendered, _ = render_readme(tpl, met, out, repo_root=tmp_path)

        assert "102.9" in rendered
        assert "[`logs/run.log`](logs/run.log)" in rendered
        assert "```" in rendered
