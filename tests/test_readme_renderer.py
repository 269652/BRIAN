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
