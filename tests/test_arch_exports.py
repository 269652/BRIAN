"""Tests for neuroslm.arch_exports — RED before GREEN.

Run with:
    brian test tests/test_arch_exports.py
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from neuroslm.arch_exports import (
    parse_expert_blocks,
    parse_export_directives,
    resolve_export_expr,
    collect_arch_exports,
    write_neuro_exports,
)


# ── fixtures ──────────────────────────────────────────────────────────

ARCH_SAMPLE = textwrap.dedent("""\
    expert GeneralExpert {
        model:  "smollm2_360m",
        role:   "general",
        frozen: true
    }

    expert CodeExpert {
        model:  "microsoft/CodeGPT-small-py",
        role:   "code",
        frozen: true
    }

    expert ReasoningExpert {
        model:  "Qwen/Qwen2.5-0.5B",
        role:   "reasoning",
        frozen: true
    }

    # @export EXPERT_GENERAL_MODEL = GeneralExpert.model
    # @export EXPERT_GENERAL_ROLE  = GeneralExpert.role
    # @export EXPERT_CODE_MODEL    = CodeExpert.model
    # @export EXPERT_CODE_ROLE     = CodeExpert.role
    # @export EXPERT_REASONING_MODEL = ReasoningExpert.model
    # @export EXPERT_REASONING_ROLE  = ReasoningExpert.role
    # @export TRUNK_TRAINABLE_PARAMS = "~147M"
    # @export EXPERT_GENERAL_PARAMS  = "~360M"
    # @export EXPERT_CODE_PARAMS     = "~125M"
    # @export EXPERT_REASONING_PARAMS = "~500M"
""")


# ── parse_expert_blocks ───────────────────────────────────────────────

def test_parse_expert_blocks_extracts_all_three():
    experts = parse_expert_blocks(ARCH_SAMPLE)
    assert set(experts.keys()) == {"GeneralExpert", "CodeExpert", "ReasoningExpert"}


def test_parse_expert_blocks_model_field():
    experts = parse_expert_blocks(ARCH_SAMPLE)
    assert experts["GeneralExpert"]["model"] == "smollm2_360m"
    assert experts["CodeExpert"]["model"] == "microsoft/CodeGPT-small-py"
    assert experts["ReasoningExpert"]["model"] == "Qwen/Qwen2.5-0.5B"


def test_parse_expert_blocks_role_field():
    experts = parse_expert_blocks(ARCH_SAMPLE)
    assert experts["GeneralExpert"]["role"] == "general"
    assert experts["CodeExpert"]["role"] == "code"
    assert experts["ReasoningExpert"]["role"] == "reasoning"


def test_parse_expert_blocks_empty_text():
    assert parse_expert_blocks("no expert blocks here") == {}


def test_parse_expert_blocks_strips_quotes_and_whitespace():
    text = 'expert MyExpert {\n    model: "  spaced  ",\n    role: "r"\n}'
    experts = parse_expert_blocks(text)
    assert experts["MyExpert"]["model"] == "spaced"


# ── parse_export_directives ───────────────────────────────────────────

def test_parse_export_directives_returns_list_of_tuples():
    directives = parse_export_directives(ARCH_SAMPLE)
    keys = [k for k, _ in directives]
    assert "EXPERT_GENERAL_MODEL" in keys
    assert "EXPERT_CODE_MODEL" in keys
    assert "EXPERT_REASONING_MODEL" in keys


def test_parse_export_directives_captures_expressions():
    directives = parse_export_directives(ARCH_SAMPLE)
    d = dict(directives)
    assert d["EXPERT_GENERAL_MODEL"] == "GeneralExpert.model"
    assert d["EXPERT_CODE_MODEL"] == "CodeExpert.model"
    assert d["TRUNK_TRAINABLE_PARAMS"] == '"~147M"'


def test_parse_export_directives_empty_text():
    assert parse_export_directives("no exports here") == []


def test_parse_export_directives_ignores_uncommented_at_export():
    text = "@export SHOULD_IGNORE = foo\n# @export SHOULD_CAPTURE = bar"
    directives = dict(parse_export_directives(text))
    assert "SHOULD_IGNORE" not in directives
    assert "SHOULD_CAPTURE" in directives


# ── resolve_export_expr ───────────────────────────────────────────────

EXPERTS = {
    "GeneralExpert": {"model": "smollm2_360m", "role": "general"},
    "CodeExpert": {"model": "microsoft/CodeGPT-small-py", "role": "code"},
}


def test_resolve_dot_path_expert_model():
    assert resolve_export_expr("GeneralExpert.model", EXPERTS) == "smollm2_360m"


def test_resolve_dot_path_expert_role():
    assert resolve_export_expr("CodeExpert.role", EXPERTS) == "code"


def test_resolve_string_literal_with_quotes():
    assert resolve_export_expr('"~147M"', EXPERTS) == "~147M"


def test_resolve_string_literal_single_quotes():
    assert resolve_export_expr("'some value'", EXPERTS) == "some value"


def test_resolve_unknown_expert_returns_empty():
    assert resolve_export_expr("UnknownExpert.model", EXPERTS) == ""


def test_resolve_unknown_field_returns_empty():
    assert resolve_export_expr("GeneralExpert.nonexistent", EXPERTS) == ""


def test_resolve_bare_string_no_quotes():
    # A bare word with no dot and no expert match returns it as-is
    assert resolve_export_expr("somevalue", EXPERTS) == "somevalue"


# ── collect_arch_exports ──────────────────────────────────────────────

def test_collect_arch_exports_from_file(tmp_path):
    arch_neuro = tmp_path / "arch.neuro"
    arch_neuro.write_text(ARCH_SAMPLE, encoding="utf-8")

    exports = collect_arch_exports(arch_neuro)

    assert exports["EXPERT_GENERAL_MODEL"] == "smollm2_360m"
    assert exports["EXPERT_CODE_MODEL"] == "microsoft/CodeGPT-small-py"
    assert exports["EXPERT_REASONING_MODEL"] == "Qwen/Qwen2.5-0.5B"
    assert exports["EXPERT_GENERAL_ROLE"] == "general"
    assert exports["TRUNK_TRAINABLE_PARAMS"] == "~147M"


def test_collect_arch_exports_all_string_values(tmp_path):
    arch_neuro = tmp_path / "arch.neuro"
    arch_neuro.write_text(ARCH_SAMPLE, encoding="utf-8")

    exports = collect_arch_exports(arch_neuro)
    for key, value in exports.items():
        assert isinstance(value, str), f"{key} is not a string: {value!r}"


def test_collect_arch_exports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_arch_exports(tmp_path / "nonexistent.neuro")


def test_collect_arch_exports_skips_unresolvable(tmp_path):
    text = textwrap.dedent("""\
        # @export GOOD = "hello"
        # @export BAD = UnknownExpert.model
    """)
    arch_neuro = tmp_path / "arch.neuro"
    arch_neuro.write_text(text, encoding="utf-8")

    exports = collect_arch_exports(arch_neuro)
    assert exports["GOOD"] == "hello"
    # Unresolvable expression → key is omitted (empty string is excluded)
    assert "BAD" not in exports


# ── write_neuro_exports ───────────────────────────────────────────────

def test_write_neuro_exports_creates_toml(tmp_path):
    exports = {"EXPERT_GENERAL_MODEL": "smollm2_360m", "TRUNK_TRAINABLE_PARAMS": "~147M"}
    out = write_neuro_exports(exports, tmp_path)

    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "EXPERT_GENERAL_MODEL" in text
    assert "smollm2_360m" in text


def test_write_neuro_exports_is_valid_toml(tmp_path):
    exports = {"KEY_A": "val1", "KEY_B": "val2"}
    out = write_neuro_exports(exports, tmp_path)

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore

    with open(out, "rb") as fh:
        parsed = tomllib.load(fh)

    flat = {}
    for section in parsed.values():
        if isinstance(section, dict):
            flat.update(section)
    assert flat.get("KEY_A") == "val1"
    assert flat.get("KEY_B") == "val2"


def test_write_neuro_exports_overwrites_existing(tmp_path):
    (tmp_path / "exports.toml").write_text("old content", encoding="utf-8")
    write_neuro_exports({"NEW_KEY": "new_val"}, tmp_path)
    text = (tmp_path / "exports.toml").read_text(encoding="utf-8")
    assert "NEW_KEY" in text
    assert "old content" not in text
