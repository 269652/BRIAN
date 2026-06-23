# -*- coding: utf-8 -*-
"""Tests for the cross-platform secrets helper.

Coverage matrix (each row → at least one test):

  SH-1   env var hit short-circuits the chain
  SH-2   provider chain order is respected (low priority wins)
  SH-3   aliases are tried within each backend before moving on
  SH-4   ``default=`` is returned when nothing resolves
  SH-5   ``cache_env=True`` exports to os.environ; ``False`` doesn't
  SH-6   provider that raises is treated as a miss, not propagated
  SH-7   ``register_secret_provider`` honours priority ordering
  SH-8   re-registering a provider name replaces the old one
  SH-9   ``unregister_secret_provider`` removes it
  SH-10  ``.env`` parsing: quoted, unquoted, exported, with-comments
  SH-11  ``.env`` walk: child dir finds parent's .env
  SH-12  ``bootstrap_secrets`` resolves & reports
  SH-13  ``bootstrap_secrets(required=…)`` raises on missing
  SH-14  ``detect_environment`` returns a known label
  SH-15  Colab/Kaggle providers degrade silently when the SDK is absent

Tests run in pure Python — no Colab/Kaggle SDKs required.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from neuroslm.utils import secrets as sec
from neuroslm.utils.secrets import (
    bootstrap_secrets,
    detect_environment,
    get_secret,
    list_secret_providers,
    register_secret_provider,
    unregister_secret_provider,
)


# ── helpers ────────────────────────────────────────────────────────


@pytest.fixture
def isolated_providers():
    """Save & restore the global provider registry around a test."""
    saved = list(sec._PROVIDERS)  # shallow copy is fine — tuples are frozen
    yield
    sec._PROVIDERS[:] = saved


@pytest.fixture
def clean_env(monkeypatch):
    """Wipe test-namespaced env keys + the actual secret names the
    helper manages, so cross-test state doesn't bleed."""
    for k in list(os.environ):
        if k.startswith("BRIAN_TEST_"):
            monkeypatch.delenv(k, raising=False)
    # Also clear the canonical secret names — tests below explicitly
    # set+assert these, so any value from a prior test must be gone.
    for k in ("GITHUB", "GITHUB_TOKEN", "GH_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


# ── SH-1: env var short-circuit ────────────────────────────────────


def test_env_var_hit_short_circuits(clean_env):
    clean_env.setenv("BRIAN_TEST_GH", "abc123")
    assert get_secret("BRIAN_TEST_GH") == "abc123"


def test_env_value_stripped(clean_env):
    clean_env.setenv("BRIAN_TEST_GH", "   abc123\n  ")
    assert get_secret("BRIAN_TEST_GH") == "abc123"


def test_empty_env_treated_as_missing(clean_env):
    clean_env.setenv("BRIAN_TEST_GH", "   ")
    # Empty after strip → miss → falls through to default
    assert get_secret("BRIAN_TEST_GH", default="fallback") == "fallback"


# ── SH-2/SH-7/SH-8: provider chain ordering ────────────────────────


def test_chain_order_respects_priority(clean_env, isolated_providers):
    """Provider with the lowest priority number runs first."""
    calls: list[str] = []

    def low(_):
        calls.append("low")
        return "from-low"

    def high(_):
        calls.append("high")
        return "from-high"

    register_secret_provider("low", low, priority=5)
    register_secret_provider("high", high, priority=99)

    assert get_secret("BRIAN_TEST_GH") == "from-low"
    assert calls == ["low"]   # high never invoked


def test_higher_priority_takes_over_after_low_misses(clean_env, isolated_providers):
    def low(_): return None
    def high(_): return "from-high"

    register_secret_provider("low", low, priority=5)
    register_secret_provider("high", high, priority=99)

    assert get_secret("BRIAN_TEST_GH") == "from-high"


def test_re_registration_replaces(clean_env, isolated_providers):
    register_secret_provider("dup", lambda _: "v1", priority=5)
    register_secret_provider("dup", lambda _: "v2", priority=5)
    names = [n for _, n in list_secret_providers()]
    assert names.count("dup") == 1
    assert get_secret("BRIAN_TEST_GH") == "v2"


def test_unregister_removes(clean_env, isolated_providers):
    register_secret_provider("temp", lambda _: "x", priority=5)
    assert unregister_secret_provider("temp") is True
    assert unregister_secret_provider("temp") is False    # second call → no-op


# ── SH-3: alias chain within each backend ──────────────────────────


def test_alias_tried_when_primary_missing(clean_env):
    clean_env.setenv("GITHUB_TOKEN", "xyz789")
    val = get_secret("GITHUB", aliases=("GITHUB_TOKEN", "GH_TOKEN"))
    assert val == "xyz789"


def test_alias_caches_under_primary_name(clean_env):
    clean_env.setenv("GITHUB_TOKEN", "xyz789")
    get_secret("GITHUB", aliases=("GITHUB_TOKEN",), cache_env=True)
    # Primary name now exported even though the value came via alias
    assert os.environ.get("GITHUB") == "xyz789"


def test_primary_wins_over_alias(clean_env):
    clean_env.setenv("GITHUB", "primary-val")
    clean_env.setenv("GITHUB_TOKEN", "alias-val")
    assert get_secret("GITHUB", aliases=("GITHUB_TOKEN",)) == "primary-val"


# ── SH-4: default ──────────────────────────────────────────────────


def test_default_returned_when_nothing_resolves(clean_env, isolated_providers):
    # Disable all built-in providers so we're guaranteed a miss
    for _, name, _ in list(sec._PROVIDERS):
        unregister_secret_provider(name)
    assert get_secret("BRIAN_TEST_NX", default="DEFLT") == "DEFLT"
    assert get_secret("BRIAN_TEST_NX") is None


# ── SH-5: cache_env toggle ─────────────────────────────────────────


def test_cache_env_true_exports(clean_env, isolated_providers):
    register_secret_provider("custom", lambda _: "cached-val", priority=5)
    get_secret("BRIAN_TEST_GH", cache_env=True)
    assert os.environ.get("BRIAN_TEST_GH") == "cached-val"


def test_cache_env_false_does_not_export(clean_env, isolated_providers):
    register_secret_provider("custom", lambda _: "transient", priority=5)
    val = get_secret("BRIAN_TEST_GH", cache_env=False)
    assert val == "transient"
    assert "BRIAN_TEST_GH" not in os.environ


# ── SH-6: provider exceptions are swallowed ────────────────────────


def test_raising_provider_does_not_break_chain(clean_env, isolated_providers):
    def boom(_):
        raise RuntimeError("network down")

    def ok(_):
        return "survived"

    register_secret_provider("boom", boom, priority=5)
    register_secret_provider("ok", ok, priority=10)
    assert get_secret("BRIAN_TEST_GH") == "survived"


# ── SH-10/SH-11: .env file parsing ─────────────────────────────────


def test_dotenv_simple_kv(tmp_path, monkeypatch, clean_env):
    (tmp_path / ".env").write_text("BRIAN_TEST_GH=plain-value\n")
    monkeypatch.chdir(tmp_path)
    assert get_secret("BRIAN_TEST_GH") == "plain-value"


def test_dotenv_quoted_values(tmp_path, monkeypatch, clean_env):
    (tmp_path / ".env").write_text(
        'BRIAN_TEST_DQ="double-q"\n'
        "BRIAN_TEST_SQ='single-q'\n"
    )
    monkeypatch.chdir(tmp_path)
    assert get_secret("BRIAN_TEST_DQ") == "double-q"
    assert get_secret("BRIAN_TEST_SQ") == "single-q"


def test_dotenv_export_prefix(tmp_path, monkeypatch, clean_env):
    (tmp_path / ".env").write_text("export BRIAN_TEST_X=shell-style\n")
    monkeypatch.chdir(tmp_path)
    assert get_secret("BRIAN_TEST_X") == "shell-style"


def test_dotenv_comments_and_blank_lines(tmp_path, monkeypatch, clean_env):
    (tmp_path / ".env").write_text(
        "# this is a comment\n"
        "\n"
        "BRIAN_TEST_REAL=here\n"
        "# trailing comment\n"
    )
    monkeypatch.chdir(tmp_path)
    assert get_secret("BRIAN_TEST_REAL") == "here"


def test_dotenv_walk_finds_parent_env(tmp_path, monkeypatch, clean_env):
    (tmp_path / ".env").write_text("BRIAN_TEST_PARENT=found\n")
    child = tmp_path / "deep" / "nested"
    child.mkdir(parents=True)
    monkeypatch.chdir(child)
    assert get_secret("BRIAN_TEST_PARENT") == "found"


def test_dotenv_first_match_wins(tmp_path, monkeypatch, clean_env):
    """Closer .env shadows ancestors (POSIX dotenv semantics)."""
    (tmp_path / ".env").write_text("BRIAN_TEST_SHADOW=parent\n")
    child = tmp_path / "deep"
    child.mkdir()
    (child / ".env").write_text("BRIAN_TEST_SHADOW=child\n")
    monkeypatch.chdir(child)
    assert get_secret("BRIAN_TEST_SHADOW") == "child"


def test_dotenv_missing_key_does_not_continue_walking(tmp_path, monkeypatch, clean_env):
    """If the first .env doesn't have the key, we DO fall through (the
    walk stops at the first .env file but get_secret continues with the
    NEXT provider — which is the custom chain after dotenv)."""
    (tmp_path / ".env").write_text("OTHER_KEY=x\n")
    monkeypatch.chdir(tmp_path)
    # Key isn't anywhere → None
    assert get_secret("BRIAN_TEST_NOT_THERE") is None


# ── SH-12/SH-13: bootstrap_secrets ─────────────────────────────────


def test_bootstrap_resolves_multiple(clean_env, capsys):
    clean_env.setenv("BRIAN_TEST_A", "a-val")
    clean_env.setenv("BRIAN_TEST_B", "b-val")
    out = bootstrap_secrets(["BRIAN_TEST_A", "BRIAN_TEST_B"], verbose=False)
    assert out == {"BRIAN_TEST_A": "a-val", "BRIAN_TEST_B": "b-val"}


def test_bootstrap_reports_missing(clean_env, capsys):
    clean_env.setenv("BRIAN_TEST_A", "a-val")
    out = bootstrap_secrets(
        ["BRIAN_TEST_A", "BRIAN_TEST_MISSING"], verbose=True
    )
    captured = capsys.readouterr().out
    assert "BRIAN_TEST_A: set" in captured
    assert "BRIAN_TEST_MISSING: missing" in captured
    assert out["BRIAN_TEST_MISSING"] is None


def test_bootstrap_required_raises(clean_env):
    with pytest.raises(RuntimeError, match="BRIAN_TEST_REQUIRED"):
        bootstrap_secrets(
            ["BRIAN_TEST_REQUIRED"],
            required=["BRIAN_TEST_REQUIRED"],
            verbose=False,
        )


def test_bootstrap_aliases_routed_per_name(clean_env):
    clean_env.setenv("GITHUB", "from-alias")
    out = bootstrap_secrets(
        ["GH_TOKEN"],
        aliases={"GH_TOKEN": ["GITHUB_TOKEN", "GITHUB", "GITHUB_PAT"]},
        verbose=False,
    )
    assert out["GH_TOKEN"] == "from-alias"
    assert os.environ.get("GH_TOKEN") == "from-alias"


# ── SH-14: detect_environment ──────────────────────────────────────


def test_detect_environment_returns_known_label(monkeypatch):
    # Clear all known markers so we don't depend on the test runner
    for k in ("COLAB_GPU", "COLAB_RELEASE_TAG",
              "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE"):
        monkeypatch.delenv(k, raising=False)
    env = detect_environment()
    assert env in {"colab", "kaggle", "jupyter", "ipython", "script"}


def test_detect_environment_colab(monkeypatch):
    monkeypatch.setenv("COLAB_GPU", "1")
    assert detect_environment() == "colab"


def test_detect_environment_kaggle(monkeypatch):
    # Make sure colab isn't also set in the test env
    monkeypatch.delenv("COLAB_GPU", raising=False)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    assert detect_environment() == "kaggle"


# ── SH-15: optional SDKs degrade silently ──────────────────────────


def test_colab_provider_missing_sdk_returns_none(monkeypatch):
    """When google.colab isn't importable, the Colab provider returns
    None rather than raising. Regression guard for the original
    ``from google.colab import userdata`` try/except chain."""
    # Force the import to fail by removing any installed shim
    sys.modules.pop("google.colab", None)
    sys.modules.pop("google.colab.userdata", None)
    # Block future imports
    monkeypatch.setitem(sys.modules, "google.colab", None)
    assert sec._provider_colab("WHATEVER") is None


def test_kaggle_provider_missing_sdk_returns_none(monkeypatch):
    sys.modules.pop("kaggle_secrets", None)
    monkeypatch.setitem(sys.modules, "kaggle_secrets", None)
    assert sec._provider_kaggle("WHATEVER") is None


# ── Built-in provider inventory sanity check ───────────────────────


def test_default_providers_present():
    names = {n for _, n in list_secret_providers()}
    assert {"env", "colab", "kaggle", "dotenv"}.issubset(names)


def test_default_providers_priority_order():
    pairs = list_secret_providers()
    # env (10) before colab (20) before kaggle (30) before dotenv (40)
    by_name = {n: p for p, n in pairs}
    assert by_name["env"] < by_name["colab"] < by_name["kaggle"] < by_name["dotenv"]
