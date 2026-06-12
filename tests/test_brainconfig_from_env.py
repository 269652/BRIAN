"""BrainConfig.from_env — env-var overlay for the four ablation flags.

Spec
----
Adds a classmethod that takes a base BrainConfig and returns a NEW
BrainConfig with any ``BRIAN_*`` env vars overlaid. Used by
``neuroslm/train.py`` so ablation runs can flip flags from the shell
without editing CLI parsing in every script.

Env-var schema (all optional; absence = no change):

    BRIAN_USE_TDW                   bool   → cfg.use_tdw
    BRIAN_USE_DIFF_ATTN             bool   → cfg.use_diff_attn
    BRIAN_USE_TONNETZ_PRIOR         bool   → cfg.use_tonnetz_prior
    BRIAN_USE_EXPERT_ENSEMBLE       bool   → cfg.use_expert_ensemble
    BRIAN_TONNETZ_GAP_THRESHOLD     float  → cfg.tonnetz_gap_threshold
    BRIAN_W_TONNETZ                 float  → cfg.w_tonnetz

Bool parsing: '1', 'true', 'yes', 'on' → True (case-insensitive)
              '0', 'false', 'no', 'off', '' → False
              anything else → ValueError (no silent fallthrough)

Per CLAUDE.md §10 the function never mutates the input cfg; it
returns a copy with the overlays applied, so the original preset
object is reusable.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

from neuroslm.config import BrainConfig, tiny


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

_ALL_ENV_KEYS = (
    "BRIAN_USE_TDW",
    "BRIAN_USE_DIFF_ATTN",
    "BRIAN_USE_TONNETZ_PRIOR",
    "BRIAN_USE_EXPERT_ENSEMBLE",
    "BRIAN_TONNETZ_GAP_THRESHOLD",
    "BRIAN_W_TONNETZ",
)


@contextmanager
def _env(**overrides: str):
    """Temporarily set env vars; restore prior state on exit."""
    prior = {k: os.environ.get(k) for k in _ALL_ENV_KEYS}
    try:
        # Clear all BRIAN_* keys first so leakage from prior tests can't
        # taint the run, then apply only what the test wants.
        for k in _ALL_ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in overrides.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ─────────────────────────────────────────────────────────────────────────
# Phase 1: existence + no-op contract
# ─────────────────────────────────────────────────────────────────────────

def test_from_env_classmethod_exists():
    assert hasattr(BrainConfig, "from_env")
    assert callable(BrainConfig.from_env)


def test_from_env_with_no_envvars_returns_unchanged_copy():
    """No BRIAN_* env vars set → returned cfg equals the base."""
    base = tiny()
    base.vocab_size = 256
    with _env():
        out = BrainConfig.from_env(base)
    assert out.use_tdw is base.use_tdw
    assert out.use_diff_attn is base.use_diff_attn
    assert out.use_tonnetz_prior is base.use_tonnetz_prior
    assert out.use_expert_ensemble is base.use_expert_ensemble
    assert out.tonnetz_gap_threshold == base.tonnetz_gap_threshold
    assert out.w_tonnetz == base.w_tonnetz
    assert out.vocab_size == base.vocab_size  # preset values preserved


def test_from_env_does_not_mutate_input():
    """Overlay creates a new object; the base config is untouched."""
    base = tiny()
    base_use_tdw_before = base.use_tdw
    with _env(BRIAN_USE_TDW="1"):
        out = BrainConfig.from_env(base)
    assert out.use_tdw is True
    assert base.use_tdw is base_use_tdw_before  # untouched


# ─────────────────────────────────────────────────────────────────────────
# Phase 2: bool overlays for the four ablation flags
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("env_value", ["1", "true", "True", "TRUE", "yes", "on"])
def test_brian_use_tdw_truthy_strings_set_true(env_value):
    with _env(BRIAN_USE_TDW=env_value):
        out = BrainConfig.from_env(tiny())
    assert out.use_tdw is True


@pytest.mark.parametrize("env_value", ["0", "false", "False", "no", "off", ""])
def test_brian_use_tdw_falsy_strings_set_false(env_value):
    with _env(BRIAN_USE_TDW=env_value):
        out = BrainConfig.from_env(tiny())
    assert out.use_tdw is False


def test_brian_use_diff_attn_overlay():
    with _env(BRIAN_USE_DIFF_ATTN="1"):
        out = BrainConfig.from_env(tiny())
    assert out.use_diff_attn is True


def test_brian_use_tonnetz_prior_overlay():
    with _env(BRIAN_USE_TONNETZ_PRIOR="true"):
        out = BrainConfig.from_env(tiny())
    assert out.use_tonnetz_prior is True


def test_brian_use_expert_ensemble_overlay():
    with _env(BRIAN_USE_EXPERT_ENSEMBLE="yes"):
        out = BrainConfig.from_env(tiny())
    assert out.use_expert_ensemble is True


def test_invalid_bool_string_raises_valueerror():
    """No silent fallthrough — typo'd env vars must fail loudly."""
    with _env(BRIAN_USE_TDW="maybe"):
        with pytest.raises(ValueError, match="BRIAN_USE_TDW"):
            BrainConfig.from_env(tiny())


# ─────────────────────────────────────────────────────────────────────────
# Phase 3: float overlays for tonnetz scalars
# ─────────────────────────────────────────────────────────────────────────

def test_brian_tonnetz_gap_threshold_overlay():
    with _env(BRIAN_TONNETZ_GAP_THRESHOLD="0.75"):
        out = BrainConfig.from_env(tiny())
    assert out.tonnetz_gap_threshold == pytest.approx(0.75)


def test_brian_w_tonnetz_overlay():
    with _env(BRIAN_W_TONNETZ="0.05"):
        out = BrainConfig.from_env(tiny())
    assert out.w_tonnetz == pytest.approx(0.05)


def test_invalid_float_string_raises_valueerror():
    with _env(BRIAN_TONNETZ_GAP_THRESHOLD="not_a_number"):
        with pytest.raises(ValueError, match="BRIAN_TONNETZ_GAP_THRESHOLD"):
            BrainConfig.from_env(tiny())


# ─────────────────────────────────────────────────────────────────────────
# Phase 4: combined overlay (full ablation matrix in one env block)
# ─────────────────────────────────────────────────────────────────────────

def test_all_flags_overlaid_simultaneously():
    """The full ablation matrix can be set in a single shell invocation."""
    with _env(
        BRIAN_USE_TDW="1",
        BRIAN_USE_DIFF_ATTN="1",
        BRIAN_USE_TONNETZ_PRIOR="1",
        BRIAN_USE_EXPERT_ENSEMBLE="1",
        BRIAN_TONNETZ_GAP_THRESHOLD="0.5",
        BRIAN_W_TONNETZ="0.02",
    ):
        out = BrainConfig.from_env(tiny())
    assert out.use_tdw is True
    assert out.use_diff_attn is True
    assert out.use_tonnetz_prior is True
    assert out.use_expert_ensemble is True
    assert out.tonnetz_gap_threshold == pytest.approx(0.5)
    assert out.w_tonnetz == pytest.approx(0.02)


def test_partial_overlay_leaves_other_flags_at_base():
    """Setting only one env var must not silently flip the others."""
    base = tiny()
    with _env(BRIAN_USE_TDW="1"):
        out = BrainConfig.from_env(base)
    assert out.use_tdw is True
    assert out.use_diff_attn is False
    assert out.use_tonnetz_prior is False
    assert out.use_expert_ensemble is False


# ─────────────────────────────────────────────────────────────────────────
# Phase 5: integration — train.py calls the overlay
# ─────────────────────────────────────────────────────────────────────────

def test_train_py_imports_from_env():
    """train.py must reference BrainConfig.from_env so the overlay is live."""
    import inspect
    from neuroslm import train as train_mod
    src = inspect.getsource(train_mod)
    assert "BrainConfig.from_env" in src, (
        "train.py must call BrainConfig.from_env(cfg) after building the "
        "preset so the BRIAN_* ablation flags actually reach the model.")
