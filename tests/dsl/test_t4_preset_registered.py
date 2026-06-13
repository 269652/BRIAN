# -*- coding: utf-8 -*-
"""TDD guard: t4_2k and cheap_2k must be resolvable via preset_bridge."""
import pytest
from neuroslm.dsl.preset_bridge import dsl_lm_config_from_preset, _DSL_NATIVE_PRESETS


def test_t4_2k_in_dsl_native_presets():
    assert "t4_2k" in _DSL_NATIVE_PRESETS


def test_cheap_2k_in_dsl_native_presets():
    assert "cheap_2k" in _DSL_NATIVE_PRESETS


def test_t4_2k_lookup_does_not_raise():
    cfg = dsl_lm_config_from_preset("t4_2k")
    assert cfg is not None


def test_t4_2k_uses_fp16():
    cfg = dsl_lm_config_from_preset("t4_2k")
    assert cfg.get("mixed_precision") == "fp16", (
        "T4 is Turing sm_75 — no native bf16, must use fp16"
    )


def test_cheap_2k_uses_bf16():
    cfg = dsl_lm_config_from_preset("cheap_2k")
    assert cfg.get("mixed_precision") == "bf16"


def test_t4_2k_returns_copy():
    """Mutations must not affect the canonical dict."""
    cfg1 = dsl_lm_config_from_preset("t4_2k")
    cfg1["lr"] = 999.0
    cfg2 = dsl_lm_config_from_preset("t4_2k")
    assert cfg2["lr"] != 999.0


def test_t4_2k_dims():
    cfg = dsl_lm_config_from_preset("t4_2k")
    assert cfg["d_model"] == 384
    assert cfg["depth"] == 6
    assert cfg["n_heads"] == 6
    assert cfg["vocab"] == 50257
    assert cfg["max_ctx"] == 512
    assert cfg["batch_size"] == 4
    assert cfg["grad_accum"] == 2


def test_unknown_preset_error_lists_dsl_natives():
    with pytest.raises(KeyError) as exc:
        dsl_lm_config_from_preset("does_not_exist_xyz")
    msg = str(exc.value)
    assert "t4_2k" in msg or "cheap_2k" in msg
