"""Config / preset tests."""
from __future__ import annotations
import pytest
from neuroslm.config import PRESETS, BrainConfig, tiny, small, medium, large, xl, xxl


def test_all_presets_buildable():
    for name, factory in PRESETS.items():
        cfg = factory()
        assert isinstance(cfg, BrainConfig)
        # Basic invariants
        assert cfg.d_sem >= 64
        assert cfg.d_hidden >= 128
        assert cfg.lang_layers >= 1
        assert cfg.lang_heads >= 1
        assert cfg.lang_ctx >= 64
        # New Φ objective is on by default
        assert cfg.enable_phi_objective is True
        assert 0.0 <= cfg.w_phi <= 1.0


def test_phi_threshold_in_range():
    cfg = BrainConfig()
    assert 0.0 <= cfg.phi_lock_threshold <= 5.0


def test_preset_names_match():
    expected = {"tiny", "small", "medium", "large", "xl", "xxl"}
    assert set(PRESETS) == expected
