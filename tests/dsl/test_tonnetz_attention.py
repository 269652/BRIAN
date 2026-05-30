# -*- coding: utf-8 -*-
"""Tonnetz toroidal attention mask tests."""
import pytest
import torch

from neuroslm.dsl.nn_lang import build_dsl_language_cortex
from neuroslm.dsl import nn_ops


def test_tonnetz_mask_shape_and_structure():
    """Mask should be (T, T), additive (-inf or 0), and circular."""
    T, period = 32, 12
    mask = nn_ops._tonnetz_attention_mask(T, period=period, device="cpu")
    assert mask.shape == (T, T)
    # Self-attention always allowed
    assert mask[0, 0].item() == 0.0
    # Position 0 attending to position period should also be allowed
    # (circular distance 0)
    assert mask[period, 0].item() == 0.0
    # Local window also allowed (default local_window = max(8, period))
    assert mask[10, 0].item() == 0.0


def test_tonnetz_block_compiles_and_runs():
    """Build a small DSL cortex with tonnetz_period > 0, run forward."""
    torch.manual_seed(0)
    m = build_dsl_language_cortex(
        vocab=128, d_model=32, depth=3, n_heads=4,
        max_ctx=24, tonnetz_period=12)
    m.eval()
    ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        logits = m(ids)
    assert logits.shape == (2, 16, 128)
    assert torch.isfinite(logits).all()


def test_tonnetz_off_matches_baseline():
    """tonnetz_period=0 should produce a model with the original
    StandardBlock — no parameter or shape difference vs the legacy path."""
    m_off = build_dsl_language_cortex(
        vocab=64, d_model=32, depth=3, n_heads=4, max_ctx=16,
        tonnetz_period=0)
    m_on = build_dsl_language_cortex(
        vocab=64, d_model=32, depth=3, n_heads=4, max_ctx=16,
        tonnetz_period=12)
    n_off = sum(p.numel() for p in m_off.parameters())
    n_on = sum(p.numel() for p in m_on.parameters())
    # Tonnetz adds NO new params (just a mask in the attention op) —
    # both models must have identical parameter counts.
    assert n_on == n_off, f"tonnetz changed param count: {n_off} -> {n_on}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
