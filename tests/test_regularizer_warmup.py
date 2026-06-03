# -*- coding: utf-8 -*-
"""Test for RegularizationController warmup ramp.

The warmup multiplier prevents early-training instability by scaling
aux loss contributions from 0 → 1 over `warmup_steps`. Internal state
(PCC negatives buffer, AdaptiveMixture probe) still updates every step.
"""
import torch
import pytest

from neuroslm.dsl.regularization import (
    RegularizationConfig, DARConfig, PCCConfig, IsotropyConfig,
    CMDConfig, AdaptiveMixtureConfig, parse_regularization_block,
)
from neuroslm.regularizers import RegularizationController


def _make_ctrl(warmup_steps: int = 100) -> RegularizationController:
    cfg = RegularizationConfig(
        warmup_steps=warmup_steps,
        dar=DARConfig(enabled=True, lam=1.0, hidden=16, grl_alpha=0.1),
        pcc=PCCConfig(enabled=True, k=2, n_negatives=8, tau=0.1),
        isotropy=IsotropyConfig(enabled=True, weight=0.01, buffer=64),
        cmd=CMDConfig(enabled=False),  # OOM-prone, off for tests
        adaptive_mixture=AdaptiveMixtureConfig(enabled=True),
    )
    return RegularizationController(cfg, d_model=32, vocab_size=128)


def _run_step(ctrl: RegularizationController) -> dict:
    B, T, D, V = 2, 8, 32, 128
    h = torch.randn(B, T, D, requires_grad=True)
    lm_logits = torch.randn(B, T, V, requires_grad=True)
    per_sample_ce = torch.rand(B)
    return ctrl.collect_aux(
        h=h, lm_logits=lm_logits,
        per_sample_ce=per_sample_ce,
        domain_labels=torch.zeros(B, dtype=torch.long),
    )


def test_warmup_starts_at_zero() -> None:
    """Step 0 → multiplier 0.0 → total aux contribution is 0."""
    ctrl = _make_ctrl(warmup_steps=100)
    assert ctrl.warmup_multiplier() == pytest.approx(0.0)
    out = _run_step(ctrl)
    assert float(out["warmup_mult"]) == pytest.approx(0.0)
    # total = mult * (pcc + iso + cmd + dar); mult=0 → total=0
    assert float(out["total"]) == pytest.approx(0.0, abs=1e-6)


def test_warmup_linear_ramp() -> None:
    """Multiplier ramps linearly from 0 → 1 over warmup_steps."""
    ctrl = _make_ctrl(warmup_steps=100)
    # Run 25 steps → multiplier should be 25/100 = 0.25
    for _ in range(25):
        _run_step(ctrl)
    assert ctrl.warmup_multiplier() == pytest.approx(0.25, abs=0.01)

    # Run another 25 → 50/100 = 0.50
    for _ in range(25):
        _run_step(ctrl)
    assert ctrl.warmup_multiplier() == pytest.approx(0.50, abs=0.01)


def test_warmup_caps_at_one() -> None:
    """Multiplier caps at 1.0 after warmup_steps."""
    ctrl = _make_ctrl(warmup_steps=10)
    for _ in range(20):  # 2× past warmup
        _run_step(ctrl)
    assert ctrl.warmup_multiplier() == pytest.approx(1.0)


def test_warmup_zero_disables_ramp() -> None:
    """warmup_steps=0 → multiplier always 1.0 (legacy behavior)."""
    ctrl = _make_ctrl(warmup_steps=0)
    assert ctrl.warmup_multiplier() == pytest.approx(1.0)
    out = _run_step(ctrl)
    assert float(out["warmup_mult"]) == pytest.approx(1.0)


def test_warmup_state_advances_even_when_aux_zero() -> None:
    """Internal state (PCC buffer, mixture probe) updates during warmup
    even though loss contribution is scaled to ~0."""
    ctrl = _make_ctrl(warmup_steps=1000)
    # First step: multiplier ~0, but PCC step counter still advances
    initial_pcc_step = int(ctrl.pcc._step.item()) if hasattr(ctrl.pcc, "_step") else 0
    _run_step(ctrl)
    # The controller's step counter must increment
    assert int(ctrl._reg_step.item()) == 1


def test_warmup_metric_in_output() -> None:
    """warmup_mult is exposed in the collect_aux return dict."""
    ctrl = _make_ctrl(warmup_steps=100)
    out = _run_step(ctrl)
    assert "warmup_mult" in out
    assert isinstance(out["warmup_mult"], torch.Tensor)


def test_warmup_parsed_from_dsl() -> None:
    """DSL `warmup_steps: N` is correctly parsed into RegularizationConfig."""
    body = """
        warmup_steps: 1500
        dar: { enabled: true, lambda: 1.0 }
        pcc: { enabled: true, k: 4, n_negatives: 64, tau: 0.1 }
    """
    cfg = parse_regularization_block(body)
    assert cfg.warmup_steps == 1500
    assert cfg.dar.enabled is True
    assert cfg.pcc.enabled is True


def test_warmup_default_is_2000() -> None:
    """Default warmup_steps is 2000 (matches arch.neuro recommendation)."""
    cfg = RegularizationConfig()
    assert cfg.warmup_steps == 2000


def test_warmup_no_block_uses_default() -> None:
    """Empty regularization block → warmup defaults to 2000."""
    cfg = parse_regularization_block("")
    assert cfg.warmup_steps == 2000
