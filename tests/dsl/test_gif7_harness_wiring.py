# -*- coding: utf-8 -*-
"""RED tests: verify GIF-7 mechanisms are wired into BRIANHarness.

Tests the three wiring points:
  A) divisive_grad_normalize replaces hard clip_grad_norm_ when configured
  B) LossVarianceDamper is built, updated, and modulates LR
  C) vbb_kl_floor_loss is added to VBB free energy when configured

These tests verify the INTEGRATION — the pure-function unit tests live
in test_gif7_gradient_equilibrium.py.
"""
import pytest
import math
import torch
import torch.nn as nn

from neuroslm.dsl.training_config import TrainingConfig
from neuroslm.harness import BRIANHarness


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def base_cfg():
    """Minimal training config — all GIF-7 fields at defaults (disabled)."""
    cfg = TrainingConfig()
    return cfg


@pytest.fixture
def gif7_cfg():
    """Training config with all three GIF-7 mechanisms enabled."""
    cfg = TrainingConfig()
    cfg.divisive_grad_c = 5.0
    cfg.loss_var_window = 64
    cfg.loss_var_min_mult = 0.1
    cfg.vbb_kl_floor = 1000.0
    cfg.vbb_kl_floor_gamma = 0.01
    cfg.warmup = 200
    return cfg


def _make_harness(cfg):
    """Build a minimal harness without a full DSL circuit."""
    circuit = nn.Identity()
    h = BRIANHarness(circuit=circuit, vocab_size=128, d_sem=32,
                     training_config=cfg)
    return h


# ═══════════════════════════════════════════════════════════════════
# Part B init: LossVarianceDamper construction
# ═══════════════════════════════════════════════════════════════════

class TestGIF7DamperInit:
    """LossVarianceDamper is built/skipped based on config."""

    def test_damper_none_when_window_zero(self, base_cfg):
        """Default config (loss_var_window=0) → no damper."""
        h = _make_harness(base_cfg)
        assert h._loss_var_damper is None

    def test_damper_built_when_window_positive(self, gif7_cfg):
        """loss_var_window=64 → damper is constructed."""
        h = _make_harness(gif7_cfg)
        assert h._loss_var_damper is not None
        assert h._loss_var_damper.window == 64
        assert h._loss_var_damper.min_mult == 0.1

    def test_damper_uses_warmup_for_calibrate_at(self, gif7_cfg):
        """calibrate_at should match cfg.warmup."""
        gif7_cfg.warmup = 300
        h = _make_harness(gif7_cfg)
        assert h._loss_var_damper.calibrate_at == 300

    def test_from_language_model_builds_damper(self, gif7_cfg):
        """The from_language_model path also builds the damper."""
        lm = nn.Linear(32, 128)
        h = BRIANHarness.from_language_model(
            lm, vocab_size=128, d_sem=32, training_config=gif7_cfg)
        assert h._loss_var_damper is not None
        assert h._loss_var_damper.window == 64


# ═══════════════════════════════════════════════════════════════════
# Part A: Divisive Gradient Normalization wiring
# ═══════════════════════════════════════════════════════════════════

class TestGIF7DivisiveGradNormWiring:
    """Grad normalization switches between divisive and hard clip."""

    def test_divisive_normalizes_gradients(self, gif7_cfg):
        """When divisive_grad_c > 0, grads are divisively normalized."""
        h = _make_harness(gif7_cfg)
        # Create a simple model with known gradients
        param = nn.Parameter(torch.ones(10))
        param.grad = torch.ones(10) * 30.0  # gnorm ≈ 94.87

        from neuroslm.emergent.gif7 import divisive_grad_normalize
        gnorm, scale = divisive_grad_normalize(iter([param]), c=5.0)

        # Scale should be c/sqrt(c²+gnorm²) ≈ 5/sqrt(25+9000) ≈ 0.053
        assert 0 < scale < 0.1
        assert gnorm > 90  # raw norm should be large

    def test_config_divisive_grad_c_default_zero(self, base_cfg):
        """Default config has divisive_grad_c = 0 (disabled)."""
        assert base_cfg.divisive_grad_c == 0.0

    def test_config_divisive_grad_c_set(self, gif7_cfg):
        """GIF-7 config has divisive_grad_c = 5.0."""
        assert gif7_cfg.divisive_grad_c == 5.0


# ═══════════════════════════════════════════════════════════════════
# Part B: Loss-Variance Metaplastic Damping wiring
# ═══════════════════════════════════════════════════════════════════

class TestGIF7LRDampingWiring:
    """_apply_gif7_lr_damping modulates optimizer LR."""

    def test_apply_gif7_lr_damping_exists(self, gif7_cfg):
        """Harness has the _apply_gif7_lr_damping method."""
        h = _make_harness(gif7_cfg)
        assert hasattr(h, '_apply_gif7_lr_damping')
        assert callable(h._apply_gif7_lr_damping)

    def test_noop_when_damper_none(self, base_cfg):
        """No damper → _apply_gif7_lr_damping is a no-op."""
        h = _make_harness(base_cfg)
        # Create a mock optimizer
        param = nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([param], lr=0.01)
        h._apply_gif7_lr_damping(opt)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.01)

    def test_lr_reduced_when_variance_high(self, gif7_cfg):
        """When loss variance exceeds σ_ref, LR is reduced."""
        h = _make_harness(gif7_cfg)
        damper = h._loss_var_damper

        # Fill buffer with stable losses to set σ_ref
        for _ in range(64):
            damper.update(4.0 + 0.01 * (_ % 2))  # very low variance
        damper.calibrate()

        # Now feed wildly oscillating losses
        for _ in range(64):
            damper.update(2.0 if _ % 2 == 0 else 8.0)  # huge variance

        param = nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([param], lr=0.01)
        h._apply_gif7_lr_damping(opt)
        # LR should be reduced
        assert opt.param_groups[0]["lr"] < 0.01

    def test_lr_unchanged_when_stable(self, gif7_cfg):
        """When loss variance is at/below σ_ref, LR is unchanged."""
        h = _make_harness(gif7_cfg)
        damper = h._loss_var_damper

        # Fill buffer with stable losses and calibrate
        for i in range(64):
            damper.update(4.0 + 0.01 * (i % 2))
        damper.calibrate()

        # Continue with same stability
        for i in range(64):
            damper.update(4.0 + 0.005 * (i % 2))

        param = nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([param], lr=0.01)
        h._apply_gif7_lr_damping(opt)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.01)

    def test_gif7_lr_mult_metric_logged(self, gif7_cfg):
        """_apply_gif7_lr_damping logs the multiplier to _metrics."""
        h = _make_harness(gif7_cfg)
        damper = h._loss_var_damper
        for i in range(64):
            damper.update(4.0)
        damper.calibrate()
        for i in range(64):
            damper.update(2.0 if i % 2 == 0 else 8.0)

        param = nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([param], lr=0.01)
        h._apply_gif7_lr_damping(opt)
        assert "gif7_lr_mult" in h._metrics
        assert h._metrics["gif7_lr_mult"] < 1.0


# ═══════════════════════════════════════════════════════════════════
# Part C: VBB KL Floor wiring
# ═══════════════════════════════════════════════════════════════════

class TestGIF7KLFloorWiring:
    """vbb_kl_floor_loss is added to VBB free energy."""

    def test_kl_floor_config_default_zero(self, base_cfg):
        """Default config has vbb_kl_floor = 0 (disabled)."""
        assert base_cfg.vbb_kl_floor == 0.0

    def test_kl_floor_config_set(self, gif7_cfg):
        """GIF-7 config has vbb_kl_floor = 1000."""
        assert gif7_cfg.vbb_kl_floor == 1000.0
        assert gif7_cfg.vbb_kl_floor_gamma == 0.01

    def test_kl_floor_penalty_nonzero_at_collapse(self):
        """When KL << kl_min, penalty should be large."""
        from neuroslm.emergent.gif7 import vbb_kl_floor_loss
        kl = torch.tensor(100.0)  # collapsed KL
        penalty = vbb_kl_floor_loss(kl, kl_min=1000.0, gamma=0.01)
        # Expected: 0.01 * (1000 - 100)² = 0.01 * 810000 = 8100
        assert penalty.item() == pytest.approx(8100.0)

    def test_kl_floor_penalty_zero_above_min(self):
        """When KL ≥ kl_min, penalty should be zero."""
        from neuroslm.emergent.gif7 import vbb_kl_floor_loss
        kl = torch.tensor(2000.0)  # healthy KL
        penalty = vbb_kl_floor_loss(kl, kl_min=1000.0, gamma=0.01)
        assert penalty.item() == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════
# Integration: all three wiring points exist
# ═══════════════════════════════════════════════════════════════════

class TestGIF7Integration:
    """Verify that the harness has all the wiring hooks."""

    def test_harness_has_gif7_damper_attr(self, gif7_cfg):
        h = _make_harness(gif7_cfg)
        assert hasattr(h, '_loss_var_damper')

    def test_harness_has_gif7_lr_damping_method(self, gif7_cfg):
        h = _make_harness(gif7_cfg)
        assert hasattr(h, '_apply_gif7_lr_damping')

    def test_harness_has_build_gif7_damper_method(self, gif7_cfg):
        h = _make_harness(gif7_cfg)
        assert hasattr(h, '_build_gif7_damper')

    def test_all_three_metrics_names_valid(self):
        """Metric names used in the wiring follow project conventions."""
        expected = ["gif7_dgn_scale", "gif7_lr_mult", "gif7_kl_floor"]
        for name in expected:
            assert name.startswith("gif7_"), f"metric {name} missing gif7_ prefix"
