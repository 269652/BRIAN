"""RED tests for GIF-7: Homeostatic Gradient Equilibrium.

Part A — Divisive Gradient Normalization (cortical gain control)
Part B — Loss-Variance Metaplastic Damping (BCM rule)
Part C — VBB KL Floor (anti-collapse guard)
"""
import math
import pytest
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════
# Part A: Divisive Gradient Normalization
# ═══════════════════════════════════════════════════════════════════

class TestDivisiveGradNorm:
    """Smooth gain-control normalization replacing hard clip."""

    def test_import(self):
        from neuroslm.emergent.gif7 import divisive_grad_normalize
        assert callable(divisive_grad_normalize)

    def test_small_gnorm_passes_through(self):
        """When ||g|| << c, scale ≈ 1 (no attenuation)."""
        from neuroslm.emergent.gif7 import divisive_grad_normalize
        model = nn.Linear(10, 10)
        # Set tiny gradients
        model.weight.grad = torch.ones_like(model.weight) * 0.001
        model.bias.grad = torch.ones_like(model.bias) * 0.001
        gnorm_before = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float('inf'))
        # Reset and apply divisive norm
        model.weight.grad = torch.ones_like(model.weight) * 0.001
        model.bias.grad = torch.ones_like(model.bias) * 0.001
        c = 1.0
        gnorm, scale = divisive_grad_normalize(model.parameters(), c)
        # Scale should be very close to 1.0
        assert scale > 0.99, f"Expected scale ≈ 1.0 for small gnorm, got {scale}"

    def test_large_gnorm_attenuated(self):
        """When ||g|| >> c, scale ≈ c/||g|| (strong suppression)."""
        from neuroslm.emergent.gif7 import divisive_grad_normalize
        model = nn.Linear(10, 10)
        # Set huge gradients
        model.weight.grad = torch.ones_like(model.weight) * 100.0
        model.bias.grad = torch.ones_like(model.bias) * 100.0
        c = 1.0
        gnorm, scale = divisive_grad_normalize(model.parameters(), c)
        # For gnorm=~331, c=1: scale ≈ 1/sqrt(1 + 331²) ≈ 0.003
        assert scale < 0.1, f"Expected strong attenuation, got scale={scale}"

    def test_at_semi_saturation_scale_is_half(self):
        """At ||g|| = c, scale = c/sqrt(2c²) = 1/√2 ≈ 0.707."""
        from neuroslm.emergent.gif7 import divisive_grad_normalize
        # We want gnorm ≈ c. Create params where gnorm will be exactly c.
        c = 5.0
        model = nn.Linear(1, 1, bias=False)
        # gnorm = sqrt(sum of grad²) = |grad_value| * sqrt(n_params)
        # For 1 param: gnorm = |grad_value|
        model.weight.grad = torch.tensor([[c]])
        gnorm, scale = divisive_grad_normalize(model.parameters(), c)
        expected_scale = c / math.sqrt(c**2 + c**2)  # = 1/√2
        assert abs(scale - expected_scale) < 0.01, \
            f"At semi-saturation, expected {expected_scale:.4f}, got {scale:.4f}"

    def test_returns_pre_norm_gnorm(self):
        """Returns the raw gradient norm before normalization."""
        from neuroslm.emergent.gif7 import divisive_grad_normalize
        model = nn.Linear(5, 5, bias=False)
        model.weight.grad = torch.ones(5, 5) * 2.0
        # gnorm should be sqrt(25 * 4) = 10.0
        gnorm, scale = divisive_grad_normalize(model.parameters(), c=1.0)
        assert abs(gnorm - 10.0) < 0.01

    def test_gradients_actually_scaled(self):
        """After call, actual gradient values are multiplied by scale."""
        from neuroslm.emergent.gif7 import divisive_grad_normalize
        model = nn.Linear(1, 1, bias=False)
        model.weight.grad = torch.tensor([[30.0]])
        c = 1.0
        gnorm, scale = divisive_grad_normalize(model.parameters(), c)
        # grad should now be 30 * scale
        actual = model.weight.grad.item()
        expected = 30.0 * scale
        assert abs(actual - expected) < 0.001

    def test_smoothness_no_discontinuity(self):
        """Unlike hard clip, the scaling function is smooth (C∞)."""
        from neuroslm.emergent.gif7 import divisive_grad_normalize
        c = 1.0
        scales = []
        for gnorm_val in [0.9, 0.95, 1.0, 1.05, 1.1]:
            model = nn.Linear(1, 1, bias=False)
            model.weight.grad = torch.tensor([[gnorm_val]])
            _, scale = divisive_grad_normalize(model.parameters(), c)
            scales.append(scale)
        # Check monotonically decreasing (no discontinuity)
        for i in range(len(scales) - 1):
            assert scales[i] > scales[i + 1], \
                f"Scale not monotonically decreasing: {scales}"

    def test_zero_grad_safe(self):
        """Handles zero gradients without NaN."""
        from neuroslm.emergent.gif7 import divisive_grad_normalize
        model = nn.Linear(5, 5)
        model.weight.grad = torch.zeros_like(model.weight)
        model.bias.grad = torch.zeros_like(model.bias)
        gnorm, scale = divisive_grad_normalize(model.parameters(), c=1.0)
        assert gnorm == 0.0
        assert scale == 1.0  # No scaling when gnorm is 0


# ═══════════════════════════════════════════════════════════════════
# Part B: Loss-Variance Metaplastic Damping (BCM Rule)
# ═══════════════════════════════════════════════════════════════════

class TestLossVarianceDamping:
    """LR modulation by inverse loss variance."""

    def test_import(self):
        from neuroslm.emergent.gif7 import LossVarianceDamper
        assert LossVarianceDamper is not None

    def test_init_with_window(self):
        from neuroslm.emergent.gif7 import LossVarianceDamper
        d = LossVarianceDamper(window=64)
        assert d.window == 64

    def test_multiplier_is_one_during_warmup(self):
        """Before window fills, multiplier should be 1.0 (no damping)."""
        from neuroslm.emergent.gif7 import LossVarianceDamper
        d = LossVarianceDamper(window=100)
        # Feed only 10 samples (< window)
        for i in range(10):
            d.update(4.0 + i * 0.1)
        assert d.lr_multiplier() == 1.0

    def test_stable_loss_multiplier_near_one(self):
        """When loss variance is low (stable), multiplier ≈ 1."""
        from neuroslm.emergent.gif7 import LossVarianceDamper
        d = LossVarianceDamper(window=32)
        # Feed very stable losses
        for i in range(64):
            d.update(4.0 + 0.001 * (i % 2))  # tiny jitter
        mult = d.lr_multiplier()
        assert mult > 0.9, f"Stable loss should give mult ≈ 1, got {mult}"

    def test_volatile_loss_reduces_multiplier(self):
        """When loss variance is high, multiplier drops below 1."""
        from neuroslm.emergent.gif7 import LossVarianceDamper
        d = LossVarianceDamper(window=32)
        # Feed stable losses first to establish reference
        for i in range(32):
            d.update(4.0)
        d.calibrate()  # Set σ_ref from current stable state
        # Now feed highly volatile losses
        for i in range(32):
            d.update(4.0 + 3.0 * ((-1) ** i))  # oscillating ±3
        mult = d.lr_multiplier()
        assert mult < 0.5, f"Volatile loss should damp LR, got mult={mult}"

    def test_multiplier_bounded_below(self):
        """Multiplier never goes below a minimum floor (e.g., 0.1)."""
        from neuroslm.emergent.gif7 import LossVarianceDamper
        d = LossVarianceDamper(window=32, min_mult=0.1)
        # Establish tiny reference
        for i in range(32):
            d.update(4.0)
        d.calibrate()
        # Extreme volatility
        for i in range(32):
            d.update(4.0 + 100.0 * ((-1) ** i))
        mult = d.lr_multiplier()
        assert mult >= 0.1, f"Multiplier breached floor: {mult}"

    def test_calibrate_sets_reference(self):
        """calibrate() captures current std as σ_ref."""
        from neuroslm.emergent.gif7 import LossVarianceDamper
        d = LossVarianceDamper(window=32)
        for i in range(32):
            d.update(4.0 + 0.1 * ((-1) ** i))  # σ ≈ 0.1
        d.calibrate()
        assert d.sigma_ref > 0
        assert abs(d.sigma_ref - 0.1) < 0.05

    def test_auto_calibrate_at_step(self):
        """Auto-calibrates at specified step (e.g., end of warmup)."""
        from neuroslm.emergent.gif7 import LossVarianceDamper
        d = LossVarianceDamper(window=32, calibrate_at=32)
        for i in range(33):
            d.update(4.0 + 0.05 * ((-1) ** i))
        # Should have auto-calibrated at step 32
        assert d.sigma_ref > 0


# ═══════════════════════════════════════════════════════════════════
# Part C: VBB KL Floor (Anti-Collapse Guard)
# ═══════════════════════════════════════════════════════════════════

class TestVBBKLFloor:
    """Prevents posterior collapse by penalizing low KL."""

    def test_import(self):
        from neuroslm.emergent.gif7 import vbb_kl_floor_loss
        assert callable(vbb_kl_floor_loss)

    def test_healthy_kl_no_penalty(self):
        """When KL > floor, penalty is zero."""
        from neuroslm.emergent.gif7 import vbb_kl_floor_loss
        kl = torch.tensor(500.0)
        loss = vbb_kl_floor_loss(kl, kl_min=100.0, gamma=0.01)
        assert loss.item() == 0.0

    def test_collapsed_kl_produces_penalty(self):
        """When KL < floor, penalty is γ * (kl_min - kl)²."""
        from neuroslm.emergent.gif7 import vbb_kl_floor_loss
        kl = torch.tensor(50.0)  # collapsed below floor of 100
        loss = vbb_kl_floor_loss(kl, kl_min=100.0, gamma=0.01)
        expected = 0.01 * (100.0 - 50.0) ** 2  # = 0.01 * 2500 = 25.0
        assert abs(loss.item() - expected) < 0.01

    def test_at_floor_no_penalty(self):
        """Exactly at floor → zero penalty."""
        from neuroslm.emergent.gif7 import vbb_kl_floor_loss
        kl = torch.tensor(100.0)
        loss = vbb_kl_floor_loss(kl, kl_min=100.0, gamma=0.01)
        assert loss.item() == 0.0

    def test_gradient_flows(self):
        """Gradient flows through the penalty to the KL source."""
        from neuroslm.emergent.gif7 import vbb_kl_floor_loss
        kl = torch.tensor(30.0, requires_grad=True)
        loss = vbb_kl_floor_loss(kl, kl_min=100.0, gamma=0.01)
        loss.backward()
        # dL/d(kl) = γ * 2 * (kl_min - kl) * (-1) = -0.01 * 2 * 70 = -1.4
        assert kl.grad is not None
        assert kl.grad.item() < 0  # pushes KL up (gradient is negative)

    def test_zero_gamma_disables(self):
        """gamma=0 means no floor enforcement."""
        from neuroslm.emergent.gif7 import vbb_kl_floor_loss
        kl = torch.tensor(1.0)  # extremely collapsed
        loss = vbb_kl_floor_loss(kl, kl_min=100.0, gamma=0.0)
        assert loss.item() == 0.0

    def test_quadratic_shape(self):
        """Penalty is quadratic — steeper the further below floor."""
        from neuroslm.emergent.gif7 import vbb_kl_floor_loss
        loss_90 = vbb_kl_floor_loss(torch.tensor(90.0), kl_min=100.0, gamma=0.01)
        loss_50 = vbb_kl_floor_loss(torch.tensor(50.0), kl_min=100.0, gamma=0.01)
        loss_10 = vbb_kl_floor_loss(torch.tensor(10.0), kl_min=100.0, gamma=0.01)
        # Quadratic: penalty at 50 should be 25x penalty at 90
        assert loss_50.item() > loss_90.item()
        assert loss_10.item() > loss_50.item()
        # ratio: (50)² / (10)² = 2500 / 100 = 25
        ratio = loss_50.item() / max(loss_90.item(), 1e-9)
        assert ratio > 20  # (50²=2500) / (10²=100) = 25


# ═══════════════════════════════════════════════════════════════════
# Integration: Config parsing
# ═══════════════════════════════════════════════════════════════════

class TestGIF7Config:
    """TrainingConfig fields for GIF-7."""

    def test_divisive_grad_norm_field_exists(self):
        from neuroslm.dsl.training_config import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, 'divisive_grad_c')
        assert cfg.divisive_grad_c == 0.0  # 0 = disabled (use hard clip)

    def test_loss_variance_window_field(self):
        from neuroslm.dsl.training_config import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, 'loss_var_window')
        assert cfg.loss_var_window == 0  # 0 = disabled

    def test_loss_variance_min_mult_field(self):
        from neuroslm.dsl.training_config import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, 'loss_var_min_mult')
        assert cfg.loss_var_min_mult == 0.1

    def test_vbb_kl_floor_field(self):
        from neuroslm.dsl.training_config import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, 'vbb_kl_floor')
        assert cfg.vbb_kl_floor == 0.0  # 0 = disabled

    def test_vbb_kl_floor_gamma_field(self):
        from neuroslm.dsl.training_config import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, 'vbb_kl_floor_gamma')
        assert cfg.vbb_kl_floor_gamma == 0.01

    def test_parse_divisive_grad_c(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config("divisive_grad_c: 5.0")
        assert cfg.divisive_grad_c == 5.0

    def test_parse_loss_var_window(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config("loss_var_window: 64")
        assert cfg.loss_var_window == 64

    def test_parse_vbb_kl_floor(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config("vbb_kl_floor: 1000.0")
        assert cfg.vbb_kl_floor == 1000.0


# ═══════════════════════════════════════════════════════════════════
# DSL equation tests
# ═══════════════════════════════════════════════════════════════════

class TestGIF7DSLEquations:
    """DSL equations exist in lib/gif.neuro."""

    def test_divisive_grad_equation_in_lib(self):
        from pathlib import Path
        lib = Path(__file__).parents[2] / "lib" / "gif.neuro"
        content = lib.read_text()
        assert "gif_divisive_grad_norm" in content

    def test_loss_variance_damping_equation_in_lib(self):
        from pathlib import Path
        lib = Path(__file__).parents[2] / "lib" / "gif.neuro"
        content = lib.read_text()
        assert "gif_loss_variance_damping" in content

    def test_vbb_kl_floor_equation_in_lib(self):
        from pathlib import Path
        lib = Path(__file__).parents[2] / "lib" / "gif.neuro"
        content = lib.read_text()
        assert "gif_vbb_kl_floor" in content
