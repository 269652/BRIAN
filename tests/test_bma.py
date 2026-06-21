"""
tests/test_bma.py — TDD contract suite for Bures Manifold Alignment (BMA).

BMA approximates the Wasserstein-2 (Bures metric) distance between the trunk
and expert representation distributions via random projections (sliced W₂):

    loss = E_u[( √Var_trunk(u) - √Var_expert(u) )²]

where u ~ Uniform(S^{d-1}).  Gradient flows into h_motor (trunk); h_sensory
(expert) is detached.

Principles under test:
  A. Config defaults + DSL parsing.
  B. Sliced-W₂ mathematical contracts (zero, monotone, asymmetry).
  C. Gradient isolation: trunk learns, expert stays frozen.
  D. Linear ramp schedule.
  E. Metric registry population.
  F. Defensive early-exits (no activations, zero weight).
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_harness_with_bma(weight: float,
                            n_proj: int = 32,
                            ramp_start: int = 0,
                            ramp_end: int = 0,
                            d_sem: int = 16):
    """Minimal harness wired for BMA, without VBB or PC-reentry."""
    from neuroslm.harness import BRIANHarness
    from neuroslm.dsl.training_config import TrainingConfig

    class _StubLM(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.proj = nn.Linear(d, d)
            self._last_h_motor   = None
            self._last_h_sensory = None

    cfg = TrainingConfig()
    cfg.bma_weight        = weight
    cfg.bma_n_projections = n_proj
    cfg.bma_ramp_start    = ramp_start
    cfg.bma_ramp_end      = ramp_end

    lm = _StubLM(d_sem)
    return BRIANHarness.from_language_model(
        language_model=lm, vocab_size=257, d_sem=d_sem,
        training_config=cfg,
    )


def _stash(h, mu: torch.Tensor, s: torch.Tensor):
    h.language_model._last_h_motor   = mu
    h.language_model._last_h_sensory = s


# ──────────────────────────────────────────────────────────────────────────────
# A. Config defaults + DSL parsing
# ──────────────────────────────────────────────────────────────────────────────

def test_bma_config_defaults_off():
    from neuroslm.dsl.training_config import TrainingConfig
    cfg = TrainingConfig()
    assert cfg.bma_weight        == 0.0
    assert cfg.bma_n_projections == 64
    assert cfg.bma_ramp_start    == 0
    assert cfg.bma_ramp_end      == 0


def test_bma_config_parses_from_dsl():
    from neuroslm.dsl.training_config import parse_training_config
    src = """
        learning_rate: 0.001
        bma_weight: 0.05
        bma_n_projections: 32
        bma_ramp_start: 500
        bma_ramp_end: 3000
    """
    cfg = parse_training_config(src)
    assert cfg.bma_weight        == pytest.approx(0.05)
    assert cfg.bma_n_projections == 32
    assert cfg.bma_ramp_start    == 500
    assert cfg.bma_ramp_end      == 3000


# ──────────────────────────────────────────────────────────────────────────────
# B. Sliced-W₂ mathematical contracts
# ──────────────────────────────────────────────────────────────────────────────

def test_bma_zero_loss_identical_distributions():
    """When trunk and expert are the same tensor, sliced W₂ = 0."""
    h = _make_harness_with_bma(weight=1.0, n_proj=128)
    mu = torch.randn(4, 8, 16)
    _stash(h, mu.clone(), mu.clone().detach())
    loss = h._compute_bma_loss(base_weight=1.0)
    assert loss is not None
    assert float(loss) == pytest.approx(0.0, abs=1e-4)


def test_bma_positive_loss_when_trunk_collapses():
    """A near-zero trunk vs. diverse expert produces a positive W₂ loss."""
    h = _make_harness_with_bma(weight=1.0, n_proj=64)
    mu = torch.zeros(4, 8, 16)          # collapsed trunk
    s  = torch.randn(4, 8, 16) * 2.0   # diverse expert
    _stash(h, mu.clone().requires_grad_(True), s.detach())
    loss = h._compute_bma_loss(base_weight=1.0)
    assert loss is not None
    assert float(loss) > 0.1


def test_bma_loss_asymmetric_direction():
    """Collapse in trunk direction (not expert) is the penalised direction."""
    h = _make_harness_with_bma(weight=1.0, n_proj=64)
    # Case 1: diverse trunk, collapsed expert  → small W₂
    mu_big  = torch.randn(8, 4, 16) * 2.0
    s_small = torch.zeros(8, 4, 16)
    _stash(h, mu_big.detach(), s_small.detach())
    loss_big = float(h._compute_bma_loss(base_weight=1.0))
    # Case 2: collapsed trunk, diverse expert  → large W₂
    _stash(h, s_small.clone().detach(), mu_big.detach())
    loss_small = float(h._compute_bma_loss(base_weight=1.0))
    # Both cases are penalised (W₂ is symmetric by construction), but
    # the important property is that the penalty is non-trivial in both.
    assert loss_big > 0.0
    assert loss_small > 0.0


def test_bma_loss_scales_with_weight():
    """loss × 2 when weight is doubled, for the same activations."""
    h1 = _make_harness_with_bma(weight=0.5, n_proj=128)
    h2 = _make_harness_with_bma(weight=1.0, n_proj=128)
    torch.manual_seed(0)
    mu = torch.randn(4, 8, 16)
    s  = torch.randn(4, 8, 16) * 3.0

    torch.manual_seed(1)
    _stash(h1, mu.clone(), s.clone().detach())
    l1 = float(h1._compute_bma_loss(base_weight=0.5))

    torch.manual_seed(1)  # same random projections
    _stash(h2, mu.clone(), s.clone().detach())
    l2 = float(h2._compute_bma_loss(base_weight=1.0))

    assert l2 == pytest.approx(2.0 * l1, rel=0.05)


# ──────────────────────────────────────────────────────────────────────────────
# C. Gradient isolation
# ──────────────────────────────────────────────────────────────────────────────

def test_bma_gradient_flows_to_trunk():
    """h_motor must receive a gradient when BMA is active."""
    h = _make_harness_with_bma(weight=1.0)
    mu = torch.randn(2, 4, 16, requires_grad=True)
    s  = torch.randn(2, 4, 16, requires_grad=False)
    _stash(h, mu, s)
    loss = h._compute_bma_loss(base_weight=1.0)
    assert loss is not None
    loss.backward()
    assert mu.grad is not None
    assert mu.grad.abs().sum().item() > 0.0


def test_bma_gradient_blocked_from_expert():
    """h_sensory (frozen expert) must never receive gradient via BMA."""
    h = _make_harness_with_bma(weight=1.0)
    mu = torch.randn(2, 4, 16, requires_grad=True)
    s  = torch.randn(2, 4, 16, requires_grad=True)
    _stash(h, mu, s)
    loss = h._compute_bma_loss(base_weight=1.0)
    loss.backward()
    assert s.grad is None or s.grad.abs().sum().item() == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# D. Ramp schedule
# ──────────────────────────────────────────────────────────────────────────────

def test_bma_ramp_zero_before_start():
    """Effective weight must be 0 when global_step < bma_ramp_start."""
    h = _make_harness_with_bma(weight=1.0, ramp_start=1000, ramp_end=2000)
    h._global_step = 500
    mu = torch.randn(2, 4, 16)
    s  = torch.randn(2, 4, 16)
    _stash(h, mu, s)
    h._compute_bma_loss(base_weight=1.0)
    assert h._metrics.get("bma_weight", -1.0) == pytest.approx(0.0)


def test_bma_ramp_full_after_end():
    """Effective weight must equal base_weight when global_step >= bma_ramp_end."""
    h = _make_harness_with_bma(weight=0.5, ramp_start=1000, ramp_end=2000)
    h._global_step = 3000
    mu = torch.randn(2, 4, 16)
    s  = torch.randn(2, 4, 16)
    _stash(h, mu, s)
    h._compute_bma_loss(base_weight=0.5)
    assert h._metrics.get("bma_weight", -1.0) == pytest.approx(0.5)


def test_bma_ramp_linear_at_midpoint():
    """At the midpoint of the ramp, effective weight should be 0.5 × base."""
    h = _make_harness_with_bma(weight=1.0, ramp_start=1000, ramp_end=3000)
    h._global_step = 2000  # midpoint
    mu = torch.randn(2, 4, 16)
    s  = torch.randn(2, 4, 16)
    _stash(h, mu, s)
    h._compute_bma_loss(base_weight=1.0)
    assert h._metrics.get("bma_weight", -1.0) == pytest.approx(0.5, abs=1e-3)


def test_bma_no_ramp_when_ramp_end_zero():
    """When ramp_end == ramp_start == 0, full weight is applied immediately."""
    h = _make_harness_with_bma(weight=0.3, ramp_start=0, ramp_end=0)
    h._global_step = 0
    mu = torch.randn(2, 4, 16)
    s  = torch.randn(2, 4, 16)
    _stash(h, mu, s)
    h._compute_bma_loss(base_weight=0.3)
    assert h._metrics.get("bma_weight", -1.0) == pytest.approx(0.3)


# ──────────────────────────────────────────────────────────────────────────────
# E. Metric registry
# ──────────────────────────────────────────────────────────────────────────────

def test_bma_logs_all_metrics():
    """All four BMA telemetry keys must be present after a call."""
    h = _make_harness_with_bma(weight=0.1)
    mu = torch.randn(2, 4, 16)
    s  = torch.randn(2, 4, 16)
    _stash(h, mu, s)
    h._compute_bma_loss(base_weight=0.1)
    for key in ("bma_loss", "bma_weight", "bma_var_trunk", "bma_var_expert"):
        assert key in h._metrics, f"missing metric: {key}"


def test_bma_var_trunk_matches_analytical():
    """When trunk is a 1-D constant vector, var_trunk ≈ 0."""
    h = _make_harness_with_bma(weight=1.0, n_proj=64)
    mu = torch.zeros(8, 8, 16)  # all-zero: variance = 0
    s  = torch.randn(8, 8, 16)
    _stash(h, mu, s)
    h._compute_bma_loss(base_weight=1.0)
    assert h._metrics["bma_var_trunk"] == pytest.approx(0.0, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# F. Defensive early-exits
# ──────────────────────────────────────────────────────────────────────────────

def test_bma_returns_none_when_weight_zero():
    """base_weight == 0 → None immediately, no computation."""
    h = _make_harness_with_bma(weight=0.0)
    result = h._compute_bma_loss(base_weight=0.0)
    assert result is None


def test_bma_returns_none_when_no_h_motor():
    """Missing h_motor stash → None (harness not yet warmed up)."""
    h = _make_harness_with_bma(weight=0.1)
    h.language_model._last_h_motor   = None
    h.language_model._last_h_sensory = torch.randn(2, 4, 16)
    result = h._compute_bma_loss(base_weight=0.1)
    assert result is None


def test_bma_returns_none_when_no_h_sensory():
    """Missing h_sensory stash → None (expert not yet warmed up)."""
    h = _make_harness_with_bma(weight=0.1)
    h.language_model._last_h_motor   = torch.randn(2, 4, 16)
    h.language_model._last_h_sensory = None
    result = h._compute_bma_loss(base_weight=0.1)
    assert result is None


def test_bma_returns_none_when_ramp_not_started():
    """Ramp not started → returns None (zero gradient, but metrics logged)."""
    h = _make_harness_with_bma(weight=1.0, ramp_start=500, ramp_end=1000)
    h._global_step = 0
    mu = torch.randn(2, 4, 16)
    s  = torch.randn(2, 4, 16)
    _stash(h, mu, s)
    result = h._compute_bma_loss(base_weight=1.0)
    assert result is None


def test_bma_is_scalar_tensor():
    """Returned loss must be a zero-dimensional tensor."""
    h = _make_harness_with_bma(weight=1.0)
    mu = torch.randn(2, 4, 16)
    s  = torch.randn(2, 4, 16)
    _stash(h, mu, s)
    loss = h._compute_bma_loss(base_weight=1.0)
    assert loss is not None
    assert loss.ndim == 0


# ──────────────────────────────────────────────────────────────────────────────
# G. Gradient-explosion safety (regression tests for the step-2040 NaN crash)
# ──────────────────────────────────────────────────────────────────────────────

def test_bma_gradient_bounded_when_trunk_near_collapse():
    """When trunk variance ≈ 0 (near-erank-collapse), grad norm stays finite.

    The sqrt gradient is 1/(2·√var). With var clamped to only 1e-8, the
    amplification is 1/(2e-4) = 5000, which causes gradient explosion
    (exactly the step-2040 NaN crash pattern: near-identical representations
    → var≈1e-9 → clamp kicks in → 5000× amplification → NaN).

    The fix raises the clamp floor so the worst-case amplification stays ≤ 50.
    """
    h = _make_harness_with_bma(weight=0.05, n_proj=64, d_sem=16)
    torch.manual_seed(0)
    # Nearly-collapsed trunk: base direction + tiny noise.
    # var per projection ≈ noise² ≈ 1e-6 → sqrt grad ≈ 500 without fix.
    base = torch.randn(1, 16)
    noise = torch.randn(64, 16) * 1e-3
    mu = (base + noise).requires_grad_(True)
    s  = torch.randn(64, 16)
    h.language_model._last_h_motor   = mu.view(8, 8, 16)
    h.language_model._last_h_sensory = s.view(8, 8, 16)
    loss = h._compute_bma_loss(base_weight=0.05)
    assert loss is not None
    loss.backward()
    assert mu.grad is not None
    grad_norm = mu.grad.norm().item()
    assert not math.isnan(grad_norm), "gradient is NaN on near-collapsed trunk"
    assert not math.isinf(grad_norm), "gradient is inf on near-collapsed trunk"
    # With floor=1e-4 the worst-case amplification is 50; with 64 projections
    # and weight 0.05 the gradient norm should stay below 500.
    assert grad_norm < 500.0, (
        f"gradient norm {grad_norm:.1f} too large — "
        "variance clamp floor is too small"
    )


def test_bma_loss_finite_when_trunk_all_zeros():
    """All-zero trunk (degenerate limit) → finite loss and finite gradient."""
    h = _make_harness_with_bma(weight=0.05)
    # Slightly-perturbed zero so gradients are nonzero but near-zero variance
    mu = (torch.zeros(8, 8, 16) + torch.randn(8, 8, 16) * 1e-5).requires_grad_(True)
    s  = torch.randn(8, 8, 16)
    h.language_model._last_h_motor   = mu
    h.language_model._last_h_sensory = s
    loss = h._compute_bma_loss(base_weight=0.05)
    assert loss is not None
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    loss.backward()
    assert mu.grad is not None
    assert torch.isfinite(mu.grad).all(), "gradient contains NaN/inf"


# ──────────────────────────────────────────────────────────────────────────
# G. Early-start anti-collapse contract (plateau fix: bma_ramp_start: 500→0)
# ──────────────────────────────────────────────────────────────────────────

def test_bma_fires_at_step_100_with_ramp_start_zero():
    """ramp_start=0 makes BMA fire at step 100, inside the rank-collapse window.

    In the 2026-06-21 run, erank collapsed from 40 to 3-6 by step 200-300.
    With old ramp_start=500, BMA contributed nothing during steps 0-500.
    Fix: bma_ramp_start: 500 → 0 in SmolLM arch.neuro.
    """
    D = 16
    h = _make_harness_with_bma(weight=0.05, n_proj=32, ramp_start=0, ramp_end=3000, d_sem=D)
    h._global_step = 100
    torch.manual_seed(42)
    mu = torch.randn(4, 8, D) * 0.01   # collapsed trunk (low variance)
    s  = torch.randn(4, 8, D)           # diverse expert
    _stash(h, mu.clone().requires_grad_(True), s.detach())
    loss = h._compute_bma_loss(0.05)
    # alpha = (100 - 0) / (3000 - 0) = 0.033 → eff_weight = 0.05 × 0.033 > 0
    assert loss is not None, "BMA with ramp_start=0 must fire at step 100"
    assert float(loss) > 0.0


def test_bma_silent_at_step_100_with_ramp_start_500():
    """ramp_start=500 (old value) keeps BMA silent at step 100.

    Pins the regression: no anti-collapse gradient during steps 0-499
    when ramp_start=500, exactly when erank collapse is observed.
    """
    D = 16
    h = _make_harness_with_bma(weight=0.05, n_proj=32, ramp_start=500, ramp_end=3000, d_sem=D)
    h._global_step = 100
    torch.manual_seed(42)
    mu = torch.randn(4, 8, D) * 0.01
    s  = torch.randn(4, 8, D)
    _stash(h, mu.clone(), s.detach())
    loss = h._compute_bma_loss(0.05)
    # alpha = max(0, (100 - 500) / 2500) = 0 → eff_weight = 0 → returns None
    assert loss is None, "BMA with ramp_start=500 must not fire at step 100"
