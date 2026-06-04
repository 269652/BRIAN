# -*- coding: utf-8 -*-
"""HPB Phase 4 — Hyperbolic Bowtie Waist (HBW).

Replaces the Euclidean VBB posterior with a wrapped Gaussian on the
Poincaré ball B^d_c of curvature −c. The KL acquires an extra
Jacobian-log-det term:

    KL[q || W(o, R)]  =  KL_euclid  −  (d-1) · log( sinh(√c · ‖μ‖) / (√c · ‖μ‖) )

This matches Nagano-Skopek-Hayashi-Sasaki 2019 ("Wrapped Normal").

Contract under test
-------------------
1. Möbius operations satisfy their algebraic identities:
   - ``mobius_add(o, x) = x`` (origin is the identity)
   - ``mobius_add(x, −x) = o``
   - ``exp_map(o, 0) = o``
   - ``log_map(o, exp_map(o, v)) = v`` for ‖v‖ small
2. Curvature ``c → 0`` reduces to Euclidean addition.
3. Wrapped-Normal KL is ≥ Euclidean KL for any non-zero μ
   (the sinh correction is non-negative because log(sinh(x)/x) ≥ 0).
4. Wrapped-Normal KL equals Euclidean KL at μ=0 (origin).
5. ``TrainingConfig`` accepts a ``vbb_curvature`` field.
6. With ``vbb_curvature > 0`` the harness uses the hyperbolic KL.
"""
from __future__ import annotations
import math
import pytest
import torch

from neuroslm.dsl.training_config import parse_training_config


# ── 1. Möbius algebra identities ─────────────────────────────────────

def test_mobius_add_origin_is_identity():
    from neuroslm.emergent.hyperbolic import mobius_add
    torch.manual_seed(0)
    x = torch.randn(3, 5) * 0.1     # stay inside unit ball
    o = torch.zeros_like(x)
    c = torch.tensor(1.0)
    out = mobius_add(o, x, c)
    assert torch.allclose(out, x, atol=1e-6), (
        f"mobius_add(o, x) ≠ x; max-diff "
        f"{(out - x).abs().max().item():.2e}"
    )


def test_mobius_add_inverse():
    from neuroslm.emergent.hyperbolic import mobius_add
    torch.manual_seed(1)
    x = torch.randn(3, 5) * 0.1
    c = torch.tensor(1.0)
    out = mobius_add(x, -x, c)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5), (
        f"mobius_add(x, -x) ≠ 0; max-diff "
        f"{out.abs().max().item():.2e}"
    )


def test_exp_log_map_inverse_at_small_radius():
    """log_map(o, exp_map(o, v)) = v for small ‖v‖ (numerical sanity)."""
    from neuroslm.emergent.hyperbolic import exp_map_zero, log_map_zero
    torch.manual_seed(2)
    v = torch.randn(4, 6) * 0.01
    c = torch.tensor(1.0)
    v_round = log_map_zero(exp_map_zero(v, c), c)
    assert torch.allclose(v_round, v, atol=1e-4), (
        f"exp/log inverse failed; max-diff "
        f"{(v_round - v).abs().max().item():.2e}"
    )


def test_curvature_zero_limit_is_euclidean():
    """As c → 0, mobius_add reduces to vector addition."""
    from neuroslm.emergent.hyperbolic import mobius_add
    torch.manual_seed(3)
    x = torch.randn(2, 4) * 0.05
    y = torch.randn(2, 4) * 0.05
    c_small = torch.tensor(1e-8)
    out = mobius_add(x, y, c_small)
    assert torch.allclose(out, x + y, atol=1e-5), (
        "mobius_add at c≈0 must equal Euclidean +"
    )


# ── 2. Wrapped-Normal KL math ────────────────────────────────────────

def test_hyperbolic_kl_at_origin_equals_euclidean():
    """At μ=0 the sinh correction term is (d-1)·log(1) = 0, so
    hyperbolic KL collapses to the Euclidean closed form."""
    from neuroslm.emergent.hyperbolic import wrapped_normal_kl
    D = 8
    mu = torch.zeros(4, D)
    log_var = torch.zeros(4, D)
    c = torch.tensor(1.0)
    kl_hyp = wrapped_normal_kl(mu, log_var, c)
    # Euclidean KL: ½ Σ (σ² + μ² − 1 − log σ²) = 0 at the prior.
    assert float(kl_hyp) == pytest.approx(0.0, abs=1e-6)


def test_hyperbolic_kl_geq_euclidean_for_nonzero_mu():
    """The Jacobian-correction (d-1) · log(sinh(x)/x) is ≥ 0 for x > 0,
    so the hyperbolic KL is always ≥ Euclidean KL at the same (μ, σ)."""
    from neuroslm.emergent.hyperbolic import wrapped_normal_kl
    torch.manual_seed(11)
    D = 16
    mu = torch.randn(4, D) * 0.3
    log_var = torch.zeros(4, D)
    c = torch.tensor(0.5)
    kl_hyp = float(wrapped_normal_kl(mu, log_var, c))
    # Euclidean reference
    kl_eu = float(0.5 * (log_var.exp() + mu.pow(2) - 1.0 - log_var).mean())
    assert kl_hyp >= kl_eu - 1e-6, (
        f"wrapped-Normal KL ({kl_hyp:.4f}) must be ≥ Euclidean "
        f"({kl_eu:.4f}); sinh correction sign flipped?"
    )


def test_hyperbolic_kl_is_finite_at_boundary():
    """Test KL stays finite even when ‖μ‖ is close to (but inside)
    the Poincaré ball boundary ‖μ‖ < 1/√c."""
    from neuroslm.emergent.hyperbolic import wrapped_normal_kl
    D = 4
    # ‖μ‖ ≈ 0.5 for c=1; well inside the ball.
    mu = torch.full((2, D), 0.25)
    log_var = torch.full((2, D), -2.0)
    c = torch.tensor(1.0)
    kl = wrapped_normal_kl(mu, log_var, c)
    assert torch.isfinite(kl).all()
    assert float(kl) > 0.0


def test_hyperbolic_kl_gradient_flows():
    """Both μ and log_var must receive gradient through KL."""
    from neuroslm.emergent.hyperbolic import wrapped_normal_kl
    torch.manual_seed(13)
    D = 8
    # Build leaf tensors directly (do NOT scale after requires_grad=True
    # or the tensor becomes a non-leaf and .grad is silently None).
    mu = (torch.randn(4, D) * 0.1).requires_grad_(True)
    # Initialise log_var to a non-zero value so the KL has a non-zero
    # gradient w.r.t. log_var.  At log_var=0 the d/d(log_var) of the
    # per-dim KL ½(e^lv + μ² − 1 − lv) is ½(e^lv − 1) = 0 — a math fact
    # of the unit-variance prior, not a code bug.  Real training never
    # sits at exactly log_var = 0 because the sigma_head bias is -8.
    log_var = (torch.full((4, D), -0.5)).requires_grad_(True)
    c = torch.tensor(0.5)
    kl = wrapped_normal_kl(mu, log_var, c)
    kl.backward()
    assert mu.grad is not None and mu.grad.abs().sum().item() > 0
    assert log_var.grad is not None and log_var.grad.abs().sum().item() > 0


# ── 3. TrainingConfig + harness integration ──────────────────────────

def test_training_config_curvature_defaults_zero():
    """Default curvature is 0 = Euclidean ⇒ HBW disabled by default."""
    from neuroslm.dsl.training_config import TrainingConfig
    cfg = TrainingConfig()
    assert getattr(cfg, "vbb_curvature", 0.0) == 0.0


def test_parse_vbb_curvature():
    src = """
        learning_rate: 0.0003
        vbb_alpha: 0.001
        vbb_curvature: 1.0
    """
    cfg = parse_training_config(src)
    assert math.isclose(cfg.vbb_curvature, 1.0)


def test_harness_uses_hyperbolic_kl_when_curvature_positive():
    """With vbb_curvature > 0, _compute_pc_reentry_loss must call
    the wrapped-Normal KL instead of the Euclidean one. We verify
    via the side-effect on cfg.vbb_kl metric (should differ from
    the Euclidean baseline at the same (μ, σ))."""
    from neuroslm.harness import BRIANHarness
    from neuroslm.dsl.training_config import TrainingConfig
    import torch.nn as nn

    class _StubLM(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.proj = nn.Linear(d, d)
            self._last_h_motor = torch.randn(2, 4, d, requires_grad=True) * 0.2
            self._last_h_sensory = torch.randn(2, 4, d, requires_grad=True) * 0.2

        def forward(self, x):
            return self.proj(x)

    D = 8
    VOCAB = 64
    # Euclidean run
    cfg_e = TrainingConfig()
    cfg_e.vbb_alpha = 0.001
    cfg_e.vbb_curvature = 0.0
    h_e = BRIANHarness.from_language_model(
        _StubLM(D), vocab_size=VOCAB, d_sem=D, training_config=cfg_e)
    # Hyperbolic run with same inputs
    cfg_h = TrainingConfig()
    cfg_h.vbb_alpha = 0.001
    cfg_h.vbb_curvature = 1.0
    h_h = BRIANHarness.from_language_model(
        _StubLM(D), vocab_size=VOCAB, d_sem=D, training_config=cfg_h)
    # Drive both with the same μ/σ so the KL difference is structural.
    # We set the same stash on both LMs and read the published metric.
    torch.manual_seed(99)
    mu = torch.randn(2, 4, D, requires_grad=True) * 0.3
    sens = torch.randn(2, 4, D, requires_grad=True) * 0.3
    h_e.language_model._last_h_motor = mu
    h_e.language_model._last_h_sensory = sens
    h_h.language_model._last_h_motor = mu
    h_h.language_model._last_h_sensory = sens
    loss_e = h_e._compute_pc_reentry_loss(base_weight=0.1)
    loss_h = h_h._compute_pc_reentry_loss(base_weight=0.1)
    assert loss_e is not None and loss_h is not None
    # KL differs by structural Jacobian term — losses must not be equal.
    assert abs(float(loss_e) - float(loss_h)) > 1e-6, (
        f"Hyperbolic vs Euclidean loss identical ({float(loss_e):.4f}) — "
        "HBW switch is dead code"
    )


# ── 4. Composition with MDRV ─────────────────────────────────────────

def test_hbw_composes_with_free_bits():
    """The free-bits clamp must still apply on the per-dim KL even
    when curvature > 0 (we clamp the per-dim Euclidean part, then add
    the scalar sinh correction)."""
    from neuroslm.emergent.hyperbolic import wrapped_normal_kl
    torch.manual_seed(31)
    D = 8
    mu = torch.randn(2, D) * 0.05      # tiny μ → small KL
    log_var = torch.full((2, D), -6.0)  # very small σ
    c = torch.tensor(0.5)
    kl_no_fb = float(wrapped_normal_kl(mu, log_var, c, free_bits=0.0))
    kl_fb = float(wrapped_normal_kl(mu, log_var, c, free_bits=0.5))
    # With free-bits, each per-dim contribution is clamped UP to 0.5,
    # so the KL must be ≥ the unclamped version.
    assert kl_fb >= kl_no_fb - 1e-6
    # And strictly greater than the unclamped case when σ is tiny.
    assert kl_fb > 0.4, (
        f"free-bits floor not respected in hyperbolic KL (got {kl_fb:.3f})"
    )
