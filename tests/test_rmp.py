"""
tests/test_rmp.py — TDD contract suite for Riemannian Motor Projection (RMP).

RMP maps h_motor onto the Poincaré ball of curvature R before it enters the
VBB posterior.  The tanh-map:

    h_proj = ρ · tanh(‖h‖ / ρ) · (h / ‖h‖),   ρ = 1 / √R

has these analytical properties:
  * ||h_proj|| ≤ ρ  for all h  (magnitude is capped at the ball radius)
  * h_proj / ||h_proj|| = h / ||h||  (direction is preserved)
  * h_proj → h as R → 0  (identity limit; standard Euclidean residual)
  * Gradient flows through the tanh (non-zero for finite h)

When composed with the existing LayerNorm in the VBB path, RMP provides a
geometrically motivated pre-normalisation that prevents KL explosion for any
activation magnitude.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_harness_with_rmp(curvature: float, d_sem: int = 16,
                            with_vbb: bool = False,
                            vbb_alpha: float = 1e-3):
    from neuroslm.harness import BRIANHarness
    from neuroslm.dsl.training_config import TrainingConfig

    class _StubLM(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.proj = nn.Linear(d, d)
            self._last_h_motor   = None
            self._last_h_sensory = None

    cfg = TrainingConfig()
    cfg.motor_curvature    = curvature
    cfg.pc_reentry_nt_gate = False
    if with_vbb:
        cfg.vbb_alpha      = vbb_alpha
        cfg.pc_reentry_weight = 0.1

    lm = _StubLM(d_sem)
    return BRIANHarness.from_language_model(
        language_model=lm, vocab_size=257, d_sem=d_sem,
        training_config=cfg,
    )


def _apply_rmp(h, mu: torch.Tensor) -> torch.Tensor:
    """Extract the RMP tanh-map from a harness and apply it to mu."""
    R   = F.softplus(h._motor_R_raw)
    rho = R.rsqrt()
    norm = mu.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return rho * torch.tanh(norm / rho) * mu / norm


# ──────────────────────────────────────────────────────────────────────────────
# A. Config defaults + DSL parsing
# ──────────────────────────────────────────────────────────────────────────────

def test_rmp_config_defaults_off():
    from neuroslm.dsl.training_config import TrainingConfig
    cfg = TrainingConfig()
    assert cfg.motor_curvature == 0.0


def test_rmp_config_parses_from_dsl():
    from neuroslm.dsl.training_config import parse_training_config
    src = "learning_rate: 0.001\nmotor_curvature: 0.5"
    cfg = parse_training_config(src)
    assert cfg.motor_curvature == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────────────────
# B. Module construction
# ──────────────────────────────────────────────────────────────────────────────

def test_rmp_disabled_when_curvature_zero():
    """curvature=0 → _motor_R_raw must be None (no optimizer pollution)."""
    h = _make_harness_with_rmp(curvature=0.0)
    assert getattr(h, "_motor_R_raw", None) is None


def test_rmp_enabled_when_curvature_positive():
    """curvature>0 → _motor_R_raw is an nn.Parameter."""
    h = _make_harness_with_rmp(curvature=1.0)
    assert h._motor_R_raw is not None
    assert isinstance(h._motor_R_raw, nn.Parameter)


def test_rmp_initial_curvature_matches_config():
    """softplus(_motor_R_raw) ≈ motor_curvature at init."""
    curvature = 2.0
    h = _make_harness_with_rmp(curvature=curvature)
    R_eff = F.softplus(h._motor_R_raw).item()
    assert R_eff == pytest.approx(curvature, rel=1e-3)


def test_rmp_parameter_in_harness_parameters():
    """_motor_R_raw must appear in harness.parameters() so AdamW updates it."""
    h = _make_harness_with_rmp(curvature=1.0)
    p_ids = {id(p) for p in h.parameters()}
    assert id(h._motor_R_raw) in p_ids


def test_rmp_exists_independently_of_vbb():
    """RMP must be built even when vbb_alpha=0 (legacy PC-reentry path)."""
    from neuroslm.dsl.training_config import TrainingConfig
    from neuroslm.harness import BRIANHarness

    class _Stub(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.proj = nn.Linear(d, d)
            self._last_h_motor   = None
            self._last_h_sensory = None

    cfg = TrainingConfig()
    cfg.motor_curvature = 1.0
    cfg.vbb_alpha       = 0.0  # VBB disabled
    lm = _Stub(16)
    h  = BRIANHarness.from_language_model(lm, vocab_size=257, d_sem=16,
                                           training_config=cfg)
    assert h._motor_R_raw is not None


# ──────────────────────────────────────────────────────────────────────────────
# C. Mathematical contracts
# ──────────────────────────────────────────────────────────────────────────────

def test_rmp_caps_large_norms():
    """‖h_proj‖ ≤ ρ for any h, including activations with magnitude ~277."""
    h   = _make_harness_with_rmp(curvature=1.0, d_sem=16)
    R   = F.softplus(h._motor_R_raw).item()
    rho = 1.0 / math.sqrt(R)

    mu = torch.full((4, 8, 16), 277.0)
    proj = _apply_rmp(h, mu)
    norms = proj.norm(dim=-1)
    assert (norms <= rho + 1e-4).all(), (
        f"Max projected norm {norms.max():.4f} exceeds ρ={rho:.4f}")


def test_rmp_preserves_direction():
    """The tanh-map is radial: h_proj and h point in the same direction."""
    h  = _make_harness_with_rmp(curvature=0.5, d_sem=16)
    mu = torch.randn(4, 8, 16)
    proj = _apply_rmp(h, mu)
    flat_mu   = mu.reshape(-1, 16)
    flat_proj = proj.reshape(-1, 16)
    cos = F.cosine_similarity(flat_mu, flat_proj, dim=-1)
    assert (cos > 0.999).all(), f"Direction not preserved; min cos={cos.min():.6f}"


def test_rmp_identity_at_zero_input():
    """Zero vector maps to zero (degenerate case — tanh(0)=0)."""
    h  = _make_harness_with_rmp(curvature=1.0, d_sem=16)
    mu = torch.zeros(2, 4, 16)
    proj = _apply_rmp(h, mu)
    assert proj.abs().max().item() < 1e-6


def test_rmp_small_input_near_identity():
    """For ‖h‖ ≪ ρ, tanh(‖h‖/ρ) ≈ ‖h‖/ρ, so h_proj ≈ h."""
    h = _make_harness_with_rmp(curvature=0.01, d_sem=16)  # large ρ
    R   = F.softplus(h._motor_R_raw).item()
    rho = 1.0 / math.sqrt(R)
    # tiny vectors: norm ≪ ρ
    mu = torch.randn(4, 8, 16) * 0.01
    proj = _apply_rmp(h, mu)
    # relative error should be < 1%
    rel_err = ((proj - mu).norm() / mu.norm()).item()
    assert rel_err < 0.01, f"RMP deviates {rel_err*100:.2f}% from identity for small inputs"


def test_rmp_monotone_in_curvature():
    """Higher curvature → smaller projected norms (tighter ball)."""
    mu = torch.full((4, 8, 16), 10.0)
    norms = []
    for c in [0.1, 0.5, 1.0, 2.0, 5.0]:
        h = _make_harness_with_rmp(curvature=c, d_sem=16)
        proj = _apply_rmp(h, mu)
        norms.append(proj.norm(dim=-1).mean().item())
    # monotonically decreasing norms as curvature increases
    assert all(norms[i] > norms[i+1] for i in range(len(norms)-1)), (
        f"Norms not monotone in curvature: {norms}")


# ──────────────────────────────────────────────────────────────────────────────
# D. Gradient flow
# ──────────────────────────────────────────────────────────────────────────────

def test_rmp_gradient_flows_through_projection():
    """Gradient must flow from the projection output back into h_m."""
    h  = _make_harness_with_rmp(curvature=1.0, d_sem=16)
    mu = torch.randn(2, 4, 16, requires_grad=True)
    proj = _apply_rmp(h, mu)
    proj.sum().backward()
    assert mu.grad is not None
    assert mu.grad.abs().sum().item() > 0.0


def test_rmp_curvature_parameter_receives_gradient():
    """_motor_R_raw must also receive gradient (so optimizer can tune R)."""
    h  = _make_harness_with_rmp(curvature=1.0, d_sem=16)
    mu = torch.randn(2, 4, 16)
    proj = _apply_rmp(h, mu)
    proj.sum().backward()
    assert h._motor_R_raw.grad is not None
    assert h._motor_R_raw.grad.abs().item() > 0.0


# ──────────────────────────────────────────────────────────────────────────────
# E. Integration — RMP + VBB keeps KL bounded
# ──────────────────────────────────────────────────────────────────────────────

def test_rmp_vbb_kl_bounded_for_large_hm():
    """With RMP enabled, VBB KL must stay < 10 even for h_m magnitude ~277."""
    h = _make_harness_with_rmp(curvature=1.0, with_vbb=True, d_sem=16)
    mu = torch.full((2, 4, 16), 277.0)
    s  = torch.randn(2, 4, 16)
    h.language_model._last_h_motor   = mu
    h.language_model._last_h_sensory = s
    h._compute_pc_reentry_loss(base_weight=0.1)
    kl = h._metrics.get("vbb_kl", float("inf"))
    assert kl < 10.0, f"VBB KL = {kl:.1f} despite RMP — projection not applied"


def test_rmp_logs_curvature_metrics():
    """When RMP fires, motor_curvature_R and motor_rho must appear in _metrics."""
    h = _make_harness_with_rmp(curvature=1.0, with_vbb=True, d_sem=16)
    mu = torch.randn(2, 4, 16)
    s  = torch.randn(2, 4, 16)
    h.language_model._last_h_motor   = mu
    h.language_model._last_h_sensory = s
    h._compute_pc_reentry_loss(base_weight=0.1)
    assert "motor_curvature_R" in h._metrics
    assert "motor_rho" in h._metrics
    assert h._metrics["motor_curvature_R"] > 0.0
    assert h._metrics["motor_rho"] > 0.0
