"""Tests for the convergence / stabilization mechanics.

A fresh full run reached PPL ~60 (below baseline) at step ~7k then DIVERGED to
PPL ~154 (run 1); engaging the aux losses harder/earlier diverged at ~step 2k
with gnorm pinned ~14 (run 2). Root cause: the auxiliary-loss gradients flow
back into the SHARED LM trunk and corrupt the representation the LM head
depends on. Fixes, in order of importance:

  PRIMARY — trunk gradient isolation: the bio/cognitive modules read a
    stop-gradient copy of the trunk's `sem`, so aux losses train their own
    modules but cannot reshape the LM trunk. (detach_trunk_from_aux)
  A. maturity-gated aux weight (maturity_aux_gate): aux strengthens only as
     the LM matures, and backs off if it regresses. Secondary guard.
  B. asymmetric maturity EMA: rise fast, fall slow (damped) — no whipsaw but
     keeps the maturity-fall recovery valve. Hard ratchet is opt-in.
  C. freeze structural pruning post-maturation.
  D. gradient-spike rejection.
"""
from __future__ import annotations
import torch

from neuroslm.train import maturity_aux_gate
from neuroslm.xla_utils import optimizer_step
from neuroslm.config import tiny, large, BrainConfig
from neuroslm.brain import Brain


def _tiny() -> BrainConfig:
    c = tiny()
    c.vocab_size = 256
    return c


# ── ReZero-style forward-injection gates ────────────────────────────────────

def test_rezero_lambdas_initialized_to_zero():
    """Every module → LM forward-injection gate starts at exactly zero, so
    the model behaves identically to the pure isolated-trunk LM at t=0 (no
    awakening discontinuity)."""
    c = _tiny(); c.use_rezero_injection_gates = True
    torch.manual_seed(0); b = Brain(c)
    assert float(b.lambda_motor.item()) == 0.0
    assert float(b.lambda_mem.item()) == 0.0
    assert float(b.lambda_thought.item()) == 0.0
    assert b.lambda_motor.requires_grad
    assert b.lambda_mem.requires_grad
    assert b.lambda_thought.requires_grad


def test_rezero_lambda_is_in_lm_forward_graph():
    """When a module output is non-zero, setting λ != 0 must change the LM
    logits — proving the gate is actually in the computation graph."""
    c = _tiny(); c.use_rezero_injection_gates = True
    torch.manual_seed(0); b = Brain(c); b.eval()
    # Force motor cortex output to be non-trivial so the gate has something
    # to scale — replace motor_lang_bias generation with a constant ones bias
    # via a hook on `self.motor`.
    orig = b.motor.forward
    def _hook(action, survival=None):
        out = orig(action, survival=survival)
        # out = (_mt, motor_lang_bias, action_idx, action_logits, action_probs)
        ones = torch.ones_like(out[1])
        return (out[0], ones, out[2], out[3], out[4])
    b.motor.forward = _hook  # type: ignore[assignment]

    ids = torch.randint(0, 256, (1, 16))
    torch.manual_seed(1)
    with torch.no_grad():
        b.lambda_motor.fill_(0.0)
        L0 = b.forward_lm(ids)["logits"].detach().clone()
        b.lambda_motor.fill_(1.0)
        L1 = b.forward_lm(ids)["logits"].detach().clone()
    assert not torch.allclose(L0, L1, atol=1e-5), (
        "λ_motor must scale the motor injection in the LM forward path")


def test_rezero_default_on():
    c = BrainConfig()
    assert c.use_rezero_injection_gates is True


def test_large_preset_has_rezero_on():
    assert large().use_rezero_injection_gates is True


# ── PRIMARY: trunk gradient isolation ───────────────────────────────────────

def _trunk_grad(detach: bool, w_world: float):
    c = _tiny()
    c.detach_trunk_from_aux = detach
    c.w_world = w_world
    torch.manual_seed(0); b = Brain(c); b.train()
    ids = torch.randint(0, 256, (2, 16))
    tgt = torch.randint(0, 256, (2, 16))
    torch.manual_seed(1)
    b.zero_grad()
    b.forward_lm(ids, tgt)["loss"].backward()
    g = b.language.tok_emb.weight.grad
    return None if g is None else g.detach().clone()


def test_trunk_isolation_makes_trunk_grad_invariant_to_aux_weight():
    """With isolation ON, scaling an auxiliary loss must NOT change the LM
    trunk's gradient — proof that aux gradients can't reshape the trunk."""
    g_lo = _trunk_grad(detach=True, w_world=0.3)
    g_hi = _trunk_grad(detach=True, w_world=50.0)
    assert g_lo is not None and torch.isfinite(g_lo).all()
    assert torch.allclose(g_lo, g_hi, atol=1e-6)


def test_without_isolation_aux_weight_reshapes_trunk():
    """With isolation OFF (legacy), the aux weight DOES change the trunk
    gradient — the backward path that drove the divergence."""
    g_lo = _trunk_grad(detach=False, w_world=0.3)
    g_hi = _trunk_grad(detach=False, w_world=50.0)
    assert not torch.allclose(g_lo, g_hi, atol=1e-6)


def test_trunk_still_trains_under_lm_loss_with_isolation():
    """Isolation must not starve the trunk — the LM loss still trains it."""
    g = _trunk_grad(detach=True, w_world=0.3)
    assert g is not None and g.abs().sum() > 0


# ── A. Maturity-gated aux weight ────────────────────────────────────────────

def test_maturity_aux_gate_ramps_between_lo_and_hi():
    assert maturity_aux_gate(0.40, 0.50, 0.65) == 0.0      # below lo
    assert maturity_aux_gate(0.50, 0.50, 0.65) == 0.0      # at lo
    assert abs(maturity_aux_gate(0.575, 0.50, 0.65) - 0.5) < 1e-6  # midpoint
    assert maturity_aux_gate(0.65, 0.50, 0.65) == 1.0      # at hi
    assert maturity_aux_gate(0.90, 0.50, 0.65) == 1.0      # above hi


def test_maturity_aux_gate_is_self_correcting():
    """If the LM regresses (MAT drops back below lo) the gate closes again."""
    hi = maturity_aux_gate(0.64, 0.50, 0.65)
    regressed = maturity_aux_gate(0.45, 0.50, 0.65)
    assert hi > 0 and regressed == 0.0


def test_maturity_aux_gate_degenerate_window():
    assert maturity_aux_gate(0.7, 0.6, 0.6) == 1.0
    assert maturity_aux_gate(0.5, 0.6, 0.6) == 0.0


# ── B. Asymmetric maturity EMA ──────────────────────────────────────────────

def test_maturity_falls_slower_than_it_rises():
    """For SYMMETRIC moves around a mid MAT, the downward step must be more
    damped than the upward one — so a loss spike barely dents MAT (no
    whipsaw) while the maturity-fall recovery valve is still present."""
    from neuroslm.neurochem.transmitters import L_RANDOM_DEFAULT as L
    c = _tiny(); c.maturity_ratchet = False
    c.maturity_ema_alpha = 0.5; c.maturity_fall_alpha = 0.02
    torch.manual_seed(0); b = Brain(c)
    loss_up = (1.0 - 0.70) * L   # → m_now ≈ 0.70 (gap +0.20 from 0.5)
    loss_dn = (1.0 - 0.30) * L   # → m_now ≈ 0.30 (gap −0.20 from 0.5)
    b.maturity.fill_(0.5); rise = b.update_maturity(loss_up) - 0.5
    b.maturity.fill_(0.5); fall = 0.5 - b.update_maturity(loss_dn)
    assert rise > 0 and fall > 0
    assert fall < rise, "equal-magnitude fall must be more damped than the rise"


def test_maturity_ratchet_opt_in_holds_through_spike():
    c = _tiny(); c.maturity_ratchet = True; c.maturity_ema_alpha = 1.0
    torch.manual_seed(0); b = Brain(c)
    m_good = b.update_maturity(2.0)
    m_spike = b.update_maturity(9.5)
    assert m_spike >= m_good


def test_maturity_ratchet_off_by_default():
    assert BrainConfig().maturity_ratchet is False


# ── C. Freeze pruning after maturation ──────────────────────────────────────

def test_pruning_latches_off_after_maturation():
    c = _tiny()
    c.freeze_pruning_after_maturation = True
    c.prune_freeze_mat = 0.6
    c.maturity_ema_alpha = 1.0
    torch.manual_seed(0); b = Brain(c)
    b.update_maturity(2.0)   # MAT ~0.81 ≥ 0.6 → hwm crosses freeze threshold
    assert getattr(b, "_maturity_hwm", 0.0) >= c.prune_freeze_mat
    ids = torch.randint(0, 256, (1, 16))
    tgt = torch.randint(0, 256, (1, 16))
    b.eval()
    with torch.no_grad():
        b.forward_lm(ids, tgt)
    assert getattr(b, "_pruning_frozen", False) is True


# ── D. Gradient-spike rejection ─────────────────────────────────────────────

def _one_param_opt():
    lin = torch.nn.Linear(4, 4)
    return lin, torch.optim.SGD(lin.parameters(), lr=1.0)


def test_grad_spike_is_skipped():
    lin, opt = _one_param_opt()
    w0 = lin.weight.detach().clone()
    lin.weight.grad = torch.ones_like(lin.weight) * 100.0
    gnorm = optimizer_step(opt, lin.parameters(), max_norm=1.0, skip_threshold=5.0)
    assert gnorm > 5.0 and torch.allclose(lin.weight.detach(), w0)


def test_normal_grad_steps_through():
    lin, opt = _one_param_opt()
    w0 = lin.weight.detach().clone()
    lin.weight.grad = torch.ones_like(lin.weight) * 0.1
    optimizer_step(opt, lin.parameters(), max_norm=1.0, skip_threshold=5.0)
    assert not torch.allclose(lin.weight.detach(), w0)


def test_skip_threshold_none_always_steps():
    lin, opt = _one_param_opt()
    w0 = lin.weight.detach().clone()
    lin.weight.grad = torch.ones_like(lin.weight) * 100.0
    optimizer_step(opt, lin.parameters(), max_norm=1.0, skip_threshold=None)
    assert not torch.allclose(lin.weight.detach(), w0)


# ── Config defaults ─────────────────────────────────────────────────────────

def test_convergence_defaults():
    c = BrainConfig()
    assert c.detach_trunk_from_aux is True
    assert c.aux_gate_mat_lo == 0.50 and c.aux_gate_mat_hi == 0.65
    assert c.maturity_ratchet is False
    assert c.maturity_fall_alpha == 0.01
    assert c.freeze_pruning_after_maturation is True
    assert c.grad_spike_factor == 3.0


def test_large_preset_inherits_convergence_defaults():
    c = large()
    assert c.detach_trunk_from_aux is True
    assert c.freeze_pruning_after_maturation is True
