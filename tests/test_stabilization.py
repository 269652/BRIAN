"""Tests for the post-awakening convergence / stabilization fixes (A–D).

Background: a fresh full run reached PPL ~60 (below the baseline) around step
7k, then DIVERGED in the second half — a gnorm spike triggered a
self-amplifying collapse (lm_loss↑ → maturity↓ → aux-gate/pruning shift →
bigger perturbation → lm_loss↑), ending at PPL ~154. These four fixes target
the control loop, not the LM architecture:

  A. aux_ramp_fraction — fixed-length, horizon-independent aux-loss ramp
     (the old `steps_ramped/(total-step)` form blew up in the final ~10%).
  B. maturity ratchet — MAT is monotonic non-decreasing post-awakening, so a
     loss spike can't unwind the control state.
  C. freeze pruning after maturation — no projection pruning once mature.
  D. grad-spike rejection — skip the optimizer step on a gnorm spike.
"""
from __future__ import annotations
import torch

from neuroslm.train import aux_ramp_fraction
from neuroslm.xla_utils import optimizer_step
from neuroslm.config import tiny, large, BrainConfig
from neuroslm.brain import Brain


def _tiny() -> BrainConfig:
    c = tiny()
    c.vocab_size = 256
    return c


# ── A. Auxiliary-loss ramp ──────────────────────────────────────────────────

def test_aux_ramp_is_linear_over_window():
    assert aux_ramp_fraction(1000, 1000, 2000) == 0.0      # at start
    assert aux_ramp_fraction(2000, 1000, 2000) == 0.5      # halfway
    assert aux_ramp_fraction(3000, 1000, 2000) == 1.0      # full
    assert aux_ramp_fraction(9999, 1000, 2000) == 1.0      # clamped past window


def test_aux_ramp_is_horizon_independent():
    """The core fix: the ramp depends only on (step - start) and window length,
    NOT on the training horizon — so it can't blow up near total_steps."""
    # Same step/start/window must give the same fraction regardless of how
    # long the run is. (The old form divided by (total_steps - step).)
    f = aux_ramp_fraction(2500, 1000, 2000)
    assert f == aux_ramp_fraction(2500, 1000, 2000)
    assert 0.0 < f < 1.0
    # Never exceeds 1.0 even deep into a long run.
    assert aux_ramp_fraction(95000, 1000, 2000) == 1.0


def test_aux_ramp_safe_before_start():
    assert aux_ramp_fraction(500, None, 2000) == 0.0
    assert aux_ramp_fraction(500, 1000, 2000) == 0.0       # step < start
    assert aux_ramp_fraction(5000, 1000, 0) == 0.0         # degenerate window


# ── B. Maturity ratchet ─────────────────────────────────────────────────────

def test_maturity_ratchet_holds_through_loss_spike():
    c = _tiny(); c.maturity_ratchet = True
    torch.manual_seed(0); b = Brain(c)
    b._maturity_ema_alpha = 1.0   # disable smoothing to isolate the ratchet
    m_good = b.update_maturity(2.0)    # low loss → high MAT (awakened)
    m_spike = b.update_maturity(9.5)   # loss spike → would normally drop MAT
    assert m_good > c.maturity_awaken_floor
    assert m_spike >= m_good, "ratchet must prevent MAT from falling post-awakening"


def test_maturity_can_fall_when_ratchet_disabled():
    c = _tiny(); c.maturity_ratchet = False
    torch.manual_seed(0); b = Brain(c)
    b._maturity_ema_alpha = 1.0
    m_good = b.update_maturity(2.0)
    m_spike = b.update_maturity(9.5)
    assert m_spike < m_good, "without ratchet, a loss spike unwinds MAT (the bug)"


def test_maturity_ratchet_inactive_below_awaken_floor():
    """Before the high-water mark crosses the floor, MAT tracks freely (so a
    bad-init early model isn't locked at a spuriously high value)."""
    c = _tiny(); c.maturity_ratchet = True; c.maturity_awaken_floor = 0.9
    torch.manual_seed(0); b = Brain(c)
    b._maturity_ema_alpha = 1.0
    m1 = b.update_maturity(7.0)   # MAT ~0.35, below the 0.9 floor
    m2 = b.update_maturity(9.5)   # can still fall (floor not reached)
    assert m2 < m1


# ── C. Freeze pruning after maturation ──────────────────────────────────────

def test_pruning_latches_off_after_maturation():
    c = _tiny()
    c.freeze_pruning_after_maturation = True
    c.prune_freeze_mat = 0.6
    torch.manual_seed(0); b = Brain(c)
    b._maturity_ema_alpha = 1.0
    # Push maturity high-water mark above the freeze threshold.
    b.update_maturity(2.0)   # MAT ~0.81 ≥ 0.6
    assert getattr(b, "_maturity_hwm", 0.0) >= c.prune_freeze_mat
    # A forward pass with targets runs the trophic block, which should latch.
    ids = torch.randint(0, 256, (1, 16))
    tgt = torch.randint(0, 256, (1, 16))
    b.eval()
    with torch.no_grad():
        b.forward_lm(ids, tgt)
    assert getattr(b, "_pruning_frozen", False) is True


# ── D. Gradient-spike rejection ─────────────────────────────────────────────

def _one_param_opt():
    lin = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(lin.parameters(), lr=1.0)
    return lin, opt


def test_grad_spike_is_skipped():
    lin, opt = _one_param_opt()
    w0 = lin.weight.detach().clone()
    lin.weight.grad = torch.ones_like(lin.weight) * 100.0  # spike
    gnorm = optimizer_step(opt, lin.parameters(), max_norm=1.0, skip_threshold=5.0)
    assert gnorm > 5.0
    assert torch.allclose(lin.weight.detach(), w0), "spiked step must be skipped"


def test_normal_grad_steps_through():
    lin, opt = _one_param_opt()
    w0 = lin.weight.detach().clone()
    lin.weight.grad = torch.ones_like(lin.weight) * 0.1
    gnorm = optimizer_step(opt, lin.parameters(), max_norm=1.0, skip_threshold=5.0)
    assert gnorm <= 5.0
    assert not torch.allclose(lin.weight.detach(), w0), "normal step must apply"


def test_skip_threshold_none_always_steps():
    lin, opt = _one_param_opt()
    w0 = lin.weight.detach().clone()
    lin.weight.grad = torch.ones_like(lin.weight) * 100.0
    optimizer_step(opt, lin.parameters(), max_norm=1.0, skip_threshold=None)
    assert not torch.allclose(lin.weight.detach(), w0), "no threshold → always steps"


# ── Config defaults ─────────────────────────────────────────────────────────

def test_stabilization_defaults():
    c = BrainConfig()
    assert c.aux_ramp_steps == 2000
    assert c.maturity_ratchet is True
    assert c.maturity_awaken_floor == 0.3
    assert c.freeze_pruning_after_maturation is True
    assert c.prune_freeze_mat == 0.6
    assert c.grad_spike_factor == 3.0
    assert c.grad_spike_warmup == 100


def test_large_preset_inherits_stabilization_defaults():
    c = large()
    assert c.maturity_ratchet is True
    assert c.freeze_pruning_after_maturation is True
    assert c.grad_spike_factor == 3.0
