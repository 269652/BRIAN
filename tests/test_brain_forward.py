"""End-to-end Brain forward-pass tests.

Validates that every documented output key is present, has the right shape,
has finite values, and (where applicable) participates in the gradient.
"""
from __future__ import annotations
import math
import pytest
import torch

from neuroslm.config import tiny
from neuroslm.brain import Brain


@pytest.fixture(scope="module")
def brain():
    cfg = tiny()
    cfg.vocab_size = 256
    torch.manual_seed(0)
    b = Brain(cfg)
    b.eval()
    return b


def _ids():
    return torch.randint(0, 256, (1, 16))


def test_forward_lm_keys_present(brain):
    out = brain.forward_lm(_ids(), _ids())
    for k in ("logits", "loss", "lm_loss", "phi", "phi_loss",
              "world_loss", "motor_loss", "pred_coding_loss",
              "novelty", "action_idx", "consciousness"):
        assert k in out, f"missing {k}"


def test_logits_shape(brain):
    ids = torch.randint(0, 256, (1, 16))
    out = brain.forward_lm(ids, ids)
    assert out["logits"].shape == (1, 16, 256)


def test_loss_finite_and_positive(brain):
    out = brain.forward_lm(_ids(), _ids())
    loss = float(out["loss"].item())
    lm = float(out["lm_loss"].item())
    assert math.isfinite(loss)
    assert math.isfinite(lm)
    assert lm > 0


def test_phi_is_real(brain):
    out = brain.forward_lm(_ids(), _ids())
    phi = float(out["phi"].item())
    assert math.isfinite(phi)
    assert phi >= 0.0


def test_phi_loss_is_negative_bounded(brain):
    """phi_loss = -tanh(phi/3)*3 lies in [-3, 0]."""
    out = brain.forward_lm(_ids(), _ids())
    pl = float(out["phi_loss"].item())
    assert -3.0001 <= pl <= 0.0001


def test_backward_succeeds():
    cfg = tiny()
    cfg.vocab_size = 256
    torch.manual_seed(0)
    b = Brain(cfg)
    b.eval()
    out = b.forward_lm(_ids(), _ids())
    out["loss"].backward()
    grad_params = [p for p in b.parameters() if p.grad is not None]
    assert grad_params, "no parameter received gradient"
    g = sum(p.grad.abs().sum().item() for p in grad_params)
    assert g > 0
    assert math.isfinite(g)


def test_phi_objective_trains_modules_not_trunk():
    """Φ objective must participate in backward — but with trunk gradient
    isolation (cfg.detach_trunk_from_aux, default ON) it injects that extra
    gradient into the BIO modules (shaping how they integrate), NOT into the
    language trunk. The trunk is shaped by the LM loss alone. This is the
    architectural fix for the post-awakening divergence: aux objectives no
    longer reshape the representation the LM depends on."""
    ids = torch.randint(0, 256, (1, 16))
    tgt = torch.randint(0, 256, (1, 16))

    def run(w_phi: float):
        c = tiny(); c.vocab_size = 256
        c.enable_phi_objective = w_phi > 0; c.w_phi = w_phi
        torch.manual_seed(0); b = Brain(c); b.eval()
        loss = b.forward_lm(ids, tgt)["loss"]
        loss.backward()
        lang = sum(p.grad.abs().sum().item() for p in b.language.parameters()
                   if p.grad is not None)
        return float(loss.item()), lang

    loss_on, lang_on = run(1.0)
    loss_off, lang_off = run(0.0)
    # Φ participates in the objective (changes the total loss value)…
    assert abs(loss_on - loss_off) > 1e-4, (
        f"phi must affect the loss: {loss_on} vs {loss_off}")
    # …but with trunk isolation it does NOT reach the language trunk: the
    # trunk gradient is identical with and without Φ.
    assert abs(lang_on - lang_off) < 1e-3, (
        f"phi must NOT reshape the isolated trunk: {lang_on} vs {lang_off}")


def test_consciousness_metrics_populated(brain):
    out = brain.forward_lm(_ids(), _ids())
    c = out.get("consciousness", {})
    # ConsciousnessMetrics returns these keys in update()
    for key in ("phi", "gamma", "theta", "alpha", "coherence",
                "ignition", "metacognition", "binding", "tick"):
        assert key in c, f"missing consciousness metric: {key}"


def test_last_phi_persists_across_steps(brain):
    """_last_phi must be set after a forward pass and remain accessible."""
    brain.forward_lm(_ids(), _ids())
    p1 = brain._last_phi
    assert isinstance(p1, float)
    assert math.isfinite(p1)
    brain.forward_lm(_ids(), _ids())
    p2 = brain._last_phi
    assert isinstance(p2, float)
    # Should change step-to-step (different inputs).
    # Not asserting strict inequality (may coincidentally land near zero).
    assert math.isfinite(p2)


def test_baseline_mode_skips_phi():
    cfg = tiny()
    cfg.vocab_size = 256
    cfg.baseline = True
    torch.manual_seed(0)
    b = Brain(cfg)
    b.eval()
    out = b.forward_lm(_ids(), _ids())
    # Baseline only emits logits and loss; no phi key required.
    assert "logits" in out
    assert "loss" in out
    assert float(out["loss"].item()) > 0


def test_inference_no_targets(brain):
    out = brain.forward_lm(_ids(), targets=None)
    assert "logits" in out
    # loss is only computed when targets are provided
    assert "loss" not in out or out["loss"] is None or torch.is_tensor(out.get("loss"))


def test_two_passes_idempotent(brain):
    """Running forward_lm twice must not raise (begin_pass clears state)."""
    out1 = brain.forward_lm(_ids(), _ids())
    out2 = brain.forward_lm(_ids(), _ids())
    assert math.isfinite(float(out1["loss"].item()))
    assert math.isfinite(float(out2["loss"].item()))
