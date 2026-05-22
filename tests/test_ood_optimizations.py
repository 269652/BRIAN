"""Tests for the OOD-generalization optimizations.

Covers the four levers added to narrow the train→WikiText (OOD) perplexity
gap for a model that is trained to ~10k steps:

  1. cosine_lr   — anneals to a floor by the DECAY HORIZON (not the run
                   length), so an early-stopped run actually sees the LR
                   annealing phase (the biggest single PPL lever).
  2. dropout     — embedding / standard-block / pre-head dropout that is
                   active in train() and disabled in eval(), and adds NO new
                   state_dict keys (so existing checkpoints still load).
  3. decoupled WD — weight decay applied to 2-D matrices only; 1-D params
                    (norms, biases, NT levels) are exempt.
  4. label smoothing — wired into the chunked LM cross-entropy.

Plus a guard that the `large` preset ships the new generalization defaults.
"""
from __future__ import annotations
import math
import torch

from neuroslm.train import cosine_lr, build_param_groups
from neuroslm.config import tiny, large, BrainConfig
from neuroslm.brain import Brain


# ── 1. LR schedule ──────────────────────────────────────────────────────────

def test_cosine_lr_warmup_is_linear():
    peak = 2.5e-4
    assert cosine_lr(0, 500, 10000, peak) == 0.0
    assert math.isclose(cosine_lr(250, 500, 10000, peak), peak * 0.5, rel_tol=1e-6)
    assert math.isclose(cosine_lr(500, 500, 10000, peak), peak, rel_tol=1e-6)


def test_cosine_lr_anneals_to_floor_by_horizon():
    """At the decay horizon the LR equals peak * min_ratio (the floor)."""
    peak = 2.5e-4
    lr_end = cosine_lr(10000, 500, 10000, peak, min_ratio=0.1)
    assert math.isclose(lr_end, peak * 0.1, rel_tol=1e-6)
    # min_ratio=0 → fully decays to ~0.
    assert cosine_lr(10000, 500, 10000, peak, min_ratio=0.0) < peak * 1e-3


def test_cosine_lr_horizon_is_decoupled_from_run_length():
    """The core fix: a 10k-targeted run must NOT leave the LR near peak.

    With the old behavior (horizon = --steps = 100000) the LR at step 10k is
    still ~98% of peak — the model never anneals. Targeting the horizon at
    10k drops it to the floor.
    """
    peak = 2.5e-4
    near_peak = cosine_lr(10000, 500, 100000, peak, min_ratio=0.0) / peak
    annealed  = cosine_lr(10000, 500, 10000, peak, min_ratio=0.1) / peak
    assert near_peak > 0.95          # old: barely moved
    assert annealed <= 0.11          # fix: at the floor
    assert near_peak - annealed > 0.8


def test_cosine_lr_clamps_past_horizon():
    peak = 2.5e-4
    # Past the horizon the cosine arg is clamped, so LR stays at the floor.
    assert math.isclose(cosine_lr(99999, 500, 10000, peak, min_ratio=0.1),
                        peak * 0.1, rel_tol=1e-6)


# ── 2. Dropout ──────────────────────────────────────────────────────────────

def _tiny(dropout: float) -> BrainConfig:
    c = tiny()
    c.vocab_size = 256
    c.dropout = dropout
    return c


def test_dropout_modules_active_only_when_configured():
    torch.manual_seed(0)
    b_on = Brain(_tiny(0.1))
    n_active = sum(1 for m in b_on.modules()
                   if isinstance(m, torch.nn.Dropout) and m.p > 0)
    assert n_active > 0, "dropout=0.1 should instantiate active Dropout layers"

    torch.manual_seed(0)
    b_off = Brain(_tiny(0.0))
    n_off = sum(1 for m in b_off.modules()
                if isinstance(m, torch.nn.Dropout) and m.p > 0)
    assert n_off == 0, "dropout=0.0 must add no active Dropout layers"


def test_dropout_adds_no_state_dict_keys():
    """Existing checkpoints (trained dropout-free) must still load: Dropout
    has no parameters, so the key set is identical regardless of dropout."""
    torch.manual_seed(0); k_off = set(Brain(_tiny(0.0)).state_dict().keys())
    torch.manual_seed(0); k_on  = set(Brain(_tiny(0.1)).state_dict().keys())
    assert k_off == k_on


def test_dropout_stochastic_in_train_deterministic_in_eval():
    torch.manual_seed(0)
    b = Brain(_tiny(0.3))   # high p so the effect is unmistakable
    ids = torch.randint(0, 256, (2, 16))
    tgt = torch.randint(0, 256, (2, 16))

    # eval(): two passes must be identical (dropout disabled).
    b.eval()
    with torch.no_grad():
        e1 = b.forward_lm(ids, tgt)["lm_loss"]
        e2 = b.forward_lm(ids, tgt)["lm_loss"]
    assert torch.allclose(e1, e2), "eval forward must be deterministic"

    # train(): the language logits should differ across passes.
    b.train()
    torch.manual_seed(1); l1 = b.language(ids)[0]
    torch.manual_seed(2); l2 = b.language(ids)[0]
    assert not torch.allclose(l1, l2), "train-mode dropout should perturb logits"


def test_dropout_forward_backward_runs():
    torch.manual_seed(0)
    b = Brain(_tiny(0.1)); b.train()
    ids = torch.randint(0, 256, (2, 16))
    tgt = torch.randint(0, 256, (2, 16))
    loss = b.forward_lm(ids, tgt)["loss"]
    loss.backward()
    assert torch.isfinite(loss)


# ── 3. Decoupled weight decay ───────────────────────────────────────────────

def test_build_param_groups_decoupled_splits_by_dim():
    torch.manual_seed(0)
    b = Brain(_tiny(0.0))
    groups = build_param_groups(b.named_parameters(), weight_decay=0.05,
                                decoupled=True)
    assert isinstance(groups, list) and len(groups) == 2
    decay, nodecay = groups
    assert decay["weight_decay"] == 0.05
    assert nodecay["weight_decay"] == 0.0
    # Every decayed param is a matrix; every exempt param is 1-D.
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in nodecay["params"])
    assert len(decay["params"]) > 0 and len(nodecay["params"]) > 0


def test_build_param_groups_legacy_is_flat_list():
    torch.manual_seed(0)
    b = Brain(_tiny(0.0))
    flat = build_param_groups(b.named_parameters(), weight_decay=0.05,
                              decoupled=False)
    assert isinstance(flat, list)
    assert all(isinstance(p, torch.nn.Parameter) for p in flat)


def test_build_param_groups_excludes_learned_opt():
    torch.manual_seed(0)
    b = Brain(_tiny(0.0))
    has_lopt = any(n.startswith("learned_opt.") for n, _ in b.named_parameters())
    groups = build_param_groups(b.named_parameters(), weight_decay=0.05,
                                decoupled=True)
    grouped = {id(p) for g in groups for p in g["params"]}
    lopt = {id(p) for n, p in b.named_parameters() if n.startswith("learned_opt.")}
    if has_lopt:
        assert grouped.isdisjoint(lopt), "learned_opt params must be excluded"


def test_param_groups_cover_all_trainable_model_params():
    """No trainable model param (outside learned_opt) is silently dropped."""
    torch.manual_seed(0)
    b = Brain(_tiny(0.0))
    groups = build_param_groups(b.named_parameters(), weight_decay=0.05,
                                decoupled=True)
    grouped = {id(p) for g in groups for p in g["params"]}
    expected = {id(p) for n, p in b.named_parameters()
                if not n.startswith("learned_opt.") and p.requires_grad}
    assert grouped == expected


# ── 4. Label smoothing ──────────────────────────────────────────────────────

def test_label_smoothing_changes_lm_loss():
    logits = torch.randn(2, 8, 256)
    tgt = torch.randint(0, 256, (2, 8))
    ce_plain = Brain._chunked_ce(logits, tgt, label_smoothing=0.0)
    ce_smooth = Brain._chunked_ce(logits, tgt, label_smoothing=0.1)
    assert ce_plain.shape == (2,)
    assert not torch.allclose(ce_plain, ce_smooth)


def test_label_smoothing_defaults_off():
    logits = torch.randn(2, 8, 256)
    tgt = torch.randint(0, 256, (2, 8))
    default = Brain._chunked_ce(logits, tgt)
    explicit_off = Brain._chunked_ce(logits, tgt, label_smoothing=0.0)
    assert torch.allclose(default, explicit_off)


# ── 5. Preset defaults ──────────────────────────────────────────────────────

def test_large_preset_ships_generalization_defaults():
    c = large()
    assert c.dropout == 0.1
    assert c.weight_decay == 0.05
    assert c.decoupled_wd is True
    assert c.min_lr_ratio == 0.1


def test_brainconfig_new_fields_have_safe_defaults():
    c = BrainConfig()
    # lr_decay_steps=0 → schedule falls back to the run's total --steps.
    assert c.lr_decay_steps == 0
    assert c.label_smoothing == 0.0
    assert c.decoupled_wd is True
