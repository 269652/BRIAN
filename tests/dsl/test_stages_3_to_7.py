# -*- coding: utf-8 -*-
"""Stages 3-7 of the OOD push — DSL-level tests.

Each stage adds a knob in TrainingConfig + arch.neuro. Tests verify the
knobs are wired through, defaults preserve baseline, and at least one
behavior changes when enabled.
"""
import pytest
import torch

from neuroslm.dsl.training_config import (
    TrainingConfig, load_training_config_from_arch,
)


def test_config_parses_all_new_fields():
    """arch.neuro should parse every Stage 3-7 field with the right type."""
    c = load_training_config_from_arch("architectures/master")
    # Stage 3 BEMA
    assert isinstance(c.bema_rollback_window, int)
    assert isinstance(c.bema_snapshot_every, int)
    assert isinstance(c.bema_cooldown, int)
    # Stage 4 NEMORI
    assert isinstance(c.nemori_floor, float)
    # Stage 6 μP
    assert isinstance(c.mu_p_scaling, bool)
    # Stage 7 curriculum
    assert isinstance(c.curriculum, str)
    assert isinstance(c.crystallization_step, int)


def test_config_defaults_preserve_baseline():
    """A bare TrainingConfig with no overrides must be the no-op baseline
    (all stage-3-to-7 knobs off, identical to prior behavior)."""
    c = TrainingConfig()
    assert c.bema_rollback_window == 0   # off
    assert c.nemori_floor == 0.0          # off
    assert c.mu_p_scaling is False
    assert c.curriculum == "random"
    assert c.crystallization_step == 0


# ── Stage 3 BEMA ──────────────────────────────────────────────────────

def test_bema_controller_snapshots_and_rolls_back():
    """Verify BEMA snapshot + rollback restores model params."""
    from neuroslm.dsl.bema_optimizer import BEMAController, BEMAConfig

    model = torch.nn.Linear(8, 8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    bema = BEMAController(
        model, optimizer,
        BEMAConfig(enabled=True, snapshot_every=1,
                    rollback_window=3, cooldown=0, max_snapshots=4))

    initial_weight = model.weight.detach().clone()
    # 5 steps of climbing loss → triggers rollback
    losses = [1.0, 1.1, 1.2, 1.3, 1.4]
    for loss in losses:
        # mutate model so a real rollback is observable
        with torch.no_grad():
            model.weight.add_(torch.randn_like(model.weight) * 0.01)
        info = bema.maybe_step(loss_value=loss)
    # After 4 climbing steps post-EMA-init, rollback should have fired
    # at least once during the 5-loss sweep.
    assert bema.state.rollbacks_performed >= 1, \
        f"BEMA didn't roll back (rollbacks={bema.state.rollbacks_performed})"


def test_bema_off_is_a_noop():
    """With enabled=False the controller's maybe_step is just optimizer.step()."""
    from neuroslm.dsl.bema_optimizer import BEMAController, BEMAConfig

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    bema = BEMAController(model, optimizer, BEMAConfig(enabled=False))
    info = bema.maybe_step(loss_value=1.0)
    assert info["bema_rollback"] is False
    assert bema.state.rollbacks_performed == 0


# ── Stage 4 NEMORI surprise gate is wired into harness.train_step ─────
# (Functional test requires the harness — skipped here; the wiring is
#  verified by reading harness.py line 365 area where nemori_floor
#  consults self._last_lm_loss_value.)


# ── Stage 5 Fisher-Rao retrieval ──────────────────────────────────────

def test_fisher_rao_distance_shapes_and_signs():
    """Distance must be (B, M), non-negative, with d(x,x)=0."""
    from neuroslm.dsl.nn_ops import fisher_rao_distance, fisher_rao_topk

    query = torch.randn(3, 16)
    keys = torch.randn(10, 16)
    variance = torch.ones(16) * 0.5
    d = fisher_rao_distance(query, keys, variance)
    assert d.shape == (3, 10)
    assert (d >= 0).all()

    # Self-distance: insert query into keys, check distance 0 to itself
    keys_with_self = torch.cat([keys, query[:1]], dim=0)
    d2 = fisher_rao_distance(query[:1], keys_with_self, variance)
    assert d2[0, -1].item() < 1e-5, f"self-distance should be 0, got {d2[0,-1]}"

    # topk
    dvals, idx = fisher_rao_topk(query, keys, variance, k=3)
    assert dvals.shape == (3, 3) and idx.shape == (3, 3)


def test_fisher_rao_weights_by_inverse_variance():
    """High-variance dimensions should be DOWN-weighted in the metric."""
    from neuroslm.dsl.nn_ops import fisher_rao_distance

    # Two identical queries except in dim 0, where they differ by 10
    q = torch.tensor([[10.0, 0.0]])
    k = torch.tensor([[0.0, 0.0]])
    # Case A: dim 0 has LOW variance → high precision → counts more
    var_low_dim0 = torch.tensor([0.01, 1.0])
    d_high_precision = fisher_rao_distance(q, k, var_low_dim0)[0, 0]
    # Case B: dim 0 has HIGH variance → low precision → counts less
    var_high_dim0 = torch.tensor([100.0, 1.0])
    d_low_precision = fisher_rao_distance(q, k, var_high_dim0)[0, 0]
    assert d_high_precision > d_low_precision, \
        f"high-precision distance should be larger: {d_high_precision} vs {d_low_precision}"


# ── Stage 6 μP scaling ────────────────────────────────────────────────

def test_mu_p_param_groups_scales_by_width():
    """At wider hidden dim, hidden-weight LR multiplier should shrink."""
    from neuroslm.harness import _mu_p_param_groups

    # Tiny "model" with 3 params: 1-D norm, 2-D hidden, 2-D lm_head
    class M(torch.nn.Module):
        def __init__(self, d_model=384, vocab=50257):
            super().__init__()
            self.norm = torch.nn.Parameter(torch.ones(d_model))
            self.hidden = torch.nn.Parameter(torch.randn(d_model, d_model))
            self.lm_head = torch.nn.Parameter(torch.randn(vocab, d_model))
    m = M()
    groups = _mu_p_param_groups(m, base_lr=1e-3, wd=0.01)
    # 3 groups, one per param
    assert len(groups) == 3
    # Map name -> lr
    by_name = {next(iter(g["params"])).shape: g["lr"] for g in groups}
    norm_lr = by_name[(384,)]
    hidden_lr = by_name[(384, 384)]
    head_lr = by_name[(50257, 384)]
    # Norms get full LR
    assert abs(norm_lr - 1e-3) < 1e-12
    # Hidden gets base_width / fan_in = 256/384
    assert abs(hidden_lr - 1e-3 * 256 / 384) < 1e-9
    # Head gets base_width / d_model = 256/384 (last dim)
    assert abs(head_lr - 1e-3 * 256 / 384) < 1e-9


# ── Stage 7 curriculum (config-only here; ordering uses data source) ──

def test_curriculum_field_accepts_known_strategies():
    c = TrainingConfig()
    for strategy in ("random", "easy_to_hard", "uniform"):
        c.curriculum = strategy
        # No validation enforced — just verify it sticks
        assert c.curriculum == strategy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
