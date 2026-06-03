# -*- coding: utf-8 -*-
"""TDD spec for PR-B: Cross-Distribution Gradient Alignment (CDGA).

CDGA performs gradient surgery against a frozen OOD anchor batch.

Reference: architectures/rcc_bowtie/lib/cdga.neuro  (math-first DSL)
            docs/CDGA.md                              (full design doc)

Pipeline contract
─────────────────
After the standard `loss.backward()` accumulates g_train into every
parameter's .grad slot, CDGAController.apply_surgery(model, anchor_fn)
performs:

    1. Snapshot g_train as a list of detached tensors.
    2. Zero grads.
    3. Run anchor_fn() → loss_anchor → backward() → g_anchor in .grad.
    4. Compute scalars dot, g2, c = max(0, -dot/g2).
    5. Write g_aligned = g_train - α·c·g_anchor into .grad.

The optimizer then proceeds normally.

This file tests CDGAController in isolation using a tiny MLP and
hand-crafted gradient scenarios that exercise:
  (T1) No-op when anchor disabled
  (T2) No-op when α = 0
  (T3) No-op when gradients aligned (dot ≥ 0)
  (T4) Subtraction when gradients oppose (dot < 0)
  (T5) Magnitude bounded by α · ||g_train||
  (T6) Warmup ramp on α
  (T7) Anchor refresh schedule
  (T8) Cosine telemetry published
  (T9) DSL parser accepts the block
  (T10) End-to-end on real autograd backward
"""
import pytest
import torch
import torch.nn as nn

from neuroslm.dsl.regularization import (
    CDGAConfig, parse_regularization_block)
from neuroslm.regularizers import CDGAController


# ── Toy fixtures ────────────────────────────────────────────────────

class _Tiny(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.lin(x)


def _set_grads(model: nn.Module, vec: torch.Tensor) -> None:
    """Write a flat tensor as the .grad of every model param."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = vec[offset:offset + n].view_as(p).clone()
        offset += n


def _flat_grads(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.flatten() for p in model.parameters()])


# ── Config + parser ─────────────────────────────────────────────────

class TestCDGAConfig:
    def test_disabled_by_default(self):
        cfg = CDGAConfig()
        assert cfg.enabled is False

    def test_default_alpha_max_is_one(self):
        cfg = CDGAConfig()
        assert cfg.alpha_max == pytest.approx(1.0)

    def test_default_warmup_steps_matches_outer(self):
        # Default warmup_steps for the CDGA module is independent of
        # the top-level regularization warmup (because the surgery
        # only makes sense once g_train is non-random).
        cfg = CDGAConfig()
        assert cfg.warmup_steps == 2000

    def test_default_refresh_every_is_modest(self):
        cfg = CDGAConfig()
        assert cfg.refresh_every == 4


class TestCDGAParser:
    def test_parser_accepts_block(self):
        body = """cdga: { enabled: true, alpha_max: 0.5,
                          warmup_steps: 1000, refresh_every: 8 }"""
        cfg = parse_regularization_block(body)
        assert cfg.cdga.enabled is True
        assert cfg.cdga.alpha_max == pytest.approx(0.5)
        assert cfg.cdga.warmup_steps == 1000
        assert cfg.cdga.refresh_every == 8


# ── Conflict coefficient maths ──────────────────────────────────────

class TestCDGAMath:
    """Test the pure scalar math in isolation."""

    def test_conflict_zero_when_aligned(self):
        g_train = torch.tensor([1.0, 2.0, 3.0])
        g_anchor = torch.tensor([0.5, 1.0, 1.5])  # parallel to g_train
        ctrl = CDGAController(CDGAConfig(enabled=True, alpha_max=1.0,
                                          warmup_steps=0))
        c = ctrl.conflict_coefficient(g_train, g_anchor)
        assert float(c) == pytest.approx(0.0, abs=1e-6)

    def test_conflict_zero_when_orthogonal(self):
        g_train = torch.tensor([1.0, 0.0])
        g_anchor = torch.tensor([0.0, 1.0])
        ctrl = CDGAController(CDGAConfig(enabled=True, alpha_max=1.0,
                                          warmup_steps=0))
        c = ctrl.conflict_coefficient(g_train, g_anchor)
        assert float(c) == pytest.approx(0.0, abs=1e-6)

    def test_conflict_positive_when_opposed(self):
        g_train = torch.tensor([1.0, 2.0])
        g_anchor = torch.tensor([-1.0, -2.0])  # anti-parallel
        ctrl = CDGAController(CDGAConfig(enabled=True, alpha_max=1.0,
                                          warmup_steps=0))
        c = ctrl.conflict_coefficient(g_train, g_anchor)
        # dot = -5, g2 = 5, c = max(0, -(-5)/5) = 1.0
        assert float(c) == pytest.approx(1.0)

    def test_aligned_gradient_subtracts_conflict(self):
        g_train = torch.tensor([1.0, 0.0])
        g_anchor = torch.tensor([-1.0, 0.0])
        ctrl = CDGAController(CDGAConfig(enabled=True, alpha_max=1.0,
                                          warmup_steps=0))
        c = ctrl.conflict_coefficient(g_train, g_anchor)
        # g_aligned = g_train - 1 * 1 * g_anchor = [1, 0] - [-1, 0] = [2, 0]
        g_al = g_train - 1.0 * c * g_anchor
        assert torch.allclose(g_al, torch.tensor([2.0, 0.0]))

    def test_cosine_published(self):
        g_train = torch.tensor([1.0, 1.0])
        g_anchor = torch.tensor([1.0, 0.0])
        ctrl = CDGAController(CDGAConfig(enabled=True, alpha_max=1.0,
                                          warmup_steps=0))
        cos = ctrl.cosine_similarity(g_train, g_anchor)
        # cos = 1 / (√2 * 1) ≈ 0.7071
        assert float(cos) == pytest.approx(1.0 / 2.0 ** 0.5, abs=1e-4)


# ── Strength schedule ───────────────────────────────────────────────

class TestCDGAWarmup:
    def test_alpha_starts_at_zero(self):
        ctrl = CDGAController(
            CDGAConfig(enabled=True, alpha_max=1.0, warmup_steps=100))
        assert ctrl.current_alpha() == pytest.approx(0.0)

    def test_alpha_ramps_linearly(self):
        ctrl = CDGAController(
            CDGAConfig(enabled=True, alpha_max=1.0, warmup_steps=100))
        for _ in range(50):
            ctrl._tick()
        assert ctrl.current_alpha() == pytest.approx(0.5, abs=1e-3)

    def test_alpha_caps_at_max(self):
        ctrl = CDGAController(
            CDGAConfig(enabled=True, alpha_max=0.7, warmup_steps=10))
        for _ in range(100):
            ctrl._tick()
        assert ctrl.current_alpha() == pytest.approx(0.7)

    def test_alpha_zero_when_warmup_zero(self):
        """warmup_steps=0 means immediate full strength."""
        ctrl = CDGAController(
            CDGAConfig(enabled=True, alpha_max=1.0, warmup_steps=0))
        assert ctrl.current_alpha() == pytest.approx(1.0)


# ── apply_surgery on a real module ──────────────────────────────────

class TestCDGAApplySurgery:
    def _make_model_with_grad(self, grad_vec):
        m = _Tiny(d=2)
        # Pre-set g_train so we can call apply_surgery with a stub
        # anchor function that overwrites with the anchor grad.
        _set_grads(m, grad_vec)
        return m

    def test_no_op_when_disabled(self):
        ctrl = CDGAController(CDGAConfig(enabled=False))
        m = self._make_model_with_grad(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        before = _flat_grads(m).clone()
        out = ctrl.apply_surgery(m, anchor_loss_fn=lambda: None)
        after = _flat_grads(m)
        assert torch.allclose(before, after)
        assert out["applied"] is False

    def test_no_op_when_alpha_zero(self):
        ctrl = CDGAController(
            CDGAConfig(enabled=True, alpha_max=0.0, warmup_steps=0))
        m = self._make_model_with_grad(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        before = _flat_grads(m).clone()

        def anchor_fn():
            # Write some opposing grad
            _set_grads(m, torch.tensor([-1.0, -2.0, -3.0, -4.0]))
            return torch.tensor(0.0)
        out = ctrl.apply_surgery(m, anchor_loss_fn=anchor_fn)
        after = _flat_grads(m)
        assert torch.allclose(before, after)

    def test_subtracts_conflict_when_opposed(self):
        ctrl = CDGAController(
            CDGAConfig(enabled=True, alpha_max=1.0, warmup_steps=0))
        m = self._make_model_with_grad(torch.tensor([1.0, 1.0, 1.0, 1.0]))

        def anchor_fn():
            _set_grads(m, torch.tensor([-1.0, -1.0, -1.0, -1.0]))
            return torch.tensor(0.0)
        out = ctrl.apply_surgery(m, anchor_loss_fn=anchor_fn)
        after = _flat_grads(m)
        # g_train=[1,1,1,1], g_anchor=[-1,-1,-1,-1]
        # dot = -4, g2 = 4, c = max(0, -(-4)/4) = 1.0
        # g_aligned = [1,1,1,1] - 1*1*[-1,-1,-1,-1] = [2,2,2,2]
        assert torch.allclose(after, torch.tensor([2.0, 2.0, 2.0, 2.0]))
        assert out["applied"] is True
        assert out["conflict_coef"] == pytest.approx(1.0)

    def test_noop_when_aligned(self):
        ctrl = CDGAController(
            CDGAConfig(enabled=True, alpha_max=1.0, warmup_steps=0))
        m = self._make_model_with_grad(torch.tensor([1.0, 1.0, 1.0, 1.0]))

        def anchor_fn():
            _set_grads(m, torch.tensor([0.5, 0.5, 0.5, 0.5]))
            return torch.tensor(0.0)
        out = ctrl.apply_surgery(m, anchor_loss_fn=anchor_fn)
        after = _flat_grads(m)
        assert torch.allclose(after, torch.tensor([1.0, 1.0, 1.0, 1.0]))
        assert out["conflict_coef"] == pytest.approx(0.0)

    def test_perturbation_bounded(self):
        """Property P2: ||g_aligned - g_train|| ≤ α·||g_train||."""
        ctrl = CDGAController(
            CDGAConfig(enabled=True, alpha_max=1.0, warmup_steps=0))
        torch.manual_seed(42)
        for _ in range(20):
            g_train = torch.randn(4)
            g_anchor = torch.randn(4)
            m = self._make_model_with_grad(g_train.clone())

            def anchor_fn(ga=g_anchor):
                _set_grads(m, ga.clone())
                return torch.tensor(0.0)
            ctrl.apply_surgery(m, anchor_loss_fn=anchor_fn)
            after = _flat_grads(m)
            delta = (after - g_train).norm()
            bound = 1.0 * g_train.norm()
            assert float(delta) <= float(bound) + 1e-5

    def test_refresh_schedule_skips_steps(self):
        """When refresh_every > 1, only every Nth step runs the anchor
        forward (the rest reuse cached g_anchor or skip)."""
        ctrl = CDGAController(
            CDGAConfig(enabled=True, alpha_max=1.0, warmup_steps=0,
                       refresh_every=4))
        m = self._make_model_with_grad(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        call_count = {"n": 0}

        def anchor_fn():
            call_count["n"] += 1
            _set_grads(m, torch.tensor([-1.0, -1.0, -1.0, -1.0]))
            return torch.tensor(0.0)

        # Run 8 surgery passes; expect anchor_fn called every 4 → 2 times
        for _ in range(8):
            m.zero_grad()
            _set_grads(m, torch.tensor([1.0, 1.0, 1.0, 1.0]))
            ctrl.apply_surgery(m, anchor_loss_fn=anchor_fn)
        assert call_count["n"] == 2  # called on tick 0 and tick 4

    def test_telemetry_has_cosine(self):
        ctrl = CDGAController(
            CDGAConfig(enabled=True, alpha_max=1.0, warmup_steps=0))
        m = self._make_model_with_grad(torch.tensor([1.0, 0.0, 0.0, 0.0]))

        def anchor_fn():
            _set_grads(m, torch.tensor([1.0, 0.0, 0.0, 0.0]))  # same dir
            return torch.tensor(0.0)
        out = ctrl.apply_surgery(m, anchor_loss_fn=anchor_fn)
        assert "cosine" in out
        assert out["cosine"] == pytest.approx(1.0, abs=1e-5)


class TestCDGAEndToEnd:
    """Run apply_surgery against real autograd-backed grads."""

    def test_real_backward_then_surgery(self):
        torch.manual_seed(0)
        model = _Tiny(d=4)
        x_train = torch.randn(2, 4)
        y_train = torch.randn(2, 4)
        x_anchor = torch.randn(2, 4)
        y_anchor = torch.randn(2, 4)

        # Standard training step
        loss_t = ((model(x_train) - y_train) ** 2).mean()
        loss_t.backward()
        g_train_snapshot = _flat_grads(model).clone()

        ctrl = CDGAController(
            CDGAConfig(enabled=True, alpha_max=1.0, warmup_steps=0))

        def anchor_fn():
            model.zero_grad()
            loss_a = ((model(x_anchor) - y_anchor) ** 2).mean()
            loss_a.backward()
            return loss_a

        out = ctrl.apply_surgery(model, anchor_loss_fn=anchor_fn)
        g_after = _flat_grads(model)

        # The surgery must produce a valid gradient (finite, same shape)
        assert torch.isfinite(g_after).all()
        assert g_after.shape == g_train_snapshot.shape
        # And the perturbation bound P2 must hold
        delta = (g_after - g_train_snapshot).norm()
        bound = 1.0 * g_train_snapshot.norm()
        assert float(delta) <= float(bound) + 1e-4
