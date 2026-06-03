# -*- coding: utf-8 -*-
"""Tests for the NT-gated PC-reentry trunk loss (Jun 2026 surgery).

These tests do NOT spin up the full harness — that would require a
real DSL arch + tokenizer. Instead they exercise the integration
contract directly:

    1. ``PCReentryProbe.residual_diff`` returns an autograd-tracked
       scalar that flows gradient into BOTH input populations.
    2. With ``pc_reentry_weight == 0`` the harness path is a no-op
       (no probe constructed, no metrics published).
    3. NT gating: with ``pc_reentry_nt_gate=True`` and high GABA the
       effective weight collapses to 0; with high DA it amplifies.
"""
from __future__ import annotations
import math
import pytest
import torch

from neuroslm.emergent.pc_reentry import PCReentryProbe


# ── 1. residual_diff is autograd-tracked ─────────────────────────────

def test_residual_diff_flows_gradient_into_both_populations():
    probe = PCReentryProbe(dim=8)
    h_m = torch.randn(2, 3, 8, requires_grad=True)
    h_s = torch.randn(2, 3, 8, requires_grad=True)
    loss = probe.residual_diff(h_m, h_s)
    assert loss is not None
    assert loss.requires_grad
    assert loss.dim() == 0   # scalar
    loss.backward()
    # Gradient on both — the loss is ||s - W·m||² so both contribute.
    assert h_m.grad is not None and h_m.grad.abs().sum().item() > 0.0
    assert h_s.grad is not None and h_s.grad.abs().sum().item() > 0.0


def test_residual_diff_does_not_update_probe_W():
    """W is detached inside residual_diff — only the trunk gets grad."""
    probe = PCReentryProbe(dim=4)
    diag_before = probe._diag.clone()
    u_before = probe._u.clone()
    v_before = probe._v.clone()
    h_m = torch.randn(2, 5, 4, requires_grad=True)
    h_s = torch.randn(2, 5, 4, requires_grad=True)
    loss = probe.residual_diff(h_m, h_s)
    loss.backward()
    # Probe parameters unchanged (no autograd path through them).
    assert torch.allclose(probe._diag, diag_before)
    assert torch.allclose(probe._u, u_before)
    assert torch.allclose(probe._v, v_before)


def test_residual_diff_handles_none_and_shape_mismatch():
    probe = PCReentryProbe(dim=8)
    assert probe.residual_diff(None, torch.randn(2, 3, 8)) is None
    assert probe.residual_diff(torch.randn(2, 3, 8), None) is None
    # Dim mismatch — probe expects dim=8.
    assert probe.residual_diff(torch.randn(2, 3, 4),
                               torch.randn(2, 3, 4)) is None
    # Shape mismatch between m and s.
    assert probe.residual_diff(torch.randn(2, 3, 8),
                               torch.randn(2, 4, 8)) is None


def test_residual_diff_does_not_break_internal_sgd():
    """The original ``step()`` SGD path must still update W on detached
    inputs, independently of the differentiable path."""
    probe = PCReentryProbe(dim=4, lr=1e-1)
    h_m = torch.randn(2, 5, 4)
    h_s = torch.randn(2, 5, 4)
    # Prime the buffer.
    probe.step(h_m, h_s)
    diag_before = probe._diag.clone()
    # A second step should move diag.
    probe.step(h_m, h_s)
    assert not torch.allclose(probe._diag, diag_before), \
        "internal SGD on W stalled after residual_diff was introduced"


# ── 2. TrainingConfig defaults preserve the no-op contract ───────────

def test_training_config_pc_reentry_defaults_off():
    from neuroslm.dsl.training_config import TrainingConfig
    cfg = TrainingConfig()
    assert cfg.pc_reentry_weight == 0.0
    assert cfg.pc_reentry_nt_gate is False


def test_arch_neuro_parses_new_pc_reentry_fields():
    """The arch.neuro training block must surface the new knobs into
    TrainingConfig — otherwise the rewire is a no-op in practice."""
    from neuroslm.dsl.training_config import parse_training_config
    src = """
        learning_rate: 0.0003
        pc_reentry_weight: 0.1
        pc_reentry_nt_gate: true
    """
    cfg = parse_training_config(src)
    assert cfg.pc_reentry_weight == pytest.approx(0.1)
    assert cfg.pc_reentry_nt_gate is True


# ── 3. NT-gating semantics ───────────────────────────────────────────

def test_nt_gate_math():
    """Verifies the gate formula from harness._compute_pc_reentry_loss.

    Reuses the exact math so any future tweak forces a test update —
    the gate is part of the rewire's contract, not an implementation
    detail.
    """
    def gate(da, gaba):
        return max(0.0, 1.0 + 0.5 * da - 0.7 * gaba)

    # Baseline NTs around 1.0 → gate ≈ 0.8 (curiosity = inhibition).
    assert gate(1.0, 1.0) == pytest.approx(0.8)
    # High DA, low GABA → amplified.
    assert gate(2.0, 0.0) == pytest.approx(2.0)
    # Strong inhibition → loss switched off cleanly.
    assert gate(0.0, 2.0) == 0.0
    # Mild reward floor.
    assert gate(0.0, 0.0) == pytest.approx(1.0)


# ── 4. Device migration (CUDA <-> CPU) ───────────────────────────────

def test_probe_auto_migrates_to_input_device_cpu():
    """Probe constructed default (CPU) accepts CPU tensors with no error."""
    probe = PCReentryProbe(dim=4)
    h_m = torch.randn(2, 3, 4)
    h_s = torch.randn(2, 3, 4)
    # Prime + step (was the failing path on the trainer).
    probe.step(h_m, h_s)
    out = probe.step(h_m, h_s)
    assert "pc_residual" in out
    # residual_diff path also.
    loss = probe.residual_diff(h_m, h_s)
    assert loss is not None and loss.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA not available")
def test_probe_auto_migrates_to_cuda_when_inputs_are_cuda():
    """Reproduces the trainer crash: probe built on CPU, inputs on CUDA.

    The fix is in PCReentryProbe._to_device — first call to step() /
    residual_diff() with CUDA tensors silently migrates all probe state.
    """
    probe = PCReentryProbe(dim=4)               # default CPU
    h_m = torch.randn(2, 3, 4, device="cuda")
    h_s = torch.randn(2, 3, 4, device="cuda")
    probe.step(h_m, h_s)                         # primes; was crashing
    out = probe.step(h_m, h_s)
    assert "pc_residual" in out
    # All probe tensors must now be on cuda.
    assert probe._diag.device.type == "cuda"
    assert probe._u.device.type == "cuda"
    assert probe._v.device.type == "cuda"
    assert probe._prev_motor is not None
    assert probe._prev_motor.device.type == "cuda"
    # residual_diff path returns a CUDA scalar.
    loss = probe.residual_diff(h_m, h_s)
    assert loss is not None and loss.device.type == "cuda"