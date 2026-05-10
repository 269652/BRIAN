"""Orchestrator tests: gates, reentry, stage routing, Φ buffer hygiene."""
from __future__ import annotations
import math
import torch
import pytest

from neuroslm.intelligence.orchestrator import (
    NeuralOrchestrator, HomeostaticGate, LateralGridMixer,
    STAGE_SENSORY, STAGE_COGNITIVE_CTL, STAGE_GWS,
)


def test_homeostatic_gate_preserves_shape():
    g = HomeostaticGate(64, n_heads=4)
    x = torch.randn(2, 8, 64)
    y = g(x)
    assert y.shape == x.shape


def test_homeostatic_gate_2d_squeeze():
    g = HomeostaticGate(64, n_heads=4)
    x = torch.randn(2, 64)
    y = g(x)
    assert y.shape == (2, 64)


def test_homeostatic_gate_running_stats_updated():
    g = HomeostaticGate(32, n_heads=4)
    g.eval()
    n_before = int(g.n_updates.item())
    for _ in range(3):
        g(torch.randn(2, 4, 32))
    assert int(g.n_updates.item()) == n_before + 3


def test_lateral_mixer_preserves_shape():
    m = LateralGridMixer(64, n_heads=4)
    outs = [torch.randn(2, 64) for _ in range(4)]
    mixed = m(outs)
    assert len(mixed) == 4
    for o in mixed:
        assert o.shape == (2, 64)


def test_lateral_mixer_short_list_passthrough():
    m = LateralGridMixer(64, n_heads=4)
    one = [torch.randn(2, 64)]
    assert m(one) is one


def test_reentry_state_smoothed():
    o = NeuralOrchestrator(d_sem=32, module_names=["a"])
    pre = o._reentry_state.clone()
    o.update_reentry(torch.randn(4, 32))
    post = o._reentry_state.clone()
    # EMA α=0.15 means most weight stays on the prior (zero) on first step
    assert (post - pre).abs().sum() > 0
    assert int(o._reentry_count.item()) == 1


def test_reentry_bias_safe_under_inplace_update():
    """Issuing get_reentry_bias followed by update_reentry then backward
    on a downstream loss must not raise the 'modified by inplace' error."""
    o = NeuralOrchestrator(d_sem=32, module_names=["a"])
    bias = o.get_reentry_bias(2, torch.device("cpu"))
    # Trigger the inplace lerp_ that previously corrupted the saved view.
    o.update_reentry(torch.randn(2, 32))
    loss = bias.pow(2).sum()
    loss.backward()  # would raise if `bias` was a non-cloned view of the buffer


def test_gws_broadcast_lifecycle():
    o = NeuralOrchestrator(d_sem=32, module_names=["a", "b"])
    assert not bool(o._gws_broadcast_ready.item())
    o.set_gws_broadcast(torch.randn(4, 5, 32))   # (B, S, D)
    assert bool(o._gws_broadcast_ready.item())
    o.begin_pass()
    assert not bool(o._gws_broadcast_ready.item())


def test_begin_pass_clears_stage_buffer():
    o = NeuralOrchestrator(d_sem=32, module_names=["a", "b"])
    for _ in range(3):
        o.record_stage_output(torch.randn(2, 32))
    assert len(o._last_stage_outputs) == 3
    o.begin_pass()
    assert len(o._last_stage_outputs) == 0


def test_record_bounded_to_16():
    o = NeuralOrchestrator(d_sem=16, module_names=["a"])
    for _ in range(30):
        o.record_stage_output(torch.randn(2, 16))
    assert len(o._last_stage_outputs) == 16


def test_reset_fast_weights():
    o = NeuralOrchestrator(d_sem=32, module_names=["a"])
    o._hfw_states["x"] = torch.randn(2, 4, 8, 8)
    o.reset_fast_weights()
    assert o._hfw_states == {}


def test_baseline_mode_short_circuits():
    o = NeuralOrchestrator(d_sem=32, module_names=["a", "b"], baseline=True)
    sig = torch.randn(2, 32)
    out, met = o.route(sig, modules={"a": None, "b": None})
    assert torch.equal(out, sig)
    assert met["mode"] == "baseline"


def test_stage_routing_with_simple_module():
    o = NeuralOrchestrator(d_sem=16, module_names=[])
    # Register a no-op linear at stage 2
    mod = torch.nn.Linear(16, 16)
    o.register_module_brain("worker", 2, mod)
    sig = torch.randn(2, 16)
    out, met = o.route_stage(2, sig)
    assert out.shape == sig.shape
    assert met["stage"] == 2
    assert met["n_active"] >= 1


def test_stability_report_when_active():
    o = NeuralOrchestrator(d_sem=16, module_names=["x"], n_heads=2)
    rep = o.stability_report()
    assert "x_pre" in rep
    assert "x_post" in rep
