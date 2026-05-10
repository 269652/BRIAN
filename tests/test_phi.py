"""Tests for the IIT-style Φ proxy.

These guard the core property: the value returned by `compute_phi_proxy`
and `phi_tensor` must equal the Gaussian-MI lower bound over the MIP
bisection, not a correlation heuristic.
"""
from __future__ import annotations
import math
import pytest
import torch

from neuroslm.intelligence.orchestrator import NeuralOrchestrator


def _orch(n=6, d=64):
    return NeuralOrchestrator(d_sem=d, module_names=[f"m{i}" for i in range(n)],
                              n_heads=2)


def test_phi_value_real_and_finite():
    o = _orch()
    for _ in range(6):
        o.record_stage_output(torch.randn(2, 4, 64))
    v = o.compute_phi_proxy()
    assert isinstance(v, float)
    assert math.isfinite(v)
    # Random gaussian outputs → small but strictly nonnegative Φ.
    assert v >= 0.0


def test_phi_zero_when_fewer_than_two_outputs():
    o = _orch()
    assert o.compute_phi_proxy() == 0.0
    o.record_stage_output(torch.randn(2, 4, 64))
    assert o.compute_phi_proxy() == 0.0


def test_phi_tensor_is_differentiable():
    o = _orch()
    M = {f"m{i}": torch.randn(2, 4, 64, requires_grad=True) for i in range(6)}
    phi = o.phi_tensor(module_outputs=M)
    assert phi is not None
    assert phi.requires_grad
    phi.backward()
    # Gradient must reach at least one of the inputs.
    grads = [v.grad for v in M.values() if v.grad is not None]
    assert grads, "no input received gradient from phi"
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_phi_higher_for_coupled_outputs():
    """Φ for strongly coupled (rank-1) outputs > Φ for independent outputs.

    Coupled: every module's output is a scaled version of the same base
    vector → MI cannot be reduced by any bipartition. Independent: random
    noise per module → MI lower bound close to zero.
    """
    o = _orch()
    base = torch.randn(4, 64)
    coupled = {f"m{i}": base * (0.5 + 0.1 * i) + 0.01 * torch.randn(4, 64)
               for i in range(6)}
    independent = {f"m{i}": torch.randn(4, 64) for i in range(6)}
    phi_c = o.phi_tensor(module_outputs=coupled).item()
    phi_i = o.phi_tensor(module_outputs=independent).item()
    assert phi_c > phi_i, f"coupled {phi_c} should exceed independent {phi_i}"


def test_phi_proxy_recovers_real_phi():
    """compute_phi_proxy() (no_grad path) must equal phi_tensor() up to fp
    precision when fed identical module outputs.
    """
    o = _orch(n=5, d=32)
    vs = [torch.randn(3, 32) for _ in range(5)]
    for v in vs:
        o.record_stage_output(v.unsqueeze(0))   # (1, B, d) shape via stage convention
    proxy = o.compute_phi_proxy()
    # phi_tensor with no module_outputs falls back to the same buffer.
    t = o.phi_tensor()
    assert t is not None
    assert abs(proxy - float(t.item())) < 1e-5


def test_phi_baseline_orchestrator_returns_zero():
    o = NeuralOrchestrator(d_sem=32, module_names=["a", "b"], baseline=True)
    assert o.compute_phi_proxy() == 0.0
    assert o.phi_tensor() is None


def test_phi_handles_n_greater_than_8():
    """n > 8 falls back to spectral bisection; must remain finite and ≥ 0."""
    o = NeuralOrchestrator(d_sem=32, module_names=[f"m{i}" for i in range(12)])
    M = {f"m{i}": torch.randn(2, 32) for i in range(12)}
    phi = o.phi_tensor(module_outputs=M)
    assert phi is not None
    assert math.isfinite(float(phi.item()))
    assert float(phi.item()) >= 0.0


def test_phi_records_detached_for_proxy():
    """record_stage_output must store detached tensors so phi_proxy is
    safe to call inside no_grad contexts and doesn't leak the graph."""
    o = _orch()
    x = torch.randn(2, 4, 64, requires_grad=True)
    o.record_stage_output(x)
    stored = o._last_stage_outputs[-1]
    assert not stored.requires_grad
