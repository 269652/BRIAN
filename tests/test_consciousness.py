"""Tests for ConsciousnessMetrics / estimate_fiedler."""
from __future__ import annotations
import math
import torch
import pytest

from neuroslm.modules.consciousness import ConsciousnessMetrics, estimate_fiedler


def test_estimate_fiedler_basic():
    outs = {f"m{i}": torch.randn(2, 16) for i in range(6)}
    val, vec = estimate_fiedler(outs)
    assert math.isfinite(val)
    assert 0.0 <= val <= 2.0
    assert vec is None or vec.shape[0] == 6


def test_estimate_fiedler_too_few_modules():
    outs = {"a": torch.randn(2, 16)}
    val, vec = estimate_fiedler(outs)
    assert val == 0.0
    assert vec is None


def test_consciousness_update_returns_metrics():
    cm = ConsciousnessMetrics(d_sem=16, n_modules=4)
    outs = {f"m{i}": torch.randn(2, 16) for i in range(4)}
    gws_slots = torch.randn(2, 4, 16)
    thought = torch.randn(2, 16)
    novelty = torch.rand(2)
    routing = torch.softmax(torch.randn(2, 4), dim=-1)
    metrics = cm.update(module_outputs=outs, gws_slots=gws_slots,
                        floating_thought=thought, novelty=novelty,
                        routing=routing)
    for k in ("gamma", "theta", "alpha", "phi", "coherence",
              "ignition", "metacognition", "binding", "tick"):
        assert k in metrics


def test_phi_enumerate_vs_spectral_consistent_for_n_lte_8():
    """For n ≤ 8 enumerate runs; for n > 8 spectral; ensure both return finite."""
    cm = ConsciousnessMetrics(d_sem=16, n_modules=4)
    small = {f"m{i}": torch.randn(2, 16) for i in range(5)}
    large = {f"m{i}": torch.randn(2, 16) for i in range(10)}
    phi_s = cm._compute_phi_mip(small, torch.device("cpu"))
    phi_l = cm._compute_phi_mip(large, torch.device("cpu"))
    assert math.isfinite(phi_s) and phi_s >= 0.0
    assert math.isfinite(phi_l) and phi_l >= 0.0


def test_oscillation_spectrum_summary():
    cm = ConsciousnessMetrics(d_sem=16, n_modules=4, history_len=8)
    outs = {f"m{i}": torch.randn(2, 16) for i in range(4)}
    for _ in range(4):
        cm.update(outs, torch.randn(2, 4, 16), torch.randn(2, 16),
                  torch.rand(2), torch.softmax(torch.randn(2, 4), dim=-1))
    spec = cm.oscillation_spectrum()
    for k in ("gamma_mean", "theta_mean", "alpha_mean",
              "phi_mean", "coherence_mean"):
        assert k in spec
        assert math.isfinite(spec[k])
