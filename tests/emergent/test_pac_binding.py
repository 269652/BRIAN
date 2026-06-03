"""Tests for C6 — PACBindingProbe."""
from __future__ import annotations
import math
import torch

from neuroslm.emergent.pac_binding import PACBindingProbe


def test_init_validates():
    import pytest
    with pytest.raises(ValueError):
        PACBindingProbe(window=8)
    with pytest.raises(ValueError):
        PACBindingProbe(n_phase_bins=2)


def test_empty_buffer_returns_zero():
    p = PACBindingProbe(window=64)
    s = p.compute()
    assert s["pac"] == 0.0


def test_pac_near_zero_for_white_noise():
    torch.manual_seed(0)
    p = PACBindingProbe(window=512, n_phase_bins=18)
    for _ in range(512):
        p.observe(float(torch.randn(()).item()))
    s = p.compute()
    # Pure noise: very little coupling.
    assert s["pac"] < 0.2


def test_pac_high_for_synthetic_modulated_signal():
    """Build a signal cos(θ_t) + A_t·γ_t with A_t = 1 + cos(θ_t)
    (theta carrier + theta-modulated gamma amplitude). PAC of gamma
    envelope onto theta phase should be large."""
    p = PACBindingProbe(window=2048, n_phase_bins=18,
                        theta=(0.01, 0.05), gamma=(0.2, 0.45))
    T = 2048
    t = torch.arange(T, dtype=torch.float32)
    theta = 2 * math.pi * 0.02 * t       # 0.02 cyc/sample → 0.04 of Nyquist
    gamma = torch.sin(2 * math.pi * 0.15 * t)   # 0.15 cyc/sample → 0.30 of Nyquist
    amp = 1.0 + torch.cos(theta)
    # Theta carrier so the theta band has phase info, plus modulated gamma.
    sig = torch.cos(theta) + amp * gamma
    for v in sig.tolist():
        p.observe(v)
    s = p.compute()
    assert s["pac"] > 0.1


def test_pac_in_unit_interval():
    p = PACBindingProbe(window=256)
    for _ in range(256):
        p.observe(float(torch.randn(()).item()) * 100.0)
    s = p.compute()
    assert 0.0 <= s["pac"] <= 1.0
    assert -math.pi <= s["pac_pref_phase"] <= math.pi


def test_pac_invariant_to_dc_offset():
    """Adding a constant to every sample must not change PAC."""
    torch.manual_seed(0)
    base = [float(x) for x in torch.randn(256).tolist()]
    p1 = PACBindingProbe(window=256)
    p2 = PACBindingProbe(window=256)
    for x in base:
        p1.observe(x)
        p2.observe(x + 1000.0)
    s1 = p1.compute()
    s2 = p2.compute()
    assert abs(s1["pac"] - s2["pac"]) < 1e-4


def test_stats_returns_last_computed():
    p = PACBindingProbe(window=64)
    for _ in range(64):
        p.observe(0.0)
    s_compute = p.compute()
    s_stats = p.stats()
    assert s_compute == s_stats
