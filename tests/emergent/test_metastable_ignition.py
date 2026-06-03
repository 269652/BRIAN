"""Tests for C2 — MetastableIgnition."""
from __future__ import annotations
import torch

from neuroslm.emergent.metastable_ignition import MetastableIgnition


def test_init_validates_target_rate():
    import pytest
    with pytest.raises(ValueError):
        MetastableIgnition(target_rate=0.0)
    with pytest.raises(ValueError):
        MetastableIgnition(target_rate=1.0)


def test_peak_in_unit_interval():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32)
    p = MetastableIgnition.peak(x)
    assert 0.0 <= p <= 1.0


def test_peak_one_for_one_hot_input():
    x = torch.full((1, 1, 16), -10.0)
    x[..., 0] = 10.0
    p = MetastableIgnition.peak(x)
    assert p > 0.99


def test_rate_converges_to_target_under_random_peaks():
    """Under iid uniform peaks ∈ [0,1] the adaptive threshold should drive
    the firing rate toward `target_rate`. Tolerance is loose because the
    rate is measured from a finite EMA window."""
    torch.manual_seed(42)
    m = MetastableIgnition(target_rate=0.3, threshold_eta=0.2,
                           softness=0.02, history=400)
    for _ in range(6000):
        p = float(torch.rand(()).item())
        m.step(p)
    rate = m.stats()["ign_rate"]
    # Direction is the load-bearing assertion: rate started at 0.5 and
    # must have been driven toward 0.3.
    assert rate < 0.5
    assert abs(rate - 0.3) < 0.12


def test_ne_lowers_effective_threshold():
    """High NE should make ignition easier — same peak, more events."""
    torch.manual_seed(0)
    m_low = MetastableIgnition(target_rate=0.5, threshold_eta=0.0,
                               softness=0.05, ne_coupling=0.3)
    m_low.threshold = 0.5
    m_high = MetastableIgnition(target_rate=0.5, threshold_eta=0.0,
                                softness=0.05, ne_coupling=0.3)
    m_high.threshold = 0.5
    fires_low = 0
    fires_high = 0
    for _ in range(200):
        p = 0.45
        if m_low.step(p, ne=0.0)["ign_event"]:
            fires_low += 1
        if m_high.step(p, ne=1.0)["ign_event"]:
            fires_high += 1
    assert fires_high > fires_low


def test_stats_keys_complete():
    m = MetastableIgnition()
    s = m.step(0.5, ne=0.1)
    expected = {"ign_rate", "ign_strength", "ign_threshold",
                "ign_event", "ign_g", "ign_peak"}
    assert set(s) == expected


def test_event_is_zero_or_one():
    m = MetastableIgnition()
    for _ in range(50):
        s = m.step(float(torch.rand(()).item()))
        assert s["ign_event"] in (0, 1)


def test_threshold_stays_bounded():
    m = MetastableIgnition(threshold_eta=1.0)  # huge gain
    for _ in range(500):
        m.step(0.99)         # would push threshold up forever
    assert 0.0 <= m.threshold <= 1.0
