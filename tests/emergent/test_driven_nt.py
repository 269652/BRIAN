"""Tests for C1 — DrivenNTSystem.

Guards:
  * channels stay in [0, 1] under any input
  * each channel responds to its intended driver in the expected direction
  * compat shim `step(activity=...)` behaves like the legacy NTSystem
  * baseline-recovery: drivers absent → channels relax toward baseline
"""
from __future__ import annotations
import math
import pytest

from neuroslm.emergent.driven_nt import DrivenNTSystem


def _all_in_unit(d):
    return all(0.0 <= v <= 1.0 for v in d.values())


def test_initial_levels_match_baselines():
    nt = DrivenNTSystem()
    lvl = nt.levels()
    assert lvl == nt.baselines
    assert _all_in_unit(lvl)


def test_channels_bounded_under_extreme_drivers():
    nt = DrivenNTSystem()
    for _ in range(50):
        nt.step_full(loss=1e6, grad_norm=1e6, activation=1e6,
                     ignition_rate=1.0, attn_entropy_norm=0.0)
        assert _all_in_unit(nt.levels())
    for _ in range(50):
        nt.step_full(loss=-1e6, grad_norm=0.0, activation=0.0,
                     ignition_rate=0.0, attn_entropy_norm=1.0)
        assert _all_in_unit(nt.levels())


def test_ne_rises_with_grad_norm():
    nt = DrivenNTSystem()
    # Drive with small gnorm for a while, then a big one.
    for _ in range(30):
        nt.step_full(grad_norm=0.1)
    low = nt.levels()["NE"]
    for _ in range(30):
        nt.step_full(grad_norm=5.0)
    high = nt.levels()["NE"]
    assert high > low + 0.05


def test_gaba_inhibits_when_workspace_saturates():
    nt = DrivenNTSystem()
    nt.step_full(ignition_rate=0.0)
    high_gaba = nt.levels()["GABA"]
    nt.step_full(ignition_rate=1.0)
    low_gaba = nt.levels()["GABA"]
    assert high_gaba > low_gaba


def test_ach_rises_with_sharp_attention():
    nt = DrivenNTSystem()
    nt.step_full(attn_entropy_norm=1.0)        # diffuse → ACh low
    diffuse = nt.levels()["ACh"]
    nt.step_full(attn_entropy_norm=0.0)        # sharp → ACh high
    sharp = nt.levels()["ACh"]
    assert sharp > diffuse


def test_da_spikes_on_negative_surprise():
    nt = DrivenNTSystem()
    # Plateau at high loss → fill the window.
    for _ in range(40):
        nt.step_full(loss=5.0)
    # Then a sudden drop ("good" surprise).
    nt.step_full(loss=2.0)
    da_good = nt.levels()["DA"]
    # And a sudden jump ("bad" surprise).
    for _ in range(40):
        nt.step_full(loss=5.0)
    nt.step_full(loss=8.0)
    da_bad = nt.levels()["DA"]
    assert da_good > da_bad


def test_compat_shim_step_relaxes_other_channels():
    """Legacy step(activity=...) only drives Glu/eCB; others relax."""
    nt = DrivenNTSystem()
    # Force NE far from baseline via step_full first.
    for _ in range(20):
        nt.step_full(grad_norm=10.0)
    pushed = nt.levels()["NE"]
    base_ne = nt.baselines["NE"]
    assert pushed > base_ne + 0.1
    # Many shim-step calls without driver should relax it back.
    for _ in range(200):
        nt.step(activity=0.1)
    final = nt.levels()["NE"]
    # Should have moved at least halfway back to baseline.
    assert abs(final - base_ne) < abs(pushed - base_ne) / 2


def test_levels_returns_fresh_dict():
    nt = DrivenNTSystem()
    a = nt.levels()
    a["DA"] = 999.0
    b = nt.levels()
    assert b["DA"] != 999.0


def test_keys_match_legacy_baselines():
    """Schema parity with metrics.NTSystem (so the log column doesn't change)."""
    nt = DrivenNTSystem()
    expected = {"DA", "NE", "5HT", "ACh", "eCB", "Glu", "GABA"}
    assert set(nt.levels()) == expected


def test_5ht_rises_as_long_term_loss_falls():
    nt = DrivenNTSystem()
    for _ in range(200):
        nt.step_full(loss=10.0)
    early = nt.levels()["5HT"]
    for _ in range(800):
        nt.step_full(loss=2.0)
    late = nt.levels()["5HT"]
    assert late > early + 0.05


def test_custom_baselines_respected():
    nt = DrivenNTSystem(baselines={"DA": 0.5})
    assert math.isclose(nt.baselines["DA"], 0.5)
    assert math.isclose(nt.levels()["DA"], 0.5)
