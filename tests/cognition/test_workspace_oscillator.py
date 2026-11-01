# -*- coding: utf-8 -*-
"""Oscillatory global workspace (architecture.md §14.13).

A Kuramoto phase model integrated in sub-tick steps (1 ms) over the
runtime's modules. Each module contributes a population of gamma-band
oscillators whose participation is scaled by that module's activation
this tick. Ignition = the global order parameter
R = |Σ a_j e^{iθ_j}| / N staying above threshold over the last window:
many oscillators both active AND phase-locked. ACh scales the coupling
and GABA slows the frequency; ignition feeds back into both through
DrivenNTSystem, closing a second-order loop.

These contracts pin the mathematics (Kuramoto critical coupling, the
spectral peak, NT modulation, determinism), not shapes.
"""
from __future__ import annotations

import math

import pytest

ALL_ON = {"sensory": 1.0, "memory": 1.0, "language": 1.0,
          "selection": 1.0, "novelty": 1.0}
BASE_NT = {"ACh": 0.30, "GABA": 0.15}


def _ws(**kw):
    from neuroslm.cognition.oscillator import GammaWorkspace
    base = dict(n_per_module=40, f0_hz=40.0, f_sd_hz=4.0, noise=0.0,
                ms_per_tick=1000, seed=0)
    base.update(kw)
    return GammaWorkspace(**base)


def _settle(ws, act=ALL_ON, nt=BASE_NT, ticks=2):
    out = None
    for _ in range(ticks):
        out = ws.step(act, nt)
    return out


class TestKuramotoLimits:
    def test_critical_coupling_matches_gaussian_formula(self):
        """Kc = 2 / (π g(0)) for a Gaussian of angular sd σ, i.e.
        Kc = 2σ√(2π)/π (Kuramoto 1975)."""
        ws = _ws()
        sigma = 2 * math.pi * 4.0
        assert ws.critical_coupling == pytest.approx(
            2 * sigma * math.sqrt(2 * math.pi) / math.pi)

    def test_subcritical_coupling_stays_incoherent(self):
        out = _settle(_ws(coupling_ratio=0.4))
        assert out.sync_r < 0.3
        assert not out.ignited

    def test_supercritical_coupling_phase_locks_and_ignites(self):
        out = _settle(_ws(coupling_ratio=3.0))
        assert out.sync_r > 0.8
        assert out.ignited

    def test_synchrony_increases_with_coupling(self):
        rs = [_settle(_ws(coupling_ratio=k)).sync_r for k in (0.5, 1.5, 3.0)]
        assert rs[0] < rs[1] < rs[2]


class TestSpectrum:
    def test_locked_population_peaks_in_gamma_at_f0(self):
        out = _settle(_ws(coupling_ratio=3.0))
        assert out.peak_hz == pytest.approx(40.0, abs=4.0)
        assert out.gamma_power > 0.5, "most power in the 30-100 Hz band"

    def test_gaba_slows_the_rhythm(self):
        fast = _settle(_ws(coupling_ratio=3.0), nt={"ACh": 0.3, "GABA": 0.15})
        slow = _settle(_ws(coupling_ratio=3.0), nt={"ACh": 0.3, "GABA": 0.45})
        assert slow.peak_hz < fast.peak_hz - 4.0


class TestNeuromodulation:
    def test_ach_raises_synchrony_near_criticality(self):
        lo = _settle(_ws(coupling_ratio=1.1), nt={"ACh": 0.15, "GABA": 0.15})
        hi = _settle(_ws(coupling_ratio=1.1), nt={"ACh": 0.45, "GABA": 0.15})
        assert hi.sync_r > lo.sync_r + 0.1

    def test_feedback_off_ignores_nt(self):
        a = _settle(_ws(coupling_ratio=1.1, nt_feedback=False),
                    nt={"ACh": 0.15, "GABA": 0.15})
        b = _settle(_ws(coupling_ratio=1.1, nt_feedback=False),
                    nt={"ACh": 0.45, "GABA": 0.45})
        assert a.sync_r == pytest.approx(b.sync_r)
        assert a.peak_hz == pytest.approx(b.peak_hz)


class TestIgnitionIsGlobal:
    def test_silent_modules_cannot_ignite(self):
        out = _settle(_ws(coupling_ratio=3.0),
                      act={k: 0.0 for k in ALL_ON})
        assert out.sync_r == 0.0 and not out.ignited

    def test_one_locked_module_is_not_a_global_ignition(self):
        """R is normalised by ALL oscillators: one perfectly locked
        module out of five caps R at 0.2."""
        act = {k: 0.0 for k in ALL_ON}
        act["language"] = 1.0
        out = _settle(_ws(coupling_ratio=3.0), act=act)
        assert out.sync_r <= 0.2 + 1e-9
        assert not out.ignited


class TestStateAndDeterminism:
    def test_same_seed_same_trajectory(self):
        a = _settle(_ws(coupling_ratio=1.2, noise=2.0), ticks=3)
        b = _settle(_ws(coupling_ratio=1.2, noise=2.0), ticks=3)
        assert a.sync_r == b.sync_r and a.peak_hz == b.peak_hz

    def test_phases_carry_across_ticks(self):
        ws = _ws(coupling_ratio=3.0, ms_per_tick=100)
        first = ws.step(ALL_ON, BASE_NT).sync_r
        for _ in range(10):
            last = ws.step(ALL_ON, BASE_NT).sync_r
        assert last > first, "locking accumulates over ticks"
