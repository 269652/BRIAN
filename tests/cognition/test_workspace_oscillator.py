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
    # locking physics against a FIXED bar; adaptive ignition (which
    # habituates to a steady locked state) has its own contracts
    base = dict(n_per_module=40, f0_hz=40.0, f_sd_hz=4.0, noise=0.0,
                ms_per_tick=1000, seed=0, ignition_mode="fixed")
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


class TestAdaptiveIgnition:
    """H66 (run 58676023): with the FIXED threshold 0.25, R1-7B's idle
    ticks never ignited (R 0.05-0.23) — the operating point rested on a
    guessed novelty level. Adaptive mode: ignite when R exceeds its own
    recent baseline, θ_t = max(floor, μ_R + k·σ_R), stats updated AFTER
    the decision, seeded at the old operating point."""

    MOD = {"sensory": 0.0, "memory": 0.7, "language": 0.7,
           "selection": 0.4, "novelty": 0.35}

    def _ws(self, **kw):
        from neuroslm.cognition.oscillator import GammaWorkspace
        base = dict(seed=0, ignition_mode="adaptive")
        base.update(kw)
        return GammaWorkspace(**base)

    def test_baseline_is_seeded_from_the_first_observed_r(self):
        """Run 58678026: seeding at the old bar (0.25) while R1's R sat
        near 0.12 spent ~8-10 of each 12-tick run in a warm-up
        transient (ignition 2/30). The baseline now starts at the
        system's own first R; tick 1 cannot ignite."""
        ws = self._ws(ignition_threshold=0.25)
        out = ws.step(self.MOD, BASE_NT)
        assert out.threshold == pytest.approx(
            max(0.1, out.sync_r + 1.0 * 0.05))
        assert not out.ignited
        assert ws._r_mu == pytest.approx(out.sync_r)

    def test_steady_input_habituates(self):
        """Stationary input → the threshold settles just above the
        system's own R: steady activity is not ignition."""
        ws = self._ws()
        outs = [ws.step(self.MOD, BASE_NT) for _ in range(60)]
        late = outs[30:]
        assert sum(o.ignited for o in late) / len(late) < 0.35
        assert abs(late[-1].threshold - late[-1].sync_r) < 0.15

    def test_a_jump_above_baseline_ignites_even_below_the_old_fixed_bar(self):
        ws = self._ws()
        dull = {k: 0.3 for k in ALL_ON}
        dull["sensory"] = 0.0
        for _ in range(30):
            ws.step(dull, BASE_NT)
        rich = dict(self.MOD, memory=0.9, language=0.85, selection=0.7,
                    novelty=0.6)
        hits = [ws.step(rich, BASE_NT) for _ in range(3)]
        assert any(h.ignited for h in hits)
        assert any(h.sync_r < 0.25 and h.ignited for h in hits) or \
            all(h.sync_r >= 0.25 for h in hits)

    def test_floor_blocks_noise_level_ignition(self):
        ws = self._ws(ignition_floor=0.1)
        silent = {k: 0.05 for k in ALL_ON}
        outs = [ws.step(silent, BASE_NT) for _ in range(40)]
        assert all(o.threshold >= 0.1 for o in outs)
        assert not any(o.ignited for o in outs[5:])

    def test_fixed_mode_is_unchanged(self):
        from neuroslm.cognition.oscillator import GammaWorkspace
        ws = GammaWorkspace(seed=0, ignition_mode="fixed",
                            ignition_threshold=0.25)
        out = ws.step(self.MOD, BASE_NT)
        assert out.threshold == 0.25
        assert out.ignited == (out.sync_r >= 0.25)

    def test_rejects_unknown_mode(self):
        from neuroslm.cognition.oscillator import GammaWorkspace
        with pytest.raises(ValueError):
            GammaWorkspace(ignition_mode="vibes")



class TestBaselineFromWanderingOnly:
    """Run 58683237: the adaptive baseline was seeded by the REPLY tick
    (sensory module on, R≈0.3-0.5) → θ started at 0.50 while idle R ran
    0.03-0.20: 0 ignitions in 26 ticks, nothing stored, recall stuck on
    the one seed episode. Reply ticks now neither seed nor move the
    idle baseline; before the first idle tick they use the fixed bar."""

    MOD = {"sensory": 0.0, "memory": 0.6, "language": 0.7,
           "selection": 0.5, "novelty": 0.4}

    def _ws(self):
        from neuroslm.cognition.oscillator import GammaWorkspace
        return GammaWorkspace(seed=0, ignition_mode="adaptive",
                              ignition_threshold=0.25)

    def test_reply_tick_does_not_touch_the_baseline(self):
        ws = self._ws()
        reply = dict(self.MOD, sensory=1.0)
        out = ws.step(reply, BASE_NT, update_baseline=False)
        assert not ws._r_seen
        assert out.threshold == pytest.approx(0.25), (
            "unseeded: decide against the fixed bar")

    def test_first_idle_tick_seeds_the_baseline(self):
        ws = self._ws()
        ws.step(dict(self.MOD, sensory=1.0), BASE_NT, update_baseline=False)
        idle = ws.step(self.MOD, BASE_NT)
        assert ws._r_seen
        assert idle.threshold == pytest.approx(
            max(0.1, idle.sync_r + 0.05))

    def test_reply_ticks_later_leave_the_idle_baseline_alone(self):
        ws = self._ws()
        for _ in range(5):
            ws.step(self.MOD, BASE_NT)
        mu, var = ws._r_mu, ws._r_var
        ws.step(dict(self.MOD, sensory=1.0), BASE_NT, update_baseline=False)
        assert (ws._r_mu, ws._r_var) == (mu, var)
