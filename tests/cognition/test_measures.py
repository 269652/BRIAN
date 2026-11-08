"""Consciousness / IIT measures over a run's per-tick state
(neuroslm/cognition/measures.py). Each contract pins a known analytic
case: independent parts integrate nothing, coupled parts do, a constant
stream has no complexity, a shuffled one has the most."""
from __future__ import annotations

import math

import numpy as np
import pytest


def _ar(n_steps, A, noise=1.0, seed=0):
    """x_t = A x_{t-1} + ε_t."""
    rng = np.random.default_rng(seed)
    A = np.asarray(A, float)
    x = np.zeros((n_steps, A.shape[0]))
    for t in range(1, n_steps):
        x[t] = A @ x[t - 1] + noise * rng.standard_normal(A.shape[0])
    return x


class TestPhiAR:
    """Autoregressive Φ (Barrett & Seth 2011): information the whole
    system's past carries about its present beyond what its parts' pasts
    carry about their own presents, at the minimum-information
    bipartition (normalised for partition size)."""

    def test_independent_units_integrate_nothing(self):
        from neuroslm.cognition.measures import phi_ar
        X = _ar(4000, np.diag([0.8, 0.7, 0.6, 0.5]))
        assert phi_ar(X)["phi"] == pytest.approx(0.0, abs=0.02)

    def test_cross_coupled_units_integrate(self):
        from neuroslm.cognition.measures import phi_ar
        A = [[0.4, 0.5, 0.0], [0.0, 0.4, 0.5], [0.5, 0.0, 0.4]]
        assert phi_ar(_ar(4000, A))["phi"] > 0.1

    def test_stronger_coupling_integrates_more(self):
        from neuroslm.cognition.measures import phi_ar
        weak = [[0.4, 0.15, 0.0], [0.0, 0.4, 0.15], [0.15, 0.0, 0.4]]
        strong = [[0.4, 0.5, 0.0], [0.0, 0.4, 0.5], [0.5, 0.0, 0.4]]
        assert phi_ar(_ar(4000, strong))["phi"] > \
            phi_ar(_ar(4000, weak))["phi"]

    def test_two_independent_blocks_split_at_the_seam(self):
        from neuroslm.cognition.measures import phi_ar
        A = np.zeros((4, 4))
        A[:2, :2] = [[0.4, 0.5], [0.5, 0.4]]
        A[2:, 2:] = [[0.4, 0.5], [0.5, 0.4]]
        out = phi_ar(_ar(4000, A))
        assert out["phi"] == pytest.approx(0.0, abs=0.02)
        assert sorted(map(sorted, out["mib"])) == [[0, 1], [2, 3]]

    def test_too_short_or_constant_series_has_no_phi(self):
        from neuroslm.cognition.measures import phi_ar
        assert phi_ar(np.zeros((3, 3)))["phi"] is None
        assert phi_ar(np.ones((50, 3)))["phi"] is None


def _common_driver(n_steps, n_units=3, rho=0.95, noise=0.3, seed=0):
    """Units that only copy one slow latent: redundant, never interacting."""
    rng = np.random.default_rng(seed)
    z = _ar(n_steps, [[rho]], seed=seed)[:, 0]
    return z[:, None] + noise * rng.standard_normal((n_steps, n_units))


class TestPhiR:
    """Φ_R (Mediano et al. 2021, ΦID): whole-minus-sum Φ plus the MMI
    double-redundancy min_{i,j} I(M^i_past; M^j_now) the subtraction
    counts twice. Run 59588081: modules sharing a driver gave
    Φ_AR z = −15 against an independent-channel null."""

    def test_redundant_units_make_phi_ar_negative_but_not_phi_r(self):
        from neuroslm.cognition.measures import phi_ar, phi_r
        X = _common_driver(4000)
        assert phi_ar(X)["phi"] < -0.05
        assert phi_r(X)["phi"] > -0.01

    def test_phi_r_is_phi_ar_plus_the_redundancy_at_its_partition(self):
        from neuroslm.cognition.measures import (_gauss_mi, phi_ar_split,
                                                 phi_r)
        X = _common_driver(4000)
        out = phi_r(X)
        a, b = out["mib"]
        Z = (X - X.mean(0)) / X.std(0)
        P, F = Z[:-1], Z[1:]
        red = min(_gauss_mi(P[:, i], F[:, j]) for i in (a, b) for j in (a, b))
        assert out["redundancy"] == pytest.approx(red, abs=1e-9)
        assert out["phi"] == pytest.approx(
            phi_ar_split(P, F, a, b) + red, abs=1e-9)

    def test_independent_units_integrate_nothing(self):
        from neuroslm.cognition.measures import phi_r
        X = _ar(4000, np.diag([0.8, 0.7, 0.6, 0.5]))
        assert phi_r(X)["phi"] == pytest.approx(0.0, abs=0.02)

    def test_cross_coupled_units_integrate(self):
        from neuroslm.cognition.measures import phi_r
        A = [[0.4, 0.5, 0.0], [0.0, 0.4, 0.5], [0.5, 0.0, 0.4]]
        assert phi_r(_ar(4000, A))["phi"] > 0.1

    def test_two_independent_blocks_split_at_the_seam(self):
        from neuroslm.cognition.measures import phi_r
        A = np.zeros((4, 4))
        A[:2, :2] = [[0.4, 0.5], [0.5, 0.4]]
        A[2:, 2:] = [[0.4, 0.5], [0.5, 0.4]]
        out = phi_r(_ar(4000, A))
        assert out["phi"] == pytest.approx(0.0, abs=0.02)
        assert sorted(map(sorted, out["mib"])) == [[0, 1], [2, 3]]

    def test_same_guards_as_phi_ar(self):
        from neuroslm.cognition.measures import phi_r
        assert phi_r(np.zeros((3, 3)))["phi"] is None
        assert phi_r(np.ones((50, 3)))["phi"] is None
        assert phi_r(_ar(100, np.eye(5) * 0.5))["phi"] is None


class TestNeuralComplexity:
    """Tononi–Sporns–Edelman complexity: integration and differentiation
    together — 0 for independent units, 0 for perfectly redundant ones,
    highest in between."""

    def _gauss(self, cov, n=20000, seed=0):
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(np.zeros(len(cov)), cov, n)

    def test_independent_units_have_no_complexity(self):
        from neuroslm.cognition.measures import neural_complexity
        X = self._gauss(np.eye(4))
        assert neural_complexity(X) == pytest.approx(0.0, abs=0.02)

    def test_structured_beats_independent_and_near_redundant(self):
        from neuroslm.cognition.measures import neural_complexity
        mid = np.full((4, 4), 0.5) + 0.5 * np.eye(4)
        hi = np.full((4, 4), 0.99) + 0.01 * np.eye(4)
        c_mid = neural_complexity(self._gauss(mid))
        assert c_mid > neural_complexity(self._gauss(np.eye(4)))
        assert c_mid > 0.0
        # near-redundant: integration is high but the whole adds little
        # beyond any part — complexity grows far slower than integration
        from neuroslm.cognition.measures import integration
        assert neural_complexity(self._gauss(hi)) / integration(
            self._gauss(hi)) < c_mid / integration(self._gauss(mid))


class TestStateComplexity:
    def test_constant_stream_is_simple_random_is_complex(self):
        from neuroslm.cognition.measures import lz_state_complexity
        rng = np.random.default_rng(0)
        assert lz_state_complexity(np.ones((200, 4))) == pytest.approx(0.0)
        assert lz_state_complexity(rng.standard_normal((200, 4))) > 0.8

    def test_periodic_is_less_complex_than_random(self):
        from neuroslm.cognition.measures import lz_state_complexity
        t = np.arange(200)[:, None]
        periodic = np.sin(2 * np.pi * t / 10 + np.arange(4))
        rng = np.random.default_rng(1)
        assert lz_state_complexity(periodic) < \
            lz_state_complexity(rng.standard_normal((200, 4)))


class TestWorkspaceDynamics:
    def test_ignition_rate_and_episode_length(self):
        from neuroslm.cognition.measures import ignition_stats
        s = ignition_stats([0.1, 0.5, 0.5, 0.1, 0.5, 0.1],
                           [False, True, True, False, True, False])
        assert s["ignition_rate"] == pytest.approx(0.5)
        assert s["mean_episode"] == pytest.approx(1.5)
        assert s["metastability"] == pytest.approx(np.std(
            [0.1, 0.5, 0.5, 0.1, 0.5, 0.1]))

    def test_all_or_none_ignition_is_bimodal(self):
        from neuroslm.cognition.measures import ignition_stats
        rng = np.random.default_rng(0)
        bi = np.r_[rng.normal(0.1, 0.02, 100), rng.normal(0.6, 0.02, 100)]
        uni = rng.normal(0.35, 0.1, 200)
        assert ignition_stats(bi, bi > 0.35)["bimodality"] > 0.555
        assert ignition_stats(uni, uni > 0.45)["bimodality"] < 0.555

    def test_phase_locking_between_modules(self):
        from neuroslm.cognition.measures import phase_locking
        T = 100
        locked = np.c_[np.linspace(0, 20, T), np.linspace(0, 20, T) + 0.3]
        rng = np.random.default_rng(0)
        free = np.c_[np.linspace(0, 20, T), rng.uniform(0, 2 * np.pi, T)]
        assert phase_locking(locked)[0][1] == pytest.approx(1.0)
        assert phase_locking(free)[0][1] < 0.3


class TestThoughtQuality:
    @staticmethod
    def _embed(text):
        axes = ("river", "mill", "bench", "dog", "rain", "chess")
        return [1.0 if w in text else 0.05 for w in axes]

    def test_a_repeating_stream_scores_as_repetitive(self):
        from neuroslm.cognition.measures import thought_quality
        q = thought_quality(["the river"] * 5, self._embed)
        assert q["repetition_rate"] == pytest.approx(1.0)
        assert q["mean_novelty"] == pytest.approx(0.0, abs=1e-9)

    def test_a_flowing_stream_is_novel_and_coherent(self):
        from neuroslm.cognition.measures import thought_quality
        stream = ["the river", "the river and the mill", "the mill bench",
                  "a dog on the bench", "rain on the dog"]
        q = thought_quality(stream, self._embed)
        assert q["repetition_rate"] == 0.0
        assert q["mean_novelty"] > 0.2
        assert q["mean_coherence"] > 0.3, "each thought follows the last"

    def test_grounding_in_what_was_perceived(self):
        from neuroslm.cognition.measures import thought_quality
        q = thought_quality(["a dog on the bench", "chess openings"],
                            self._embed,
                            context=[["A dog, close."], ["A dog, close."]])
        assert q["grounding"][0] > q["grounding"][1]
        assert q["mean_grounding"] == pytest.approx(np.mean(q["grounding"]))

    def test_lexical_diversity(self):
        from neuroslm.cognition.measures import thought_quality
        q = thought_quality(["a a a a", "b c d e"], self._embed)
        assert q["distinct_2"] == pytest.approx(4 / 6)


class TestSampleSizeGuards:
    """A Gaussian estimate needs enough data per fitted parameter.
    Run 59316881 fitted Φ_AR's 12×12 VAR(1) from 119 transitions (~1 per
    parameter) and got −0.008: noise. Below ``min_obs_per_param`` (10)
    the measure is refused, with the reason."""

    def test_phi_ar_needs_ten_transitions_per_var_coefficient(self):
        from neuroslm.cognition.measures import phi_ar
        A = [[0.4, 0.5, 0.0, 0.0], [0.0, 0.4, 0.5, 0.0],
             [0.0, 0.0, 0.4, 0.5], [0.5, 0.0, 0.0, 0.4]]
        short = phi_ar(_ar(150, A))              # 149 < 10·4²
        assert short["phi"] is None and "transitions" in short["reason"]
        assert phi_ar(_ar(170, A))["phi"] is not None

    def test_integration_and_complexity_need_ten_per_covariance_entry(self):
        from neuroslm.cognition.measures import (integration,
                                                 neural_complexity)
        X = np.random.default_rng(0).standard_normal((99, 4))   # < 10·10
        assert integration(X) is None and neural_complexity(X) is None
        Y = np.random.default_rng(0).standard_normal((100, 4))
        assert integration(Y) is not None


class TestPhiOnlyAcrossConsecutiveTicks:
    """Ticks without a workspace reading (inhibited) leave gaps; a
    transition across a gap is not a one-step transition."""

    def test_only_consecutive_ticks_make_transitions(self):
        from neuroslm.cognition.measures import phi_ar
        A = [[0.4, 0.5], [0.5, 0.4]]
        X = _ar(400, A)
        ticks = list(range(200)) + list(range(300, 500))
        assert phi_ar(X, ticks=ticks)["n_transitions"] == 398
        assert phi_ar(X)["n_transitions"] == 399

    def test_no_consecutive_ticks_no_phi(self):
        from neuroslm.cognition.measures import phi_ar
        X = _ar(400, [[0.4, 0.5], [0.5, 0.4]])
        assert phi_ar(X, ticks=list(range(0, 800, 2)))["phi"] is None


class TestSurrogates:
    def test_phase_randomising_keeps_spectra_breaks_coupling(self):
        from neuroslm.cognition.measures import phase_randomise
        x = _ar(1024, [[0.9]])[:, 0]
        X = np.c_[x, x]
        S = phase_randomise(X, np.random.default_rng(0))
        for k in range(2):
            assert np.allclose(np.abs(np.fft.rfft(S[:, k] - S[:, k].mean())),
                               np.abs(np.fft.rfft(X[:, k] - X[:, k].mean())))
        assert abs(np.corrcoef(S.T)[0, 1]) < 0.5

    def test_time_shuffling_keeps_states_breaks_order(self):
        from neuroslm.cognition.measures import time_shuffle
        X = _ar(500, [[0.9, 0.0], [0.0, 0.9]])
        S = time_shuffle(X, np.random.default_rng(0))
        assert sorted(map(tuple, S)) == sorted(map(tuple, X))
        lag1 = lambda v: np.corrcoef(v[:-1], v[1:])[0, 1]
        assert lag1(X[:, 0]) > 0.7 and abs(lag1(S[:, 0])) < 0.2

    def test_coupled_dynamics_beat_their_null(self):
        from neuroslm.cognition.measures import null_test, phi_ar, \
            phase_randomise
        A = [[0.4, 0.5, 0.0], [0.0, 0.4, 0.5], [0.5, 0.0, 0.4]]
        r = null_test(lambda Z: phi_ar(Z)["phi"], _ar(1000, A),
                      phase_randomise, n=40, seed=0)
        assert r["z"] > 3 and r["p"] < 0.05

    def test_independent_dynamics_do_not(self):
        from neuroslm.cognition.measures import null_test, phi_ar, \
            phase_randomise
        r = null_test(lambda Z: phi_ar(Z)["phi"],
                      _ar(1000, np.diag([0.8, 0.6, 0.4])),
                      phase_randomise, n=40, seed=0)
        assert r["p"] > 0.05

    def test_no_value_no_test(self):
        from neuroslm.cognition.measures import null_test, time_shuffle
        r = null_test(lambda Z: None, np.ones((10, 2)), time_shuffle, n=5)
        assert r["value"] is None and r["z"] is None and r["p"] is None


class TestBootstrap:
    def test_block_bootstrap_interval_covers_the_estimate(self):
        from neuroslm.cognition.measures import block_bootstrap_ci
        X = _ar(1000, [[0.5, 0.3], [0.3, 0.5]])
        lo, hi = block_bootstrap_ci(lambda Z: float(Z[:, 0].mean()), X,
                                    n=200, seed=0)
        assert lo < float(X[:, 0].mean()) < hi
        assert lo < 0.0 < hi                       # the true mean
        assert hi - lo < 0.5


class TestAccessTest:
    """Global workspace theory: content that ignites becomes accessible
    — later reported, remembered, used. Units: percepts held in the
    workspace; exposure: held on an ignited tick; outcome: later
    access. Stratified by salience (salience drives both), with a
    within-stratum permutation null."""

    def _events(self, rule, n=400, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            sal = float(rng.uniform())
            ign = bool(rng.uniform() < 0.3 + 0.4 * sal)
            out.append((ign, bool(rng.uniform() < rule(ign, sal)), sal))
        return out

    def test_ignition_that_grants_access_is_detected(self):
        from neuroslm.cognition.measures import access_test
        r = access_test(self._events(lambda i, s: 0.6 if i else 0.2),
                        n_perm=500, seed=0)
        assert r["diff"] > 0.2 and r["p"] < 0.01

    def test_epiphenomenal_ignition_is_not(self):
        from neuroslm.cognition.measures import access_test
        r = access_test(self._events(lambda i, s: 0.4), n_perm=500, seed=0)
        assert r["p"] > 0.05

    def test_salience_alone_is_not_mistaken_for_ignition(self):
        from neuroslm.cognition.measures import access_test
        ev = self._events(lambda i, s: 0.05 + 0.9 * s, n=2000)
        raw = (np.mean([a for i, a, _ in ev if i])
               - np.mean([a for i, a, _ in ev if not i]))
        r = access_test(ev, n_perm=500, seed=0)
        assert raw > 0.05, "the confound is real"
        assert abs(r["diff"]) < 0.05 and r["p"] > 0.05
