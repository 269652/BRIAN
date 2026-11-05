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
