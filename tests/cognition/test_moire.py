# -*- coding: utf-8 -*-
"""Moiré E-I bilayer (H70): two rings of excitatory-inhibitory units
with mismatched spacing, coupled by distance between layers.

No unit has a natural frequency. Each unit is a linearised
Wilson-Cowan E-I pair; its rhythm comes from the E-I loop and the
time constants (PING). With identical local loops, every eigenmode κ
of the lateral matrix W becomes one 2×2 damped oscillator, so the
spectrum of W is the "band structure" and the loop turns bands into
rhythms. These contracts pin that mathematics, the measures (top-edge
density of states, inverse participation ratio) against known answers
(tight-binding cosine bands, the Aubry-André transition), and the
nonlinear simulation against its linearisation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


def _m():
    from neuroslm.cognition import moire
    return moire


class TestLateralMatrix:
    def test_uncoupled_layers_are_two_tight_binding_rings(self):
        m = _m()
        W = m.lateral_matrix(12, 9, g=0.0, t=1.0)
        want = sorted([2 * math.cos(2 * math.pi * k / 12) for k in range(12)]
                      + [2 * math.cos(2 * math.pi * k / 9) for k in range(9)])
        assert np.allclose(np.linalg.eigvalsh(W), want)

    def test_interlayer_coupling_falls_with_distance_around_the_ring(self):
        m = _m()
        g, s = 0.7, 0.5
        W = m.lateral_matrix(10, 8, g=g, s=s)
        # layer A at 0,1,…,9; layer B every 10/8 = 1.25 on the same ring
        d = abs(0.0 - 7 * 1.25)            # 8.75 → wraps to 1.25
        d = min(d, 10 - d)
        assert W[0, 10 + 7] == pytest.approx(g * math.exp(-d * d / (2 * s * s)))
        assert np.allclose(W, W.T)

    def test_shuffled_layer_keeps_its_size_and_is_seeded(self):
        m = _m()
        a = m.shuffled_positions(10, 8, np.random.default_rng(3))
        b = m.shuffled_positions(10, 8, np.random.default_rng(3))
        assert len(a) == 8 and np.all(np.diff(a) >= 0)
        assert np.all((a >= 0) & (a < 10))
        assert np.array_equal(a, b)
        W = m.lateral_matrix(10, 8, g=0.5, positions_b=a)
        assert W.shape == (18, 18) and np.allclose(W, W.T)


class TestEmergentRhythm:
    def test_the_spectrum_is_the_2x2_map_of_the_lateral_modes(self):
        m = _m()
        p = m.EIParams()
        W = m.lateral_matrix(9, 7, g=0.4)
        w_ee = 0.3
        full = np.linalg.eigvals(m.jacobian(W, w_ee, p))
        kap = np.linalg.eigvalsh(W)
        mapped = np.concatenate([m.ei_eigs(k, w_ee, p) for k in kap])
        key = lambda z: (round(z.real, 8), round(z.imag, 8))
        assert sorted(full, key=key) == pytest.approx(sorted(mapped, key=key))

    def test_the_rhythm_is_set_by_time_constants_not_by_any_frequency(self):
        m = _m()
        p = m.EIParams()
        fast = m.EIParams(tau_e=p.tau_e / 2, tau_i=p.tau_i / 2,
                          criticality=p.criticality)
        assert m.rhythm_hz(fast) == pytest.approx(2 * m.rhythm_hz(p))

    def test_criticality_places_the_leading_mode_at_the_hopf_point(self):
        m = _m()
        W = m.lateral_matrix(20, 15, g=0.3)
        at = m.spectrum(W, m.EIParams(criticality=1.0))
        below = m.spectrum(W, m.EIParams(criticality=0.9))
        assert at["lead"].real == pytest.approx(0.0, abs=1e-12)
        assert below["lead"].real < 0 and abs(below["lead"].imag) > 0

    def test_default_loop_rings_in_the_gamma_band(self):
        """A design choice (τ_E 3 ms, τ_I 6 ms, loop gains), checked."""
        m = _m()
        assert 30.0 <= m.rhythm_hz(m.EIParams()) <= 100.0


class TestMeasures:
    def test_ipr_is_one_over_n_for_a_plane_wave_and_one_for_a_spike(self):
        m = _m()
        n = 50
        assert m.ipr(np.ones(n)) == pytest.approx(1 / n)
        e = np.zeros(n)
        e[7] = 3.0
        assert m.ipr(e) == pytest.approx(1.0)

    def test_ipr_sees_the_aubry_andre_transition(self):
        """Known answer: a chain with on-site λ·cos(2πβi), β irrational,
        has all modes extended for λ < 2t and localised for λ > 2t."""
        m = _m()
        n, beta = 377, (math.sqrt(5) - 1) / 2

        def lead_ipr(lam):
            H = m.lateral_matrix(n, 0, g=0.0)
            H = H + np.diag(lam * np.cos(2 * math.pi * beta * np.arange(n)))
            _, v = np.linalg.eigh(H)
            return m.ipr(v[:, -1])

        assert lead_ipr(1.0) * n < 10
        assert lead_ipr(4.0) > 0.2

    def test_top_dos_counts_modes_near_the_edge(self):
        m = _m()
        kap = np.array([0.0, 5.0, 9.0, 9.9, 9.95, 10.0])
        assert m.top_dos(kap, eps_frac=0.01) == 3      # within 0.1 of 10
        assert m.top_dos(kap, eps_frac=0.2) == 4


class TestSweeps:
    def test_mismatch_sweep_reads_each_point_against_its_null(self):
        m = _m()
        rows = m.mismatch_sweep(30, [30, 25, 20], g=0.5, n_null=5, seed=0)
        assert [r["n_b"] for r in rows] == [30, 25, 20]
        assert rows[1]["delta"] == pytest.approx(30 / 25 - 1)
        for r in rows:
            assert len(r["null_top_dos"]) == 5
            assert r["top_dos"] >= 1
            assert 0 < r["lead_ipr"] <= 1
        again = m.mismatch_sweep(30, [30, 25, 20], g=0.5, n_null=5, seed=0)
        assert rows == again

    def test_localization_sweep_rows_per_coupling(self):
        m = _m()
        rows = m.localization_sweep(34, 21, [0.0, 1.0], n_null=3, seed=1)
        assert [r["g"] for r in rows] == [0.0, 1.0]
        assert all(len(r["null_lead_ipr"]) == 3 for r in rows)
        # uncoupled rings: the leading mode is a plane wave
        assert rows[0]["lead_ipr"] * (34 + 21) < 3


class TestNonlinearSimulation:
    def test_noise_driven_rhythm_peaks_where_the_linearisation_says(self):
        m = _m()
        p = m.EIParams(criticality=0.95)
        W = m.lateral_matrix(40, 0, g=0.0)        # leading mode is uniform
        E = m.simulate(W, p, ms=4000, dt=0.05, noise=0.02, seed=0)
        f = m.peak_hz(E.mean(axis=1), dt_ms=0.05 * 10)
        assert f == pytest.approx(m.rhythm_hz(p), rel=0.15)

    def test_above_the_hopf_point_activity_stays_bounded(self):
        m = _m()
        p = m.EIParams(criticality=1.2)
        W = m.lateral_matrix(20, 15, g=0.5)
        E = m.simulate(W, p, ms=500, dt=0.05, noise=0.01, seed=0)
        assert np.all(np.isfinite(E)) and np.abs(E).max() <= 1.1
        assert E[len(E) // 2:].std() > 0.05            # a limit cycle, not decay

    def test_simulation_is_seeded(self):
        m = _m()
        p = m.EIParams()
        W = m.lateral_matrix(10, 8, g=0.3)
        a = m.simulate(W, p, ms=50, dt=0.05, noise=0.05, seed=4)
        b = m.simulate(W, p, ms=50, dt=0.05, noise=0.05, seed=4)
        assert np.array_equal(a, b)


class TestKagome:
    """Kagome (H71): an exactly flat band without any twist. Under
    lateral inhibition (W = −c·A) the flat band is the TOP of the
    spectrum, so its compact hexagon modes are the ones that ring."""

    def test_every_site_has_four_neighbours(self):
        m = _m()
        A = m.kagome_adjacency(4)
        assert A.shape == (48, 48) and np.allclose(A, A.T)
        assert np.all(A.sum(axis=1) == 4)

    def test_the_flat_band_sits_at_minus_two_with_l2_plus_one_modes(self):
        m = _m()
        L = 5
        k = np.linalg.eigvalsh(m.kagome_adjacency(L))
        assert np.sum(np.isclose(k, -2.0)) == L * L + 1
        assert k.min() == pytest.approx(-2.0) and k.max() == pytest.approx(4.0)

    def test_an_alternating_hexagon_is_a_compact_flat_band_mode(self):
        m = _m()
        L = 4
        A = m.kagome_adjacency(L)
        ring = m.kagome_hexagon(L, 1, 2)
        assert len(set(ring)) == 6
        for a, b in zip(ring, ring[1:] + ring[:1]):
            assert A[a, b] == 1
        v = np.zeros(3 * L * L)
        v[ring] = [1, -1, 1, -1, 1, -1]
        assert np.allclose(A @ v, -2 * v)

    def test_lateral_inhibition_puts_the_flat_band_on_top(self):
        m = _m()
        L = 5
        A = m.kagome_adjacency(L)
        inhib = np.linalg.eigvalsh(-0.5 * A)
        excit = np.linalg.eigvalsh(0.5 * A)
        assert m.top_dos(inhib, eps_frac=1e-6) == L * L + 1
        assert m.top_dos(excit, eps_frac=1e-6) == 1


class TestAssemblies:
    def test_variance_share_of_the_top_tenth(self):
        m = _m()
        rng = np.random.default_rng(0)
        one = np.zeros((200, 20))
        one[:, 3] = rng.standard_normal(200)
        assert m.assembly_stats(one)["top10_share"] == pytest.approx(1.0)
        flat = rng.standard_normal((5000, 20))
        assert m.assembly_stats(flat)["top10_share"] == pytest.approx(0.1, abs=0.02)

    def test_pattern_overlap_is_one_for_the_same_pattern(self):
        m = _m()
        rng = np.random.default_rng(1)
        X = rng.standard_normal((300, 12)) * np.arange(1, 13)
        a, b = m.assembly_stats(X), m.assembly_stats(2 * X)
        assert m.pattern_overlap(a["pattern"], b["pattern"]) == pytest.approx(1.0)

    def test_assembly_sweep_runs_one_simulation_per_seed(self):
        m = _m()
        W = -0.5 * m.kagome_adjacency(3)
        out = m.assembly_sweep(W, m.EIParams(criticality=1.1), seeds=[0, 1],
                               ms=200)
        assert len(out["runs"]) == 2
        assert 0 <= out["mean_overlap"] <= 1
        assert all(0 < r["top10_share"] <= 1 for r in out["runs"])
        assert out == m.assembly_sweep(W, m.EIParams(criticality=1.1),
                                       seeds=[0, 1], ms=200)
