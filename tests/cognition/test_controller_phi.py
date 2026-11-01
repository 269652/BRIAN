# -*- coding: utf-8 -*-
"""Controller φ (architecture.md §14.14).

Integrated information φ (IIT 2.0, Balduzzi & Tononi 2008), computed
EXACTLY from an interventional state-by-node TPM over five binary
macro-nodes of the cognition runtime's controller. Rigorous at that
grain; explicitly not IIT 3.0/4.0 Φ^max and not a property of the LLM
or the hardware.

TPM convention (PyPhi's): row index = Σ_i b_i 2^i (little-endian),
column i = P(node i is ON at t+1 | row state at t).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from neuroslm.cognition.runtime import MindConfig, MindSeams, ThoughtScore


def _swap_tpm():
    """A(t+1) = B(t), B(t+1) = A(t) — each node only reads the other."""
    tpm = np.zeros((4, 2))
    for s in range(4):
        a, b = s & 1, (s >> 1) & 1
        tpm[s] = [b, a]
    return tpm


def _self_copy_tpm():
    """A(t+1) = A(t), B(t+1) = B(t) — two disconnected units."""
    tpm = np.zeros((4, 2))
    for s in range(4):
        a, b = s & 1, (s >> 1) & 1
        tpm[s] = [a, b]
    return tpm


class TestPhiIIT2:
    def test_cross_coupled_pair_integrates_two_bits(self):
        from neuroslm.cognition.controller_phi import phi_iit2
        phi, mip = phi_iit2(_swap_tpm(), (1, 0))
        assert phi == pytest.approx(2.0, abs=1e-9)
        assert sorted(map(sorted, mip)) == [[0], [1]]

    def test_disconnected_units_integrate_nothing(self):
        from neuroslm.cognition.controller_phi import phi_iit2
        for state in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            phi, _ = phi_iit2(_self_copy_tpm(), state)
            assert phi == pytest.approx(0.0, abs=1e-9)

    def test_noise_reduces_integration(self):
        from neuroslm.cognition.controller_phi import phi_iit2
        noisy = 0.8 * _swap_tpm() + 0.1
        phi, _ = phi_iit2(noisy, (1, 0))
        assert 0.0 < phi < 2.0

    def test_three_node_ring_is_integrated(self):
        """A←C, B←A, C←B: every bipartition cuts a dependency."""
        from neuroslm.cognition.controller_phi import phi_iit2
        tpm = np.zeros((8, 3))
        for s in range(8):
            b = [(s >> i) & 1 for i in range(3)]
            tpm[s] = [b[2], b[0], b[1]]
        phi, _ = phi_iit2(tpm, (1, 0, 1))
        assert phi > 0.9

    def test_rejects_a_malformed_tpm(self):
        from neuroslm.cognition.controller_phi import phi_iit2
        with pytest.raises(ValueError):
            phi_iit2(np.zeros((3, 2)), (0, 1))


# ── interventions on the live runtime (fakes) ────────────────────────

def _vec(text):
    axes = ("river", "code", "moon", "coffee")
    v = [0.05] * 4
    for i, w in enumerate(axes):
        if w in (text or "").lower():
            v[i] += 1.0
    return v


def _seams():
    topics = ["river", "code", "moon", "coffee"]
    state = {"n": 0}

    def gen(prompt, n):
        state["n"] += 1
        return f"thinking about the {topics[state['n'] % 4]}."

    return MindSeams(generate_fn=gen,
                     score_fn=lambda t: ThoughtScore(3.0, 0.4),
                     embed_fn=_vec)


class TestInterventions:
    def test_set_and_read_every_node(self):
        from neuroslm.cognition.controller_phi import (
            CONTROLLER_NODES, apply_state, read_state)
        rt = _seams().runtime(cfg=MindConfig(n_candidates=1))
        for target in [(1, 1, 0, 1, 0), (0, 0, 1, 0, 1)]:
            apply_state(rt, target)
            assert read_state(rt) == target, CONTROLLER_NODES

    def test_sync_intervention_locks_or_scrambles_phases(self):
        from neuroslm.cognition.controller_phi import apply_state
        rt = _seams().runtime(cfg=MindConfig(n_candidates=1))
        apply_state(rt, (1, 0, 0, 0, 0))
        z = abs(np.mean(np.exp(1j * rt._osc.theta)))
        assert z > 0.99
        apply_state(rt, (0, 0, 0, 0, 0))
        z = abs(np.mean(np.exp(1j * rt._osc.theta)))
        assert z < 0.5


class TestTPMEstimation:
    def test_shape_range_and_determinism(self):
        from neuroslm.cognition.controller_phi import estimate_controller_tpm
        a = estimate_controller_tpm(_seams(), MindConfig(n_candidates=1),
                                    samples=2, seed=0)
        b = estimate_controller_tpm(_seams(), MindConfig(n_candidates=1),
                                    samples=2, seed=0)
        assert a.shape == (32, 5)
        assert np.all((a >= 0) & (a <= 1))
        assert np.array_equal(a, b)

    def test_probe_reports_phi_per_condition(self):
        from neuroslm.cognition.controller_phi import probe_controller_phi
        res = probe_controller_phi(_seams(), cfg=MindConfig(n_candidates=1),
                                   samples=1, seed=0)
        json.dumps(res)
        assert res["indicator"] == "IIT/φ (IIT 2.0, controller macro-nodes)"
        for cond in ("full", "no_osc_feedback"):
            c = res["conditions"][cond]
            assert len(c["phi_per_state"]) == 32
            assert c["phi_mean"] >= 0.0
            assert c["phi_max"] >= c["phi_mean"]
            assert len(c["tpm"]) == 32
