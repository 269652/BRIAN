# -*- coding: utf-8 -*-
"""Bit-identical parity tests for DSL maturity + phase-gate utilities.

The DSL Brain aggregator (next layer up) sums Brain's `total = w_lm*lm + Σ
aux_w*phase*w_aux*loss_aux` formula. To get bit-identical aggregation, every
component of that formula must match Brain's implementation EXACTLY:

    1. `compute_mat`             ← neurochem.transmitters.compute_mat
    2. `phase_gate`              ← brain.Brain._phase_gate
    3. `MaturityTracker.update`  ← brain.Brain.update_maturity (rise-fast/fall-slow EMA)
    4. `AuxWeights` constants    ← brain.py:1794-1810 (centers, widths, weights)

These tests verify (1)-(3) numerically and (4) by configuration audit.
"""
import math
import pytest
import torch

from neuroslm.dsl.maturity import (
    compute_mat, phase_gate, MaturityTracker, AuxWeights, L_RANDOM_DEFAULT,
)
from neuroslm.neurochem.transmitters import (
    compute_mat as brain_compute_mat,
    L_RANDOM_DEFAULT as BRAIN_L_RANDOM,
)


def test_l_random_default_matches_brain():
    """The vocab-loss floor constant must be identical."""
    assert L_RANDOM_DEFAULT == BRAIN_L_RANDOM
    assert abs(L_RANDOM_DEFAULT - math.log(50257)) < 1e-12


def test_compute_mat_bit_identical():
    """DSL compute_mat must equal neurochem.transmitters.compute_mat."""
    # Sample a representative LM-loss curve from random-init (~10.8) down
    # to a converged ~3.0 nats, plus pathological spikes.
    losses = [10.8, 8.0, 6.0, 4.5, 4.0, 3.5, 3.0, 11.0, 0.5, -1.0, 1e6]
    for lm in losses:
        d = compute_mat(lm)
        b = brain_compute_mat(lm)
        assert d == b, f"compute_mat({lm}) diverges: dsl={d} brain={b}"


def test_phase_gate_bit_identical():
    """DSL phase_gate must equal brain.Brain._phase_gate at every MAT.

    Brain's gate is `0.5 * (1 + tanh((mat-center)/width))`. We sweep a
    grid of (mat, center, width) and check exact float equality.
    """
    from neuroslm.brain import Brain
    for mat in (0.0, 0.1, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0):
        for center, width in (
            (0.35, 0.08),  # ph_pred
            (0.45, 0.08),  # ph_world
            (0.50, 0.08),  # ph_motor / ph_fwd
            (0.55, 0.08),  # ph_novel / ph_cpc
            (0.60, 0.08),  # ph_kl / ph_phi
            (0.10, 0.10),  # default
        ):
            d = phase_gate(mat, center, width)
            b = Brain._phase_gate(mat, center, width)
            assert d == b, f"phase_gate diverges at mat={mat} c={center} w={width}"


def test_maturity_tracker_rise_fast_fall_slow():
    """The asymmetric EMA must rise fast (alpha=0.20) and fall slow (0.05)."""
    t = MaturityTracker()
    # Start at 0, push an LM-loss = 5.4 (= 0.5 of L_random ≈ 10.8 → mat=0.5)
    t.update(0.5 * L_RANDOM_DEFAULT)
    # After one rise step: mat = 0 + 0.20 * (0.5 - 0) = 0.10
    assert abs(t.value() - 0.10) < 1e-9, t.value()

    # Push the same input — rise EMA continues
    t.update(0.5 * L_RANDOM_DEFAULT)
    # 0.10 + 0.20 * (0.5 - 0.10) = 0.18
    assert abs(t.value() - 0.18) < 1e-9, t.value()

    # Now spike LM-loss high → MAT drops → slow alpha
    t.update(L_RANDOM_DEFAULT)  # mat_now = 0
    # 0.18 + 0.05 * (0 - 0.18) = 0.171
    assert abs(t.value() - 0.171) < 1e-9, t.value()


def test_aux_weights_match_brain_constants():
    """AuxWeights' (weight, center, width) tuples must mirror brain.py."""
    aw = AuxWeights()
    # Reference values from brain.py:1794-1810 (and config rcc_bowtie_30m_p4)
    expected = {
        "pred_coding": (0.10, 0.35, 0.08),
        "world":       (0.30, 0.45, 0.08),
        "forward":     (0.20 * 0.01, 0.50, 0.08),   # *0.01 bonus in formula
        "motor":       (0.05, 0.50, 0.08),
        "kl_world":    (0.10, 0.60, 0.08),
        "novel":       (0.05, 0.55, 0.08),
        "cpc":         (0.05, 0.55, 0.08),
        "phi":         (0.02, 0.60, 0.08),
    }
    for k, (w, c, width) in expected.items():
        got = getattr(aw, k)
        assert got == (w, c, width), f"{k}: got {got}, want {(w, c, width)}"


def test_aux_weights_scaled_at_low_maturity():
    """All aux losses should be near-zero at infancy (mat=0).

    The phase-gate centers are 0.35-0.60 with width 0.08, so at mat=0 the
    tanh argument is ≤ -4.4 → gate ≈ 0.0006 — essentially zero. This is
    what `detach_trunk_from_aux=True` plus low MAT achieves: an entirely
    LM-only trunk update during infancy.
    """
    aw = AuxWeights(master_scale=1.0)
    scaled = aw.all_scaled(mat=0.0)
    for k, v in scaled.items():
        assert v < 0.01, f"aux {k} not suppressed at mat=0 (got {v})"


def test_aux_weights_scaled_at_full_maturity():
    """At mat=1.0, all phase gates are saturated → weights equal raw values."""
    aw = AuxWeights(master_scale=1.0)
    scaled = aw.all_scaled(mat=1.0)
    # phase_gate(1.0, 0.6, 0.08) = 0.5 * (1 + tanh(5)) ≈ 0.99996
    # So scaled value ≈ raw weight
    for k in ("pred_coding", "world", "motor", "kl_world", "novel", "cpc", "phi"):
        raw_w = getattr(aw, k)[0]
        assert abs(scaled[k] - raw_w) < 1e-3, \
            f"aux {k} at mat=1 not at full weight: got {scaled[k]} expected {raw_w}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
