# -*- coding: utf-8 -*-
"""Contracts for the discrete information probe (neuroslm/information.py).

Why this exists
===============
To detect a "fractional" integration mode we first need an *instrument* that
measures higher-order information exactly. Net synergy (co-information) and the
Williams–Beer PID synergy atom have KNOWN analytic values on canonical logic
gates — those values are the calibration of the instrument:

    gate    I(X1;Y)  I(X2;Y)  I(X1X2;Y)  WB-redundancy  WB-synergy  net-synergy
    XOR      0        0        1           0              1           +1
    AND      0.311    0.311    0.811       0.311          0.5         +0.189
    COPY     1        1        1           1              0           -1
    unique   1        0        1           0              0            0   (Y=X1)

All gates are enumerated as exhaustive truth tables (each input combo once =
uniform), so the measures are EXACT — no sampling tolerance needed beyond float
rounding. If these don't hold, the instrument is miscalibrated and every
downstream "we found synergy" claim is void.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# Canonical 2-bit truth tables (each row once → uniform inputs).
X1 = np.array([0, 0, 1, 1])
X2 = np.array([0, 1, 0, 1])
Y_XOR = np.array([0, 1, 1, 0])
Y_AND = np.array([0, 0, 0, 1])
Y_OR = np.array([0, 1, 1, 1])

# AND-gate canonical PID values (Williams & Beer 2010).
AND_REDUNDANCY = 0.31127812445913283
AND_SYNERGY = 0.5
AND_NET_SYNERGY = AND_SYNERGY - AND_REDUNDANCY  # = co-information = 0.18872...


class TestEntropy:
    def test_fair_coin_is_one_bit(self):
        from neuroslm.information import entropy
        assert abs(entropy(np.array([0, 1])) - 1.0) < 1e-9

    def test_constant_is_zero(self):
        from neuroslm.information import entropy
        assert abs(entropy(np.array([7, 7, 7, 7]))) < 1e-12

    def test_fair_die_is_log2_six(self):
        from neuroslm.information import entropy
        assert abs(entropy(np.arange(6)) - math.log2(6)) < 1e-9


class TestMutualInformation:
    def test_self_information_equals_entropy(self):
        from neuroslm.information import entropy, mutual_information
        x = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        assert abs(mutual_information(x, x) - entropy(x)) < 1e-9

    def test_independent_is_zero(self):
        from neuroslm.information import mutual_information
        # Exhaustive product of two fair bits → exactly independent.
        assert abs(mutual_information(X1, X2)) < 1e-12

    def test_copy_is_entropy(self):
        from neuroslm.information import mutual_information
        assert abs(mutual_information(X1, X1.copy()) - 1.0) < 1e-9

    def test_xor_marginals_are_zero(self):
        """The signature of a synergistic source: each input alone carries
        ZERO information about the XOR output."""
        from neuroslm.information import mutual_information
        assert abs(mutual_information(X1, Y_XOR)) < 1e-12
        assert abs(mutual_information(X2, Y_XOR)) < 1e-12

    def test_xor_pair_is_one_bit(self):
        from neuroslm.information import mutual_information
        pair = np.stack([X1, X2], axis=1)
        assert abs(mutual_information(pair, Y_XOR) - 1.0) < 1e-9


class TestConditionalMutualInformation:
    def test_xor_conditional_is_full(self):
        """Given X2, X1 fully determines the XOR output → I(X1;Y|X2)=H(X1)=1."""
        from neuroslm.information import conditional_mutual_information
        cmi = conditional_mutual_information(X1, Y_XOR, X2)
        assert abs(cmi - 1.0) < 1e-9

    def test_conditioning_on_target_drops_to_zero(self):
        from neuroslm.information import conditional_mutual_information
        # I(X1;X1 | X1) = 0 (target known).
        assert abs(conditional_mutual_information(X1, X1.copy(), X1.copy())) < 1e-12


class TestNetSynergy:
    """net_synergy = I(X1X2;Y) - I(X1;Y) - I(X2;Y) = -co_information.
    Positive ⇒ the whole says more than the sum of parts (synergy)."""

    def test_xor_is_maximal_positive(self):
        from neuroslm.information import net_synergy
        assert abs(net_synergy(X1, X2, Y_XOR) - 1.0) < 1e-9

    def test_copy_is_maximal_negative(self):
        from neuroslm.information import net_synergy
        # Y = X1 = X2 → redundant, net synergy = -1.
        assert abs(net_synergy(X1, X1.copy(), X1.copy()) + 1.0) < 1e-9

    def test_and_is_known_value(self):
        from neuroslm.information import net_synergy
        assert abs(net_synergy(X1, X2, Y_AND) - AND_NET_SYNERGY) < 1e-6

    def test_unique_is_zero(self):
        from neuroslm.information import net_synergy
        # Y = X1, X2 independent and irrelevant → net synergy 0.
        assert abs(net_synergy(X1, X2, X1.copy())) < 1e-9

    def test_is_negative_co_information(self):
        from neuroslm.information import net_synergy, co_information
        for Y in (Y_XOR, Y_AND, Y_OR):
            assert abs(net_synergy(X1, X2, Y) + co_information(X1, X2, Y)) < 1e-9


class TestPIDSynergy:
    """Williams–Beer Partial Information Decomposition. The four atoms must sum
    to I(X1X2;Y) and match the canonical gate values."""

    def test_atoms_sum_to_total(self):
        from neuroslm.information import pid_synergy, mutual_information
        pair = np.stack([X1, X2], axis=1)
        for Y in (Y_XOR, Y_AND, Y_OR):
            d = pid_synergy(X1, X2, Y)
            total = (d["redundancy"] + d["unique1"]
                     + d["unique2"] + d["synergy"])
            assert abs(total - mutual_information(pair, Y)) < 1e-9
            assert abs(total - d["total"]) < 1e-9

    def test_xor_is_pure_synergy(self):
        from neuroslm.information import pid_synergy
        d = pid_synergy(X1, X2, Y_XOR)
        assert abs(d["synergy"] - 1.0) < 1e-9
        assert abs(d["redundancy"]) < 1e-9
        assert abs(d["unique1"]) < 1e-9 and abs(d["unique2"]) < 1e-9

    def test_and_canonical_decomposition(self):
        from neuroslm.information import pid_synergy
        d = pid_synergy(X1, X2, Y_AND)
        assert abs(d["redundancy"] - AND_REDUNDANCY) < 1e-6
        assert abs(d["synergy"] - AND_SYNERGY) < 1e-6
        assert abs(d["unique1"]) < 1e-6 and abs(d["unique2"]) < 1e-6

    def test_copy_is_pure_redundancy(self):
        from neuroslm.information import pid_synergy
        d = pid_synergy(X1, X1.copy(), X1.copy())
        assert abs(d["redundancy"] - 1.0) < 1e-9
        assert abs(d["synergy"]) < 1e-9

    def test_unique_information(self):
        from neuroslm.information import pid_synergy
        # Y = X1 → all information is unique to X1.
        d = pid_synergy(X1, X2, X1.copy())
        assert abs(d["unique1"] - 1.0) < 1e-9
        assert abs(d["unique2"]) < 1e-9
        assert abs(d["redundancy"]) < 1e-9 and abs(d["synergy"]) < 1e-9

    def test_synergy_atoms_are_nonnegative(self):
        from neuroslm.information import pid_synergy
        for Y in (Y_XOR, Y_AND, Y_OR):
            d = pid_synergy(X1, X2, Y)
            for k in ("redundancy", "unique1", "unique2", "synergy"):
                assert d[k] >= -1e-9, f"{k} must be ≥0, got {d[k]}"


class TestTotalCorrelation:
    """Multi-information TC = ΣH(Xi) − H(X1..Xn): total dependence among a set."""

    def test_independent_is_zero(self):
        from neuroslm.information import total_correlation
        # 3 fair bits, exhaustive product (8 rows) → independent.
        bits = np.array([[a, b, c] for a in (0, 1)
                         for b in (0, 1) for c in (0, 1)])
        assert abs(total_correlation(bits[:, 0], bits[:, 1], bits[:, 2])) < 1e-12

    def test_fully_copied_is_n_minus_one_bits(self):
        from neuroslm.information import total_correlation
        x = np.array([0, 1])
        tc = total_correlation(x, x.copy(), x.copy())
        assert abs(tc - 2.0) < 1e-9
