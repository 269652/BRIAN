# -*- coding: utf-8 -*-
"""Contracts for the higher-order synthetic testbed (neuroslm/synthetic_tasks.py).

Why these tasks
===============
To test whether a substrate can *capture* synergy we need sources whose signal
is provably synergistic — present in the joint, absent from every strict
subset. Two such tasks:

  * k-way PARITY  — y = XOR of k bits. Any (k−1) bits are independent of y;
    only the full k-tuple determines it. The canonical "no low-order shortcut"
    problem.
  * MODULAR ADDITION — y = (a+b) mod m. Each operand alone is uniform w.r.t. y
    (I=0); the pair determines it (I=H(y)=log2 m). The grokking task.

These tests double as an end-to-end calibration: we run the
``neuroslm.information`` probe ON the generated data and assert the synergy is
exactly where the task definition says it is. If the probe + testbed agree on
these analytic values, the apparatus is sound.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from neuroslm.information import mutual_information, net_synergy, entropy


class TestParityStructure:
    def test_exhaustive_shape_and_labels(self):
        from neuroslm.synthetic_tasks import parity_task
        X, y = parity_task(n_bits=3)  # exhaustive
        assert X.shape == (8, 3)
        assert y.shape == (8,)
        # y must equal the XOR (parity) of all bits.
        assert np.array_equal(y, X.sum(axis=1) % 2)
        assert set(np.unique(X)) <= {0, 1}

    def test_no_single_bit_shortcut(self):
        """Each bit alone is independent of the parity output."""
        from neuroslm.synthetic_tasks import parity_task
        X, y = parity_task(n_bits=4)
        for i in range(4):
            assert abs(mutual_information(X[:, i], y)) < 1e-12

    def test_no_order_minus_one_shortcut(self):
        """For k-way parity, even (k−1) bits jointly carry zero info — the
        defining 'no low-order shortcut' property."""
        from neuroslm.synthetic_tasks import parity_task
        X, y = parity_task(n_bits=4, order=4)
        first_three = X[:, :3]  # any 3 of the 4 relevant bits
        assert abs(mutual_information(first_three, y)) < 1e-12

    def test_full_tuple_determines_output(self):
        from neuroslm.synthetic_tasks import parity_task
        X, y = parity_task(n_bits=4, order=4)
        assert abs(mutual_information(X, y) - entropy(y)) < 1e-9
        assert abs(entropy(y) - 1.0) < 1e-9  # balanced parity → 1 bit

    def test_order_below_nbits_leaves_irrelevant_bits(self):
        from neuroslm.synthetic_tasks import parity_task
        X, y = parity_task(n_bits=3, order=2)
        # y depends only on bits 0,1 → pure synergy there, bit 2 irrelevant.
        assert abs(net_synergy(X[:, 0], X[:, 1], y) - 1.0) < 1e-9
        assert abs(mutual_information(X[:, 2], y)) < 1e-12

    def test_sampled_mode_shapes_and_determinism(self):
        from neuroslm.synthetic_tasks import parity_task
        Xa, ya = parity_task(n_bits=5, n_samples=256, seed=0)
        Xb, yb = parity_task(n_bits=5, n_samples=256, seed=0)
        assert Xa.shape == (256, 5) and ya.shape == (256,)
        assert np.array_equal(Xa, Xb) and np.array_equal(ya, yb)  # seed determinism
        assert np.array_equal(ya, Xa.sum(axis=1) % 2)


class TestModularAddition:
    def test_exhaustive_shape_and_labels(self):
        from neuroslm.synthetic_tasks import modular_addition_task
        X, y = modular_addition_task(modulus=5)  # exhaustive 25 rows
        assert X.shape == (25, 2)
        assert y.shape == (25,)
        assert np.array_equal(y, (X[:, 0] + X[:, 1]) % 5)

    def test_operands_alone_are_uninformative(self):
        from neuroslm.synthetic_tasks import modular_addition_task
        X, y = modular_addition_task(modulus=5)
        assert abs(mutual_information(X[:, 0], y)) < 1e-12
        assert abs(mutual_information(X[:, 1], y)) < 1e-12

    def test_pair_carries_full_entropy_as_synergy(self):
        from neuroslm.synthetic_tasks import modular_addition_task
        X, y = modular_addition_task(modulus=5)
        assert abs(mutual_information(X, y) - math.log2(5)) < 1e-9
        # All of it is synergy (marginals are zero).
        assert abs(net_synergy(X[:, 0], X[:, 1], y) - math.log2(5)) < 1e-9

    def test_sampled_determinism(self):
        from neuroslm.synthetic_tasks import modular_addition_task
        Xa, ya = modular_addition_task(modulus=7, n_samples=128, seed=1)
        Xb, yb = modular_addition_task(modulus=7, n_samples=128, seed=1)
        assert Xa.shape == (128, 2)
        assert np.array_equal(Xa, Xb) and np.array_equal(ya, yb)
        assert np.array_equal(ya, (Xa[:, 0] + Xa[:, 1]) % 7)
