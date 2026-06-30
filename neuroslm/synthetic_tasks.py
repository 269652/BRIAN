# -*- coding: utf-8 -*-
"""Synthetic tasks with provable higher-order (synergistic) structure.

These are the calibrated sources for the integration apparatus: tasks whose
signal lives in the *joint* of several inputs and is absent from every strict
subset. A model that can only couple inputs pairwise (or marginally) provably
cannot solve them; one that captures the full-order interaction can. Paired
with ``neuroslm.information`` they let us measure whether a substrate has
entered a synergistic regime, and — eventually — search for one.

- ``parity_task`` — k-way XOR. The canonical "no low-order shortcut" problem.
- ``modular_addition_task`` — (a+b) mod m. The grokking task; pure synergy.

Each returns ``(X, y)`` as integer ``numpy`` arrays: ``X`` of shape
``(N, n_inputs)``, ``y`` of shape ``(N,)``. With ``n_samples=None`` the task is
enumerated exhaustively (every input combination once → exactly uniform inputs,
so the information measures are analytic); otherwise ``n_samples`` rows are
drawn i.i.d. with the given ``seed``.
"""
from __future__ import annotations

import itertools
from typing import Optional, Tuple

import numpy as np


def parity_task(
    n_bits: int,
    order: Optional[int] = None,
    n_samples: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """k-way parity: ``y = XOR(x_0, …, x_{order-1})``.

    ``order`` (default ``n_bits``) sets how many of the ``n_bits`` inputs the
    output depends on; any strict subset of those ``order`` bits is independent
    of ``y``. Bits beyond ``order`` are irrelevant distractors.
    """
    if order is None:
        order = n_bits
    if not (1 <= order <= n_bits):
        raise ValueError(f"order must be in [1, n_bits]; got {order} (n_bits={n_bits})")

    if n_samples is None:
        X = np.array(list(itertools.product((0, 1), repeat=n_bits)), dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        X = rng.integers(0, 2, size=(n_samples, n_bits)).astype(np.int64)

    y = (X[:, :order].sum(axis=1) % 2).astype(np.int64)
    return X, y


def modular_addition_task(
    modulus: int,
    n_samples: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Modular addition: inputs ``(a, b) ∈ Z_m²``, target ``y = (a+b) mod m``.

    Each operand alone is uniform with respect to ``y`` (zero mutual
    information); the pair determines it (I = log2 m). All of the predictive
    information is synergy.
    """
    if modulus < 2:
        raise ValueError(f"modulus must be ≥ 2; got {modulus}")

    if n_samples is None:
        a, b = np.meshgrid(np.arange(modulus), np.arange(modulus), indexing="ij")
        X = np.stack([a.ravel(), b.ravel()], axis=1).astype(np.int64)
    else:
        rng = np.random.default_rng(seed)
        X = rng.integers(0, modulus, size=(n_samples, 2)).astype(np.int64)

    y = ((X[:, 0] + X[:, 1]) % modulus).astype(np.int64)
    return X, y
