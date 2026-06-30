# -*- coding: utf-8 -*-
"""Exact discrete information-theoretic probe.

The measurement apparatus for higher-order information integration. Everything
here is computed exactly from empirical joint distributions of integer-labelled
samples (no kernel density, no estimator bias beyond finite-sample histogram
counts), so on small/enumerable systems the values are analytic.

The instrument's job is to detect *synergy* — information carried only by the
joint of several variables, irreducible to any subset (the XOR signature). That
is the information-theoretic analogue of fractionalisation: a quantity that
belongs to the collective and cannot be assigned to the parts. Standard
pairwise architectures are structurally biased away from it; this probe is how
we tell whether a substrate has entered a synergistic regime.

Measures
--------
- ``entropy`` / ``joint_entropy`` — Shannon H (bits by default).
- ``mutual_information`` — I(X;Y); X and/or Y may be multi-column (a joint).
- ``conditional_mutual_information`` — I(X;Y|Z).
- ``total_correlation`` — multi-information ΣH(Xi)−H(X), total dependence.
- ``co_information`` — McGill 3-way interaction; >0 net redundancy, <0 net synergy.
- ``net_synergy`` — −co_information = I(X1X2;Y)−I(X1;Y)−I(X2;Y).
- ``pid_synergy`` — Williams–Beer Partial Information Decomposition (2 sources):
  redundancy / unique1 / unique2 / synergy atoms summing to I(X1X2;Y).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = [
    "entropy",
    "joint_entropy",
    "mutual_information",
    "conditional_mutual_information",
    "total_correlation",
    "co_information",
    "net_synergy",
    "specific_information",
    "pid_synergy",
]


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _codes(*xs) -> np.ndarray:
    """Pack one or more label arrays into a single 1-D array of row-codes.

    Each column is a variable; identical joint rows map to the same integer
    code. This reduces every entropy to a 1-D histogram while supporting
    multi-column (joint) variables transparently.
    """
    cols = []
    for x in xs:
        a = _to_np(x)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        elif a.ndim != 2:
            raise ValueError(f"expected 1-D or 2-D labels, got ndim={a.ndim}")
        cols.append(a)
    n = cols[0].shape[0]
    for a in cols:
        if a.shape[0] != n:
            raise ValueError("all variables must have the same number of samples")
    matrix = np.concatenate(cols, axis=1)
    _, inverse = np.unique(matrix, axis=0, return_inverse=True)
    return inverse.reshape(-1)


def _entropy_from_codes(codes: np.ndarray, base: float) -> float:
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * (np.log(p) / np.log(base))).sum())


def entropy(x, base: float = 2.0) -> float:
    """Shannon entropy H(X). ``x`` may be 1-D or a 2-D joint (N, k)."""
    return _entropy_from_codes(_codes(x), base)


def joint_entropy(*xs, base: float = 2.0) -> float:
    """Joint entropy H(X1, …, Xk)."""
    return _entropy_from_codes(_codes(*xs), base)


def mutual_information(x, y, base: float = 2.0) -> float:
    """I(X;Y) = H(X) + H(Y) − H(X,Y). Either argument may be a joint."""
    return entropy(x, base) + entropy(y, base) - joint_entropy(x, y, base=base)


def conditional_mutual_information(x, y, z, base: float = 2.0) -> float:
    """I(X;Y|Z) = H(X,Z) + H(Y,Z) − H(X,Y,Z) − H(Z)."""
    return (joint_entropy(x, z, base=base)
            + joint_entropy(y, z, base=base)
            - joint_entropy(x, y, z, base=base)
            - entropy(z, base))


def total_correlation(*xs, base: float = 2.0) -> float:
    """Multi-information TC = ΣH(Xi) − H(X1,…,Xn) ≥ 0."""
    return sum(entropy(x, base) for x in xs) - joint_entropy(*xs, base=base)


def co_information(x1, x2, y, base: float = 2.0) -> float:
    """McGill interaction information I(X1;X2;Y) = I(X1;X2) − I(X1;X2|Y).

    Sign: > 0 ⇒ the three variables are net-redundant; < 0 ⇒ net-synergistic.
    """
    return (mutual_information(x1, x2, base)
            - conditional_mutual_information(x1, x2, y, base))


def net_synergy(x1, x2, y, base: float = 2.0) -> float:
    """I(X1X2;Y) − I(X1;Y) − I(X2;Y) = −co_information(X1;X2;Y).

    The whole-minus-sum: positive when the pair predicts Y beyond what the
    marginals do (synergy), negative when they overlap (redundancy).
    """
    return -co_information(x1, x2, y, base)


def specific_information(source, target, target_value, base: float = 2.0) -> float:
    """DeWeese–Meister specific information I(Y=y0; X) about one target value.

        I(y0; X) = Σ_x p(x|y0) · log[ p(x|y0) / p(x) ]   (= D_KL(p(X|y0)‖p(X)))

    Averaging over y with weights p(y) recovers I(X;Y). This is the per-target
    surprise term the Williams–Beer redundancy minimises over sources.
    """
    sx = _codes(source)
    sy = _codes(target)
    n = sx.shape[0]
    px = np.bincount(sx) / n
    mask = sy == target_value
    if not mask.any():
        return 0.0
    sub = sx[mask]
    px_given = np.bincount(sub, minlength=px.shape[0]) / sub.shape[0]
    nz = px_given > 0
    return float(np.sum(px_given[nz]
                        * (np.log(px_given[nz] / px[nz]) / np.log(base))))


def pid_synergy(x1, x2, y, base: float = 2.0) -> Dict[str, float]:
    """Williams–Beer Partial Information Decomposition for two sources.

    Redundancy is the I_min over sources of the per-target specific information;
    the unique and synergy atoms follow from the PID lattice identities. The
    four atoms sum to I(X1X2;Y).

    Returns ``{redundancy, unique1, unique2, synergy, total}`` in ``base`` bits.
    """
    sy = _codes(y)
    n = sy.shape[0]
    vals, counts = np.unique(sy, return_counts=True)
    py = counts / n

    redundancy = 0.0
    for v, p in zip(vals, py):
        i1 = specific_information(x1, y, v, base)
        i2 = specific_information(x2, y, v, base)
        redundancy += p * min(i1, i2)

    i1y = mutual_information(x1, y, base)
    i2y = mutual_information(x2, y, base)
    pair = np.stack([_codes(x1), _codes(x2)], axis=1)
    i_pair_y = mutual_information(pair, y, base)

    unique1 = i1y - redundancy
    unique2 = i2y - redundancy
    synergy = i_pair_y - redundancy - unique1 - unique2
    return {
        "redundancy": float(redundancy),
        "unique1": float(unique1),
        "unique2": float(unique2),
        "synergy": float(synergy),
        "total": float(i_pair_y),
    }
