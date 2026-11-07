# -*- coding: utf-8 -*-
"""Moiré E-I bilayer: rhythms from structure, not from tuned oscillators (H70).

Two rings of units share a circumference C = n_a. Layer A sits at
0, 1, …, n_a−1 and layer B at spacing C/n_b, so the mismatch is
δ = n_a/n_b − 1. Within a layer, units couple to their nearest
neighbours (t, a tight-binding ring). Between layers, excitatory units
couple by distance: g·exp(−d²/2s²), d measured around the ring. This
is the 1-D analogue of twisted-bilayer tunnelling.

Each unit is a Wilson-Cowan E-I pair (rates, tanh transfer):

    τ_E dE = −E + tanh((w_EE + W) E − w_EI I) + noise
    τ_I dI = −I + tanh(w_IE E − w_II I)

No unit has a natural frequency. The rhythm is the E-I loop's (PING):
it comes from τ_E, τ_I and the loop gains. Linearised at rest, every
eigenmode κ of the lateral matrix W gives one 2×2 system

    M(κ) = [[(w_EE + κ − 1)/τ_E, −w_EI/τ_E], [w_IE/τ_I, −(1 + w_II)/τ_I]],

so the spectrum of W is the band structure and M turns each band into
a damped rhythm. The Hopf point of the top mode is
w_EE + κ_max = a_c = 1 + (τ_E/τ_I)(1 + w_II). ``criticality`` sets
w_EE = criticality·a_c − κ_max: 1 is the Hopf point, < 1 is a damped
rhythm ringing under noise.

The measures for "does the moiré do anything" are:
- ``top_dos``: how many spatial modes share the top band, i.e. how
  many patterns ring at nearly the leading rhythm (a flat band);
- ``ipr``: whether the leading mode is global (1/N) or localised (→ 1).
Both are read against layers with randomly placed B units (same n_b
and g, no moiré geometry).

Kagome (H71): an exactly flat band without any twist. Under lateral
inhibition (W = −c·A) it is the top of the spectrum, so its compact
hexagon modes are the ones that ring. ``assembly_sweep`` asks whether
noise-driven activity settles into localised, seed-dependent
assemblies sharing one rhythm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class EIParams:
    tau_e: float = 3.0
    """ms, AMPA-scale population time constant."""
    tau_i: float = 6.0
    """ms, GABA_A-scale."""
    w_ei: float = 2.0
    w_ie: float = 1.5
    w_ii: float = 0.5
    criticality: float = 0.95


def _ring_positions(n: int, circumference: float) -> np.ndarray:
    return np.arange(n) * circumference / n


def shuffled_positions(n_a: int, n_b: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.uniform(0.0, float(n_a), n_b))


def _ring(n: int, t: float) -> np.ndarray:
    R = np.zeros((n, n))
    for i in range(n):
        R[i, (i + 1) % n] += t
        R[i, (i - 1) % n] += t
    return R


def lateral_matrix(n_a: int, n_b: int, g: float, t: float = 1.0,
                   s: float = 0.5,
                   positions_b: Optional[np.ndarray] = None) -> np.ndarray:
    C = float(n_a)
    W = np.zeros((n_a + n_b, n_a + n_b))
    W[:n_a, :n_a] = _ring(n_a, t)
    if n_b == 0:
        return W
    W[n_a:, n_a:] = _ring(n_b, t)
    xb = _ring_positions(n_b, C) if positions_b is None else positions_b
    d = np.abs(_ring_positions(n_a, C)[:, None] - xb[None, :])
    d = np.minimum(d, C - d)
    G = g * np.exp(-d * d / (2 * s * s))
    W[:n_a, n_a:] = G
    W[n_a:, :n_a] = G.T
    return W


def hopf_drive(p: EIParams) -> float:
    return 1.0 + (p.tau_e / p.tau_i) * (1.0 + p.w_ii)


def ei_eigs(kappa: float, w_ee: float, p: EIParams) -> np.ndarray:
    a = w_ee + kappa
    tr = (a - 1.0) / p.tau_e - (1.0 + p.w_ii) / p.tau_i
    det = ((a - 1.0) * -(1.0 + p.w_ii) + p.w_ei * p.w_ie) / (p.tau_e * p.tau_i)
    root = np.sqrt(complex(tr * tr / 4.0 - det))
    return np.array([tr / 2.0 + root, tr / 2.0 - root])


def jacobian(W: np.ndarray, w_ee: float, p: EIParams) -> np.ndarray:
    n = len(W)
    eye = np.eye(n)
    return np.block([
        [((w_ee - 1.0) * eye + W) / p.tau_e, -p.w_ei * eye / p.tau_e],
        [p.w_ie * eye / p.tau_i, -(1.0 + p.w_ii) * eye / p.tau_i],
    ])


def _lead_eig(p: EIParams) -> complex:
    return complex(ei_eigs(0.0, p.criticality * hopf_drive(p), p)[0])


def rhythm_hz(p: EIParams) -> float:
    """Frequency of the top mode, whatever W is: w_EE places it."""
    return abs(_lead_eig(p).imag) * 1000.0 / (2 * math.pi)


def spectrum(W: np.ndarray, p: EIParams) -> Dict:
    kappa, vecs = np.linalg.eigh(W)
    w_ee = p.criticality * hopf_drive(p) - kappa[-1]
    return {"kappa": kappa[::-1], "vecs": vecs[:, ::-1], "w_ee": w_ee,
            "lead": complex(ei_eigs(kappa[-1], w_ee, p)[0])}


def ipr(v: np.ndarray) -> float:
    q = np.abs(np.asarray(v, dtype=float)) ** 2
    return float((q * q).sum() / q.sum() ** 2)


def top_dos(kappa: np.ndarray, eps_frac: float = 0.01) -> int:
    hi, lo = float(np.max(kappa)), float(np.min(kappa))
    return int(np.sum(kappa >= hi - eps_frac * (hi - lo) - 1e-12))


def _top(W: np.ndarray, eps_frac: float):
    kappa, vecs = np.linalg.eigh(W)
    return top_dos(kappa, eps_frac), ipr(vecs[:, -1])


def mismatch_sweep(n_a: int, n_bs: Sequence[int], g: float, n_null: int = 20,
                   seed: int = 0, eps_frac: float = 0.01, t: float = 1.0,
                   s: float = 0.5) -> List[Dict]:
    rows = []
    for n_b in n_bs:
        dos, lead = _top(lateral_matrix(n_a, n_b, g, t, s), eps_frac)
        rng = np.random.default_rng([seed, n_b])
        nulls = [_top(lateral_matrix(n_a, n_b, g, t, s,
                                     shuffled_positions(n_a, n_b, rng)),
                      eps_frac) for _ in range(n_null)]
        rows.append({"n_b": n_b, "delta": n_a / n_b - 1, "top_dos": dos,
                     "lead_ipr": lead,
                     "null_top_dos": [d for d, _ in nulls],
                     "null_lead_ipr": [q for _, q in nulls]})
    return rows


def localization_sweep(n_a: int, n_b: int, gs: Sequence[float],
                       n_null: int = 20, seed: int = 0, t: float = 1.0,
                       s: float = 0.5) -> List[Dict]:
    rows = []
    for i, g in enumerate(gs):
        _, lead = _top(lateral_matrix(n_a, n_b, g, t, s), 0.01)
        rng = np.random.default_rng([seed, i])
        nulls = [_top(lateral_matrix(n_a, n_b, g, t, s,
                                     shuffled_positions(n_a, n_b, rng)), 0.01)[1]
                 for _ in range(n_null)]
        rows.append({"g": g, "lead_ipr": lead, "null_lead_ipr": nulls})
    return rows


def _kagome_site(L: int, i: int, j: int, s: int) -> int:
    return 3 * ((i % L) * L + (j % L)) + s


def kagome_adjacency(L: int) -> np.ndarray:
    """L×L cells on a torus, sites A (cell origin), B (+a1/2), C (+a2/2)."""
    A = np.zeros((3 * L * L, 3 * L * L))

    def bond(a, b):
        A[a, b] = A[b, a] = 1.0

    for i in range(L):
        for j in range(L):
            a, b, c = (_kagome_site(L, i, j, s) for s in range(3))
            bond(a, b)
            bond(a, c)
            bond(b, c)
            bond(b, _kagome_site(L, i + 1, j, 0))
            bond(c, _kagome_site(L, i, j + 1, 0))
            bond(b, _kagome_site(L, i + 1, j - 1, 2))
    return A


def kagome_hexagon(L: int, i: int, j: int) -> List[int]:
    """The six sites around the hexagon centred at R + (a1+a2)/2, in
    ring order."""
    s = lambda di, dj, sub: _kagome_site(L, i + di, j + dj, sub)
    return [s(0, 0, 1), s(1, 0, 0), s(1, 0, 2), s(0, 1, 1), s(0, 1, 0),
            s(0, 0, 2)]


def simulate(W: np.ndarray, p: EIParams, ms: float, dt: float = 0.05,
             noise: float = 0.02, seed: int = 0,
             record_every: int = 10) -> np.ndarray:
    """Euler-Maruyama on the nonlinear rates from rest. Returns E,
    (steps // record_every, n)."""
    rng = np.random.default_rng(seed)
    n = len(W)
    A = (p.criticality * hopf_drive(p) - np.linalg.eigvalsh(W)[-1]) * np.eye(n) + W
    E = np.zeros(n)
    I = np.zeros(n)
    out = []
    sq = noise * math.sqrt(dt)
    for k in range(int(round(ms / dt))):
        dE = (-E + np.tanh(A @ E - p.w_ei * I)) / p.tau_e
        dI = (-I + np.tanh(p.w_ie * E - p.w_ii * I)) / p.tau_i
        E = E + dt * dE + sq * rng.standard_normal(n)
        I = I + dt * dI
        if k % record_every == 0:
            out.append(E.copy())
    return np.array(out)


def peak_hz(x: np.ndarray, dt_ms: float, n_seg: int = 8) -> float:
    """Welch peak: mean periodogram over ``n_seg`` Hann segments."""
    x = np.asarray(x, dtype=float)
    L = len(x) // n_seg
    p = 0.0
    for k in range(n_seg):
        seg = x[k * L:(k + 1) * L]
        seg = seg - seg.mean()
        p = p + np.abs(np.fft.rfft(seg * np.hanning(L))) ** 2
    f = np.fft.rfftfreq(L, d=dt_ms / 1000.0)
    return float(f[1:][int(np.argmax(p[1:]))])


def assembly_stats(E: np.ndarray, discard: float = 0.5) -> Dict:
    """Where the activity lives after the transient: share of E
    variance carried by the top tenth of units, and the per-unit
    amplitude pattern (std)."""
    X = np.asarray(E, dtype=float)[int(len(E) * discard):]
    var = X.var(axis=0)
    k = max(1, int(math.ceil(len(var) / 10)))
    total = float(var.sum())
    share = float(np.sort(var)[-k:].sum() / total) if total > 0 else 0.0
    return {"top10_share": share, "pattern": np.sqrt(var).tolist()}


def pattern_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def assembly_sweep(W: np.ndarray, p: EIParams, seeds: Sequence[int],
                   ms: float, dt: float = 0.05, noise: float = 0.02,
                   record_every: int = 10) -> Dict:
    """One noise-driven run per seed: is the activity localised, does
    each seed settle on a different pattern, and at what rhythm?"""
    runs = []
    for seed in seeds:
        E = simulate(W, p, ms, dt, noise, seed, record_every)
        st = assembly_stats(E)
        top = int(np.argmax(st["pattern"]))
        half = E[len(E) // 2:, top]
        runs.append({"seed": seed, "top10_share": st["top10_share"],
                     "peak_hz": peak_hz(half, dt * record_every),
                     "pattern": st["pattern"]})
    pairs = [pattern_overlap(a["pattern"], b["pattern"])
             for i, a in enumerate(runs) for b in runs[i + 1:]]
    return {"runs": runs,
            "mean_overlap": float(np.mean(pairs)) if pairs else 1.0}
