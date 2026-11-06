"""Consciousness / IIT measures over a run's per-tick state.

All operate on recorded time series (T ticks × n channels), so they can
be computed after any run:

- ``phi_ar``: autoregressive integrated information (Barrett & Seth 2011)
- ``integration`` / ``neural_complexity``: Tononi–Sporns–Edelman (1994)
- ``lz_state_complexity``: Lempel–Ziv complexity of the binarised state
  stream (as in PCI, Casali et al. 2013, without the perturbation)
- ``ignition_stats`` / ``phase_locking``: global-workspace signatures
- ``thought_quality``: the stream of thought itself

Gaussian estimators throughout: exact for linear-Gaussian dynamics,
a lower bound otherwise. Constant channels carry no information and are
dropped before estimation. An estimate from fewer than
``MIN_OBS_PER_PARAM`` observations per fitted parameter is refused
(None): run 59316881 fitted a 12×12 VAR(1) from 119 transitions.

A value means something only against its null: ``null_test`` scores it
against surrogates (``phase_randomise``: same spectra, no coupling;
``time_shuffle``: same states, no order), ``block_bootstrap_ci`` gives
its sampling interval, and ``access_test`` asks global-workspace
theory's question — does ignited content become accessible?
"""
from __future__ import annotations

import itertools
import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

_RIDGE = 1e-9
MIN_OBS_PER_PARAM = 10


def lz76_complexity(bits: Sequence[int]) -> int:
    """Lempel-Ziv (1976) complexity — number of distinct patterns in
    the exhaustive parsing (Kaspar & Schuster 1987 algorithm)."""
    s = list(bits)
    n = len(s)
    if n == 0:
        return 0
    if n == 1:
        return 1
    i, k, l, c, k_max = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            k_max = max(k_max, k)
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


def _varying(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        return np.zeros((X.shape[0] if X.ndim else 0, 0))
    return X[:, X.std(axis=0) > 1e-12]


def _logdet(S: np.ndarray) -> float:
    sign, ld = np.linalg.slogdet(S + _RIDGE * np.eye(len(S)))
    return ld if sign > 0 else -np.inf


def _past_present_mi(Xp: np.ndarray, Xf: np.ndarray) -> float:
    """I(past; present) for a Gaussian process: ½ log det Σ(X_t) /
    det Σ(X_t | X_{t−τ})."""
    n = Xp.shape[1]
    S = np.cov(np.hstack([Xf, Xp]).T)
    Sff, Sfp, Spp = S[:n, :n], S[:n, n:], S[n:, n:]
    Sres = Sff - Sfp @ np.linalg.solve(Spp + _RIDGE * np.eye(n), Sfp.T)
    return 0.5 * (_logdet(Sff) - _logdet(Sres))


def _transitions(T: int, tau: int, ticks) -> tuple:
    """Row index pairs (past, present) exactly ``tau`` ticks apart."""
    if ticks is None:
        cur = np.arange(tau, T)
        return cur - tau, cur
    pos = {int(t): i for i, t in enumerate(ticks)}
    pairs = [(pos[int(t) - tau], i) for i, t in enumerate(ticks)
             if int(t) - tau in pos]
    if not pairs:
        return np.array([], int), np.array([], int)
    a, b = zip(*pairs)
    return np.array(a), np.array(b)


def phi_ar(X, tau: int = 1, ticks: Optional[Sequence[int]] = None) -> Dict:
    """Φ = I(X_past; X_now) − Σ_k I(M^k_past; M^k_now) at the minimum-
    information bipartition, chosen by Φ / min(|M¹|, |M²|) (size-
    normalised, so a lone unit does not trivially win). ``ticks``: each
    row's tick — only rows exactly ``tau`` ticks apart form transitions
    (a gap is not a step). Returns ``{"phi", "mib", "channels",
    "n_transitions", "reason"}``; phi None with fewer than two varying
    channels or fewer than MIN_OBS_PER_PARAM transitions per VAR
    coefficient (n²)."""
    Xa = np.asarray(X, dtype=float)
    keep = (np.flatnonzero(Xa.std(axis=0) > 1e-12)
            if Xa.ndim == 2 and len(Xa) > 1 else np.array([], int))
    n = len(keep)
    past, now = _transitions(len(Xa) if Xa.ndim == 2 else 0, tau, ticks)
    out = {"phi": None, "mib": None, "channels": keep.tolist(),
           "n_transitions": int(len(now)), "reason": None}
    if n < 2:
        out["reason"] = "fewer than two varying channels"
        return out
    if len(now) < MIN_OBS_PER_PARAM * n * n:
        out["reason"] = (f"{len(now)} transitions < "
                         f"{MIN_OBS_PER_PARAM}·{n}² VAR coefficients")
        return out
    Z = Xa[:, keep]
    Z = (Z - Z.mean(0)) / Z.std(0)
    Xp, Xf = Z[past], Z[now]
    whole = _past_present_mi(Xp, Xf)
    best = None
    for r in range(1, n // 2 + 1):
        for part in itertools.combinations(range(n), r):
            a = list(part)
            b = [i for i in range(n) if i not in part]
            if r == n - r and 0 not in a:
                continue                         # each split once
            phi = whole - _past_present_mi(Xp[:, a], Xf[:, a]) \
                - _past_present_mi(Xp[:, b], Xf[:, b])
            score = phi / min(len(a), len(b))
            if best is None or score < best[0]:
                best = (score, phi, a, b)
    _, phi, a, b = best
    out.update(phi=float(phi),
               mib=[[int(keep[i]) for i in a], [int(keep[i]) for i in b]])
    return out


def _enough_for_covariance(Z: np.ndarray) -> bool:
    n = Z.shape[1]
    return Z.shape[0] >= MIN_OBS_PER_PARAM * n * (n + 1) // 2


def integration(X) -> float:
    """Multi-information I(X) = Σ H(x_i) − H(X) = −½ log det R (nats);
    None below MIN_OBS_PER_PARAM samples per covariance entry."""
    Z = _varying(X)
    if Z.shape[1] < 2:
        return 0.0
    if not _enough_for_covariance(Z):
        return None
    return float(-0.5 * _logdet(np.corrcoef(Z.T)))


def neural_complexity(X, max_subsets: int = 200, seed: int = 0) -> float:
    """C_N = Σ_k [(k/n)·I(X) − ⟨I(X_j^k)⟩], the mean over subsets of
    size k (exact up to ``max_subsets`` per size, sampled beyond); None
    below MIN_OBS_PER_PARAM samples per covariance entry."""
    Z = _varying(X)
    n = Z.shape[1]
    if n < 2:
        return 0.0
    if not _enough_for_covariance(Z):
        return None
    R = np.corrcoef(Z.T)
    i_all = float(-0.5 * _logdet(R))
    rng = np.random.default_rng(seed)
    total = 0.0
    for k in range(2, n + 1):
        combos = list(itertools.combinations(range(n), k))
        if len(combos) > max_subsets:
            idx = rng.choice(len(combos), max_subsets, replace=False)
            combos = [combos[i] for i in idx]
        mean_i = float(np.mean([-0.5 * _logdet(R[np.ix_(c, c)])
                                for c in combos]))
        total += (k / n) * i_all - mean_i
    return float(total)


def lz_state_complexity(X) -> float:
    """Normalised LZ76 complexity of the state stream: each varying
    channel binarised at its median, flattened time-major, c·log2(L) /
    (L·H(p)). 0 for a constant stream, ≈1 for an unstructured one."""
    Z = _varying(X)
    if Z.shape[1] == 0:
        return 0.0
    bits = (Z > np.median(Z, axis=0)).astype(int).ravel()
    L = len(bits)
    p = bits.mean()
    if L < 2 or p <= 0.0 or p >= 1.0:
        return 0.0
    h = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    return float(lz76_complexity(bits.tolist()) * math.log2(L) / (L * h))


def ignition_stats(sync_r: Sequence[float],
                   ignited: Sequence[bool]) -> Dict:
    """Ignition rate, mean ignition-episode length (ticks),
    metastability (sd of R over time) and Sarle's bimodality
    coefficient of R (> 5/9 ≈ 0.555 suggests all-or-none ignition)."""
    r = np.asarray(sync_r, dtype=float)
    ig = np.asarray(ignited, dtype=bool)
    runs, cur = [], 0
    for x in ig:
        if x:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    n = len(r)
    bc = None
    if n > 3 and r.std() > 0:
        d = r - r.mean()
        m2 = (d ** 2).mean()
        g = (d ** 3).mean() / m2 ** 1.5
        k = (d ** 4).mean() / m2 ** 2 - 3.0
        bc = float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2
                                   / ((n - 2) * (n - 3))))
    return {"ignition_rate": float(ig.mean()) if len(ig) else None,
            "mean_episode": float(np.mean(runs)) if runs else 0.0,
            "metastability": float(r.std()) if n else None,
            "mean_sync": float(r.mean()) if n else None,
            "bimodality": bc}


def phase_locking(phases) -> List[List[float]]:
    """Pairwise phase-locking value |⟨e^{i(φ_m − φ_n)}⟩_t| between
    modules' mean phases (T × M)."""
    P = np.exp(1j * np.asarray(phases, dtype=float))
    M = P.shape[1]
    return [[float(abs(np.mean(P[:, m] * np.conj(P[:, k]))))
             for k in range(M)] for m in range(M)]


def _cos(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def thought_quality(thoughts: Sequence[str],
                    embed_fn: Callable[[str], Sequence[float]],
                    context: Optional[Sequence[Sequence[str]]] = None,
                    repeat_cos: float = 0.95) -> Dict:
    """The stream of thought: per-thought novelty (1 − max cos to every
    earlier thought), coherence (cos to the previous thought),
    repetition (an earlier thought at cos ≥ ``repeat_cos``), grounding
    (max cos to that tick's ``context`` — percepts, recalled episodes),
    drift (1 − cos(last, first)) and distinct-2 over all thoughts."""
    vecs = [list(embed_fn(t)) for t in thoughts]
    nov, coh, rep = [], [], []
    for i in range(1, len(vecs)):
        best = max(_cos(vecs[i], v) for v in vecs[:i])
        nov.append(max(0.0, 1.0 - best))
        coh.append(_cos(vecs[i], vecs[i - 1]))
        rep.append(best >= repeat_cos)
    ground: List[Optional[float]] = []
    for i, v in enumerate(vecs):
        ctx = list(context[i]) if context is not None and i < len(context) \
            else []
        ground.append(max(_cos(v, embed_fn(c)) for c in ctx) if ctx
                      else None)
    g = [x for x in ground if x is not None]
    bigrams = [tuple(w[j:j + 2]) for t in thoughts
               for w in [t.lower().split()] for j in range(len(w) - 1)]

    def mean(xs):
        return float(np.mean(xs)) if len(xs) else None
    return {"n": len(thoughts),
            "mean_novelty": mean(nov), "novelty": nov,
            "mean_coherence": mean(coh), "coherence": coh,
            "repetition_rate": mean([float(x) for x in rep]),
            "grounding": ground, "mean_grounding": mean(g),
            "drift": (1.0 - _cos(vecs[-1], vecs[0])) if len(vecs) > 1
            else None,
            "distinct_2": (len(set(bigrams)) / len(bigrams)) if bigrams
            else None}


# ── Nulls, intervals, and the access test ───────────────────────────

def phase_randomise(X, rng) -> np.ndarray:
    """Each channel's Fourier phases replaced by independent uniform
    ones: every channel keeps its power spectrum (autocorrelation), the
    channels lose their coupling (Theiler et al. 1992)."""
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    out = np.empty_like(X)
    for k in range(X.shape[1]):
        m = X[:, k].mean()
        F = np.fft.rfft(X[:, k] - m)
        ph = rng.uniform(0.0, 2 * np.pi, len(F))
        ph[0] = 0.0
        if T % 2 == 0:
            ph[-1] = 0.0
        out[:, k] = np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=T) + m
    return out


def time_shuffle(X, rng) -> np.ndarray:
    """The rows (whole states) in random order: the same states, no
    temporal structure."""
    X = np.asarray(X, dtype=float)
    return X[rng.permutation(X.shape[0])]


def null_test(stat: Callable, X, surrogate: Callable, n: int = 100,
              seed: int = 0) -> Dict:
    """The statistic against ``n`` surrogates: z = (value − null mean) /
    null sd and the one-sided p = (1 + #null ≥ value) / (1 + #null)."""
    value = stat(np.asarray(X, dtype=float))
    out = {"value": value, "null_mean": None, "null_sd": None, "z": None,
           "p": None, "n_null": 0}
    if value is None:
        return out
    rng = np.random.default_rng(seed)
    null = [v for v in (stat(surrogate(X, rng)) for _ in range(int(n)))
            if v is not None and np.isfinite(v)]
    if not null:
        return out
    mu, sd = float(np.mean(null)), float(np.std(null))
    out.update(null_mean=mu, null_sd=sd, n_null=len(null),
               z=(float(value) - mu) / sd if sd > 0 else None,
               p=(1 + sum(v >= value for v in null)) / (1 + len(null)))
    return out


def block_bootstrap_ci(stat: Callable, X, n: int = 200,
                       block: Optional[int] = None, alpha: float = 0.05,
                       seed: int = 0) -> Optional[List[float]]:
    """Moving-block bootstrap (Künsch 1989) percentile interval; blocks
    of ~T^(1/3) rows keep the short-range dependence. None when the
    statistic is undefined on most resamples."""
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    b = int(block or max(1, round(T ** (1 / 3))))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n)):
        starts = rng.integers(0, T - b + 1, int(math.ceil(T / b)))
        idx = np.concatenate([np.arange(s0, s0 + b) for s0 in starts])[:T]
        v = stat(X[idx])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < max(10, n // 2):
        return None
    return [float(np.quantile(vals, alpha / 2)),
            float(np.quantile(vals, 1 - alpha / 2))]


def access_test(events: Sequence[tuple], n_perm: int = 1000,
                n_strata: int = 5, seed: int = 0) -> Dict:
    """Global-workspace access (Dehaene & Naccache 2001): does content
    held on an ignited tick become accessible more often than content
    held without ignition? ``events``: (ignited, accessed, salience).
    Salience drives both, so the difference is taken within salience
    strata (quantiles) and pooled with Cochran–Mantel–Haenszel weights
    n₁n₀/(n₁+n₀); the null shuffles ignition within strata. One-sided
    p: ignition predicts more access."""
    ev = [(bool(i), bool(a), float(s)) for i, a, s in events]
    out = {"n_ignited": sum(i for i, _, _ in ev),
           "n_not": sum(not i for i, _, _ in ev),
           "p_access_ignited": None, "p_access_not": None,
           "diff": None, "p": None}
    if out["n_ignited"]:
        out["p_access_ignited"] = float(np.mean([a for i, a, _ in ev if i]))
    if out["n_not"]:
        out["p_access_not"] = float(np.mean([a for i, a, _ in ev
                                             if not i]))
    if not ev:
        return out
    sal = np.array([s for _, _, s in ev])
    edges = np.quantile(sal, np.linspace(0, 1, n_strata + 1)[1:-1])
    stratum = np.searchsorted(edges, sal, side="right")
    ign = np.array([i for i, _, _ in ev])
    acc = np.array([a for _, a, _ in ev], dtype=float)
    groups = [np.flatnonzero(stratum == k) for k in range(n_strata)]

    def pooled(lab: np.ndarray) -> Optional[float]:
        num = den = 0.0
        for g in groups:
            l1 = lab[g]
            l0 = ~l1
            n1, n0 = int(l1.sum()), int(l0.sum())
            if n1 and n0:
                w = n1 * n0 / (n1 + n0)
                num += w * (acc[g][l1].mean() - acc[g][l0].mean())
                den += w
        return num / den if den else None

    diff = pooled(ign)
    if diff is None:
        return out
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(int(n_perm)):
        lab = ign.copy()
        for g in groups:
            lab[g] = rng.permutation(lab[g])
        v = pooled(lab)
        if v is not None:
            null.append(v)
    out.update(diff=float(diff),
               p=(1 + sum(v >= diff for v in null)) / (1 + len(null)))
    return out
