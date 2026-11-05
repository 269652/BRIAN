# -*- coding: utf-8 -*-
"""Oscillatory global workspace — Kuramoto phase model (§14.13).

Each runtime module (sensory, memory, language, selection, novelty)
owns a population of gamma-band phase oscillators. One runtime tick
integrates ``ms_per_tick`` milliseconds of phase dynamics (Euler-
Maruyama, ``dt_ms`` steps):

    dθ_i = ω_i·s_GABA dt + (K·s_ACh / Σ_j a_j) Σ_j a_i a_j sin(θ_j − θ_i) dt
           + σ √dt ξ

where a_i ∈ [0,1] is the activation of oscillator i's module this
tick. The global order parameter

    R(t) = |Σ_j a_j e^{iθ_j}| / N

is high only when many oscillators are both active AND phase-locked;
one perfectly locked module of five caps R at 0.2. **Ignition** =
mean R over the last ``window_ms`` crossing a threshold:

    fixed     R ≥ ignition_threshold
    adaptive  R > θ_t = max(floor, μ_R + k·σ_R)   (default, H66)

where μ_R, σ_R are an exponentially weighted mean/sd of the system's
OWN recent R, updated after each decision. μ_R is seeded with the
FIRST observed R (σ₀ = 0.05), so tick 1 cannot ignite; run 58678026
showed seeding at the old bar spent most of a 12-tick run in a
warm-up transient. ``ignition_threshold`` is the fixed-mode bar. Ignition is a departure from ongoing
activity, not an absolute level — run 58676023 showed the fixed bar
(set from guessed activations) never crossed on R1-7B's idle ticks.
The floor sits above the incoherent finite-size level (≈ ā/√N).

Neuromodulation (``nt_feedback``):
    s_ACh  = max(0, 1 + ach_gain·(ACh − ACh₀)/ACh₀)   coupling gain
    s_GABA = max(0.2, 1 − gaba_gain·(GABA − GABA₀))    frequency scale
Cholinergic enhancement of gamma synchrony and slower gamma under
stronger inhibition (PING) are the biological motivations; the
magnitudes are a-priori design choices, not fits.

K = ``coupling_ratio`` × K_c, with K_c = 2/(π g(0)) the Kuramoto
critical coupling for a Gaussian frequency spread (σ = 2π·f_sd), so
``coupling_ratio`` reads directly as "distance from criticality".

Operating point (2026-11-05): coupling_ratio=1.2 with reverberation
``ignition_gain``=1.0 — while ignited, the coupling is × (1 + gain).
Typical idle activations (~0.55) put a module at ~0.66 K_c (incoherent),
an ignited one at ~1.3 K_c (locked): ignition is all-or-none. Synthetic
activations, 5 seeds: Sarle bimodality of R 0.57 (> 5/9) vs 0.44
without reverberation, episodes ~6 ticks vs 1.6. (The earlier 2.0 was
set for the K/N mean field; under the active-mass field it left idle
modules supercritical and R unimodal.)

Phases persist across ticks; noise and natural frequencies come from
a seeded generator, so a runtime is reproducible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

MODULES = ("sensory", "memory", "language", "selection", "novelty")


def content_coupling_matrix(vectors: Mapping[str, Optional[Sequence[float]]],
                            baseline: Optional[float]
                            ) -> Tuple[List[List[float]], Optional[float]]:
    """Binding by synchrony (Singer & Gray 1995): coupling between two
    modules follows how coherent their contents are this tick,
    C_mn = 2·c/(c + c̄) with c = max(0, cos(v_m, v_n)) and c̄ the run's
    typical coherence — 1 at typical coherence, up to 2 for aligned
    content, 0 for unrelated content. Pairs where either module has no
    content, and self-coupling, stay at 1. Returns (C, mean c over the
    content pairs, or None when there are none)."""
    from neuroslm.cognition.runtime import _cosine as _cos
    from neuroslm.cognition.workspace import divisive_normalise

    M = len(MODULES)
    C = [[1.0] * M for _ in range(M)]
    cs: List[float] = []
    pairs = []
    for i in range(M):
        for j in range(i + 1, M):
            vi, vj = vectors.get(MODULES[i]), vectors.get(MODULES[j])
            if vi and vj:
                c = max(0.0, _cos(vi, vj))
                cs.append(c)
                pairs.append((i, j, c))
    if not cs:
        return C, None
    for i, j, c in pairs:
        C[i][j] = C[j][i] = 2.0 * divisive_normalise(c, baseline)
    return C, sum(cs) / len(cs)


@dataclass
class OscillatorStep:
    sync_r: float
    """Mean global order parameter R over the ignition window."""
    ignited: bool
    peak_hz: float
    """Peak frequency of the population signal Σ a_j cos θ_j / N."""
    threshold: float
    """The ignition threshold applied this tick."""
    gamma_power: float
    """Fraction of (non-DC) spectral power in 30-100 Hz."""
    module_phase: List[Optional[float]] = None  # type: ignore[assignment]
    """Each module's mean phase at the end of the tick (None if silent)."""
    module_r: List[float] = None  # type: ignore[assignment]
    """Each module's internal order parameter |⟨e^{iθ}⟩| (0 if silent)."""


class GammaWorkspace:
    def __init__(self, n_per_module: int = 8, f0_hz: float = 40.0,
                 f_sd_hz: float = 4.0, coupling_ratio: float = 1.2,
                 noise: float = 2.0, ms_per_tick: int = 250,
                 dt_ms: float = 1.0, ignition_threshold: float = 0.25,
                 window_ms: int = 50, ach_gain: float = 1.0,
                 gaba_gain: float = 1.0, nt_feedback: bool = True,
                 ach0: float = 0.30, gaba0: float = 0.15,
                 ignition_mode: str = "adaptive",
                 ignition_floor: float = 0.1, adapt_k: float = 1.0,
                 adapt_alpha: float = 0.2, ignition_gain: float = 1.0,
                 seed: int = 0) -> None:
        if ignition_mode not in ("fixed", "adaptive"):
            raise ValueError(f"ignition_mode {ignition_mode!r}: "
                             f"expected 'fixed' or 'adaptive'")
        self.ignition_mode = ignition_mode
        self.ignition_floor = float(ignition_floor)
        self.adapt_k, self.adapt_alpha = float(adapt_k), float(adapt_alpha)
        self._r_mu: float = float(ignition_threshold)
        self._r_var = 0.05 ** 2
        self._r_seen = False
        self._ignited = False
        # reverberation: an ignited workspace's long-range recurrence
        # multiplies its coupling (a-priori magnitude, like ach_gain)
        self.ignition_gain = float(ignition_gain)
        self.n_per_module = int(n_per_module)
        self.N = self.n_per_module * len(MODULES)
        self.dt = float(dt_ms) / 1000.0
        self.n_steps = max(1, int(round(ms_per_tick / dt_ms)))
        self.window = max(1, min(self.n_steps, int(round(window_ms / dt_ms))))
        self.ignition_threshold = float(ignition_threshold)
        self.noise = float(noise)
        self.ach_gain, self.gaba_gain = float(ach_gain), float(gaba_gain)
        self.ach0, self.gaba0 = float(ach0), float(gaba0)
        self.nt_feedback = bool(nt_feedback)
        sigma = 2 * math.pi * float(f_sd_hz)
        self.critical_coupling = 2 * sigma * math.sqrt(2 * math.pi) / math.pi
        self.K = float(coupling_ratio) * self.critical_coupling
        self._rng = np.random.default_rng(int(seed))
        self.omega = 2 * math.pi * self._rng.normal(f0_hz, f_sd_hz, self.N)
        self.theta = self._rng.uniform(0.0, 2 * math.pi, self.N)

    def _scales(self, nt: Mapping[str, float]):
        if not self.nt_feedback:
            return 1.0, 1.0
        ach = float(nt.get("ACh", self.ach0))
        gaba = float(nt.get("GABA", self.gaba0))
        s_ach = max(0.0, 1.0 + self.ach_gain * (ach - self.ach0) / self.ach0)
        s_gaba = max(0.2, 1.0 - self.gaba_gain * (gaba - self.gaba0))
        return s_ach, s_gaba

    def step(self, activations: Mapping[str, float],
             nt_levels: Mapping[str, float],
             update_baseline: bool = True,
             coupling: Optional[Sequence[Sequence[float]]] = None
             ) -> OscillatorStep:
        """One tick. ``update_baseline=False`` (reply ticks): the
        sensory module is on, a different regime — the idle baseline is
        neither seeded nor moved; before it exists, decide against the
        fixed bar (run 58683237: a reply-seeded θ=0.50 blocked every
        idle ignition).

        ``coupling``: optional |MODULES|×|MODULES| matrix C; oscillator
        i in module m feels Σ_n C_mn · (module n's mean field) instead
        of the global field. None ≡ C = 1 everywhere."""
        a = np.repeat([max(0.0, min(1.0, float(activations.get(m, 0.0))))
                       for m in MODULES], self.n_per_module)
        M = len(MODULES)
        C = None
        if coupling is not None:
            C = np.asarray(coupling, dtype=float)
            if C.shape != (M, M):
                raise ValueError(f"coupling must be {M}×{M}, got {C.shape}")
        s_ach, s_gaba = self._scales(nt_levels)
        # mean field over the mass each oscillator actually HEARS (its
        # in-strength Σ_n C_mn·mass_n): silent populations do not dilute
        # the coupling (run 59294820: K/N left idle ticks at ~0.4 K_c),
        # and a module cut off from the others still locks internally.
        # C = 1: Σa, and Σa = N when everything is on — the classic model.
        mass = a.reshape(M, -1).sum(axis=1)
        heard = mass.sum() if C is None else C @ mass
        k = self.K * s_ach / np.maximum(
            np.repeat(np.broadcast_to(heard, (M,)), self.n_per_module), 1e-9)
        if self._ignited:
            k = k * (1.0 + self.ignition_gain)
        omega = self.omega * s_gaba
        sq = self.noise * math.sqrt(self.dt)
        theta = self.theta
        R = np.empty(self.n_steps)
        x = np.empty(self.n_steps)
        for t in range(self.n_steps):
            phasor = a * np.exp(1j * theta)
            z = np.sum(phasor)
            R[t] = abs(z) / self.N
            x[t] = float(np.sum(a * np.cos(theta))) / self.N
            if C is None:
                field = z
            else:
                field = np.repeat(C @ phasor.reshape(M, -1).sum(axis=1),
                                  self.n_per_module)
            # Σ_j a_j sin(θ_j − θ_i) = Im(z · e^{−iθ_i})
            coupling_term = k * a * np.imag(field * np.exp(-1j * theta))
            theta = theta + (omega + coupling_term) * self.dt
            if sq:
                theta = theta + sq * self._rng.standard_normal(self.N)
        self.theta = np.mod(theta, 2 * math.pi)

        sync_r = float(np.mean(R[-self.window:]))
        peak_hz, gamma_power = self._spectrum(x)
        ignited, thr = self._decide(sync_r, update_baseline)
        phases: List[Optional[float]] = []
        rs: List[float] = []
        for m, z in enumerate(np.exp(1j * self.theta).reshape(M, -1)):
            if a[m * self.n_per_module] <= 0.0:
                phases.append(None)
                rs.append(0.0)
            else:
                mz = z.mean()
                phases.append(float(np.angle(mz)))
                rs.append(float(abs(mz)))
        return OscillatorStep(sync_r=sync_r, ignited=bool(ignited),
                              peak_hz=peak_hz, gamma_power=gamma_power,
                              threshold=float(thr), module_phase=phases,
                              module_r=rs)

    def _decide(self, sync_r: float, update_baseline: bool):
        """Ignition decision → (ignited, threshold). Adaptive mode is
        hysteretic: ENTERING ignition needs R > μ_R + k·σ_R; an ignited
        workspace reverberates while R > μ_R (self-sustaining ignition,
        Dehaene & Changeux 2005)."""
        if self.ignition_mode == "fixed":
            thr = self.ignition_threshold
            return sync_r >= thr, thr
        if not update_baseline and not self._r_seen:
            thr = self.ignition_threshold
            return sync_r >= thr, thr
        if not self._r_seen:
            self._r_mu, self._r_seen = sync_r, True
        margin = 0.0 if self._ignited else self.adapt_k * math.sqrt(
            self._r_var)
        thr = max(self.ignition_floor, self._r_mu + margin)
        ignited = sync_r > thr
        self._ignited = ignited
        if update_baseline:
            d = sync_r - self._r_mu
            self._r_mu += self.adapt_alpha * d
            self._r_var = (1 - self.adapt_alpha) * (
                self._r_var + self.adapt_alpha * d * d)
        return ignited, thr

    def _spectrum(self, x: np.ndarray):
        x = x - x.mean()
        if not np.any(x):
            return 0.0, 0.0
        p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), d=self.dt)
        p, f = p[1:], f[1:]
        total = float(p.sum())
        if total <= 0.0:
            return 0.0, 0.0
        band = (f >= 30.0) & (f <= 100.0)
        return float(f[int(np.argmax(p))]), float(p[band].sum() / total)
