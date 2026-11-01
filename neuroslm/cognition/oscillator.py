# -*- coding: utf-8 -*-
"""Oscillatory global workspace — Kuramoto phase model (§14.13).

Each runtime module (sensory, memory, language, selection, novelty)
owns a population of gamma-band phase oscillators. One runtime tick
integrates ``ms_per_tick`` milliseconds of phase dynamics (Euler-
Maruyama, ``dt_ms`` steps):

    dθ_i = ω_i·s_GABA dt + (K·s_ACh / N) Σ_j a_i a_j sin(θ_j − θ_i) dt
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

Operating point (coupling_ratio=2.0, ignition_threshold=0.25) was
fixed BEFORE any outcome was measured, from reference activation
patterns of the live runtime (run 58670004): dull idle thought
R≈0.13 < θ < novel idle thought R≈0.29 < reply R≈0.50 — ignition then
discriminates content, and sits close enough to θ for ACh to tip it.

Phases persist across ticks; noise and natural frequencies come from
a seeded generator, so a runtime is reproducible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

MODULES = ("sensory", "memory", "language", "selection", "novelty")


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


class GammaWorkspace:
    def __init__(self, n_per_module: int = 8, f0_hz: float = 40.0,
                 f_sd_hz: float = 4.0, coupling_ratio: float = 2.0,
                 noise: float = 2.0, ms_per_tick: int = 250,
                 dt_ms: float = 1.0, ignition_threshold: float = 0.25,
                 window_ms: int = 50, ach_gain: float = 1.0,
                 gaba_gain: float = 1.0, nt_feedback: bool = True,
                 ach0: float = 0.30, gaba0: float = 0.15,
                 ignition_mode: str = "adaptive",
                 ignition_floor: float = 0.1, adapt_k: float = 1.0,
                 adapt_alpha: float = 0.2, seed: int = 0) -> None:
        if ignition_mode not in ("fixed", "adaptive"):
            raise ValueError(f"ignition_mode {ignition_mode!r}: "
                             f"expected 'fixed' or 'adaptive'")
        self.ignition_mode = ignition_mode
        self.ignition_floor = float(ignition_floor)
        self.adapt_k, self.adapt_alpha = float(adapt_k), float(adapt_alpha)
        self._r_mu: float = float(ignition_threshold)
        self._r_var = 0.05 ** 2
        self._r_seen = False
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
             update_baseline: bool = True) -> OscillatorStep:
        """One tick. ``update_baseline=False`` (reply ticks): the
        sensory module is on, a different regime — the idle baseline is
        neither seeded nor moved; before it exists, decide against the
        fixed bar (run 58683237: a reply-seeded θ=0.50 blocked every
        idle ignition)."""
        a = np.repeat([max(0.0, min(1.0, float(activations.get(m, 0.0))))
                       for m in MODULES], self.n_per_module)
        s_ach, s_gaba = self._scales(nt_levels)
        k = self.K * s_ach / self.N
        omega = self.omega * s_gaba
        sq = self.noise * math.sqrt(self.dt)
        theta = self.theta
        R = np.empty(self.n_steps)
        x = np.empty(self.n_steps)
        for t in range(self.n_steps):
            z = np.sum(a * np.exp(1j * theta))
            R[t] = abs(z) / self.N
            x[t] = float(np.sum(a * np.cos(theta))) / self.N
            # Σ_j a_j sin(θ_j − θ_i) = Im(z · e^{−iθ_i})
            coupling = k * a * np.imag(z * np.exp(-1j * theta))
            theta = theta + (omega + coupling) * self.dt
            if sq:
                theta = theta + sq * self._rng.standard_normal(self.N)
        self.theta = np.mod(theta, 2 * math.pi)

        sync_r = float(np.mean(R[-self.window:]))
        peak_hz, gamma_power = self._spectrum(x)
        if self.ignition_mode == "fixed":
            thr = self.ignition_threshold
            ignited = sync_r >= thr
        else:
            if not update_baseline and not self._r_seen:
                thr = self.ignition_threshold
                ignited = sync_r >= thr
            else:
                if not self._r_seen:
                    self._r_mu, self._r_seen = sync_r, True
                thr = max(self.ignition_floor,
                          self._r_mu + self.adapt_k * math.sqrt(self._r_var))
                ignited = sync_r > thr
                if update_baseline:
                    d = sync_r - self._r_mu
                    self._r_mu += self.adapt_alpha * d
                    self._r_var = (1 - self.adapt_alpha) * (
                        self._r_var + self.adapt_alpha * d * d)
        return OscillatorStep(sync_r=sync_r, ignited=bool(ignited),
                              peak_hz=peak_hz, gamma_power=gamma_power,
                              threshold=float(thr))

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
