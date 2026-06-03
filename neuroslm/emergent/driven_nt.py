# -*- coding: utf-8 -*-
"""C1 — Driven neuromodulators (closed-loop NT observability).

A *homeostatic neuromodulatory field* — the seven classical NT channels
are realised as one leaky-linear (Ornstein–Uhlenbeck) dynamical system
in unbounded logit space, driven by *standardised* training-state
scalars and read out through a single sigmoid:

    z_k(t)   = (d_k(t) - μ̂_k(t)) / (σ̂_k(t) + ε)         # per-driver z-score
    y(t+1)   = (1-α) ⊙ y(t) + α ⊙ (μ + W · z(t))         # OU step
    nt(t)    = σ(y(t))                                     # single readout

This replaces the previous per-channel feedforward squashes which were
brittle by construction: the raw drivers had unknown training-time
scale, so the per-channel tanh/sigmoid would either rail (eCB, Glu,
ACh) or integrate monotonically with no leak (5HT) or hit a hard floor
(GABA). The homeostatic formulation *inherently* prevents all three
pathologies:

  - Drivers are z-scored → the linear map sees deviations, not raw
    magnitudes; σ at the readout stays in its responsive region.
  - α > 0 leak pulls y back to μ on timescale 1/α → no accumulation,
    no monotonic drift, guaranteed recovery from any excursion.
  - μ = logit(baseline) sets the tonic level; W rows encode biology
    (DA loves +surprise, NE loves +gnorm, GABA loves +ignition…).

Drivers (K=5):

    surprise        z-score of -loss against running stats   → DA, 5HT
    grad_norm       z-score of gnorm                         → NE, GABA
    activation      z-score of mean-|h|                      → Glu, eCB
    ignition        z-score of ignition_rate                 → GABA, -ACh
    attn_sharp      z-score of (1 - attn_entropy_norm)       → ACh

API preserved (drop-in for legacy NTSystem):

    DrivenNTSystem(baselines=...)            # baselines map kwarg
    .step(activity=...)                      # legacy shim — activation only
    .step_full(loss, grad_norm, activation, ignition_rate, attn_entropy_norm)
    .levels()                                # 7-key dict in [0,1]
    .baselines                               # property
"""
from __future__ import annotations
import math
from typing import Dict, Optional


_CHANNELS = ("DA", "NE", "5HT", "ACh", "eCB", "Glu", "GABA")
_DRIVERS = ("surprise", "grad_norm", "activation", "ignition", "attn_sharp")


# Per-channel leak rate α (≈ 1/τ in steps).  Smaller = longer memory.
_DEFAULT_ALPHA = {
    "DA":   0.30,   # fast phasic
    "NE":   0.20,
    "5HT":  0.01,   # slow integrator (~100-step time constant)
    "ACh":  0.15,
    "eCB":  0.10,
    "Glu":  0.20,
    "GABA": 0.15,
}

# Driver-to-channel coupling W ∈ R^{7×5}.  Rows in order of _CHANNELS,
# columns in order of _DRIVERS.  Magnitudes are O(1): a z=±2 driver
# produces roughly a 0.10–0.20 deviation in nt-space after the squash.
_DEFAULT_W = {
    #         surprise  gnorm     act       ign       attn
    "DA":   ( +1.20,    0.00,    0.00,    0.00,    0.00),
    "NE":   (  0.00,   +1.20,    0.00,    0.00,    0.00),
    "5HT":  ( +1.00,    0.00,    0.00,    0.00,    0.00),  # same dir as DA but slow
    "ACh":  (  0.00,    0.00,    0.00,   -0.50,   +0.80),
    "eCB":  (  0.00,    0.00,   +1.00,    0.00,    0.00),
    "Glu":  (  0.00,    0.00,   +1.00,    0.00,    0.00),
    "GABA": (  0.00,   +0.40,    0.00,   +0.80,    0.00),
}


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float, eps: float = 1e-4) -> float:
    p = max(eps, min(1.0 - eps, float(p)))
    return math.log(p / (1.0 - p))


class _RunningStats:
    """EMA mean and EMA variance of a scalar driver.

    During warmup (n < 5) returns z=0 so channels remain at their tonic
    baseline rather than being slammed by an ill-estimated mean.
    """
    __slots__ = ("alpha", "mean", "var", "n")

    def __init__(self, alpha: float = 0.02):
        self.alpha = float(alpha)
        self.mean = 0.0
        self.var = 1.0    # safe divisor for the first sample
        self.n = 0

    def update(self, x: float) -> float:
        x = float(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.var = 1.0
            return 0.0
        # Coupled EMA update for mean and variance (Welford-on-EMA).
        delta = x - self.mean
        self.mean += self.alpha * delta
        self.var = (1.0 - self.alpha) * self.var + self.alpha * delta * (x - self.mean)
        if self.n < 5:
            return 0.0
        sd = math.sqrt(max(self.var, 1e-8))
        return (x - self.mean) / sd


class DrivenNTSystem:
    """Seven-channel NT homeostat (OU in logit space, σ at readout).

    Drop-in replacement for ``dsl.metrics.NTSystem``: exposes the same
    ``step(activity=...)`` shim and ``levels()`` accessor.  For the
    full closed-loop driving signal, call :meth:`step_full` with the
    keyword drivers; any missing driver contributes z=0 (so its only
    effect that step is the natural OU leak toward the resting μ).

    Parameters
    ----------
    baselines : dict, optional
        Per-channel resting concentrations in (0,1).  Default values
        match the legacy NTSystem.  Used to set μ = logit(baseline).
    alpha : dict, optional
        Per-channel leak rate override.  Defaults in ``_DEFAULT_ALPHA``.
    W : dict, optional
        Per-channel coupling row override.  Each value is a 5-tuple
        ordered (surprise, gnorm, activation, ignition, attn_sharp).
    driver_alpha : float
        EMA rate for the per-driver running mean/std.  Default 0.005
        ⇒ ~200-step half-life: slow enough that a regime shift gives
        sustained nonzero z for the channels to integrate, fast enough
        to eventually re-baseline so the readout never ratchets.

    Back-compat kwargs (``taus``, ``surprise_window``, ``slow_loss_alpha``,
    ``fast_loss_alpha``, ``gnorm_alpha``, ``ignition_target``) are
    accepted and ignored — the homeostatic formulation makes them
    obsolete.
    """

    _DEFAULT_BASELINES = {
        "DA":   0.15,
        "NE":   0.20,
        "5HT":  0.50,    # raised from 0.35 — 5HT now sits at neutral mood
        "ACh":  0.30,    # raised from 0.25 — ACh responsive in both directions
        "eCB":  0.10,
        "Glu":  0.45,
        "GABA": 0.15,
    }

    def __init__(self,
                 baselines: Optional[Dict[str, float]] = None,
                 alpha: Optional[Dict[str, float]] = None,
                 W: Optional[Dict[str, tuple]] = None,
                 driver_alpha: float = 0.005,
                 # accepted for back-compat with previous signature; ignored.
                 taus: Optional[Dict[str, float]] = None,
                 surprise_window: int = 32,
                 slow_loss_alpha: float = 0.005,
                 fast_loss_alpha: float = 0.1,
                 gnorm_alpha: float = 0.1,
                 ignition_target: float = 0.2):
        # Baselines
        self._baselines = dict(self._DEFAULT_BASELINES)
        if baselines:
            for k, v in baselines.items():
                if k in self._baselines:
                    self._baselines[k] = float(v)
        # Per-channel leak
        self._alpha = dict(_DEFAULT_ALPHA)
        if alpha:
            for k, v in alpha.items():
                if k in self._alpha:
                    self._alpha[k] = float(v)
        # Coupling matrix
        self._W = {k: tuple(_DEFAULT_W[k]) for k in _CHANNELS}
        if W:
            for k, row in W.items():
                if k in self._W and len(row) == len(_DRIVERS):
                    self._W[k] = tuple(float(x) for x in row)

        # State: y in logit space, initialised so σ(y) = baseline.
        self._mu: Dict[str, float] = {k: _logit(self._baselines[k]) for k in _CHANNELS}
        self._y: Dict[str, float] = dict(self._mu)
        self._level: Dict[str, float] = dict(self._baselines)

        # One running standardiser per driver.
        self._stats: Dict[str, _RunningStats] = {
            k: _RunningStats(alpha=driver_alpha) for k in _DRIVERS
        }

        # Retained for API-introspection (not used in dynamics).
        self._ignition_target = float(ignition_target)

    # ── Shim for legacy `metrics.NTSystem.step(activity=...)` ────────

    def step(self, activity: float = 0.0) -> None:
        """Compat path: drives only `activation`; OU leak carries the
        other channels back toward their baseline naturally."""
        self.step_full(activation=float(activity))

    # ── Full closed-loop driver ──────────────────────────────────────

    def step_full(self,
                  loss: Optional[float] = None,
                  grad_norm: Optional[float] = None,
                  activation: Optional[float] = None,
                  ignition_rate: Optional[float] = None,
                  attn_entropy_norm: Optional[float] = None) -> None:
        """Advance one OU step with all available drivers.

        Any driver passed as ``None`` contributes z=0 to its column;
        its corresponding NT channels experience pure leak toward μ.
        """
        # Standardise each driver.  Surprise = -loss (so +z = unusually
        # good loss = drives DA/5HT up).  All others use their raw value.
        z = {k: 0.0 for k in _DRIVERS}
        if loss is not None:
            # Standardise the raw loss, then flip sign so that "loss
            # below recent mean" gives a positive z.
            z["surprise"] = -self._stats["surprise"].update(float(loss))
        if grad_norm is not None:
            z["grad_norm"] = self._stats["grad_norm"].update(float(grad_norm))
        if activation is not None:
            z["activation"] = self._stats["activation"].update(float(activation))
        if ignition_rate is not None:
            z["ignition"] = self._stats["ignition"].update(float(ignition_rate))
        if attn_entropy_norm is not None:
            z["attn_sharp"] = self._stats["attn_sharp"].update(
                1.0 - float(attn_entropy_norm))

        # Saturate z to ±5 so a single training spike can't slam the
        # logit; +5σ is already a once-in-a-million event after warmup.
        zvec = tuple(max(-5.0, min(5.0, z[k])) for k in _DRIVERS)

        # OU update per channel:  y ← (1-α) y + α (μ + W·z),  nt = σ(y).
        for c in _CHANNELS:
            a = self._alpha[c]
            wz = sum(self._W[c][i] * zvec[i] for i in range(len(_DRIVERS)))
            target = self._mu[c] + wz
            self._y[c] = (1.0 - a) * self._y[c] + a * target
            # Clamp logit just for paranoia (|y| > 20 makes σ flat).
            self._y[c] = max(-20.0, min(20.0, self._y[c]))
            self._level[c] = _sigmoid(self._y[c])

    # ── Accessors ────────────────────────────────────────────────────

    def levels(self) -> Dict[str, float]:
        """Return a fresh dict of the seven channel values."""
        return dict(self._level)

    @property
    def baselines(self) -> Dict[str, float]:
        return dict(self._baselines)
