# -*- coding: utf-8 -*-
"""C1 — Driven neuromodulators (closed-loop NT observability).

Replaces the toy `metrics.NTSystem` drift-to-baseline ODE with a system
whose seven channels are *functions of training state*. The intent is
that the NT log column becomes informative — DA spikes on surprise, NE
spikes on grad-norm shocks, GABA falls when the workspace saturates,
etc. — so the bowtie's modulation matrix has real signal to act on once
we promote this into the forward pass in a later PR.

All channels are bounded in [0,1] and update in O(7) per step.

Drivers (each fed via `step(...)` by the metric observer):

    loss           per-step LM loss
    grad_norm      pre-clip gradient L2
    activation     mean |h| of the last layer (already computed)
    ignition       most-recent ignition rate from MetastableIgnition
    attn_entropy   optional, per-step mean attention entropy normalised to log T

The channel formulas are deliberately simple closed forms (see
`docs/EMERGENT_TOPOLOGY.md §C1`). They are *not* learnable: the point of
this layer is to test whether the modulation matrix in `arch.neuro` has
been seeing dead constants. Learnable couplings are Phase 8.
"""
from __future__ import annotations
import math
from collections import deque
from typing import Dict, Optional


# Default time constants — tuned so that NT values typically span
# [0.1, 0.9] under realistic training-loop magnitudes (loss ~ 1–10,
# gnorm ~ 0.3–3, activation ~ 0.1–1).
_DEFAULT_TAUS = {
    "NE":  1.0,    # gradient-norm scale
    "eCB": 1.0,    # activation/√d scale
    "Glu": 1.0,    # activation scale
}


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class _EMA:
    """Scalar EMA with explicit ``alpha`` (smaller alpha = longer memory)."""

    __slots__ = ("alpha", "value", "_initialised")

    def __init__(self, alpha: float = 0.05, initial: float = 0.0):
        self.alpha = float(alpha)
        self.value = float(initial)
        self._initialised = False

    def update(self, x: float) -> float:
        if not self._initialised:
            self.value = float(x)
            self._initialised = True
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * float(x)
        return self.value


class DrivenNTSystem:
    """Seven-channel NT state, driven by training-state scalars.

    Drop-in replacement for `dsl.metrics.NTSystem`: exposes the same
    ``step(activity=...)`` shim (for compatibility with code paths that
    only have access to the activation magnitude) and the same
    ``levels()`` accessor returning a 7-key dict.

    For the full closed-loop driving signal, call :meth:`step_full` with
    keyword arguments. The shim ``step()`` populates only `Glu`, `eCB`,
    and gradually decays the rest toward baseline — used as a fallback
    when the richer drivers aren't available (eval-only contexts).

    Parameters
    ----------
    baselines : dict, optional
        Per-channel resting concentrations; the system contracts toward
        these when no driver signal is present. Defaults match the
        NTSystem legacy values so the column starts in the same place.
    taus : dict, optional
        Scale constants for the saturating drivers (NE, eCB, Glu).
    surprise_window : int
        Window size for the rolling-mean loss used to compute the
        per-step surprise that drives DA.
    """

    _DEFAULT_BASELINES = {
        "DA":   0.15,
        "NE":   0.20,
        "5HT":  0.35,
        "ACh":  0.25,
        "eCB":  0.10,
        "Glu":  0.45,
        "GABA": 0.15,
    }

    def __init__(self,
                 baselines: Optional[Dict[str, float]] = None,
                 taus: Optional[Dict[str, float]] = None,
                 surprise_window: int = 32,
                 slow_loss_alpha: float = 0.005,
                 fast_loss_alpha: float = 0.1,
                 gnorm_alpha: float = 0.1,
                 ignition_target: float = 0.2):
        self._baselines = dict(self._DEFAULT_BASELINES)
        if baselines:
            for k, v in baselines.items():
                if k in self._baselines:
                    self._baselines[k] = float(v)
        self._taus = dict(_DEFAULT_TAUS)
        if taus:
            for k, v in taus.items():
                self._taus[k] = float(v)

        self._level: Dict[str, float] = dict(self._baselines)

        # Driver state.
        self._loss_window: deque = deque(maxlen=int(surprise_window))
        self._loss_fast = _EMA(alpha=fast_loss_alpha)
        self._loss_slow = _EMA(alpha=slow_loss_alpha)
        self._gnorm_ema = _EMA(alpha=gnorm_alpha)
        # Reference loss for 5HT: the slow EMA tracks current loss; we
        # compare against the first ever slow value (the "starting
        # bar"). 5HT rises as long-term loss falls relative to start.
        self._loss_ref: Optional[float] = None
        self._ignition_target = float(ignition_target)

    # ── Shim for legacy `metrics.NTSystem.step(activity=...)` ────────

    def step(self, activity: float = 0.0) -> None:
        """Compat path: drives only `Glu`/`eCB` from a single activation
        magnitude scalar; the other channels relax toward baseline."""
        glu = math.tanh(max(0.0, float(activity)) / max(1e-8, self._taus["Glu"]))
        ecb = math.tanh(max(0.0, float(activity)) / max(1e-8, self._taus["eCB"]))
        self._level["Glu"] = glu
        self._level["eCB"] = ecb
        # Mild relaxation toward baseline for the un-driven channels.
        for k in ("DA", "NE", "5HT", "ACh", "GABA"):
            self._level[k] += 0.05 * (self._baselines[k] - self._level[k])

    # ── Full closed-loop driver ──────────────────────────────────────

    def step_full(self,
                  loss: Optional[float] = None,
                  grad_norm: Optional[float] = None,
                  activation: Optional[float] = None,
                  ignition_rate: Optional[float] = None,
                  attn_entropy_norm: Optional[float] = None) -> None:
        """Advance one training step with all available drivers.

        Any argument left as ``None`` causes its channel to relax toward
        the baseline. So eval-only contexts can pass just `activation`
        and the other channels will gracefully fall back.
        """
        # ── DA: phasic on negative surprise (loss below recent mean) ─
        if loss is not None:
            self._loss_window.append(float(loss))
            self._loss_fast.update(loss)
            self._loss_slow.update(loss)
            if self._loss_ref is None and self._loss_slow._initialised:
                self._loss_ref = self._loss_slow.value
            if len(self._loss_window) >= 4:
                recent = list(self._loss_window)
                mean_l = sum(recent) / len(recent)
                std_l = math.sqrt(
                    sum((x - mean_l) ** 2 for x in recent) / len(recent)
                ) + 1e-6
                surprise = (mean_l - float(loss)) / std_l
                self._level["DA"] = _sigmoid(surprise)
            # ── 5HT: rises as long-term loss falls below the start bar
            if self._loss_ref is not None:
                rel = self._loss_ref - self._loss_slow.value
                # Map to (0,1) via sigmoid; ref-scale = max(1, |ref|)
                scale = max(1.0, abs(self._loss_ref))
                self._level["5HT"] = _sigmoid(rel / scale * 3.0)
        else:
            self._level["DA"] += 0.05 * (self._baselines["DA"] - self._level["DA"])
            self._level["5HT"] += 0.05 * (self._baselines["5HT"] - self._level["5HT"])

        # ── NE: arousal — tanh of grad-norm EMA ──────────────────────
        if grad_norm is not None:
            g = self._gnorm_ema.update(grad_norm)
            self._level["NE"] = math.tanh(max(0.0, g) / max(1e-8, self._taus["NE"]))
        else:
            self._level["NE"] += 0.05 * (self._baselines["NE"] - self._level["NE"])

        # ── eCB & Glu: from activation magnitude ─────────────────────
        if activation is not None:
            a = max(0.0, float(activation))
            self._level["Glu"] = math.tanh(a / max(1e-8, self._taus["Glu"]))
            self._level["eCB"] = math.tanh(a / max(1e-8, self._taus["eCB"]))
        else:
            self._level["Glu"] += 0.05 * (self._baselines["Glu"] - self._level["Glu"])
            self._level["eCB"] += 0.05 * (self._baselines["eCB"] - self._level["eCB"])

        # ── ACh: attention sharpness ─────────────────────────────────
        if attn_entropy_norm is not None:
            # attn_entropy_norm ∈ [0,1]; sharp attention = 0 entropy → ACh = 1
            self._level["ACh"] = max(0.0, min(1.0, 1.0 - float(attn_entropy_norm)))
        else:
            self._level["ACh"] += 0.05 * (self._baselines["ACh"] - self._level["ACh"])

        # ── GABA: inhibits when workspace saturates ──────────────────
        if ignition_rate is not None:
            # GABA = 1 - ignition_rate, clamped
            self._level["GABA"] = max(0.0, min(1.0, 1.0 - float(ignition_rate)))
        else:
            self._level["GABA"] += 0.05 * (self._baselines["GABA"] - self._level["GABA"])

        # Final clamp (paranoia — every formula above is already bounded).
        for k in self._level:
            self._level[k] = max(0.0, min(1.0, self._level[k]))

    # ── Accessors ────────────────────────────────────────────────────

    def levels(self) -> Dict[str, float]:
        """Return a fresh dict of the seven channel values."""
        return dict(self._level)

    @property
    def baselines(self) -> Dict[str, float]:
        return dict(self._baselines)
