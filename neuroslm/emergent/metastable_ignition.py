# -*- coding: utf-8 -*-
"""C2 — Metastable ignition detector.

The current `metrics.gws_ignition()` returns a continuous peakiness in
[0,1] that has been observed to saturate at ~0.97 across thousands of
training steps — i.e. the GWS "ignites" every step, making it a wire,
not a workspace.

This module replaces that with a *threshold + event* model:

  g_t = sigmoid( (peak_t - θ_t - β·NE_t) / τ )         # gated strength
  e_t = 1 if g_t > 0.5 else 0                           # event indicator
  θ_{t+1} = θ_t + η · (mean(e_recent) - ρ*)             # adaptive threshold

ρ* is the target ignition rate (default 0.2, Dehaene-correct: ignition
is *rare*). The adaptive threshold is a proportional controller that
keeps the firing rate near ρ* regardless of input magnitude — so the
ignition channel becomes a robust *event detector* rather than a wire
that fires for every input above some fixed threshold.

This is observation-only: nothing in the forward pass uses `g_t` or
`e_t`. Once telemetry confirms the rate distribution is bimodal, a later
PR will gate the actual GWS broadcast on this signal.
"""
from __future__ import annotations
import math
from collections import deque
from typing import Dict, Optional

import torch


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class MetastableIgnition:
    """Adaptive-threshold event detector for GWS ignition.

    Parameters
    ----------
    target_rate : float
        Desired fraction of steps with `g_t > 0.5`. Default 0.2.
    threshold_eta : float
        Learning rate for the adaptive threshold. Default 0.01.
    softness : float
        Sigmoid temperature τ. Default 0.05 (sharp).
    ne_coupling : float
        Coefficient β by which the NE channel lowers the threshold.
        Default 0.3 (high arousal → easier to ignite).
    history : int
        Window length for the rate EMA used by the controller.
    """

    def __init__(self,
                 target_rate: float = 0.2,
                 threshold_eta: float = 0.01,
                 softness: float = 0.05,
                 ne_coupling: float = 0.3,
                 history: int = 64):
        if not 0.0 < target_rate < 1.0:
            raise ValueError("target_rate must be in (0, 1)")
        self.target_rate = float(target_rate)
        self.threshold_eta = float(threshold_eta)
        self.softness = max(1e-6, float(softness))
        self.ne_coupling = float(ne_coupling)
        # Threshold starts at 0.5 — middle of the normalised peak range.
        self.threshold = 0.5
        self._events: deque = deque(maxlen=int(history))
        self._strengths: deque = deque(maxlen=int(history))
        self._last_g = 0.0
        self._last_event = 0
        self._last_peak = 0.0

    # ── Peak computation (mirrors `metrics.gws_ignition` semantics) ──

    @staticmethod
    def peak(act: torch.Tensor) -> float:
        """Normalised peak softmax probability ∈ [0, 1]."""
        x = act.reshape(-1, act.shape[-1]).float()
        if x.numel() == 0:
            return 0.0
        p = torch.softmax(x, dim=-1)
        peak = p.max(dim=-1).values.mean().item()
        D = x.shape[-1]
        return float(max(0.0, min(1.0, (peak - 1.0 / D) / (1.0 - 1.0 / D))))

    # ── Step ─────────────────────────────────────────────────────────

    def step(self, peak: float, ne: float = 0.0) -> Dict[str, float]:
        """Advance one step given the current peak and NE level.

        Returns a dict with `rate`, `strength` (event-conditional),
        `threshold`, `event` (0/1), `g` (continuous gate).
        """
        eff_thresh = self.threshold - self.ne_coupling * float(ne)
        g = _sigmoid((float(peak) - eff_thresh) / self.softness)
        event = 1 if g > 0.5 else 0

        self._events.append(event)
        if event:
            self._strengths.append(g)

        # Controller: lower the threshold if firing too rarely, raise it
        # if firing too often. Bounded so it cannot wander outside the
        # input's natural range.
        if len(self._events) >= 8:
            rate = sum(self._events) / len(self._events)
            self.threshold += self.threshold_eta * (rate - self.target_rate)
            self.threshold = max(0.0, min(1.0, self.threshold))

        self._last_g = g
        self._last_event = event
        self._last_peak = float(peak)

        return self.stats()

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, float]:
        rate = (sum(self._events) / len(self._events)) if self._events else 0.0
        strength = (sum(self._strengths) / len(self._strengths)) \
            if self._strengths else 0.0
        return {
            "ign_rate":      float(rate),
            "ign_strength":  float(strength),
            "ign_threshold": float(self.threshold),
            "ign_event":     int(self._last_event),
            "ign_g":         float(self._last_g),
            "ign_peak":      float(self._last_peak),
        }
