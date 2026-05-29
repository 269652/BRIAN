# -*- coding: utf-8 -*-
"""Maturity + phase-gate utilities — bit-identical port of Brain's machinery.

Brain's training schedules every auxiliary loss through a *phase gate* keyed
to the **Maturity Index (MAT)** — a normalised LM-loss progress signal that
ramps 0→1 over training. Each aux loss has its own center/width window so
the awakening event is staggered (language first → world/motor → bowtie).

The trunk gradient picture for `rcc_bowtie_30m_p4`:

    total = w_lm * lm_loss
          + sum over aux: aux_w * phase_gate(mat, center, width) * w_aux * aux_loss

So getting DSL parity with Brain's *trunk trajectory* requires exactly:
  1. The same MAT estimator (`compute_mat`)
  2. The same EMA dynamics (rise-fast / fall-slow)
  3. The same phase-gate function (`phase_gate`)
  4. The same aux weights + centers/widths

This module provides all four. Validated bit-identical to:
  * `neuroslm.neurochem.transmitters.compute_mat`
  * `neuroslm.brain.Brain.update_maturity`
  * `neuroslm.brain.Brain._phase_gate`
in `tests/dsl/test_maturity_parity.py`.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

# Random-init LM loss floor for the GPT-2 50257-vocab tokenizer.
# Identical constant to `neuroslm.neurochem.transmitters.L_RANDOM_DEFAULT`.
L_RANDOM_DEFAULT: float = math.log(50257)


def compute_mat(lm_loss: float, l_random: float = L_RANDOM_DEFAULT) -> float:
    """MAT (Maturity Index) from instantaneous LM loss.

    M = clamp(1 − L_lm / L_random, 0, 1).
    Bit-identical to `neurochem.transmitters.compute_mat`.
    """
    return max(0.0, min(1.0, 1.0 - float(lm_loss) / float(l_random)))


def phase_gate(mat: float, center: float, width: float = 0.10) -> float:
    """Smooth sigmoid 0→1 across [center−width, center+width].

    Bit-identical to `Brain._phase_gate`. Used to fade each aux loss in
    around its own MAT center (e.g. pred_coding at 0.35, world at 0.45,
    motor at 0.50, novel at 0.55, kl at 0.60).
    """
    x = (float(mat) - float(center)) / max(1e-6, float(width))
    return 0.5 * (1.0 + math.tanh(x))


@dataclass
class MaturityTracker:
    """Rise-fast / fall-slow EMA on MAT — mirrors Brain.update_maturity.

    Brain uses two distinct EMA alphas so a transient LM-loss spike barely
    dents the smoothed MAT (avoiding whipsaw on the gated aux weights) but
    a sustained regression still pulls it down.

    Defaults pulled from `brain.Brain.__init__`:
      * rise alpha (fast)  = 0.20
      * fall alpha (slow)  = 0.05  (assigned to `_maturity_ema_alpha`)
    """
    rise_alpha: float = 0.20
    fall_alpha: float = 0.05
    l_random: float = L_RANDOM_DEFAULT
    mat: float = 0.0
    mat_hwm: float = 0.0   # high-water mark for hysteresis-style gates

    def update(self, lm_loss: float) -> float:
        m_now = compute_mat(lm_loss, self.l_random)
        if m_now > self.mat:
            alpha = self.rise_alpha
        else:
            alpha = self.fall_alpha
        self.mat = (1.0 - alpha) * self.mat + alpha * m_now
        self.mat_hwm = max(self.mat_hwm, self.mat)
        return self.mat

    def value(self) -> float:
        return self.mat


# ── Aux-loss weight schedule (the brain.py:1802-1810 total formula) ──

@dataclass
class AuxWeights:
    """Per-loss weight + (center, width) for the phase gate.

    Matches the constants in `brain.Brain.forward_lm` at lines 1794–1810.
    Each entry resolves to:  weight * phase_gate(mat, center, width).

    The `master_scale` is Brain's `aux_w = self._aux_w_scale` — a single
    knob that scales every aux loss uniformly (defaults to 1.0).
    """
    master_scale: float = 1.0

    # (weight, center, width) for each aux loss. Numbers verified against
    # rcc_bowtie_30m_p4 (w_world=0.3, w_forward=0.2, w_motor=0.05,
    # w_pred_coding=0.1, w_kl_world=0.1, w_cpc=0.05, w_phi=0.02).
    pred_coding: tuple = (0.10, 0.35, 0.08)
    world:       tuple = (0.30, 0.45, 0.08)
    forward:     tuple = (0.20 * 0.01, 0.50, 0.08)   # has the bonus *0.01
    motor:       tuple = (0.05, 0.50, 0.08)
    kl_world:    tuple = (0.10, 0.60, 0.08)
    novel:       tuple = (0.05, 0.55, 0.08)
    cpc:         tuple = (0.05, 0.55, 0.08)
    phi:         tuple = (0.02, 0.60, 0.08)

    def scaled(self, key: str, mat: float) -> float:
        """Return  master_scale * weight * phase_gate(mat, center, width)."""
        if not hasattr(self, key):
            raise KeyError(f"unknown aux key {key!r}")
        weight, center, width = getattr(self, key)
        return self.master_scale * weight * phase_gate(mat, center, width)

    def all_scaled(self, mat: float) -> Dict[str, float]:
        """Resolve every aux to its scalar weight at the current MAT."""
        return {k: self.scaled(k, mat) for k in
                ("pred_coding", "world", "forward", "motor",
                 "kl_world", "novel", "cpc", "phi")}


@dataclass
class TotalLossConfig:
    """Composite config the DSL aggregator consumes.

    `w_lm` and the AuxWeights table together reproduce Brain's
    `total = w_lm * lm_loss + aux_w * Σ (ph_x * w_x * loss_x)` formula
    exactly. The DSL Brain aggregator (`neuroslm.dsl.brain_aggregator`)
    uses this to apply the same weighted sum the trunk gradient sees.
    """
    w_lm: float = 1.0
    aux: AuxWeights = field(default_factory=AuxWeights)
