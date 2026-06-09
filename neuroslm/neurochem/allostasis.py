# -*- coding: utf-8 -*-
"""Synthetic HPA (hypothalamic-pituitary-adrenal) axis.

This module adds the *one* mechanism the existing neuromod stack is
missing: a slow, integrating negative-feedback loop that distinguishes
acute stress (which should not be suppressed — it's the legitimate
novelty / surprise signal) from chronic stress (which IS pathological —
positive-feedback runaway between NE → GABA → grad-norm spikes →
loss volatility → more NE → ...).

Biological model
----------------
Mammalian stress response is a two-layer cascade:

  1. **Fast loop (seconds)**: locus coeruleus releases noradrenaline (NE)
     in response to salient/novel stimuli. NE drives PFC/amygdala into
     a high-vigilance state.

  2. **Slow loop (minutes)**: sustained stress activates the HPA axis:
     hypothalamus → CRH → pituitary → ACTH → adrenal cortex → cortisol.
     Cortisol crosses the BBB, binds glucocorticoid receptors in
     hippocampus + PFC, and:
       (a) Suppresses CRH release (closes the loop — negative feedback).
       (b) Suppresses BDNF expression (Smith & Vale 2006) — chronic
           stress is bad for neurogenesis.
       (c) Inhibits LC tonic firing (Valentino & Van Bockstaele 2008) —
           damps the noradrenergic source.

The time-scale separation (cortisol kinetics ~10× slower than NE) is
what makes the system robust: a single surprising input causes a brief
NE burst that does NOT trigger cortisol because the integrator hasn't
caught up. Sustained pathological stress, by contrast, drives cortisol
up enough to engage all three suppression effects.

Software model
--------------
We track two scalar buffers:

    load(t) = (1 - α_load) · load(t-1) + α_load · stress(t)       # fast EMA
    cort(t) = (1 - α_cort) · cort(t-1) + α_cort · load(t)         # slow EMA

with ``α_load / α_cort ≈ 10`` (the physiological default).

The instantaneous ``stress(t)`` is a saturating weighted sum of:

    stress = sat( w_NE   · (NE - ne_baseline)/(1 - ne_baseline)
                + w_GABA · (GABA - gaba_baseline)/(1 - gaba_baseline)
                + w_loss · |L_t - L_{t-1}| / max(L_{t-1}, ε)
                + w_grad · max(0, gn - gn_ceil) / max(1, gn_ceil) )

Each term clips at zero (negative excursions don't count — only
above-baseline elevations register as stress).

Three downstream multipliers, all in ``[0, 1]``::

    ne_multiplier   = 1 - γ_NE       · cort        # close NE positive feedback
    trophic_mult    = 1 - γ_trophic  · cort        # stop sprouting during crisis
    lr_multiplier   = 1 - γ_LR       · cort        # consolidate, don't update

When ``cort = 0`` all multipliers are 1.0 (no effect). When
``cort = 1`` they collapse to ``1 - γ_*`` (default: NE→0.3, trophic→0,
lr→0.5). The damping is *smooth* — there are no thresholds or hard
switches that could induce control-loop oscillations.

Why this is the right abstraction
---------------------------------
* **Already present mechanisms become composable.** The existing
  Homeostasis.observe() already bumps GABA bias on high grad norm —
  that's a single-shot feedback. AllostaticController generalises it
  to a *sustained* feedback over many stress channels.
* **Bounded, by construction.** The saturating sum guarantees
  ``load ∈ [0, 1]``, and the EMA structure guarantees ``cort ∈ [0, 1]``
  too. The multipliers are clamped, so even mis-configured gammas
  cannot send LR/NE/trophic factors negative.
* **Disabled-state is bit-identical to legacy.** The controller is only
  built when ``allostasis.enabled = True``. Every existing arch.neuro
  reproduces its old behaviour.

Tests in ``tests/training/test_allostasis.py`` pin every contract.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from neuroslm.dsl.training_config import AllostasisConfig


class AllostaticController(nn.Module):
    """Synthetic HPA-axis controller.

    Holds two state buffers (``load`` and ``cort``, each shape ``(1,)``)
    that participate in checkpointing but never receive gradients
    (everything is ``torch.no_grad`` inside ``step``).

    Public surface:

    * ``step(ne_level, gaba_level, loss, grad_norm)`` — advance the
      controller by one training step. All four inputs are Python
      floats so callers don't have to worry about device placement.
    * ``ne_multiplier() / trophic_multiplier() / lr_multiplier()`` —
      three multiplicative gain factors in ``[0, 1]`` for the harness
      to apply at the relevant point of the train_step pipeline.
    * ``load.item() / cort.item()`` — readable state for telemetry.
    """

    def __init__(self, cfg: AllostasisConfig):
        super().__init__()
        self.cfg = cfg
        # State (persisted via state_dict so resumes don't reset cort)
        self.register_buffer("load", torch.zeros(1))
        self.register_buffer("cort", torch.zeros(1))
        # Previous loss — needed to compute volatility |ΔL|/L. NaN-init
        # so the very first step contributes zero loss-volatility stress
        # (no prior to compare against).
        self.register_buffer("_prev_loss",
                              torch.tensor(float("nan")))

    # ── Public API ───────────────────────────────────────────────────

    @torch.no_grad()
    def step(self, ne_level: float, gaba_level: float,
             loss: float, grad_norm: float) -> None:
        """Advance the controller by one training step.

        Args:
            ne_level:   current noradrenaline concentration ∈ [0, 1]
                         (typically ``transmitter_sys.level[batch_mean, NE]``).
            gaba_level: current GABA concentration ∈ [0, 1].
            loss:       this step's total loss (any positive scalar).
                         Used for volatility-stress only; absolute value
                         doesn't matter, only the relative jump from the
                         prior step.
            grad_norm:  this step's grad-norm-before-clipping. Counted
                         as stress only above ``cfg.grad_norm_ceiling``.

        Side effects: updates ``self.load`` (fast EMA) and ``self.cort``
        (slow EMA). NaN-safe on inputs (any non-finite gets clipped).
        """
        cfg = self.cfg

        # ── Stress sources, each clipped to [0, +∞) ──
        # NE above baseline (only counts the elevation)
        ne_excess = max(0.0, float(ne_level) - cfg.ne_baseline)
        ne_norm = ne_excess / max(1e-6, 1.0 - cfg.ne_baseline)
        ne_norm = min(1.0, max(0.0, ne_norm))

        # GABA above baseline
        gaba_excess = max(0.0, float(gaba_level) - cfg.gaba_baseline)
        gaba_norm = gaba_excess / max(1e-6, 1.0 - cfg.gaba_baseline)
        gaba_norm = min(1.0, max(0.0, gaba_norm))

        # Loss volatility = |ΔL| / L_prev (relative jump). Zero on the
        # first step (no prior). NaN guards stop a divergent loss from
        # NaN-poisoning the EMA.
        prev = float(self._prev_loss.item())
        if (math.isfinite(prev) and math.isfinite(loss)
                and abs(prev) > 1e-6):
            loss_vol = abs(float(loss) - prev) / abs(prev)
        else:
            loss_vol = 0.0
        # Saturate the volatility contribution at 1.0 so a single
        # catastrophic spike (e.g. loss 5 → 50, vol = 9) can't pin
        # the saturated sum at its ceiling forever via fast EMA.
        loss_vol = min(1.0, max(0.0, loss_vol))

        # Grad-norm spike (above the safe-operation ceiling)
        gn = float(grad_norm) if math.isfinite(float(grad_norm)) else 0.0
        gn_excess = max(0.0, gn - cfg.grad_norm_ceiling)
        # Normalise against the ceiling itself so the unit-step (one
        # ceiling-worth of overshoot) ≈ 0.5 stress. Bigger overshoots
        # saturate below 1.0 via the outer min().
        gn_norm = gn_excess / max(1.0, cfg.grad_norm_ceiling)
        gn_norm = min(1.0, gn_norm)

        # ── Saturating weighted sum (instantaneous stress in [0,1]) ──
        stress = (cfg.w_ne   * ne_norm
                + cfg.w_gaba * gaba_norm
                + cfg.w_loss * loss_vol
                + cfg.w_grad * gn_norm)
        stress = min(1.0, max(0.0, stress))

        # ── Two-layer EMA (load fast, cort slow) ──
        a_load = float(cfg.load_ema_alpha)
        a_cort = float(cfg.cort_ema_alpha)
        new_load = (1.0 - a_load) * float(self.load.item()) + a_load * stress
        # Defensive clamp — float drift over many steps can yield
        # values like 1.0000001.
        new_load = min(1.0, max(0.0, new_load))
        self.load.fill_(new_load)

        new_cort = (1.0 - a_cort) * float(self.cort.item()) + a_cort * new_load
        new_cort = min(1.0, max(0.0, new_cort))
        self.cort.fill_(new_cort)

        # Persist for next-step volatility
        if math.isfinite(float(loss)):
            self._prev_loss.fill_(float(loss))

    def ne_multiplier(self) -> float:
        """Multiplier ∈ [0, 1] for NE release. 1 = no damping, 0 = full damp.

        At ``cort = 1`` returns ``1 - γ_NE``. Per-effector kill switch:
        when ``cfg.suppress_ne = False`` returns 1.0 regardless of cort.
        """
        if not self.cfg.suppress_ne:
            return 1.0
        m = 1.0 - float(self.cfg.gamma_ne) * float(self.cort.item())
        return min(1.0, max(0.0, m))

    def trophic_multiplier(self) -> float:
        """Multiplier ∈ [0, 1] for trophic (BDNF) growth signal.

        Defaults to ``γ_T = 1.0`` so at cort=1 trophic growth halts
        completely — physiologically, chronic cortisol fully
        downregulates BDNF expression.
        """
        if not self.cfg.suppress_trophic:
            return 1.0
        m = 1.0 - float(self.cfg.gamma_trophic) * float(self.cort.item())
        return min(1.0, max(0.0, m))

    def lr_multiplier(self) -> float:
        """Multiplier ∈ [0, 1] for the optimizer learning rate.

        Applied by ``BRIANHarness.train_step`` to every param-group's
        ``lr`` right before ``optimizer.step()``. The post-damping LR
        is published back via ``param_group['lr']`` so the next
        scheduled refresh re-bases from the un-damped value.
        """
        if not self.cfg.suppress_lr:
            return 1.0
        m = 1.0 - float(self.cfg.gamma_lr) * float(self.cort.item())
        return min(1.0, max(0.0, m))

    # ── Telemetry helper for the train log ──────────────────────────

    def telemetry(self) -> dict:
        """Return a flat dict suitable for merging into ``harness._metrics``
        (then surfaced on the per-step train log line)."""
        return {
            "allostasis_load": float(self.load.item()),
            "allostasis_cort": float(self.cort.item()),
            "allostasis_ne_mult":      self.ne_multiplier(),
            "allostasis_trophic_mult": self.trophic_multiplier(),
            "allostasis_lr_mult":      self.lr_multiplier(),
        }
