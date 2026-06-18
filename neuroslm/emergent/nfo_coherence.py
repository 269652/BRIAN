# -*- coding: utf-8 -*-
"""NFO coherence probe — telemetry layer for the Neural Field Oscillator.

Consumes the ``last_state`` dict published by
:class:`neuroslm.modules.neural_field_oscillator.NeuralFieldOscillator`
on every forward and emits a flat ``Dict[str, float]`` for the harness
metric observer.  All read-only; never participates in the LM gradient.

Logged columns (one row per training step):

    nfo[R=0.42 R*=0.78 A=1.04 σA=0.31 cVar=0.18 κ=0.32 α=0.08 Φκ=0.22]

Meaning:

* ``R``      — mean Kuramoto order parameter over (batch, token, osc).
              Values near 1 ⇒ tokens are locally phase-coherent ⇒
              binding-by-synchrony active.  Climbs monotonically over
              training when the block is doing useful work.
* ``R*``     — per-token max coherence (gate magnitude).
* ``A``      — mean oscillator amplitude.  Should settle near
              ``a_star_init`` if Swift–Hohenberg damping is healthy.
* ``σA``     — amplitude std; low std + R≈R* ⇒ uniform synchronisation.
* ``cVar``   — circular variance of the phase field, in [0, 1].
              cVar = 0 ⇔ all oscillators in phase.
* ``κ``      — current Kuramoto coupling strength (sigmoid-bounded).
* ``α``      — current readout gain (the H018 zero-init starts at 0).
* ``Φκ``     — mean-field incoherence ``mean(1 − R)`` ∈ [0, 1].
              By H015, ``1 − Φκ`` is a closed-form lower bound on the
              sheaf-Laplacian Φ proxy of H001 — so a *falling* Φκ over
              training is direct evidence of integrated-information
              increase.

Compose with the existing emergent probes:

* ``TopologicalChargeProbe`` measures winding **Q** on the residual
  stream after the NFO block — Q-walls aligning with sentence boundaries
  is the per-token signature of coherence.
* ``PACBindingProbe`` measures phase-amplitude coupling in the same
  oscillator field — high PAC ⇔ gamma envelope locks to theta phase
  ⇔ binding-by-synchrony at the population level.

Together they discriminate the *type* of coherence the NFO produces:
local (Q), spectral (PAC), and integrated (Φκ).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from neuroslm.modules.neural_field_oscillator import NeuralFieldOscillator


_DEFAULT_METRICS: Dict[str, float] = {
    "nfo_R_mean": 0.0,
    "nfo_R_max": 0.0,
    "nfo_A_mean": 0.0,
    "nfo_A_std": 0.0,
    "nfo_phi_circular_var": 1.0,
    "nfo_kappa": 0.0,
    "nfo_dt": 0.0,
    "nfo_alpha": 0.0,
    "nfo_phi_kappa": 1.0,
}


class NFOCoherenceProbe:
    """Stateless observer that flattens an NFO block's ``last_state``.

    Parameters
    ----------
    block : NeuralFieldOscillator | None
        The block to observe.  May be ``None`` so the probe can be
        instantiated unconditionally and silently return the default
        zero metrics when the host arch has no NFO attached.
    """

    def __init__(self, block: Optional[NeuralFieldOscillator] = None):
        self._block = block
        self._last: Dict[str, float] = dict(_DEFAULT_METRICS)

    def attach(self, block: NeuralFieldOscillator) -> None:
        """Late-bind a block — useful when the trunk is built after the
        probe (the harness instantiates probes once and re-uses them)."""
        self._block = block

    def step(self) -> Dict[str, float]:
        """Pull the latest ``last_state`` from the bound block.

        Safe to call every training step.  Returns a stable schema dict
        so the metric observer can rely on the keys being present.
        """
        if self._block is None:
            return dict(self._last)
        state = getattr(self._block, "last_state", None)
        if not state:
            return dict(self._last)
        out: Dict[str, float] = dict(_DEFAULT_METRICS)
        for k, key in (
            ("R_mean", "nfo_R_mean"),
            ("R_max", "nfo_R_max"),
            ("A_mean", "nfo_A_mean"),
            ("A_std", "nfo_A_std"),
            ("phi_circular_var", "nfo_phi_circular_var"),
            ("kappa", "nfo_kappa"),
            ("dt", "nfo_dt"),
            ("alpha", "nfo_alpha"),
            ("phi_kappa", "nfo_phi_kappa"),
        ):
            v = state.get(k)
            if isinstance(v, torch.Tensor):
                out[key] = float(v.detach().cpu().item())
            elif v is not None:
                out[key] = float(v)
        self._last = out
        return dict(out)

    def stats(self) -> Dict[str, float]:
        """Return the last computed metrics without re-reading state."""
        return dict(self._last)


__all__ = ["NFOCoherenceProbe"]
