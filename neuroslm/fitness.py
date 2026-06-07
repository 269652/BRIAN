# -*- coding: utf-8 -*-
"""Multi-Objective Fitness Composer — Phase A/F2 of the work order.

`FitnessComposer` is the runtime counterpart to the `FitnessConfig` DSL
block.  Given a config and a per-step `LossBundle` produced by the
harness, it returns

    (total_loss : torch.Tensor,  # scalar autograd-traced
     telemetry  : Dict[str, float])

so the harness can replace its hard-coded `total_loss_config` formula
with a single declarative pipeline.

Math
----
For each enabled objective ``o`` with weight ``w_o`` and schedule ``S_o``:

.. math::
    \\mathcal{L}_{\\text{total}}
        = \\sum_{o \\in \\text{enabled}} w_o \\cdot S_o(\\text{mat}) \\cdot \\mathcal{L}_o

where :math:`\\text{mat} \\in [0, 1]` is the MaturityTracker value and
:math:`S_o(\\cdot)` is one of:

    constant — :math:`S(m) = 1`                                   (no modulation)
    gated    — :math:`S(m) = \\tfrac{1}{2}\\bigl(1 + \\tanh\\bigl((m - c)/w\\bigr)\\bigr)`   (per `phase_gate`)
    linear   — :math:`S(m) = \\max(0, \\min(1, m))`                  (warmup ramp)

Symbolic objective integration
------------------------------
When the `symbolic` objective is enabled, the composer owns a
`SymbolicHyperNeuron` sized per `cfg.symbolic_n_units / n_features`.
The harness feeds it a feature vector via ``compute_symbolic_loss(x)``;
the returned scalar is the unit's sparsity loss (entropy of the
operator + input selections) scaled by ``cfg.symbolic_sparsity_weight``.
The objective weight ``w_symbolic`` is then applied on top by
``compose``.

Legacy reproducibility
----------------------
When ``cfg.enabled == False`` the composer becomes a no-op pass-through:
``compose`` returns ``(bundle.lm, {"lm": float(bundle.lm)})`` so any
existing arch.neuro without a `fitness { ... }` block trains exactly
as before.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from neuroslm.dsl.maturity import phase_gate
from neuroslm.dsl.training_config import (
    FitnessConfig,
    FitnessObjective,
)


# ──────────────────────────────────────────────────────────────────────
# LossBundle — typed container of per-step loss scalars
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LossBundle:
    """Per-step bag of objective loss tensors produced by the harness.

    Every field is optional — a `None` value means "the harness chose
    not to compute this objective on this step" (e.g. for the genetics
    `update_every=4` case).  The composer treats `None` as zero
    contribution; the disabled objective is also omitted from telemetry.
    """
    lm:        Optional[torch.Tensor] = None
    phi:       Optional[torch.Tensor] = None
    nis_plus:  Optional[torch.Tensor] = None
    symbolic:  Optional[torch.Tensor] = None
    piso:      Optional[torch.Tensor] = None
    metabolic: Optional[torch.Tensor] = None

    def get(self, name: str) -> Optional[torch.Tensor]:
        return getattr(self, name, None)


# ──────────────────────────────────────────────────────────────────────
# Phase-gate centers per objective (matches existing AuxWeights table)
# ──────────────────────────────────────────────────────────────────────

# (center, width) for "gated" schedule per objective.  Picked to match
# the existing `dsl.maturity.AuxWeights` table so migrating an
# objective from the legacy `total_loss_config` into the new
# `fitness.objectives` table preserves its training-time activation
# curve bit-for-bit.
_GATE_TABLE: Dict[str, Tuple[float, float]] = {
    "lm":        (0.00, 0.08),   # always-on
    "phi":       (0.60, 0.08),   # matches AuxWeights.phi
    "nis_plus":  (0.55, 0.08),   # mirrors novel/cpc band
    "symbolic":  (0.40, 0.08),   # mid-training, after lexical maturity
    "piso":      (0.50, 0.08),   # alongside motor/forward
    "metabolic": (0.65, 0.08),   # late — only prune once topology is stable
}


# ──────────────────────────────────────────────────────────────────────
# FitnessComposer
# ──────────────────────────────────────────────────────────────────────

class FitnessComposer(nn.Module):
    """Compose a single weighted aggregate loss from a `LossBundle`.

    Construction also instantiates any objective-specific modules that
    have their own parameters — currently only :class:`SymbolicHyperNeuron`,
    sized per `cfg.symbolic_n_units / n_features`.

    Parameters
    ----------
    config : FitnessConfig
        The parsed `fitness { ... }` DSL block.

    Attributes
    ----------
    config         — the original `FitnessConfig`.
    symbolic_unit  — `SymbolicHyperNeuron` instance when the symbolic
                      objective is enabled, else None.
    """

    def __init__(self, config: FitnessConfig) -> None:
        super().__init__()
        self.config = config
        self.symbolic_unit: Optional[nn.Module] = None

        # Lazy import: pulling SymbolicHyperNeuron only when needed
        # keeps legacy paths free of the new module's load cost.
        sym_spec = config.objectives.get("symbolic")
        if sym_spec is not None and sym_spec.enabled:
            from neuroslm.modules.symbolic_unit import SymbolicHyperNeuron
            self.symbolic_unit = SymbolicHyperNeuron(
                n_units=config.symbolic_n_units,
                n_features=config.symbolic_n_features,
                tau=config.symbolic_tau_init,
            )

    # ── per-objective schedule resolution ──────────────────────────

    @staticmethod
    def _schedule_factor(schedule: str, maturity: float,
                         objective_name: str) -> float:
        """Return the scalar factor by which the objective weight is
        multiplied at the given maturity value."""
        if schedule == "constant":
            return 1.0
        if schedule == "linear":
            return max(0.0, min(1.0, float(maturity)))
        if schedule == "gated":
            center, width = _GATE_TABLE.get(objective_name, (0.5, 0.08))
            return phase_gate(maturity, center, width)
        # parser validates this; runtime should never see another value
        raise ValueError(
            f"unknown schedule {schedule!r} for objective {objective_name!r}"
        )

    # ── core composition ───────────────────────────────────────────

    def compose(
        self, bundle: LossBundle, maturity: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compose ``bundle`` into a single weighted scalar loss.

        Parameters
        ----------
        bundle : LossBundle
            Per-step loss tensors.  Missing / None entries contribute zero.
        maturity : float
            Current MaturityTracker value in [0, 1].  Drives the
            per-objective schedule factor.

        Returns
        -------
        total : torch.Tensor
            Scalar autograd-traced loss tensor.
        telemetry : Dict[str, float]
            Per-objective weighted contribution (for logging).  Only
            includes enabled objectives.
        """
        # Disabled master switch: pure pass-through of lm loss to
        # preserve bit-for-bit legacy reproducibility.
        if not self.config.enabled:
            if bundle.lm is None:
                # Nothing to compose at all; return a zero tensor so the
                # caller's `.backward()` is a no-op rather than crashing.
                return torch.zeros((), dtype=torch.float32), {}
            return bundle.lm, {"lm": float(bundle.lm.detach())}

        total: Optional[torch.Tensor] = None
        telemetry: Dict[str, float] = {}

        for name, spec in self.config.objectives.items():
            if not spec.enabled:
                continue
            loss_value = bundle.get(name)
            if loss_value is None:
                # Objective configured but the harness didn't produce
                # a loss this step — record zero for visibility.
                telemetry[name] = 0.0
                continue
            factor = self._schedule_factor(spec.schedule, maturity, name)
            contribution = spec.weight * factor * loss_value
            total = contribution if total is None else (total + contribution)
            telemetry[name] = float(contribution.detach())

        if total is None:
            total = torch.zeros((), dtype=torch.float32)
        return total, telemetry

    # ── symbolic-unit interface ────────────────────────────────────

    def compute_symbolic_loss(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """Run the SymbolicHyperNeuron and return the scaled sparsity loss.

        ``features`` is fed to the unit so its forward produces a
        ``(..., n_units)`` tensor (currently unused by this method but
        required to register the unit in the autograd graph for the
        backward pass through its selection logits).

        Returns
        -------
        torch.Tensor scalar
            ``cfg.symbolic_sparsity_weight * unit.sparsity_loss()``.
            Returns a zero scalar when the symbolic objective is disabled.
        """
        if self.symbolic_unit is None:
            return torch.zeros((), dtype=torch.float32)
        # Forward to register the unit in the autograd graph if the
        # caller plans to backprop through it.  We don't use the output
        # here; the symbolic objective is *purely the sparsity penalty*
        # — the unit's contribution to LM loss comes from being inserted
        # into the trunk by the harness, which is done separately.
        _ = self.symbolic_unit(features)
        return (self.config.symbolic_sparsity_weight
                * self.symbolic_unit.sparsity_loss())

    def set_symbolic_tau(self, tau: float) -> None:
        """Anneal the SymbolicHyperNeuron's Gumbel temperature.

        Typical schedule:
            * Linearly interpolate from ``cfg.symbolic_tau_init`` to
              ``cfg.symbolic_tau_final`` over the warmup steps.

        Silently no-ops when the symbolic objective is disabled (so the
        training loop can call this unconditionally).
        """
        if self.symbolic_unit is not None:
            self.symbolic_unit.set_tau(tau)

    # ── repr ───────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        enabled = sorted(
            n for n, s in self.config.objectives.items() if s.enabled
        )
        return f"enabled={self.config.enabled}, objectives={enabled}"
