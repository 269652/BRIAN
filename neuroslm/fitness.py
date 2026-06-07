# -*- coding: utf-8 -*-
"""Fitness configuration system for evolutionary training.

This module provides:
1. FitnessObjective: single metric (minimize OOD PPL, maximize Phi, etc.)
2. FitnessConfig: collection of objectives with adaptation settings
3. FitnessComposer: runtime loss composition from a LossBundle

Fitness configs can be:
- Defined in fitness.neuro per-architecture
- Stored in DNA for self-improvement
- Applied during training via epigenetic mutations

The FitnessComposer is the runtime counterpart to the `FitnessConfig` DSL
block. Given a config and a per-step `LossBundle` produced by the
harness, it returns (total_loss, telemetry) so the harness can replace
its hard-coded `total_loss_config` formula with a single declarative pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import json

import torch
import torch.nn as nn

from neuroslm.dsl.maturity import phase_gate


# ──────────────────────────────────────────────────────────────────────
# Fitness Objective & Configuration — stored per-architecture or in DNA
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FitnessObjective:
    """Single fitness objective (e.g., minimize OOD PPL, maximize Φ)."""
    name: str  # Unique identifier
    metric: str  # Metric name to track (e.g., "ood_ppl", "phi", "gap_ratio")
    direction: str  # "minimize" or "maximize"
    weight: float  # Relative importance (normalized to sum=1.0)
    target: Optional[float] = None  # Ideal value for the metric
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "name": self.name,
            "metric": self.metric,
            "direction": self.direction,
            "weight": self.weight,
            "target": self.target,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FitnessObjective":
        """Deserialize from dict."""
        return cls(**data)


@dataclass
class FitnessAdaptation:
    """Parameters for adaptive fitness mutation."""
    enabled: bool = True
    mutation_rate: float = 0.01  # Probability per step
    target_adjustment_rate: float = 0.001
    weight_adjustment_rate: float = 0.01
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "enabled": self.enabled,
            "mutation_rate": self.mutation_rate,
            "target_adjustment_rate": self.target_adjustment_rate,
            "weight_adjustment_rate": self.weight_adjustment_rate,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FitnessAdaptation":
        return cls(**data)


@dataclass
class FitnessConfig:
    """Complete fitness configuration for an architecture."""
    version: str = "1.0"
    objectives: List[FitnessObjective] = field(default_factory=list)
    adaptation: Optional[FitnessAdaptation] = None
    metadata: Dict = field(default_factory=dict)
    enabled: bool = True  # Master on/off switch

    def __post_init__(self):
        """Normalize weights to sum to 1.0."""
        if self.objectives:
            total_weight = sum(obj.weight for obj in self.objectives)
            if total_weight > 0:
                for obj in self.objectives:
                    obj.weight /= total_weight

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "version": self.version,
            "enabled": self.enabled,
            "objectives": [obj.to_dict() for obj in self.objectives],
            "adaptation": self.adaptation.to_dict() if self.adaptation else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FitnessConfig":
        """Deserialize from dict."""
        objectives = [FitnessObjective.from_dict(o) for o in data.get("objectives", [])]
        adaptation = FitnessAdaptation.from_dict(data["adaptation"]) if data.get("adaptation") else None
        return cls(
            version=data.get("version", "1.0"),
            enabled=data.get("enabled", True),
            objectives=objectives,
            adaptation=adaptation,
            metadata=data.get("metadata", {}),
        )

    def compute_loss(self, metrics: Dict[str, float]) -> float:
        """Compute multi-objective loss from metrics."""
        if not self.objectives:
            return 0.0

        total_loss = 0.0
        for obj in self.objectives:
            metric_value = metrics.get(obj.metric)
            if metric_value is None:
                continue

            target_value = obj.target if obj.target is not None else 0
            if obj.direction == "minimize":
                obj_loss = (metric_value - target_value) * obj.weight
            else:  # maximize
                obj_loss = (target_value - metric_value) * obj.weight
            total_loss += obj_loss

        return total_loss

    def save(self, path: str) -> None:
        """Save fitness config to JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FitnessConfig":
        """Load fitness config from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_or_default(cls, path: str) -> "FitnessConfig":
        """Load fitness config, return default if file doesn't exist."""
        if Path(path).exists():
            return cls.load(path)
        # Default: minimize OOD PPL, maximize Phi
        return cls(
            objectives=[
                FitnessObjective(
                    name="minimize_ood_ppl",
                    metric="ood_ppl",
                    direction="minimize",
                    weight=0.5,
                    target=180.0
                ),
                FitnessObjective(
                    name="maximize_phi",
                    metric="phi",
                    direction="maximize",
                    weight=0.3,
                    target=0.15
                ),
                FitnessObjective(
                    name="minimize_gap_ratio",
                    metric="gap_ratio",
                    direction="minimize",
                    weight=0.2,
                    target=2.0
                ),
            ],
            adaptation=FitnessAdaptation(enabled=True)
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
        enabled = [obj.name for obj in self.config.objectives]
        return f"enabled={self.config.enabled}, objectives={enabled}"


# ──────────────────────────────────────────────────────────────────────
# Fitness Mutation Helpers
# ──────────────────────────────────────────────────────────────────────

def create_fitness_mutation_vesicle(
    target_objective: str,
    delta_weight: Optional[float] = None,
    delta_target: Optional[float] = None,
    reason: str = "adaptive_improvement"
) -> Dict:
    """Create a vesicle payload for fitness mutation.

    This vesicle can be emitted during high-surprise windows to
    self-improve the fitness objectives.

    Args:
        target_objective: Name of objective to mutate
        delta_weight: Change to weight (e.g., +0.1, -0.05)
        delta_target: Change to target value (e.g., -20.0)
        reason: Why this mutation was emitted

    Returns:
        Vesicle payload dict ready to be saved as a mutation
    """
    delta = {}
    if delta_weight is not None:
        delta["weight"] = delta_weight
    if delta_target is not None:
        delta["target"] = delta_target

    return {
        "kind": "fitness_mutation",
        "target_objective": target_objective,
        "delta": delta,
        "reason": reason,
    }
