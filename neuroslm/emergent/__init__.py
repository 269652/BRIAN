# -*- coding: utf-8 -*-
"""Emergent-topology observability layer for RCC-Bowtie v2.

Six telemetry mechanisms (C1–C6) that turn the dead biology readouts of
the current RCC-Bowtie trajectory into closed-loop, signal-bearing
measurements. None of this changes the forward pass or the loss; this is
pure observability that we use to *decide* whether each mechanism is
worth promoting into a real architectural change in a later PR.

See `docs/EMERGENT_TOPOLOGY.md` for the master design and falsifiable
predictions per mechanism.

TRUNK-OPT layer (Phase 1–4):
  GradientBudgetTracker, LayerGradientProbe, BitsPerParamMeter,
  PACBayesBound, SharpnessProbe, EffectiveRankProbe, TrunkOptMonitor —
  all in `neuroslm.emergent.trunk_opt`.
"""
from neuroslm.emergent.driven_nt import DrivenNTSystem
from neuroslm.emergent.metastable_ignition import MetastableIgnition
from neuroslm.emergent.pc_reentry import PCReentryProbe
from neuroslm.emergent.topological_charge import TopologicalChargeProbe
from neuroslm.emergent.bowtie_lattice import BowtieLatticeProbe
from neuroslm.emergent.pac_binding import PACBindingProbe
from neuroslm.emergent.trunk_opt import (
    GradientBudgetTracker,
    LayerGradientProbe,
    BitsPerParamMeter,
    PACBayesBound,
    SharpnessProbe,
    EffectiveRankProbe,
    TrunkOptMonitor,
)

__all__ = [
    "DrivenNTSystem",
    "MetastableIgnition",
    "PCReentryProbe",
    "TopologicalChargeProbe",
    "BowtieLatticeProbe",
    "PACBindingProbe",
    # TRUNK-OPT
    "GradientBudgetTracker",
    "LayerGradientProbe",
    "BitsPerParamMeter",
    "PACBayesBound",
    "SharpnessProbe",
    "EffectiveRankProbe",
    "TrunkOptMonitor",
]
