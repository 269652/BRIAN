"""Receptor-gated neural projections.

Extends the basic ProjectionGraph with receptor-activity-dependent gating:
neurons control NT flow between regions based on local receptor occupancy.

Key biological circuits modeled:
  - CB1 on VTA GABA interneurons → disinhibits DA release to NAcc
  - D2 autoreceptors on VTA terminals → limits DA overflow
  - 5HT2A on cortical pyramidal cells → gates serotonergic modulation of PFC
  - α2 adrenergic autoreceptors on LC → negative feedback on NE release
  - NMDA on hippocampal CA1 → gates LTP-dependent consolidation signals
  - GABA-A on thalamic relay neurons → gates sensory throughput

Each gated projection has:
  - base projection (src → dst, carrying NT + signal)
  - gate_receptor: which receptor type controls the gate
  - gate_nt: which NT activates that receptor
  - gate_polarity: +1 (receptor activation opens gate) or -1 (closes gate)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

from .transmitters import NT_INDEX, NT_NAMES, N_NT
from .projections import Projection, ProjectionGraph


@dataclass
class GatedProjection:
    """A projection whose effective strength is gated by receptor activity."""
    src: str
    dst: str
    nt: str                         # NT released along this projection
    gate_nt: str                    # NT that controls the gate
    gate_polarity: float = 1.0      # +1: gate opens with NT, -1: closes
    release_scale: float = 1.0
    carries_signal: bool = True
    description: str = ""


# ---- Canonical gated projections (neuroscience-based) ----
GATED_PROJECTIONS = [
    GatedProjection(
        src="VTA", dst="NAcc", nt="DA", gate_nt="eCB",
        gate_polarity=1.0, release_scale=1.2,
        description="CB1 on VTA GABA interneurons: eCB disinhibits DA release to NAcc"
    ),
    GatedProjection(
        src="VTA", dst="PFC", nt="DA", gate_nt="DA",
        gate_polarity=-1.0, release_scale=0.8,
        description="D2 autoreceptor: high DA inhibits further VTA→PFC release"
    ),
    GatedProjection(
        src="Raphe", dst="PFC", nt="5HT", gate_nt="5HT",
        gate_polarity=-1.0, release_scale=0.7,
        description="5HT1A autoreceptor: limits serotonergic drive to cortex"
    ),
    GatedProjection(
        src="LC", dst="Hippo", nt="NE", gate_nt="NE",
        gate_polarity=-1.0, release_scale=0.9,
        description="α2 autoreceptor on LC: negative feedback on NE release"
    ),
    GatedProjection(
        src="Hippo", dst="PFC", nt="Glu", gate_nt="Glu",
        gate_polarity=1.0, release_scale=1.0,
        description="NMDA-gated: glutamate facilitates hippocampal→PFC consolidation"
    ),
    GatedProjection(
        src="SNr", dst="Thalamus", nt="GABA", gate_nt="DA",
        gate_polarity=-1.0, release_scale=1.0,
        description="SNr GABA to thalamus: DA from SNc reduces inhibitory gate"
    ),
    GatedProjection(
        src="NAcc", dst="VTA", nt="GABA", gate_nt="eCB",
        gate_polarity=1.0, release_scale=0.6,
        description="NAcc→VTA feedback: eCB modulates inhibitory return"
    ),
    GatedProjection(
        src="NBM", dst="PFC", nt="ACh", gate_nt="ACh",
        gate_polarity=-1.0, release_scale=0.8,
        description="M2 autoreceptor: limits ACh overflow from basal forebrain"
    ),
]


class GatedProjectionGraph(nn.Module):
    """Manages receptor-gated projections with learned gating curves.

    For each gated projection, the effective release is:
      release = base_release * gate_value
    where:
      gate_value = σ(polarity * (gate_weight * nt_level + gate_bias))
    """

    def __init__(self, gated_projections: list[GatedProjection] | None = None,
                 region_dims: dict[str, int] | None = None):
        super().__init__()
        if gated_projections is None:
            gated_projections = GATED_PROJECTIONS
        if region_dims is None:
            region_dims = {}

        self.gated = gated_projections
        n = len(gated_projections)

        # Learnable gate parameters per projection
        self.gate_weight = nn.Parameter(torch.ones(n) * 2.0)
        self.gate_bias = nn.Parameter(torch.zeros(n))

        # Signal-carrying linear maps (where dimensions are known)
        self.signal_maps = nn.ModuleDict()
        for i, gp in enumerate(gated_projections):
            if gp.carries_signal and gp.src in region_dims and gp.dst in region_dims:
                self.signal_maps[f"g{i}"] = nn.Linear(
                    region_dims[gp.src], region_dims[gp.dst], bias=False)
                nn.init.normal_(self.signal_maps[f"g{i}"].weight, std=0.02)

    def compute_gates(self, nt_levels: torch.Tensor) -> torch.Tensor:
        """Compute gate values for all gated projections.
        nt_levels: (B, N_NT) → returns (B, n_gated) gate values in [0, 1]."""
        device = nt_levels.device
        gates = []
        for i, gp in enumerate(self.gated):
            nt_idx = NT_INDEX[gp.gate_nt]
            level = nt_levels[:, nt_idx]  # (B,)
            raw = gp.gate_polarity * (self.gate_weight[i] * level + self.gate_bias[i])
            gates.append(torch.sigmoid(raw))
        return torch.stack(gates, dim=-1)  # (B, n_gated)

    def gated_release(self, nt_levels: torch.Tensor,
                      activities: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute how much each NT should be released via gated projections.
        Returns {nt_name: (B,) additional release demand}."""
        gates = self.compute_gates(nt_levels)  # (B, n_gated)
        release_by_nt: dict[str, list[torch.Tensor]] = {}
        for i, gp in enumerate(self.gated):
            src_act = activities.get(gp.src)
            if src_act is None:
                continue
            amount = src_act.clamp(0, 1) * gp.release_scale * gates[:, i]
            release_by_nt.setdefault(gp.nt, []).append(amount)
        # Sum contributions per NT
        result = {}
        for nt_name, amounts in release_by_nt.items():
            result[nt_name] = torch.stack(amounts).sum(0)
        return result

    def transmit_signal(self, idx: int, src_signal: torch.Tensor,
                        gate_value: torch.Tensor) -> Optional[torch.Tensor]:
        """Carry gated signal from src to dst.
        src_signal: (B, D_src), gate_value: (B,)"""
        key = f"g{idx}"
        if key not in self.signal_maps:
            return None
        signal = self.signal_maps[key](src_signal)
        return signal * gate_value.unsqueeze(-1)

    def info(self, nt_levels: torch.Tensor) -> dict:
        gates = self.compute_gates(nt_levels)
        return {
            f"{gp.src}→{gp.dst}({gp.nt})": float(gates[0, i])
            for i, gp in enumerate(self.gated)
        }
