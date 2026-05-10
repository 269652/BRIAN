"""Projections: directed axonal pathways between regions.

A `Projection` says: when source region S fires, NT `nt` is released onto
target region T (where T's receptors decide the effect). Optionally carries
an embedding signal (a learned linear map S_dim → T_dim) so projections also
move information, not just neurochemistry.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn


@dataclass
class Projection:
    src: str
    dst: str
    nt: str                           # neurotransmitter released
    release_scale: float = 1.0        # how strongly src activity drives release
    carries_signal: bool = True       # whether to transmit an embedding too


class ProjectionGraph(nn.Module):
    """Owns the linear maps for signal-carrying projections + provides
    helpers to compute release amounts from per-region 'activity' scalars.

    Activity scalars are scalar summaries (B,) that each region can produce
    cheaply (e.g., mean(|hidden|), or a learned readout).
    """

    def __init__(self, projections: list[Projection], region_dims: dict[str, int]):
        super().__init__()
        self.projections = projections
        self.region_dims = dict(region_dims)
        self.regions = set(region_dims.keys())
        self.maps = nn.ModuleDict()
        for i, p in enumerate(projections):
            if p.carries_signal:
                self.maps[f"p{i}"] = nn.Linear(region_dims[p.src], region_dims[p.dst], bias=False)
                # Init small so projections start as gentle modulations.
                nn.init.normal_(self.maps[f"p{i}"].weight, std=0.02)

    def add_projection(self, p: Projection) -> int:
        """Append a new projection (structural plasticity sprouting).

        Creates the corresponding signal-carrying linear map if needed and
        returns the index of the new projection.  Used by TrophicSystem to
        grow the connectome between high-activity, weakly-connected regions.
        """
        idx = len(self.projections)
        self.projections.append(p)
        if p.carries_signal:
            if p.src not in self.region_dims or p.dst not in self.region_dims:
                # Unknown regions — fall back to non-carrying NT-only edge
                p.carries_signal = False
            else:
                m = nn.Linear(self.region_dims[p.src],
                              self.region_dims[p.dst], bias=False)
                nn.init.normal_(m.weight, std=0.02)
                # Place the new map on the same device as existing ones
                if len(self.maps) > 0:
                    any_w = next(iter(self.maps.values())).weight
                    m = m.to(device=any_w.device, dtype=any_w.dtype)
                self.maps[f"p{idx}"] = m
        return idx

    def transmit(self, idx: int, src_signal: torch.Tensor) -> Optional[torch.Tensor]:
        """Carry `src_signal` (B, D_src) through the i-th projection, returning
        a (B, D_dst) contribution to add to the destination, or None if this
        projection doesn't carry signal."""
        p = self.projections[idx]
        if not p.carries_signal:
            return None
        return self.maps[f"p{idx}"](src_signal)

    def release_amount(self, idx: int, src_activity: torch.Tensor) -> torch.Tensor:
        """How much NT this projection releases this tick.
        src_activity: (B,) scalar in [0,1]."""
        return src_activity.clamp(0.0, 1.0) * self.projections[idx].release_scale
