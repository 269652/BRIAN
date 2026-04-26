"""BDNF / NGF: neurotrophic factors that grow or prune projections.

Each projection in `ProjectionGraph` gets a scalar `trophic_level` ∈ [0, 1].
Update rule (per tick):
    Δ trophic_level = BDNF · co_activation - NGF_decay - disuse
where:
  - co_activation  = cosine similarity of recent src/dst activity scalars
                     (positive Hebbian: 'fire together, wire together')
  - BDNF           = global level, raised by positive reward / RPE
  - NGF_decay      = small constant (slow forgetting)
  - disuse         = penalty when neither end fires

Effects:
  - The trophic level multiplicatively scales the projection's signal-carrying
    linear map weight (potentiation / depression).
  - When trophic_level drops below `prune_threshold`, the projection is
    DISABLED (zero contribution).
  - When it would saturate above 1.0, the system can SPAWN a new projection
    along an inferred high-coactivation edge (sprouting).

This is an inference-time / training-time process (no SGD); changes persist
in buffers and are saved with the checkpoint.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .projections import ProjectionGraph, Projection


class TrophicSystem(nn.Module):
    def __init__(self, graph: ProjectionGraph,
                 prune_threshold: float = 0.05,
                 sprout_threshold: float = 0.95,
                 bdnf_baseline: float = 0.005,
                 ngf_decay: float = 0.002,
                 max_projections: int = 64):
        super().__init__()
        self.graph = graph
        self.prune_threshold = prune_threshold
        self.sprout_threshold = sprout_threshold
        self.bdnf_baseline = bdnf_baseline
        self.ngf_decay = ngf_decay
        self.max_projections = max_projections
        n = len(graph.projections)
        # Start every existing projection at mid trophic level
        self.register_buffer("trophic", torch.full((n,), 0.5))
        self.register_buffer("active",  torch.ones(n))
        # Co-activation EMA per projection (over recent ticks)
        self.register_buffer("ema_coact", torch.zeros(n))
        self._steps = 0

    @torch.no_grad()
    def update(self, activities: dict[str, torch.Tensor], bdnf: float, ngf: float):
        """activities: {region: (B,) ∈ [0,1]}.
        bdnf, ngf: scalar floats from `Brain` (driven by reward / novelty).
        """
        # Scale neurotrophin signals so they don't overwhelm the dynamics.
        bdnf = max(0.0, min(0.05, bdnf * 0.05))
        ngf  = max(0.0, min(0.01, ngf  * 0.01))
        self._steps += 1
        for i, p in enumerate(self.graph.projections):
            a = activities.get(p.src)
            b = activities.get(p.dst)
            if a is None or b is None:
                co = 0.0
            else:
                co = float((a * b).mean().clamp(0.0, 1.0))
            self.ema_coact[i] = 0.95 * self.ema_coact[i] + 0.05 * co
            growth = (bdnf + self.bdnf_baseline) * (0.1 + self.ema_coact[i])
            decay  = ngf + self.ngf_decay + 0.001 * (1.0 - self.ema_coact[i])
            new = (self.trophic[i] + growth - decay).clamp(0.0, 1.0)
            self.trophic[i] = new
            if new < self.prune_threshold:
                self.active[i] = 0.0
            elif new > self.prune_threshold * 2.0 and self.active[i] == 0.0:
                self.active[i] = 1.0

    def gain(self, idx: int) -> float:
        """Multiplicative gain to apply to projection idx's signal map."""
        return float(self.active[idx] * (0.2 + 1.6 * self.trophic[idx]))

    def stats(self) -> dict:
        return {
            "n_projections": int(self.active.numel()),
            "n_active":      int(self.active.sum().item()),
            "n_pruned":      int((self.active == 0).sum().item()),
            "trophic_mean":  float(self.trophic.mean().item()),
            "trophic_max":   float(self.trophic.max().item()),
            "trophic_min":   float(self.trophic.min().item()),
            "saturated":     int((self.trophic > self.sprout_threshold).sum().item()),
        }
