"""Default Mode Network: scheduler / orchestrator of the cognitive loop.

Operates on slow clock (every dmn_period sensory ticks). It does not by itself
do heavy computation; it produces a query embedding that drives downstream
modules and a stop signal indicating whether to keep thinking.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .genome_configurable import GenomeConfigurable


class DefaultModeNetwork(nn.Module, GenomeConfigurable):
    def __init__(self, d_sem: int, n_slots: int, n_layers: int):
        super().__init__()
        self._genome_env = {}
        # Genome-tunable parameters
        self._wander_gate = 1.0   # how much mind-wandering noise
        self._recall_gate = 1.0   # associative recall strength
        self._oscillate_freq = 0.3  # default-mode oscillation frequency
        in_dim = d_sem * n_slots + d_sem  # GWS slots + floating thought
        layers = []
        cur = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(cur, d_sem * 2), nn.GELU()]
            cur = d_sem * 2
        self.mlp = nn.Sequential(*layers)
        self.query_head = nn.Linear(cur, d_sem)
        self.stop_head = nn.Linear(cur, 1)

    def forward(self, gws_slots: torch.Tensor, floating_thought: torch.Tensor):
        """gws_slots: (B, n_slots, d_sem). floating_thought: (B, d_sem)."""
        B = gws_slots.size(0)
        x = torch.cat([gws_slots.reshape(B, -1), floating_thought], dim=-1)
        h = self.mlp(x)
        query = self.query_head(h) * self._recall_gate  # genome-gated query
        stop_logit = self.stop_head(h).squeeze(-1)  # (B,)
        return query, stop_logit

    def configure_from_genome(self, env: dict, structural=None):
        """Apply genome params: RECALL gate, OSCILLATE freq, wander noise."""
        super().configure_from_genome(env, structural=structural)
        self._wander_gate = max(0.0, self.genv_float('wander_gate', 1.0))
        self._recall_gate = max(0.01, self.genv_float('recall_gate', 1.0))
        self._oscillate_freq = self.genv_float('oscillate_freq', 0.3)
