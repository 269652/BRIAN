"""Default Mode Network: scheduler / orchestrator of the cognitive loop.

Operates on slow clock (every dmn_period sensory ticks). It does not by itself
do heavy computation; it produces a query embedding that drives downstream
modules and a stop signal indicating whether to keep thinking.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class DefaultModeNetwork(nn.Module):
    def __init__(self, d_sem: int, n_slots: int, n_layers: int):
        super().__init__()
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
        query = self.query_head(h)
        stop_logit = self.stop_head(h).squeeze(-1)  # (B,)
        return query, stop_logit
