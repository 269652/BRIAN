"""Forward model (cerebellum-like): predicts next (world, self) given current
state and a candidate action. Used for mental simulation before execution.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ForwardModel(nn.Module):
    def __init__(self, d_sem: int, d_action: int, n_layers: int):
        super().__init__()
        in_dim = d_sem * 2 + d_action
        layers = []
        cur = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(cur, d_sem * 2), nn.GELU()]
            cur = d_sem * 2
        self.trunk = nn.Sequential(*layers)
        self.world_head = nn.Linear(cur, d_sem)
        self.self_head = nn.Linear(cur, d_sem)

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor,
                action: torch.Tensor):
        x = torch.cat([z_world, z_self, action], dim=-1)
        h = self.trunk(x)
        return self.world_head(h), self.self_head(h)
