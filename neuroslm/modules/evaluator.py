"""Evaluator (ACC / OFC analog): scalar value of predicted next state.

Decides whether a proposed action's predicted outcome justifies execution.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class Evaluator(nn.Module):
    def __init__(self, d_sem: int, n_neuromods: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_sem * 2 + n_neuromods, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, 1),
        )

    def forward(self, world_pred: torch.Tensor, self_pred: torch.Tensor,
                neuromods: torch.Tensor):
        x = torch.cat([world_pred, self_pred, neuromods], dim=-1)
        return torch.tanh(self.mlp(x)).squeeze(-1)  # (B,) in [-1, 1]
