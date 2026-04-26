"""Neuromodulator system: produces 4 global scalars (DA, NE, 5HT, ACh)
based on recent reward/novelty/uncertainty/prediction-error signals.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class Neuromodulators(nn.Module):
    """Output channel order: [DA, NE, 5HT, ACh]."""
    def __init__(self, n_neuromods: int = 4):
        super().__init__()
        # Inputs: reward, novelty, uncertainty, prediction_error
        self.mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, n_neuromods),
        )
        self.n = n_neuromods

    def forward(self, reward: torch.Tensor, novelty: torch.Tensor,
                uncertainty: torch.Tensor, pred_err: torch.Tensor):
        x = torch.stack([reward, novelty, uncertainty, pred_err], dim=-1)
        return torch.sigmoid(self.mlp(x))   # (B, n) in [0,1]
