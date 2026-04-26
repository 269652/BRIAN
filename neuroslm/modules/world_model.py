"""World model: recurrent state representing the environment.

Uses a stacked GRU as a CPU-friendly stand-in for an SSM.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class WorldModel(nn.Module):
    def __init__(self, d_sem: int, d_hidden: int, n_layers: int):
        super().__init__()
        self.rnn = nn.GRU(d_sem, d_hidden, num_layers=n_layers, batch_first=True)
        self.proj = nn.Linear(d_hidden, d_sem)
        self.predict_head = nn.Linear(d_hidden, d_sem)
        self.d_hidden = d_hidden
        self.n_layers = n_layers

    def init_state(self, batch_size: int, device) -> torch.Tensor:
        return torch.zeros(self.n_layers, batch_size, self.d_hidden, device=device)

    def forward(self, sensory: torch.Tensor, h: torch.Tensor):
        """sensory: (B, d_sem). h: (L, B, d_hidden).
        Returns z_world (B, d_sem), h_new, predicted_next_sensory (B, d_sem)."""
        x = sensory.unsqueeze(1)  # (B, 1, d_sem)
        y, h_new = self.rnn(x, h)
        z = self.proj(y.squeeze(1))
        pred = self.predict_head(y.squeeze(1))
        return z, h_new, pred
