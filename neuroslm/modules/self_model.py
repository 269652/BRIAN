"""Self model: recurrent embedding of the agent's own state.

Inputs each tick: last action embedding, neuromodulator vector, floating thought.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SelfModel(nn.Module):
    def __init__(self, d_sem: int, d_action: int, n_neuromods: int,
                 d_hidden: int, n_layers: int):
        super().__init__()
        in_dim = d_action + n_neuromods + d_sem
        self.in_proj = nn.Linear(in_dim, d_sem)
        self.rnn = nn.GRU(d_sem, d_hidden, num_layers=n_layers, batch_first=True)
        self.proj = nn.Linear(d_hidden, d_sem)
        self.d_hidden = d_hidden
        self.n_layers = n_layers

    def init_state(self, batch_size: int, device) -> torch.Tensor:
        return torch.zeros(self.n_layers, batch_size, self.d_hidden, device=device)

    def forward(self, last_action: torch.Tensor, neuromods: torch.Tensor,
                floating_thought: torch.Tensor, h: torch.Tensor):
        x = torch.cat([last_action, neuromods, floating_thought], dim=-1)
        x = self.in_proj(x).unsqueeze(1)
        y, h_new = self.rnn(x, h)
        z = self.proj(y.squeeze(1))
        return z, h_new
