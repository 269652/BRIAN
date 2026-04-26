"""Prefrontal cortex: executive selection and floating-thought gating.

Inputs: GWS slots, hippocampal recalls, current floating thought.
Outputs:
  - selected_thought (B, d_sem)
  - replace_gate (B,) in [0,1]: probability of replacing floating thought outright
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrefrontalCortex(nn.Module):
    def __init__(self, d_sem: int, n_layers: int, n_heads: int):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_sem, nhead=n_heads, dim_feedforward=d_sem * 4,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.select_query = nn.Parameter(torch.randn(1, 1, d_sem) * 0.02)
        self.replace_head = nn.Linear(d_sem, 1)

    def forward(self, gws_slots: torch.Tensor, recalls: torch.Tensor,
                floating_thought: torch.Tensor):
        B = gws_slots.size(0)
        ft = floating_thought.unsqueeze(1)
        q = self.select_query.expand(B, -1, -1)
        # Concatenate everything into a sequence; transformer attends across it.
        x = torch.cat([q, gws_slots, recalls, ft], dim=1)
        y = self.transformer(x)
        selected = y[:, 0]                        # the query-token reads out
        replace_gate = torch.sigmoid(self.replace_head(selected)).squeeze(-1)
        return selected, replace_gate
