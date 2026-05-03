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

from .genome_configurable import GenomeConfigurable


class PrefrontalCortex(nn.Module, GenomeConfigurable):
    def __init__(self, d_sem: int, n_layers: int, n_heads: int, learning_rule: str = 'backprop'):
        super().__init__()
        self._genome_env = {}
        self.learning_rule = learning_rule
        # Genome-tunable parameters
        self._attend_gate = 1.0     # attention strength
        self._select_gate = 1.0     # selection confidence
        self._modulation_gain = 1.0 # DA/5HT modulation strength
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
        selected = y[:, 0] * self._attend_gate  # genome-gated attention
        replace_gate = torch.sigmoid(
            self.replace_head(selected) * self._select_gate).squeeze(-1)
        return selected, replace_gate

    def configure_from_genome(self, env: dict, structural=None):
        """Apply genome params: ATTEND gate, PROJECT gate, MODULATE gain."""
        super().configure_from_genome(env, structural=structural)
        self._attend_gate = max(0.01, self.genv_float('attend_gate', 1.0))
        self._select_gate = max(0.01, self.genv_float('select_gate', 1.0))
        self._modulation_gain = max(0.01, self.genv_float('modulation_gain', 1.0))
