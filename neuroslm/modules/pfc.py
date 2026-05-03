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
        # Genome-driven algorithmic choices
        self._selection_strategy = 'threshold'  # 'threshold' | 'softmax' | 'inhibition_gated'
        self._recall_temperature = 1.0
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_sem, nhead=n_heads, dim_feedforward=d_sem * 4,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.select_query = nn.Parameter(torch.randn(1, 1, d_sem) * 0.02)
        self.replace_head = nn.Linear(d_sem, 1)
        # Inhibition gate for 'inhibition_gated' strategy (genome-driven)
        self.inhibition_gate = nn.Linear(d_sem, d_sem)

    def forward(self, gws_slots: torch.Tensor, recalls: torch.Tensor,
                floating_thought: torch.Tensor):
        B = gws_slots.size(0)
        ft = floating_thought.unsqueeze(1)
        q = self.select_query.expand(B, -1, -1)
        # Concatenate everything into a sequence; transformer attends across it.
        x = torch.cat([q, gws_slots, recalls, ft], dim=1)
        y = self.transformer(x)

        # Genome-driven selection strategy
        if self._selection_strategy == 'softmax':
            # Softmax attention over all slot outputs, temperature-controlled
            scores = (y[:, 0:1] * y[:, 1:]).sum(-1) / max(0.1, self._recall_temperature)
            weights = F.softmax(scores, dim=-1)            # (B, S-1)
            selected = (weights.unsqueeze(-1) * y[:, 1:]).sum(1)  # (B, d_sem)
            selected = selected * self._attend_gate
        elif self._selection_strategy == 'inhibition_gated':
            # GABA-style lateral inhibition before selection
            inhib = torch.sigmoid(self.inhibition_gate(y[:, 0]))  # (B, d_sem)
            selected = y[:, 0] * inhib * self._attend_gate
        else:
            # Default threshold selection (original behavior)
            selected = y[:, 0] * self._attend_gate

        replace_gate = torch.sigmoid(
            self.replace_head(selected) * self._select_gate).squeeze(-1)
        return selected, replace_gate

    def configure_from_genome(self, env: dict, structural=None):
        """Apply genome params: scalar gates AND algorithmic strategy."""
        super().configure_from_genome(env, structural=structural)
        self._attend_gate = max(0.01, self.genv_float('attend_gate', 1.0))
        self._select_gate = max(0.01, self.genv_float('select_gate', 1.0))
        self._modulation_gain = max(0.01, self.genv_float('modulation_gain', 1.0))
        # Algorithmic choices from genome opcode patterns
        self._selection_strategy = self.genv_str('selection_strategy', 'threshold')
        self._recall_temperature = max(0.1, self.genv_float('recall_temperature', 1.0))
