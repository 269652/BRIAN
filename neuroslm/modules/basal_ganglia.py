"""Basal ganglia: action selection via Go/NoGo gating, modulated by dopamine.

Generates n_candidates action embeddings, scores each via direct (Go) and
indirect (NoGo) pathways. Dopamine biases the balance.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .genome_configurable import GenomeConfigurable


class BasalGanglia(nn.Module, GenomeConfigurable):
    def __init__(self, d_sem: int, d_action: int, n_candidates: int):
        super().__init__()
        self._genome_env = {}
        self.n_candidates = n_candidates
        # Genome-tunable
        self._go_gate = 1.0        # direct pathway gain
        self._nogo_gate = 1.0      # indirect pathway gain
        self._da_sensitivity = 1.0  # how much DA modulates selection
        self.proposer = nn.Linear(d_sem, d_action * n_candidates)
        self.go = nn.Linear(d_action, 1)
        self.nogo = nn.Linear(d_action, 1)
        self.d_action = d_action

    def forward(self, thought: torch.Tensor, dopamine: torch.Tensor):
        """thought: (B, d_sem). dopamine: (B,) in [0,1]."""
        B = thought.size(0)
        cands = self.proposer(thought).view(B, self.n_candidates, self.d_action)
        go = self.go(cands).squeeze(-1) * self._go_gate        # genome-gated
        nogo = self.nogo(cands).squeeze(-1) * self._nogo_gate  # genome-gated
        # Dopamine boosts Go pathway, suppresses NoGo
        da = dopamine.view(B, 1) * self._da_sensitivity
        score = go * (0.5 + da) - nogo * (1.5 - da)
        probs = F.softmax(score, dim=-1)       # (B, K)
        idx = probs.argmax(dim=-1)             # (B,)
        chosen = cands[torch.arange(B), idx]   # (B, d_action)
        confidence = probs.max(dim=-1).values  # (B,)
        return chosen, confidence, probs

    def configure_from_genome(self, env: dict, structural=None):
        """Apply genome: MODULATE DA sensitivity, SIGMOID go/nogo gates."""
        super().configure_from_genome(env, structural=structural)
        self._go_gate = max(0.01, self.genv_float('go_gate', 1.0))
        self._nogo_gate = max(0.01, self.genv_float('nogo_gate', 1.0))
        self._da_sensitivity = max(0.01, self.genv_float('da_sensitivity', 1.0))
