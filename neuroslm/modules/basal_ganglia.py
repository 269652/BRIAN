"""Basal ganglia: action selection via Go/NoGo gating, modulated by dopamine.

Generates n_candidates action embeddings, scores each via direct (Go) and
indirect (NoGo) pathways. Dopamine biases the balance.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasalGanglia(nn.Module):
    def __init__(self, d_sem: int, d_action: int, n_candidates: int):
        super().__init__()
        self.n_candidates = n_candidates
        self.proposer = nn.Linear(d_sem, d_action * n_candidates)
        self.go = nn.Linear(d_action, 1)
        self.nogo = nn.Linear(d_action, 1)
        self.d_action = d_action

    def forward(self, thought: torch.Tensor, dopamine: torch.Tensor):
        """thought: (B, d_sem). dopamine: (B,) in [0,1]."""
        B = thought.size(0)
        cands = self.proposer(thought).view(B, self.n_candidates, self.d_action)
        go = self.go(cands).squeeze(-1)        # (B, K)
        nogo = self.nogo(cands).squeeze(-1)    # (B, K)
        # Dopamine boosts Go pathway, suppresses NoGo
        da = dopamine.view(B, 1)
        score = go * (0.5 + da) - nogo * (1.5 - da)
        probs = F.softmax(score, dim=-1)       # (B, K)
        idx = probs.argmax(dim=-1)             # (B,)
        chosen = cands[torch.arange(B), idx]   # (B, d_action)
        confidence = probs.max(dim=-1).values  # (B,)
        return chosen, confidence, probs
