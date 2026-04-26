"""Subconscious threat critic.

A tiny, fast classifier that runs every cognitive tick *before* the slower
PFC-mediated reasoning. It looks at the current world model embedding and
self model embedding and outputs a scalar 'threat to self' in [0, 1].

When threat exceeds `threat_threshold`, the brain enters SURVIVAL MODE:
  - LC is forced to release a high amount of NE
  - 5HT is suppressed (less patience, narrower time horizon)
  - The thalamus is biased toward the 'spatial' / fast-action streams
  - Mind wandering is interrupted

This module has *no* gradient connection to the LM loss; it is trained either
self-supervised (predicting future loss spikes) or supervised on synthetic
threat data. For now we initialize it with a simple heuristic: threat ∝
‖z_self - moving_avg(z_self)‖ (sudden self-state shifts) + ‖z_world‖ tail.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SubconsciousCritic(nn.Module):
    def __init__(self, d_sem: int, threat_threshold: float = 0.6):
        super().__init__()
        self.threat_threshold = threat_threshold
        self.mlp = nn.Sequential(
            nn.Linear(d_sem * 2, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, 1),
        )
        self.register_buffer("ema_self", torch.zeros(d_sem))
        self.register_buffer("inited", torch.zeros(1, dtype=torch.bool))

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor):
        """Returns (threat (B,), in_survival_mode (B,) bool)."""
        with torch.no_grad():
            if not bool(self.inited.item()):
                self.ema_self = z_self.detach().mean(0)
                self.inited.fill_(True)
            else:
                self.ema_self = 0.95 * self.ema_self + 0.05 * z_self.detach().mean(0)
            self_shift = (z_self.detach() - self.ema_self).pow(2).mean(-1).sqrt()
            heuristic = torch.sigmoid(2.0 * self_shift - 1.0)        # (B,)

        x = torch.cat([z_world, z_self], dim=-1)
        learned = torch.sigmoid(self.mlp(x)).squeeze(-1)              # (B,)
        # Combine learned + heuristic (learned starts ~0.5, heuristic gives signal)
        threat = 0.5 * learned + 0.5 * heuristic
        survival = (threat > self.threat_threshold)
        return threat, survival
