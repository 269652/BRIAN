"""Subconscious threat critic — fast pre-PFC survival circuit.

Runs every cognitive tick BEFORE the slower PFC-mediated reasoning.
Looks at world-model and self-model embeddings and outputs:
  threat   (B,) in [0,1] — estimated threat level
  survival (B,) bool     — True = enter survival mode (NE surge, narrow attention)

When survival is triggered:
  - LC is forced to release a high amount of NE
  - 5HT is suppressed
  - BG NoGo pathway is boosted (freeze / safe action preference)
  - Mind wandering is interrupted

Threat is estimated from two signals combined:
  1. Heuristic: sudden self-state shifts  (‖z_self - EMA(z_self)‖)
  2. Learned:   MLP(z_world ‖ z_self)  → scalar
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .brain_module import BrainModule


class SubconsciousCritic(BrainModule):
    def __init__(self, d_sem: int, threat_threshold: float = 0.55):
        super().__init__()
        self.threat_threshold = threat_threshold
        self.mlp = nn.Sequential(
            nn.Linear(d_sem * 2, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, 1),
        )
        self.register_buffer("ema_self", torch.zeros(d_sem))
        self.register_buffer("inited",   torch.zeros(1, dtype=torch.bool))

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor):
        """Returns (threat (B,), in_survival_mode (B,) bool)."""
        with torch.no_grad():
            if not bool(self.inited.item()):
                self.ema_self = z_self.detach().mean(0)
                self.inited.fill_(True)
            else:
                self.ema_self = (0.95 * self.ema_self
                                 + 0.05 * z_self.detach().mean(0))
            self_shift = (z_self.detach() - self.ema_self).pow(2).mean(-1).sqrt()
            heuristic  = torch.sigmoid(2.0 * self_shift - 1.0)   # (B,)

        x       = torch.cat([z_world, z_self], dim=-1)
        learned = torch.sigmoid(self.mlp(x)).squeeze(-1)          # (B,)
        threat  = 0.5 * learned + 0.5 * heuristic
        survival = (threat > self.threat_threshold)
        return threat, survival

    def _disabled_output(self, z_world, *_, **__):
        B = z_world.size(0)
        return (torch.zeros(B, device=z_world.device),
                torch.zeros(B, dtype=torch.bool, device=z_world.device))

    def to_device(self, device):
        return self.to(device)
