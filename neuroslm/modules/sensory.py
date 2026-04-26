"""Sensory cortices.

In this text-only prototype, the only modality is language. This module is a
thin wrapper that takes a comprehension embedding from the language cortex and
exposes it as a 'sensory token'. It is the integration point for adding vision
or audio later (each modality would own a SensoryEncoder subclass).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TextSensoryCortex(nn.Module):
    """Identity-ish wrapper; the language cortex already does the encoding work.

    Kept as a separate module so that:
      - swapping modalities later is clean
      - a salience/attention mask can be applied here (superior-colliculus analog)
    """
    def __init__(self, d_sem: int):
        super().__init__()
        self.salience = nn.Linear(d_sem, 1)
        self.proj = nn.Linear(d_sem, d_sem)

    def forward(self, sem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # sem: (B, d_sem)
        salience = torch.sigmoid(self.salience(sem))   # (B, 1)
        encoded = self.proj(sem) * salience
        return encoded, salience.squeeze(-1)
