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


class SensoryFrameEncoder(nn.Module):
    """Maps a SensoryFrame numeric vector (6-dim) to a d_sem embedding.

    Used in the continuous sensory world loop: at each forward_lm step, one
    virtual-world frame is pulled from the environment stream, encoded here,
    and added as a grounding residual to the world model's latent state.

    Input channels (must match SensoryFrame.to_vec() order):
      valence, arousal, novelty, comfort, time_pressure, social_presence
    """
    N_CHANNELS = 6

    def __init__(self, d_sem: int):
        super().__init__()
        hidden = max(32, d_sem // 4)
        self.enc = nn.Sequential(
            nn.Linear(self.N_CHANNELS, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_sem),
        )
        # Zero-init output so encoder starts as a no-op; effect grows with training
        nn.init.zeros_(self.enc[2].weight)
        nn.init.zeros_(self.enc[2].bias)

    def encode_frame(self, frame_vec: list[float], device,
                     dtype=torch.float32) -> torch.Tensor:
        """frame_vec: list of 6 floats → (1, d_sem)"""
        t = torch.tensor(frame_vec, dtype=dtype, device=device).unsqueeze(0)
        return self.enc(t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N_CHANNELS) → (B, d_sem)"""
        return self.enc(x)
