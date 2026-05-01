"""Synaptic plasticity gating: LTP / LTD modulation.

Models how neuromodulators gate learning rate / gradient flow:
  - NMDA receptor activity gates LTP (high Glu + depolarization → potentiate)
  - D1 receptor activity enables LTP in striatal direct pathway
  - D2 receptor activity enables LTD in striatal indirect pathway
  - ACh (via M1) enhances signal-to-noise in learning
  - NE broadens learning (explore) vs narrows (exploit) based on level
  - 5HT modulates learning patience (high 5HT → accept delayed reward)
  - BDNF (from trophic system) permits structural plasticity

This module produces a per-parameter or per-layer learning rate multiplier
that is applied during the training step.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .transmitters import NT_NAMES, NT_INDEX, N_NT


class PlasticityGate(nn.Module):
    """Computes a learning-rate multiplier from neuromodulator state."""

    def __init__(self):
        super().__init__()
        # Learned mapping: NT vector → scalar LR multiplier
        self.gate = nn.Sequential(
            nn.Linear(N_NT + 2, 16), nn.GELU(),  # +2 for BDNF, RPE
            nn.Linear(16, 8), nn.GELU(),
            nn.Linear(8, 1),
        )
        # Per-NT sensitivity weights (interpretable)
        self.nt_importance = nn.Parameter(torch.tensor([
            1.5,   # DA  — strong LTP enabler
            0.8,   # NE  — moderate (exploration)
            0.3,   # 5HT — patience signal
            1.0,   # ACh — attention/SNR
            -0.5,  # eCB — retrograde inhibition of plasticity
            2.0,   # Glu — NMDA/LTP driver
            -1.0,  # GABA — inhibits plasticity
        ]))

    def forward(self, nt_levels: torch.Tensor, bdnf: torch.Tensor,
                rpe: torch.Tensor) -> torch.Tensor:
        """nt_levels: (B, N_NT), bdnf: (B,), rpe: (B,)
        Returns: lr_multiplier (B,) in [0.1, 3.0]."""
        # Weight NTs by importance
        weighted = nt_levels * self.nt_importance.to(nt_levels.device)
        x = torch.cat([weighted, bdnf.unsqueeze(-1), rpe.unsqueeze(-1)], dim=-1)
        raw = self.gate(x).squeeze(-1)
        # Map to [0.1, 3.0] for stable training
        return 0.1 + 2.9 * torch.sigmoid(raw)

    def ltp_strength(self, nt_levels: torch.Tensor) -> torch.Tensor:
        """Quick estimate of LTP potential from Glu (NMDA) + DA (D1).
        Returns (B,) in [0, 1]."""
        glu = nt_levels[:, NT_INDEX["Glu"]]
        da = nt_levels[:, NT_INDEX["DA"]]
        # NMDA requires both glutamate AND depolarization (approximated by DA)
        return torch.sigmoid(2.0 * (glu + da) - 1.5)

    def ltd_strength(self, nt_levels: torch.Tensor) -> torch.Tensor:
        """LTD potential from D2 + low calcium (low Glu).
        Returns (B,) in [0, 1]."""
        glu = nt_levels[:, NT_INDEX["Glu"]]
        da = nt_levels[:, NT_INDEX["DA"]]
        gaba = nt_levels[:, NT_INDEX["GABA"]]
        # LTD: moderate DA (D2 pathway) + low glutamate + GABA
        return torch.sigmoid(1.5 * da - 2.0 * glu + gaba - 0.5)

    def explore_exploit_ratio(self, nt_levels: torch.Tensor) -> torch.Tensor:
        """NE-driven exploration vs exploitation balance.
        High NE → explore (broader learning), low NE → exploit (narrow/precise).
        Returns (B,) where >0.5 = explore, <0.5 = exploit."""
        ne = nt_levels[:, NT_INDEX["NE"]]
        ach = nt_levels[:, NT_INDEX["ACh"]]
        # NE broadens, ACh narrows
        return torch.sigmoid(3.0 * ne - 2.0 * ach)
