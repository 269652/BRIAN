"""Reuptake transporter dynamics.

Models DAT (dopamine), SERT (serotonin), NET (norepinephrine) transporters.
Each transporter has:
  - density: how many transporters are expressed (adapts slowly)
  - efficiency: current clearance rate (fast dynamics)
  - occupancy: fraction blocked by inhibitors (e.g., simulated SSRIs)

The reuptake system runs each tick and removes NT from the synaptic cleft
proportional to (density * efficiency * (1 - occupancy) * level).

Homeostasis can upregulate/downregulate transporter density over longer
timescales (hundreds of ticks) based on sustained NT levels.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .transmitters import NT_NAMES, N_NT, NT_INDEX


# Which NTs have active reuptake transporters
TRANSPORTER_MAP = {
    "DA":  "DAT",
    "NE":  "NET",
    "5HT": "SERT",
    "Glu": "EAAT",   # excitatory amino acid transporter
    "GABA": "GAT",   # GABA transporter
}

# NTs without classical reuptake (degraded enzymatically or retrograde)
# ACh → AChE (acetylcholinesterase), eCB → FAAH/MAGL
ENZYME_DEGRADED = {"ACh", "eCB"}


class ReuptakeSystem(nn.Module):
    """Per-NT reuptake transporter bank with adaptive density."""

    def __init__(self, clearance_rate: float = 0.15,
                 density_lr: float = 1e-3,
                 min_density: float = 0.2,
                 max_density: float = 2.0):
        super().__init__()
        self.clearance_rate = clearance_rate
        self.density_lr = density_lr
        self.min_density = min_density
        self.max_density = max_density

        # Transporter density per NT (learnable, adapts via homeostasis)
        self.register_buffer("density", torch.ones(N_NT))
        # Occupancy (fraction blocked) — set externally if simulating drugs
        self.register_buffer("occupancy", torch.zeros(N_NT))
        # EMA of NT levels for density adaptation
        self.register_buffer("ema_level", torch.full((N_NT,), 0.3))
        # Target level each transporter tries to maintain
        self.register_buffer("target_level", torch.tensor([
            0.10,  # DA  — low tonic
            0.15,  # NE
            0.30,  # 5HT — moderate tonic
            0.20,  # ACh
            0.05,  # eCB
            0.40,  # Glu
            0.40,  # GABA
        ]))

    @torch.no_grad()
    def clear(self, transmitters) -> torch.Tensor:
        """Apply reuptake clearance to transmitter levels.
        Returns the amount cleared per NT (B, N_NT)."""
        device = transmitters.level.device
        density = self.density.to(device)
        occupancy = self.occupancy.to(device)

        # Clearance = density * base_rate * (1 - occupancy) * current_level
        effective = density * self.clearance_rate * (1.0 - occupancy)
        cleared = transmitters.level * effective.unsqueeze(0)

        # Enzymatic degradation for ACh and eCB (faster, density-independent)
        for nt_name in ENZYME_DEGRADED:
            idx = NT_INDEX[nt_name]
            cleared[:, idx] = transmitters.level[:, idx] * 0.25

        # Apply clearance
        new_level = (transmitters.level - cleared).clamp(0.0, 1.0)
        transmitters.level = new_level
        return cleared

    @torch.no_grad()
    def adapt_density(self, transmitters):
        """Slowly adapt transporter density based on sustained NT levels.
        High sustained levels → upregulate transporters (more clearance).
        Low sustained levels → downregulate (less clearance).
        Models neuroplastic adaptation over ~100s of ticks."""
        device = transmitters.level.device
        levels = transmitters.level.detach().mean(0).to(device)
        alpha = 0.01
        self.ema_level = (1 - alpha) * self.ema_level.to(device) + alpha * levels

        # Error: positive means level is too high → increase density
        error = self.ema_level - self.target_level.to(device)
        self.density = (self.density.to(device) + self.density_lr * error).clamp(
            self.min_density, self.max_density)

    def set_occupancy(self, nt_name: str, fraction: float):
        """Simulate transporter blockade (e.g., SSRI blocks SERT)."""
        idx = NT_INDEX[nt_name]
        self.occupancy[idx] = max(0.0, min(1.0, fraction))

    def info(self) -> dict:
        return {
            "density": {n: float(self.density[i]) for i, n in enumerate(NT_NAMES)},
            "occupancy": {n: float(self.occupancy[i]) for i, n in enumerate(NT_NAMES)},
            "ema_level": {n: float(self.ema_level[i]) for i, n in enumerate(NT_NAMES)},
        }
