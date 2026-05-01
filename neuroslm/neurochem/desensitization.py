"""Receptor desensitization and sensitization dynamics.

Models how prolonged NT exposure downregulates receptor sensitivity (tolerance)
and how withdrawal upregulates it (sensitization). This is a critical
homeostatic mechanism that prevents runaway excitation/inhibition.

Each receptor type has:
  - sensitivity: current gain multiplier (starts at 1.0)
  - exposure_ema: exponential moving average of recent NT levels at that receptor
  - adaptation_rate: how fast sensitivity changes

Rules:
  - Sustained high exposure → desensitization (sensitivity decreases)
  - Sustained low exposure → sensitization (sensitivity increases)
  - Sensitivity bounds: [0.1, 3.0] to prevent total loss or runaway gain

Neuroscience basis:
  - D2 autoreceptors desensitize with chronic DA → reduced DA release
  - 5HT2A downregulates with chronic SSRI use (therapeutic delay)
  - GABA-A desensitizes with chronic benzodiazepine exposure (tolerance)
  - CB1 desensitizes with chronic cannabinoid exposure
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .transmitters import NT_NAMES, N_NT, NT_INDEX


class ReceptorAdaptation(nn.Module):
    """Tracks receptor sensitivity adaptation across the system."""

    def __init__(self, adaptation_rate: float = 5e-4,
                 recovery_rate: float = 2e-4,
                 min_sensitivity: float = 0.1,
                 max_sensitivity: float = 3.0):
        super().__init__()
        self.adaptation_rate = adaptation_rate
        self.recovery_rate = recovery_rate
        self.min_sensitivity = min_sensitivity
        self.max_sensitivity = max_sensitivity

        # Per-NT sensitivity (applied as multiplier on receptor gains)
        self.register_buffer("sensitivity", torch.ones(N_NT))
        # Exposure EMA
        self.register_buffer("exposure_ema", torch.full((N_NT,), 0.3))
        # Baseline exposure (what the system considers "normal")
        self.register_buffer("baseline_exposure", torch.tensor([
            0.10, 0.15, 0.30, 0.20, 0.05, 0.40, 0.40
        ]))
        # Steps counter for time-dependent effects
        self.register_buffer("steps", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update(self, transmitters):
        """Call each tick. Updates exposure EMA and adapts sensitivity."""
        self.steps += 1
        device = transmitters.level.device
        levels = transmitters.level.detach().mean(0).to(device)

        # Update exposure EMA (tau ~50 ticks for short-term, could add long-term too)
        alpha = 0.02
        self.exposure_ema = (1 - alpha) * self.exposure_ema.to(device) + alpha * levels

        # Compute deviation from baseline
        deviation = self.exposure_ema - self.baseline_exposure.to(device)

        # Positive deviation (overexposure) → desensitize (reduce sensitivity)
        # Negative deviation (underexposure) → sensitize (increase sensitivity)
        desens = torch.where(
            deviation > 0,
            -self.adaptation_rate * deviation,  # desensitize
            -self.recovery_rate * deviation     # sensitize (deviation is negative, so this increases)
        )

        self.sensitivity = (self.sensitivity.to(device) + desens).clamp(
            self.min_sensitivity, self.max_sensitivity)

    def get_sensitivity(self, nt_name: str) -> float:
        """Get current sensitivity for a specific NT's receptors."""
        return float(self.sensitivity[NT_INDEX[nt_name]])

    def apply_to_gain(self, base_gain: torch.Tensor, nt_indices: torch.Tensor) -> torch.Tensor:
        """Modulate receptor bank gain by sensitivity.
        base_gain: (B,), nt_indices: indices of NTs used by the receptor bank."""
        device = base_gain.device
        sens = self.sensitivity.to(device)
        # Average sensitivity across the NTs this receptor bank uses
        avg_sens = sens[nt_indices].mean()
        return base_gain * avg_sens

    def info(self) -> dict:
        return {n: float(self.sensitivity[i]) for i, n in enumerate(NT_NAMES)}
