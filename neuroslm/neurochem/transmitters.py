"""Transmitter system: per-batch state of every neurotransmitter.

Each NT has:
  - `level`     : current synaptic concentration in [0, 1]
  - `vesicles`  : reserve in [0, 1]; depletes on release, replenished slowly
  - `baseline`  : tonic level set by homeostasis
  - `tau`       : decay time constant (per tick)

Updates use simple Euler integration. State is kept as plain tensors so it
participates in the autograd graph for the duration of a batch (we detach
between batches to prevent unbounded graph growth).
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


# Canonical NT order — used everywhere.
NT_NAMES = ("DA", "NE", "5HT", "ACh", "eCB", "Glu", "GABA")
N_NT = len(NT_NAMES)
NT_INDEX = {n: i for i, n in enumerate(NT_NAMES)}


@dataclass
class NTParams:
    tau_decay: float = 0.85       # per-tick multiplicative decay
    tau_vesicle: float = 0.02     # vesicle replenishment rate
    release_cost: float = 0.3     # vesicles consumed per unit released
    baseline: float = 0.1         # tonic floor


# Sensible defaults per NT (could be learned later).
NT_DEFAULTS = {
    "DA":   NTParams(tau_decay=0.80, baseline=0.10),
    "NE":   NTParams(tau_decay=0.70, baseline=0.15),
    "5HT":  NTParams(tau_decay=0.95, baseline=0.30),  # slow / tonic
    "ACh":  NTParams(tau_decay=0.75, baseline=0.20),
    "eCB":  NTParams(tau_decay=0.60, baseline=0.05),  # retrograde, fast
    "Glu":  NTParams(tau_decay=0.50, baseline=0.40),  # fast excitatory
    "GABA": NTParams(tau_decay=0.50, baseline=0.40),  # fast inhibitory
}


class TransmitterSystem(nn.Module):
    """Holds (B, N_NT) tensors of `level` and `vesicles`.

    Provides `release(name, amount)` and `step()` which decays / replenishes.
    """

    def __init__(self):
        super().__init__()
        # Learnable per-NT modulation of the defaults (so homeostasis can adapt).
        self.bias = nn.Parameter(torch.zeros(N_NT))
        self.gain = nn.Parameter(torch.ones(N_NT))
        self._tau_decay   = torch.tensor([NT_DEFAULTS[n].tau_decay   for n in NT_NAMES])
        self._tau_vesicle = torch.tensor([NT_DEFAULTS[n].tau_vesicle for n in NT_NAMES])
        self._baseline    = torch.tensor([NT_DEFAULTS[n].baseline    for n in NT_NAMES])
        self._release_cost= torch.tensor([NT_DEFAULTS[n].release_cost for n in NT_NAMES])
        self.register_buffer("level",    torch.zeros(1, N_NT))
        self.register_buffer("vesicles", torch.ones(1, N_NT))

    # -- state management -----------------------------------------------------
    def reset(self, batch_size: int, device):
        self.level    = self._baseline.to(device).expand(batch_size, -1).clone()
        self.vesicles = torch.ones(batch_size, N_NT, device=device)

    def detach_(self):
        self.level    = self.level.detach()
        self.vesicles = self.vesicles.detach()

    # -- core dynamics --------------------------------------------------------
    def release(self, name: str, amount: torch.Tensor):
        """`amount`: (B,) request in [0,1]. Returns actually-released (B,).
        Vesicle-limited; updates internal state in place (autograd-safe via
        functional reassignment).
        """
        idx = NT_INDEX[name]
        amount = amount.clamp(0.0, 1.0)
        v = self.vesicles[:, idx]
        actual = torch.minimum(amount, v / self._release_cost[idx].to(amount.device))
        # Build new tensors (avoid in-place ops that break autograd)
        new_level = self.level.clone()
        new_ves   = self.vesicles.clone()
        new_level[:, idx] = (self.level[:, idx] + actual * self.gain[idx]).clamp(0.0, 1.0)
        new_ves[:, idx]   = v - actual * self._release_cost[idx].to(amount.device)
        self.level    = new_level
        self.vesicles = new_ves
        return actual

    def step(self):
        """Time step: decay levels toward baseline, replenish vesicles."""
        device = self.level.device
        decay    = self._tau_decay.to(device)
        baseline = (self._baseline.to(device) + self.bias).clamp(0.0, 1.0)
        repl     = self._tau_vesicle.to(device)
        new_level = self.level * decay + baseline * (1.0 - decay)
        new_ves   = (self.vesicles + repl).clamp(0.0, 1.0)
        self.level    = new_level
        self.vesicles = new_ves

    # -- accessors ------------------------------------------------------------
    def get(self, name: str) -> torch.Tensor:
        """Current synaptic level of NT `name`. Shape (B,)."""
        return self.level[:, NT_INDEX[name]]

    def vector(self) -> torch.Tensor:
        """Full NT vector (B, N_NT) for downstream consumers."""
        return self.level
