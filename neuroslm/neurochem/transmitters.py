"""Transmitter system: per-batch state of every neurotransmitter.

Each NT has:
  - `level`     : current synaptic concentration in [0, 1]
  - `vesicles`  : reserve in [0, 1]; depletes on release, replenished slowly
  - `baseline`  : tonic level set by homeostasis
  - `tau`       : decay time constant (per tick)

Updates use simple Euler integration. State is kept as plain tensors so it
participates in the autograd graph for the duration of a batch (we detach
between batches to prevent unbounded graph growth).

Maturity Index (MAT virtual protein):
  M_t = clamp(1.0 - L_lm / L_random, 0, 1)
  where L_random = log(vocab_size) ≈ 10.84 for the default GPT-2 BPE.
  Drives the continuous fade-in of expert cortices, MoD compute, and the
  GABA homeostatic dampening tolerance.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn


# Canonical NT order — used everywhere.
NT_NAMES = ("DA", "NE", "5HT", "ACh", "eCB", "Glu", "GABA")
N_NT = len(NT_NAMES)
NT_INDEX = {n: i for i, n in enumerate(NT_NAMES)}


# Random-init LM loss floor for a 50257-vocab tokenizer (GPT-2 BPE):
# log(50257) ≈ 10.825
L_RANDOM_DEFAULT: float = math.log(50257)


def compute_mat(lm_loss: float, l_random: float = L_RANDOM_DEFAULT) -> float:
    """Compute MAT (Maturity Index) from current LM loss.

        M_t = clamp(1.0 - L_lm / L_random, 0, 1)

    Returns a scalar in [0, 1]. 0 = freshly-initialised, 1 = perfect prediction.
    """
    if l_random <= 0:
        return 0.0
    m = 1.0 - (float(lm_loss) / float(l_random))
    if m < 0.0:
        return 0.0
    if m > 1.0:
        return 1.0
    return m


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
    "GABA": NTParams(tau_decay=0.90, baseline=0.10),  # homeostatic decay γ=0.9, target=0.1
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
        # Internal step counter, used by the early-training 5-HT hard cap
        # (see step()).  Auto-increments on every `step()` call so the cap
        # phases out naturally; can be set explicitly via `set_train_step()`.
        self.register_buffer("_train_step", torch.zeros(1, dtype=torch.long))

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

    # 5-HT hard cap during early training. The slow τ_5HT=0.95 + tonic releases
    # let 5HT climb toward the ceiling and over-inhibit plasticity, which has
    # been linked to gradient spikes around the awakening window (gnorm
    # excursions > 1.5 → loss regression). Capping 5HT well below ceiling
    # during the first SHT_CAP_WARMUP_STEPS prevents that. Phases out after.
    SHT_CAP_WARMUP_STEPS: int = 20_000
    SHT_CAP_WARMUP_VALUE: float = 0.65
    SHT_CAP_NORMAL_VALUE: float = 0.95

    def set_train_step(self, step: int) -> None:
        """Inform the NT system of the current global training step.

        Optional — `step()` also auto-increments an internal counter, but
        passing the real step from train.py keeps the cap aligned across
        resumes / restarts.
        """
        with torch.no_grad():
            self._train_step.fill_(int(step))

    def step(self):
        """Time step: decay levels toward baseline, replenish vesicles.

        Saturation scavenging: when any channel's level exceeds 0.9, apply
        an extra multiplicative drop of 0.85× so chronically ceiling-pinned
        transmitters (5HT/GABA at τ≈0.95 cannot escape the ceiling with the
        normal +0.5 max homeostasis bias) get aggressively returned to the
        operating band.  Mirrors physiological auto-receptor / fast-reuptake
        scavenging triggered by extracellular excess.

        5-HT cap: during the first SHT_CAP_WARMUP_STEPS, clamp 5HT to at
        most SHT_CAP_WARMUP_VALUE (=0.65). After warmup, cap at
        SHT_CAP_NORMAL_VALUE (=0.95). Prevents the over-inhibition pattern
        that's been correlated with gradient spikes at the awakening band.
        """
        device = self.level.device
        decay    = self._tau_decay.to(device)
        base_raw = self._baseline.to(device)
        # Bias can modulate baseline but not kill it — floor at 50% of default
        baseline = torch.clamp(base_raw + self.bias, min=base_raw * 0.5, max=torch.ones_like(base_raw))
        repl     = self._tau_vesicle.to(device)
        new_level = self.level * decay + baseline * (1.0 - decay)
        # Saturation scavenge — fast reuptake when level > 0.9.
        sat_mask = (new_level > 0.9).to(new_level.dtype)
        new_level = new_level * (1.0 - 0.15 * sat_mask)
        # 5-HT hard cap (early-training over-inhibition guard).
        step_now = int(self._train_step.item())
        sht_cap = (self.SHT_CAP_WARMUP_VALUE
                   if step_now < self.SHT_CAP_WARMUP_STEPS
                   else self.SHT_CAP_NORMAL_VALUE)
        sht_idx = NT_INDEX["5HT"]
        new_level[:, sht_idx] = new_level[:, sht_idx].clamp(max=sht_cap)
        new_ves   = (self.vesicles + repl).clamp(0.0, 1.0)
        self.level    = new_level
        self.vesicles = new_ves
        # Advance internal counter (lets the cap auto-phase-out if the train
        # loop never calls set_train_step explicitly).
        with torch.no_grad():
            self._train_step += 1

    # -- accessors ------------------------------------------------------------
    def get(self, name: str) -> torch.Tensor:
        """Current synaptic level of NT `name`. Shape (B,)."""
        return self.level[:, NT_INDEX[name]]

    def vector(self) -> torch.Tensor:
        """Full NT vector (B, N_NT) for downstream consumers."""
        return self.level
