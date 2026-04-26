"""Homeostasis: a slow controller that tunes neurotransmitter baselines and
gains toward target signal-to-noise ratios + healthy gradient norms.

Inspired by:
  - autoreceptor regulation (e.g., D2 short-loop feedback)
  - astroglial buffering / glutamate-glutamine cycle
  - synaptic scaling

Concretely each step we observe:
  - mean / std of each NT level over recent history
  - LM loss trend (improvement gives positive feedback)
  - gradient norm (too large → reduce excitatory tone, increase GABA)

and nudge `TransmitterSystem.bias` and `gain` slowly toward targets. This
runs OUTSIDE the autograd loop (no gradients flow through it) — pure
control.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .transmitters import N_NT, NT_INDEX, NT_NAMES


class Homeostasis(nn.Module):
    def __init__(self, target_mean: float = 0.3, target_std: float = 0.15,
                 lr: float = 5e-3):
        super().__init__()
        self.target_mean = target_mean
        self.target_std = target_std
        self.lr = lr
        self.register_buffer("ema_level", torch.zeros(N_NT))
        self.register_buffer("ema_var",   torch.zeros(N_NT))
        self.register_buffer("ema_loss",  torch.tensor(float("nan")))
        self.register_buffer("steps",     torch.zeros(1, dtype=torch.long))
        self.ema_alpha = 0.02

    @torch.no_grad()
    def observe(self, transmitters, lm_loss: float, grad_norm: float):
        """Update EMAs and apply slow corrections to transmitter bias/gain."""
        self.steps += 1
        levels = transmitters.level.detach().mean(0)            # (N_NT,)
        var    = transmitters.level.detach().var(0, unbiased=False)
        a = self.ema_alpha
        self.ema_level = (1 - a) * self.ema_level + a * levels
        self.ema_var   = (1 - a) * self.ema_var   + a * var
        if torch.isnan(self.ema_loss):
            self.ema_loss = torch.tensor(lm_loss, device=self.ema_loss.device)
        else:
            self.ema_loss = (1 - a) * self.ema_loss + a * lm_loss

        # ---- Corrections ----
        device = transmitters.bias.device
        mean_err = self.target_mean - self.ema_level.to(device)        # want >0 ⇒ raise baseline
        std_err  = self.target_std  - self.ema_var.sqrt().to(device)   # want >0 ⇒ raise gain

        # Soft updates (write directly to params; no SGD)
        with torch.no_grad():
            transmitters.bias.add_(self.lr * mean_err)
            transmitters.gain.add_(self.lr * std_err)
            transmitters.bias.clamp_(-0.5, 0.5)
            transmitters.gain.clamp_(0.2, 3.0)

            # Gradient-norm safety: if grads exploding, boost GABA tone.
            if math.isfinite(grad_norm) and grad_norm > 5.0:
                transmitters.bias[NT_INDEX["GABA"]].add_(0.05).clamp_(-0.5, 0.5)
                transmitters.bias[NT_INDEX["Glu"]].sub_(0.02).clamp_(-0.5, 0.5)
            elif math.isfinite(grad_norm) and grad_norm < 0.1:
                # Vanishing — boost Glu tone.
                transmitters.bias[NT_INDEX["Glu"]].add_(0.02).clamp_(-0.5, 0.5)

    def state_dict_summary(self) -> dict:
        return {
            "ema_level": {n: float(self.ema_level[i]) for i, n in enumerate(NT_NAMES)},
            "ema_loss":  float(self.ema_loss) if not torch.isnan(self.ema_loss) else None,
            "steps":     int(self.steps.item()),
        }
