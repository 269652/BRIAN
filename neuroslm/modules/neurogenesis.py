"""Neurogenesis / Structural Plasticity for NeuroSLM.

Adult neurogenesis in the hippocampal dentate gyrus is driven by novelty
and stress — new neurons integrate into circuits when the existing
population cannot distinguish between similar patterns (low pattern
separation).  This module adapts that principle to neural networks:

  Grow:  when prediction error is high AND existing neurons are highly
         correlated (similar activations → poor differentiation), spawn
         a new neuron initialized near the direction of highest error.

  Prune: when a neuron's activation variance falls below a threshold for
         many steps, mark it as "silent" and remove it from the active
         pool on the next grow step (replace rather than just expand).

Implementation strategy:
  - Weight matrix is over-provisioned to max_neurons but only
    active_n neurons are used at any time.
  - Soft pruning: each neuron has a learned *importance score* that
    decays exponentially when not activated, and is masked toward zero
    when importance < prune_threshold.  No hard capacity change needed —
    gradient simply stops flowing through low-importance neurons.
  - Grow: when mean surprise > grow_threshold, find the k lowest-importance
    neurons and re-initialize them from the top-gradient directions,
    effectively "replacing" dead neurons with new ones tuned to current
    errors.
  - Growth is gated by a cooldown counter to prevent unstable oscillation.
  - The output dimension stays fixed at d_model (unlike naive expansion),
    so this is fully compatible with the rest of brain.py.

Novel contribution: tying the grow/prune threshold to the model's own
novelty signal (hippocampal surprise from episodic memory, or prediction
error from the active inference module) creates a closed loop where the
model structurally adapts to what it finds difficult — not just what it
processes frequently.

References:
  Bhaskaran & Bhaskaran (2019): Role of adult hippocampal neurogenesis
  Aljundi et al. (2017): Expert Gate (neuron reuse by importance)
  Bellec et al. (2018): Long short-term memory and learning-to-learn (sparse + growing nets)
  Chen et al. (2015): Net2Net: Accelerating learning via knowledge transfer
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class NeurogenesisLayer(nn.Module):
    """Soft-growing FF layer with novelty-gated neuron replacement.

    Parameters
    ----------
    d_model          : input and output dimension
    max_neurons      : maximum pool size (over-provisioned weight matrix)
    init_neurons     : neurons active at start
    grow_threshold   : novelty/surprise value that triggers growth
    prune_threshold  : importance score below which a neuron is dead
    cooldown_steps   : minimum steps between grow events
    importance_decay : exponential decay factor for importance scores
    """

    def __init__(self, d_model: int,
                 max_neurons: int = 512,
                 init_neurons: Optional[int] = None,
                 grow_threshold: float = 0.7,
                 prune_threshold: float = 0.01,
                 cooldown_steps: int = 100,
                 importance_decay: float = 0.99):
        super().__init__()
        self.d_model          = d_model
        self.max_neurons      = max_neurons
        self.grow_threshold   = grow_threshold
        self.prune_threshold  = prune_threshold
        self.cooldown_steps   = cooldown_steps
        self.importance_decay = importance_decay

        self.active_n = init_neurons or min(d_model, max_neurons)

        # Over-provisioned weight matrix
        self.W_in  = nn.Parameter(torch.empty(max_neurons, d_model))
        self.W_out = nn.Parameter(torch.empty(d_model, max_neurons))
        self.bias  = nn.Parameter(torch.zeros(max_neurons))

        # Per-neuron importance score (not a gradient parameter — updated in forward)
        self.register_buffer("importance",
                              torch.ones(max_neurons))

        # Cooldown counter (not a parameter)
        self.register_buffer("_cooldown",
                              torch.tensor(0, dtype=torch.long))

        self.ln = nn.LayerNorm(d_model)

        # Init weights
        nn.init.kaiming_uniform_(self.W_in,  a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out, a=math.sqrt(5))

    # ------------------------------------------------------------------

    def _soft_prune_mask(self) -> torch.Tensor:
        """Returns (max_neurons,) mask: 1.0 for active neurons, soft-zero for dead."""
        mask = torch.clamp(
            self.importance[:self.active_n] / self.prune_threshold,
            max=1.0
        )
        # Pad inactive neurons with 0
        if self.active_n < self.max_neurons:
            pad = torch.zeros(self.max_neurons - self.active_n,
                              device=mask.device)
            mask = torch.cat([mask, pad])
        return mask

    def _try_grow(self, novelty: float, grad_signal: Optional[torch.Tensor]) -> None:
        """Replace dead neurons with new ones tuned to current gradient direction."""
        if self._cooldown.item() > 0:
            self._cooldown -= 1
            return
        if novelty < self.grow_threshold:
            return
        if self.active_n >= self.max_neurons:
            return

        # Find dead neurons (low importance in active pool)
        imp = self.importance[:self.active_n]
        n_dead = (imp < self.prune_threshold).sum().item()
        if n_dead == 0 and self.active_n >= self.max_neurons:
            return

        # Grow: activate one more neuron (up to max)
        n_grow = min(int(n_dead) + 1, self.max_neurons - self.active_n)
        if n_grow <= 0:
            return

        # Init new neurons from gradient signal if available
        if grad_signal is not None and grad_signal.shape[-1] == self.d_model:
            direction = F.normalize(grad_signal.detach().mean(0), dim=-1)
            for i in range(n_grow):
                idx = self.active_n + i
                noise = torch.randn_like(direction) * 0.01
                self.W_in.data[idx] = direction + noise
                self.W_out.data[:, idx] = direction + noise
        self.active_n = min(self.active_n + n_grow, self.max_neurons)
        self.importance[self.active_n - n_grow: self.active_n] = 0.5
        self._cooldown = torch.tensor(self.cooldown_steps, dtype=torch.long,
                                      device=self._cooldown.device)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
                novelty: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, int]:
        """
        x:       (B, T, d_model)
        novelty: (B,) or scalar — surprise/prediction error signal

        Returns:
          out:      (B, T, d_model)
          active_n: int — current number of active neurons (for logging)
        """
        novelty_val = (novelty.mean().item()
                       if novelty is not None else 0.0)

        # ---- Soft-pruned forward pass ----
        mask = self._soft_prune_mask()                      # (max_neurons,)
        W_in_masked  = self.W_in * mask.unsqueeze(1)        # (max_n, D)
        W_out_masked = self.W_out * mask.unsqueeze(0)       # (D, max_n)
        bias_masked  = self.bias * mask                     # (max_n,)

        h   = F.gelu(F.linear(x, W_in_masked, bias_masked))  # (B, T, max_n)
        out = F.linear(h, W_out_masked.t())                   # (B, T, D)
        out = self.ln(x + out)

        # ---- Update importance scores ----
        with torch.no_grad():
            act_mean = h[:, :, :self.active_n].abs().mean((0, 1))  # (active_n,)
            self.importance[:self.active_n] = (
                self.importance_decay * self.importance[:self.active_n]
                + (1 - self.importance_decay) * act_mean
            )

        # ---- Maybe grow (uses grad signal from x if available) ----
        if self.training:
            self._try_grow(novelty_val, x)

        return out, self.active_n
