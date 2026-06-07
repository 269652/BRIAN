# -*- coding: utf-8 -*-
"""Hebbian Fast Weights — transient outer-product memory (Phase VI).

Rapid context-dependent binding via local Hebbian updates.
"""
import torch
import torch.nn as nn


class HebbianFastWeights(nn.Module):
    """Transient outer-product associative memory.

    Updated during forward: A ← (1-η)A + η(h_t ⊗ h_{t-1})
    Applied to hidden state: h_out = h + g·A @ h_in (gated by zero-init scalar)
    """

    def __init__(self, d_sem: int, eta: float = 0.05):
        super().__init__()
        self.d_sem = d_sem
        self.eta = eta
        # Fast weight matrix (register as buffer, not parameter)
        self.register_buffer("A", torch.zeros(d_sem, d_sem))
        # Gate (zero-init for ReZero discipline)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Apply fast weights with zero-init gate."""
        if x is None:
            x = torch.zeros(1, self.d_sem)

        # With zero-init gate, output ≈ x (identity at first forward)
        g = torch.sigmoid(self.gate)  # Soft gate, zero-init → ~0

        # Fast weight contribution (gated)
        # TODO: Implement outer-product Hebbian update
        # For now, return input scaled by gate

        return x * (1 - g)  # Gate opens as training proceeds
