"""Receptors: per-region banks that translate NT levels into multiplicative
gains on input or output activations of a brain region.

Each region owns a `ReceptorBank` declaring which NT/receptor pairs it carries
and what each does (excitatory vs inhibitory, gain on what stream). The bank
applies these as: `out = signal * (1 + sum(weight_i * NT_level_i))` with a
sigmoid clamp.

Receptor types of interest (very simplified):
  D1   — excitatory on PFC/striatum direct pathway (Go)
  D2   — inhibitory on indirect pathway / autoregulatory
  alpha2 — inhibits NE release (autoreceptor)
  5HT2A — increases gain in cortex
  M1   — ACh: increases signal-to-noise (sharpens attention)
  CB1  — eCB: retrograde, suppresses presynaptic release
  NMDA — Glu: gates plasticity (used to scale grad updates)
  GABAA — GABA: divisive inhibition / dropout-like
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
import torch.nn as nn

from .transmitters import NT_INDEX


@dataclass
class Receptor:
    nt: str           # which NT binds
    sign: float       # +1 excitatory, -1 inhibitory
    weight: float = 0.5  # learnable scale init


class ReceptorBank(nn.Module):
    """A mixable bank of receptors that yields a scalar gain (B,) per call."""

    def __init__(self, receptors: Sequence[Receptor]):
        super().__init__()
        self.receptors = list(receptors)
        # Learnable per-receptor weight (so the brain can tune sensitivity).
        self.w = nn.Parameter(torch.tensor([r.weight * r.sign for r in receptors]))
        self.idx = torch.tensor([NT_INDEX[r.nt] for r in receptors], dtype=torch.long)

    def gain(self, nt_levels: torch.Tensor) -> torch.Tensor:
        """nt_levels: (B, N_NT) → gain (B,) ≈ 1 + Σ w_i · level_i, sigmoid-clamped."""
        idx = self.idx.to(nt_levels.device)
        levels = nt_levels.index_select(1, idx)        # (B, R)
        contrib = (levels * self.w).sum(-1)            # (B,)
        # Map to [0.1, 2.0] via shifted sigmoid for stable multiplicative gain.
        return 0.1 + 1.9 * torch.sigmoid(contrib)

    def modulate(self, x: torch.Tensor, nt_levels: torch.Tensor) -> torch.Tensor:
        """Apply gain to last-dim of x."""
        g = self.gain(nt_levels)
        # x: (B, ..., D) — broadcast g over middle dims
        shape = [g.size(0)] + [1] * (x.dim() - 1)
        return x * g.view(shape)
