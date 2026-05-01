"""Claustrum: cross-modal binding and consciousness integration hub.

The claustrum is a thin sheet of neurons with the highest connectivity
density in the brain (Crick & Koch, 2005). It's hypothesized to be
critical for:
  - Cross-modal binding (integrating sight, sound, touch into unified percept)
  - Consciousness (Koubeissi et al., 2014: stimulation disrupts consciousness)
  - Salience-driven attention routing
  - Synchronizing cortical oscillations

This module acts as a "binding switchboard" that:
  1. Receives projections from ALL cortical areas
  2. Detects coincident activation patterns (binding by synchrony)
  3. Routes bound representations back to relevant areas
  4. Maintains a unified "experience vector" — the claustral gestalt
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClaustralCell(nn.Module):
    """A single claustral neuron: wide input, selective output."""

    def __init__(self, d_input: int, d_state: int):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_state)
        self.recurrent = nn.Linear(d_state, d_state, bias=False)
        self.gate = nn.Linear(d_input + d_state, d_state)
        self.register_buffer("state", torch.zeros(1, d_state))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h = self.state.expand(B, -1) if self.state.size(0) != B else self.state
        gate = torch.sigmoid(self.gate(torch.cat([x, h], dim=-1)))
        candidate = torch.tanh(self.input_proj(x) + self.recurrent(h))
        new_h = gate * candidate + (1 - gate) * h
        self.state = new_h.detach()
        return new_h


class Claustrum(nn.Module):
    """Cross-modal binding hub.

    Takes outputs from multiple brain modules (sensory, language, memory,
    motor, emotional) and produces:
      1. Bound representation: unified multi-modal gestalt
      2. Synchrony signal: which modules are currently co-active
      3. Routing mask: which modules should receive the bound signal
      4. Salience map: what's most important right now
    """

    def __init__(self, d_sem: int, n_modalities: int = 8):
        super().__init__()
        self.d_sem = d_sem
        self.n_modalities = n_modalities

        # Per-modality input projections (each cortical area projects here)
        self.input_projs = nn.ModuleList([
            nn.Linear(d_sem, d_sem) for _ in range(n_modalities)
        ])

        # Coincidence detection: pairwise synchrony
        self.coincidence = nn.Sequential(
            nn.Linear(d_sem * 2, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, 1),
            nn.Sigmoid(),
        )

        # Claustral cells: integrate across modalities with state
        self.cells = ClaustralCell(d_sem * n_modalities, d_sem)

        # Binding via attention: which modalities bind together?
        self.bind_query = nn.Linear(d_sem, d_sem)
        self.bind_key = nn.Linear(d_sem, d_sem)
        self.bind_value = nn.Linear(d_sem, d_sem)

        # Output routing: which areas get the bound signal?
        self.route_proj = nn.Sequential(
            nn.Linear(d_sem, n_modalities),
            nn.Sigmoid(),
        )

        # Salience computation
        self.salience_head = nn.Sequential(
            nn.Linear(d_sem, d_sem // 2),
            nn.GELU(),
            nn.Linear(d_sem // 2, 1),
            nn.Sigmoid(),
        )

        # Gestalt output: the unified experience
        self.gestalt_proj = nn.Sequential(
            nn.Linear(d_sem * 2, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, d_sem),
        )

    def forward(self, modality_inputs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            modality_inputs: list of (B, d_sem) tensors from different brain areas

        Returns dict with bound representation, synchrony, routing, salience.
        """
        n = len(modality_inputs)
        B = modality_inputs[0].size(0)
        device = modality_inputs[0].device

        # Project each modality
        projected = []
        for i, inp in enumerate(modality_inputs):
            proj_fn = self.input_projs[i % self.n_modalities]
            projected.append(proj_fn(inp))

        # Stack for attention-based binding
        stacked = torch.stack(projected, dim=1)  # (B, N, D)

        # Self-attention binding: modalities attend to each other
        Q = self.bind_query(stacked)
        K = self.bind_key(stacked)
        V = self.bind_value(stacked)
        scale = self.d_sem ** 0.5
        attn = torch.bmm(Q, K.transpose(1, 2)) / scale  # (B, N, N)
        attn = F.softmax(attn, dim=-1)
        bound = torch.bmm(attn, V)  # (B, N, D)

        # Synchrony matrix: which modalities are co-active?
        synchrony = attn.mean(0)  # (N, N) average synchrony pattern

        # Claustral cell integration: full multi-modal state
        # Pad if fewer modalities than expected
        flat_parts = [p for p in projected]
        while len(flat_parts) < self.n_modalities:
            flat_parts.append(torch.zeros(B, self.d_sem, device=device))
        flat = torch.cat(flat_parts[:self.n_modalities], dim=-1)
        cell_out = self.cells(flat)  # (B, D)

        # Routing: which areas should receive the bound representation?
        route_mask = self.route_proj(cell_out)  # (B, n_modalities)

        # Salience: overall importance of current binding
        salience = self.salience_head(cell_out).squeeze(-1)

        # Gestalt: unified experience = cell state + attention-bound aggregate
        bound_agg = bound.mean(1)  # (B, D)
        gestalt = self.gestalt_proj(torch.cat([cell_out, bound_agg], dim=-1))

        return {
            "gestalt": gestalt,          # (B, D) unified cross-modal representation
            "bound": bound,              # (B, N, D) per-modality bound signals
            "synchrony": synchrony,      # (N, N) cross-modal synchrony
            "routing": route_mask,       # (B, n_modalities) where to send
            "salience": salience,        # (B,) overall salience
            "cell_state": cell_out,      # (B, D) claustral persistent state
        }
