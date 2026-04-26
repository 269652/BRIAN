"""Thalamus: content-aware router.

Models the medial-dorsal nucleus + pulvinar as a soft router. Given an
embedding from the GWS, it classifies which "stream" the content belongs to
(language, math/symbolic, reasoning, spatial/visual, social) and dispatches
the embedding to a small specialized adapter per stream. The combined,
gated output is returned to be consumed by downstream regions (PFC etc).

Acts like a learned mixture-of-experts gate, but conditioned both on content
and on neuromodulator state (NE sharpens routing — high NE → more peaky
softmax; ACh raises the gain of the chosen stream).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..neurochem.transmitters import NT_INDEX


STREAM_NAMES = ("language", "math", "reasoning", "spatial", "social")


class StreamAdapter(nn.Module):
    """Small per-stream specialist: 2-layer MLP with residual."""
    def __init__(self, d_sem: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_sem, hidden)
        self.fc2 = nn.Linear(hidden, d_sem)
        self.norm = nn.LayerNorm(d_sem)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        h = self.fc2(h)
        return self.norm(x + h)


class Thalamus(nn.Module):
    def __init__(self, d_sem: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or d_sem
        self.streams = nn.ModuleList([StreamAdapter(d_sem, hidden) for _ in STREAM_NAMES])
        self.router = nn.Linear(d_sem, len(STREAM_NAMES))
        self.norm = nn.LayerNorm(d_sem)

    def forward(self, x: torch.Tensor, nt_levels: torch.Tensor | None = None,
                return_routing: bool = False):
        """x: (B, d_sem). Returns (gated_output, routing_probs).
        nt_levels (B, N_NT): NE controls softmax temperature, ACh boosts top-stream."""
        logits = self.router(x)                                  # (B, S)

        if nt_levels is not None:
            ne  = nt_levels[:, NT_INDEX["NE"]].unsqueeze(-1)     # (B,1)
            ach = nt_levels[:, NT_INDEX["ACh"]].unsqueeze(-1)
            # NE sharpens (lower temperature); base T=1.0, range ~ [0.5, 1.0]
            temp = 1.0 / (0.5 + ne)
            logits = logits * temp
        probs = F.softmax(logits, dim=-1)                        # (B, S)

        # Compute each stream's contribution (vectorized)
        outs = torch.stack([s(x) for s in self.streams], dim=1)  # (B, S, d_sem)
        if nt_levels is not None:
            # ACh boosts the top stream's contribution
            top_mask = (probs == probs.max(dim=-1, keepdim=True).values).float()
            boost = 1.0 + 0.5 * ach * top_mask                   # (B, S)
            mixed = (outs * (probs * boost).unsqueeze(-1)).sum(dim=1)
        else:
            mixed = (outs * probs.unsqueeze(-1)).sum(dim=1)

        out = self.norm(mixed)
        if return_routing:
            return out, probs
        return out
