"""Global Workspace — Baars/Dehaene-style broadcast bus.

A small set of slots that compete for content via softmax routing. Modules
write candidate embeddings; a learned gate picks which fill the slots.
NE temperature controls competition sharpness.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalWorkspace(nn.Module):
    def __init__(self, d_sem: int, n_slots: int, n_heads: int = 4,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.n_slots = n_slots
        self.gradient_checkpointing = gradient_checkpointing
        self.slot_queries = nn.Parameter(torch.randn(n_slots, d_sem) * 0.02)
        self.attn = nn.MultiheadAttention(d_sem, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_sem)

    def _forward(self, candidates: torch.Tensor,
                 ne_temp: torch.Tensor | None = None) -> torch.Tensor:
        B = candidates.size(0)
        q = self.slot_queries.unsqueeze(0).expand(B, -1, -1)
        if ne_temp is not None:
            q = q * ne_temp.view(B, 1, 1)
        out, _ = self.attn(q, candidates, candidates, need_weights=False)
        return self.norm(out)

    def forward(self, candidates: torch.Tensor,
                ne_temp: torch.Tensor | None = None) -> torch.Tensor:
        """candidates: (B, K, d_sem) — embeddings competing for slot occupancy.
        Returns slots: (B, n_slots, d_sem)."""
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, candidates, ne_temp, use_reentrant=False)
        return self._forward(candidates, ne_temp)
