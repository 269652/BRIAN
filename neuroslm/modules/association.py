"""Association cortex: multimodal fusion.

In the text-only prototype this is a no-op pass-through with a learned gate,
but the API supports adding more modality streams as a list.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class AssociationCortex(nn.Module):
    def __init__(self, d_sem: int, max_modalities: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_sem, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(d_sem)

    def forward(self, modality_embeds: list[torch.Tensor]) -> torch.Tensor:
        # Stack: (B, M, d_sem)
        x = torch.stack(modality_embeds, dim=1)
        # Self-attend across modalities, then pool
        y, _ = self.attn(x, x, x, need_weights=False)
        fused = self.norm(y.mean(dim=1))  # (B, d_sem)
        return fused
