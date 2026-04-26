"""Language cortex: combined Wernicke (comprehension) + Broca (production).

Contains the token embeddings, transformer stack, and LM head.
The hidden state at the last position is exposed as the "comprehension embedding"
projected into d_sem space for downstream modules.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .common import TransformerBlock, RMSNorm


class LanguageCortex(nn.Module):
    def __init__(self, vocab_size: int, d_hidden: int, d_sem: int,
                 n_layers: int, n_heads: int, max_ctx: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_hidden)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_hidden, n_heads, max_ctx) for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_hidden)
        # Tied output head
        self.lm_head = nn.Linear(d_hidden, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        # Project last-layer hidden state into shared semantic space
        self.to_sem = nn.Linear(d_hidden, d_sem, bias=False)
        # Inverse projection: take a thought (d_sem) and condition generation
        self.from_sem = nn.Linear(d_sem, d_hidden, bias=False)

        # Proper init: small embeddings + scaled output projections.
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.to_sem.weight, mean=0.0, std=0.02)
        # Zero-init the conditioning projection so initial training is a clean LM.
        nn.init.zeros_(self.from_sem.weight)
        for blk in self.blocks:
            for p in blk.parameters():
                if p.dim() >= 2:
                    nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor, thought: torch.Tensor | None = None,
                motor_bias: torch.Tensor | None = None):
        """ids: (B, T). thought: optional (B, d_sem) injected as a prefix bias.
        motor_bias: optional (B, d_hidden) added to the LAST position's hidden
        state before the LM head — this is how the motor cortex shapes
        emission without requiring a second forward pass."""
        h = self.tok_emb(ids)
        if thought is not None:
            bias = self.from_sem(thought).unsqueeze(1)  # (B, 1, d_hidden)
            h = h + bias
        for blk in self.blocks:
            h = blk(h)
        h = self.norm_f(h)
        if motor_bias is not None:
            # Add bias only to the last position (the one that will be sampled).
            h_last = h[:, -1:, :] + motor_bias.unsqueeze(1)
            logits = torch.cat([self.lm_head(h[:, :-1, :]),
                                self.lm_head(h_last)], dim=1)
        else:
            logits = self.lm_head(h)
        sem = self.to_sem(h.mean(dim=1))
        return logits, sem, h
