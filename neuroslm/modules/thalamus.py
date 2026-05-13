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
        residual = x + h
        return self.norm(residual.float()).to(dtype=residual.dtype)


class Thalamus(nn.Module):
    """Content-aware router with optional stochastic ε-exploration.

    During training, with probability ε the router output for each batch
    item is replaced by a one-hot routing onto a *non-Language* expert
    stream (Math, Reasoning, Spatial, or Social). This injects gradient
    flow into expert cortices even when the input is plain language —
    ensuring they develop a baseline vocabulary and contribute to Φ.

    ε is also softly tied to maturity: more exploration when the network
    is young (MAT low); the user can pass `maturity` in forward() to scale.
    """

    # Stream indices we explore into (skip 0 = "language")
    EXPLORE_STREAMS = (1, 2, 3, 4)   # math / reasoning / spatial / social

    def __init__(self, d_sem: int, hidden: int | None = None,
                 epsilon: float = 0.1):
        super().__init__()
        hidden = hidden or d_sem
        self.streams = nn.ModuleList([StreamAdapter(d_sem, hidden) for _ in STREAM_NAMES])
        self.router = nn.Linear(d_sem, len(STREAM_NAMES))
        self.norm = nn.LayerNorm(d_sem)
        self.epsilon = float(epsilon)

    def forward(self, x: torch.Tensor, nt_levels: torch.Tensor | None = None,
                return_routing: bool = False,
                maturity: float | None = None):
        """x: (B, d_sem). Returns (gated_output, routing_probs).

        nt_levels (B, N_NT): NE controls softmax temperature, ACh boosts top-stream.
        maturity: optional MAT scalar in [0, 1]. Scales the ε exploration
                  probability so young networks (low M) explore more.
                  Effective ε = self.epsilon * (1 - M).

        Homeostatic max-norm enforcement: all incoming signals clamped to max norm 1.0
        to prevent signal magnitude explosion in re-entrant bowtie loops."""
        # Enforce max norm on input (homeostatic gate)
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)
        x = x / x_norm.clamp(min=1.0)  # clip signals with norm > 1.0

        logits = self.router(x)                                  # (B, S)

        if nt_levels is not None:
            nt_levels = nt_levels.to(dtype=x.dtype)
            ne  = nt_levels[:, NT_INDEX["NE"]].unsqueeze(-1)     # (B,1)
            ach = nt_levels[:, NT_INDEX["ACh"]].unsqueeze(-1)
            # NE sharpens (lower temperature); base T=1.0, range ~ [0.5, 1.0]
            temp = 1.0 / (0.5 + ne)
            logits = logits * temp
        probs = F.softmax(logits, dim=-1)                        # (B, S)

        # ── Stochastic ε-exploration ───────────────────────────────────────
        # Active during training only. With probability ε (scaled by 1-M so
        # young networks explore more), replace the routing distribution for
        # each batch item with a one-hot mass on a random *non-language*
        # expert stream. Uses torch.where → XLA-static, no Python branching.
        if self.training and self.epsilon > 0.0:
            B = probs.shape[0]
            eps_eff = self.epsilon * (1.0 - (float(maturity) if maturity is not None else 0.0))
            eps_eff = max(0.0, min(1.0, eps_eff))
            # Random stream index per batch item from EXPLORE_STREAMS
            stream_pool = torch.tensor(self.EXPLORE_STREAMS, device=probs.device)
            pick_idx = stream_pool[torch.randint(
                0, len(self.EXPLORE_STREAMS), (B,), device=probs.device)]
            one_hot = F.one_hot(pick_idx, num_classes=probs.shape[1]).to(probs.dtype)
            # Per-batch exploration mask: 1 with prob eps_eff
            explore_mask = (torch.rand(B, 1, device=probs.device) < eps_eff).to(probs.dtype)
            probs = torch.where(explore_mask.bool().expand_as(probs), one_hot, probs)

        # Compute each stream's contribution (vectorized)
        outs = torch.stack([s(x) for s in self.streams], dim=1)  # (B, S, d_sem)
        if nt_levels is not None:
            # ACh boosts the top stream's contribution
            top_mask = (probs == probs.max(dim=-1, keepdim=True).values).to(probs.dtype)
            boost = 1.0 + 0.5 * ach * top_mask                   # (B, S)
            mixed = (outs * (probs * boost).unsqueeze(-1)).sum(dim=1)
        else:
            mixed = (outs * probs.unsqueeze(-1)).sum(dim=1)

        out = self.norm(mixed.float()).to(dtype=mixed.dtype)
        if return_routing:
            return out, probs
        return out
