"""Mixture of Depths — dynamic layer skipping per token.

Paper: "Mixture-of-Depths" (Raposo et al., Google DeepMind, 2024)
Key insight: not all tokens need the same number of layers. Easy tokens
(function words, predictable continuations) can skip layers. Hard tokens
(novel concepts, reasoning steps) use all layers. A lightweight router
per layer decides which tokens proceed vs skip.

    For each layer l:
      router_score = Linear(x) → scalar per token
      top-C tokens (by score) pass through the layer
      remaining tokens skip with identity (residual only)

This gives:
  - 2× throughput at same quality (50% of tokens skip ~half the layers)
  - Better quality at same FLOPs (hard tokens get more compute)
  - Naturally genome-controllable: genome sets capacity C per layer

Integration with NeuroSLM:
  - NT modulation: NE (arousal) → increase capacity (more tokens processed)
  - Genome controls per-layer capacity ratio
  - Router scores contribute to consciousness metrics (which tokens "enter awareness")
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoDRouter(nn.Module):
    """Per-layer router: 2-layer MLP scores token hardness for top-k selection.

    Hard tokens (novel concepts, reasoning steps) receive full layer depth.
    Easy tokens (function words, predictable continuations) skip via residual.
    The MLP router captures non-linear token hardness better than a linear probe.
    """

    def __init__(self, dim: int, capacity_ratio: float = 0.5, n_nt: int = 0):
        super().__init__()
        self.capacity_ratio = capacity_ratio
        # 2-layer MLP: captures token hardness (local context, surprisal proxy)
        hidden = max(32, dim // 8)
        self.router = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, 1, bias=True),
        )
        # Zero-init so routing starts uniform (all tokens score 0)
        nn.init.zeros_(self.router[0].weight)
        nn.init.zeros_(self.router[2].weight)
        nn.init.zeros_(self.router[2].bias)

        if n_nt > 0:
            self.nt_capacity = nn.Linear(n_nt, 1, bias=False)
            nn.init.zeros_(self.nt_capacity.weight)
        else:
            self.nt_capacity = None

    def forward(self, x: torch.Tensor,
                nt: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (selected_x, indices, router_logits).

        selected_x: (B, C, D) — top-C hard tokens
        indices:    (B, C)    — sorted positions (causal order preserved)
        router_logits: (B, T, 1) — raw scores for aux loss
        """
        B, T, D = x.shape
        router_logits = self.router(x)  # (B, T, 1)

        cap = self.capacity_ratio
        if self.nt_capacity is not None and nt is not None:
            nt_mod = torch.sigmoid(self.nt_capacity(nt))   # (B, 1)
            cap = cap * (0.5 + nt_mod.squeeze(-1).mean().item())
            cap = float(min(1.0, max(0.1, cap)))

        C = max(1, int(T * cap))
        scores = router_logits.squeeze(-1)                  # (B, T)
        _, topk_idx = scores.topk(C, dim=-1, sorted=False)
        topk_idx, _ = topk_idx.sort(dim=-1)                 # preserve causal order
        selected_x = x.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, D))
        return selected_x, topk_idx, router_logits


class MoDBlock(nn.Module):
    """Transformer block with Mixture-of-Depths token routing.

    Only top-C tokens (by router score) pass through the attention+MLP.
    Remaining tokens get identity (residual passthrough).
    """

    def __init__(self, dim: int, n_heads: int, max_ctx: int,
                 n_kv_heads: int | None = None,
                 n_nt: int = 0, capacity_ratio: float = 0.5,
                 use_diff_attn: bool = True):
        super().__init__()
        from .common import SwiGLU, RMSNorm
        self.router = MoDRouter(dim, capacity_ratio, n_nt)

        if use_diff_attn:
            from .differential_attention import DifferentialAttention
            self.attn = DifferentialAttention(dim, n_heads, max_ctx, n_kv_heads, n_nt)
        else:
            from .common import CausalSelfAttention
            self.attn = CausalSelfAttention(dim, n_heads, max_ctx, n_kv_heads, n_nt=n_nt)

        self.n1 = RMSNorm(dim)
        self.n2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim)

    def forward(self, x: torch.Tensor,
                nt: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape

        # Route: select which tokens get processed
        selected_x, indices, router_logits = self.router(x, nt)
        C = selected_x.size(1)

        if C == T:
            # All tokens selected — standard path
            x = x + self.attn(self.n1(x), nt=nt)
            x = x + self.mlp(self.n2(x))
            return x

        # Process only selected tokens
        h = self.attn(self.n1(selected_x), nt=nt)
        h = h + self.mlp(self.n2(selected_x + h))

        # Scatter back: selected tokens get residual + output; others get identity
        out = x.clone()
        out.scatter_(1, indices.unsqueeze(-1).expand(-1, -1, D), selected_x + h)

        # Store router logits for auxiliary load-balancing loss
        self._last_router_logits = router_logits

        return out

    @property
    def router_aux_loss(self) -> torch.Tensor:
        """Load-balancing loss: keeps mean routing fraction ≈ capacity_ratio.
        Also penalises variance to prevent batch-level routing collapse."""
        logits = getattr(self, '_last_router_logits', None)
        if logits is None:
            return torch.tensor(0.0)
        probs = torch.sigmoid(logits.squeeze(-1))       # (B, T)
        mean_prob = probs.mean(dim=-1)                  # (B,)
        target = self.router.capacity_ratio if hasattr(self.router, 'capacity_ratio') \
                 else 0.5
        mean_loss = ((mean_prob - target) ** 2).mean()
        # Entropy bonus: reward routing diversity within each batch item
        entropy = -(probs * (probs + 1e-8).log() +
                    (1 - probs) * (1 - probs + 1e-8).log()).mean()
        return mean_loss - 0.01 * entropy
