"""Adaptive Compute Flow.

Two complementary mechanisms that let a small model spend more compute on
hard tokens (the single biggest known lever for parameter efficiency):

1. **AdaptiveComputeBlock** — a transformer block that can be re-applied
   1..K times to the *same* token positions, with weight sharing. This is
   the Universal-Transformer / DeepThinker idea: depth becomes a function
   of token difficulty rather than a fixed hyperparameter.

2. **PonderController** — a learned halting head (PonderNet style). For
   each token at each step, it produces a halting probability λ_t. The
   expected compute per token is bounded; tokens that look "easy"
   (high logit confidence, low layer-norm of residual update) halt early.

Together these emulate a much deeper transformer for a constant parameter
budget, at the cost of variable wall-clock per token (which is exactly
what reasoning needs).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────
# Halting / pondering
# ────────────────────────────────────────────────────────────────────────

class PonderController(nn.Module):
    """Per-token halting probabilities (PonderNet).

    At each compute step k it emits a halting prob λ_t,k ∈ (0,1). The
    geometric distribution p_k = λ_k Π_{j<k}(1-λ_j) gives a proper
    distribution over compute depth (capped at K with remaining mass).

    Loss term `regularization_loss(probs, K, prior_mean=4.0)` keeps the
    expected number of steps near a target so the model doesn't trivially
    halt at step 0 or run forever.
    """

    def __init__(self, d_model: int, max_steps: int = 8):
        super().__init__()
        self.max_steps = max_steps
        self.halt = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T) halting logits in (0,1)."""
        return torch.sigmoid(self.halt(x).squeeze(-1))

    @staticmethod
    def regularization_loss(halt_probs: list[torch.Tensor],
                            target_mean_steps: float = 4.0) -> torch.Tensor:
        """KL between empirical halt distribution and a geometric prior
        with mean = target_mean_steps. Drives the controller toward the
        configured compute budget without forbidding adaptive variation.
        """
        if not halt_probs:
            return torch.zeros(())
        # Build distribution over steps
        K = len(halt_probs)
        not_halted = torch.ones_like(halt_probs[0])
        p_steps = []
        for k, lam in enumerate(halt_probs):
            p_k = lam * not_halted
            p_steps.append(p_k)
            not_halted = not_halted * (1.0 - lam)
        # Remaining mass at last step
        p_steps[-1] = p_steps[-1] + not_halted
        p = torch.stack(p_steps, dim=-1).clamp_min(1e-8)  # (B, T, K)

        # Geometric prior over k=1..K with mean target_mean_steps
        lam_prior = 1.0 / max(target_mean_steps, 1.0)
        ks = torch.arange(K, device=p.device, dtype=p.dtype)
        prior = lam_prior * (1.0 - lam_prior) ** ks
        prior = prior / prior.sum()
        # KL(p || prior)
        kl = (p * (p.log() - prior.log())).sum(dim=-1).mean()
        return kl


# ────────────────────────────────────────────────────────────────────────
# Adaptive-depth block
# ────────────────────────────────────────────────────────────────────────

class _SharedTransformerBlock(nn.Module):
    """A standard pre-norm transformer block with rotary-friendly attention.
    Plain MHA used for portability; the routing wrapper (`AdaptiveCompute
    Block`) runs this block multiple times with shared weights.
    """

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        d_ff = ff_mult * d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False,
                         is_causal=attn_mask is not None)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class AdaptiveComputeBlock(nn.Module):
    """Universal-Transformer-style recurrent block with PonderNet halting.

    Forward returns the weighted sum of intermediate states, plus the list
    of per-step halting probabilities (for the ponder regularizer).

    Critical detail: we do *not* halt asynchronously per token at runtime
    (would require ragged batching). Instead we run K steps and produce
    a halting-weighted residual. This is mathematically the PonderNet
    expected-output and trains end-to-end with one extra reg term.
    """

    def __init__(self, d_model: int, n_heads: int, max_steps: int = 6,
                 ff_mult: int = 4, dropout: float = 0.0,
                 target_mean_steps: float = 3.0):
        super().__init__()
        self.block = _SharedTransformerBlock(d_model, n_heads, ff_mult, dropout)
        self.controller = PonderController(d_model, max_steps=max_steps)
        self.max_steps = max_steps
        self.target_mean_steps = target_mean_steps

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, list[torch.Tensor], dict]:
        """Returns (y, halt_probs, info).
        y: (B, T, D) — halting-weighted output state.
        halt_probs: list of (B, T) halting probs at each step (for ponder loss).
        info: diagnostics (mean_steps, max_step_used).
        """
        not_halted = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
        accumulated = torch.zeros_like(x)
        halt_probs: list[torch.Tensor] = []
        h = x

        for k in range(self.max_steps):
            h = self.block(h, attn_mask=attn_mask)
            lam = self.controller(h)  # (B, T)
            if k == self.max_steps - 1:
                # Force remaining mass to halt
                p_k = not_halted
                lam = torch.ones_like(lam)
            else:
                p_k = lam * not_halted
            halt_probs.append(lam)
            accumulated = accumulated + p_k.unsqueeze(-1) * h
            not_halted = not_halted * (1.0 - lam)
            # Optional early-exit if everything has halted (saves compute)
            if (not_halted.max() < 1e-3) and self.training is False:
                break

        # Effective mean compute steps per token
        with torch.no_grad():
            steps = torch.zeros_like(halt_probs[0])
            nh = torch.ones_like(halt_probs[0])
            for k, lam in enumerate(halt_probs):
                steps = steps + (k + 1) * lam * nh
                nh = nh * (1.0 - lam)
            mean_steps = float(steps.mean().item())

        info = {"mean_steps": mean_steps, "max_steps_used": len(halt_probs)}
        return accumulated, halt_probs, info
