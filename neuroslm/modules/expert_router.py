"""Expert-Choice Routing (Zhou et al. 2022) for SRC-TEH.

Each expert "pulls" the top-C tokens with the highest affinity, instead of
each token picking its expert. Properties:

  • No dropped tokens at the *expert* level — every expert gets exactly C
    tokens per batch (or fewer if its affinity column is degenerate).
  • Naturally load-balanced — no aux loss required.
  • Tokens may be processed by 0, 1, or multiple experts in a single pass;
    untouched tokens flow through the residual path unchanged.
  • XLA-safe: capacity C is a Python int, no dynamic shapes; only `topk`
    is needed.

Used by `brain.forward_lm` AFTER the shared trunk to dispatch hidden states
to {LanguageExpert, MathExpert, ReasoningExpert} for deeper extraction.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertChoiceRouter(nn.Module):
    """Top-C expert-choice router over `n_experts` experts.

    Args:
        d_hidden:   Per-token hidden dimension.
        n_experts:  Number of experts (typically 3: Lang / Math / Reason).
        capacity_factor: Per-expert capacity = ceil(T * capacity_factor / n_experts).
            With cf=1.5 every expert pulls roughly 50% of the tokens it would
            see under uniform routing — gives overlap (a token can be picked
            by several experts) while keeping FLOPs bounded.
        temperature_init: Initial routing softmax temperature.  Annealed to
            1.0 by the maturity scheduler outside.
    """

    def __init__(self, d_hidden: int, n_experts: int = 3,
                 capacity_factor: float = 1.5,
                 temperature_init: float = 1.0):
        super().__init__()
        self.d_hidden        = d_hidden
        self.n_experts       = int(n_experts)
        self.capacity_factor = float(capacity_factor)
        # Routing scorer: linear projection → per-expert affinity.
        # Zero-init bias, small-normal weights so each expert starts roughly
        # equal — the router has no prior over which tokens to pull.
        self.scorer = nn.Linear(d_hidden, n_experts, bias=True)
        nn.init.normal_(self.scorer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.scorer.bias)
        # Learnable inverse-temperature; clamped via softplus + 0.5 to avoid
        # collapse to delta or uniform. Starts near 1.0.
        self.log_temp = nn.Parameter(torch.tensor(math.log(temperature_init)))

    def capacity(self, T: int) -> int:
        """Per-expert capacity for sequence length T (Python int, XLA-safe)."""
        return max(1, int(math.ceil(T * self.capacity_factor / max(1, self.n_experts))))

    def forward(self, h: torch.Tensor,
                maturity: float | None = None):
        """h: (B, T, d_hidden).

        Returns dict with:
          • `assignments`: list of n_experts (B, C) long tensors — token
            indices each expert pulls (per batch item).
          • `weights`: list of n_experts (B, C) float tensors — soft gate
            weights = sigmoid(score) applied when the expert writes back.
          • `aux_loss`: small entropy-collapse penalty (kept tiny; routing
            balance is intrinsic to expert-choice, this just discourages
            degenerate uniform).
        """
        B, T, D = h.shape
        E = self.n_experts
        C = self.capacity(T)

        # Per-token, per-expert affinity logits.
        temp   = F.softplus(self.log_temp) + 0.5      # > 0.5
        logits = self.scorer(h) / temp                 # (B, T, E)

        # Maturity-aware exploration: blend in uniform when MAT is low so
        # cold-start gradients reach every expert.
        if maturity is not None and maturity < 0.5:
            blend = max(0.0, (0.5 - float(maturity)) / 0.5)   # 0..1
            uniform = torch.zeros_like(logits)
            logits = (1.0 - blend) * logits + blend * uniform

        # Score the OPPOSITE way: each expert ranks all tokens.
        # Reshape to (B, E, T) — for each expert, pick top-C tokens.
        scores = logits.transpose(1, 2)                # (B, E, T)
        topv, topi = scores.topk(C, dim=-1)            # (B, E, C)
        gate = torch.sigmoid(topv)                     # (B, E, C) — write gate

        assignments = [topi[:, e, :].contiguous() for e in range(E)]
        weights     = [gate[:, e, :].contiguous()  for e in range(E)]

        # Auxiliary penalty: prevent the router from going degenerate
        # (all experts pull identical tokens). Encourage column-rank diversity
        # of the assignment indicator matrix.
        with torch.enable_grad():
            # Soft one-hot of which expert each token *would* prefer
            soft = F.softmax(scores, dim=1)            # (B, E, T)
            mean_load = soft.mean(dim=-1)              # (B, E)
            aux_loss = ((mean_load - 1.0 / E) ** 2).mean()
        return {
            "assignments": assignments,
            "weights":     weights,
            "aux_loss":    aux_loss,
            "logits":      logits,
            "capacity":    C,
        }


def gather_tokens(h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather (B, T, D) along T using (B, C) indices → (B, C, D)."""
    B, T, D = h.shape
    C = idx.size(1)
    idx_exp = idx.unsqueeze(-1).expand(B, C, D)
    return torch.gather(h, dim=1, index=idx_exp)


def scatter_add_tokens(out: torch.Tensor, idx: torch.Tensor,
                       contrib: torch.Tensor) -> torch.Tensor:
    """Scatter-add (B, C, D) `contrib` back into (B, T, D) `out` at idx.

    Tokens that are picked by multiple experts accumulate additively (the
    "residual stream" picture). Tokens picked by zero experts pass through
    `out` unchanged.
    """
    B, T, D = out.shape
    C = idx.size(1)
    idx_exp = idx.unsqueeze(-1).expand(B, C, D)
    return out.scatter_add(dim=1, index=idx_exp, src=contrib)


class LanguageExpert(nn.Module):
    """3-block transformer expert specialised for stylistic / discourse
    refinement of routed tokens.

    Mirrors MathCortex.forward_tokens / ReasoningCortex.forward_tokens but
    without the symbolic-memory or attractor cross-attention — pure
    transformer + zero-init output projection so it starts as identity.
    """

    def __init__(self, d_hidden: int, n_blocks: int = 3,
                 n_heads: int = 8, max_ctx: int = 2048):
        super().__init__()
        from .common import TransformerBlock, RMSNorm
        nh = max(1, n_heads)
        while d_hidden % nh != 0 and nh > 1:
            nh -= 1
        self.blocks = nn.ModuleList([
            TransformerBlock(d_hidden, n_heads=nh, max_ctx=max_ctx)
            for _ in range(n_blocks)
        ])
        self.norm = RMSNorm(d_hidden)
        self.out  = nn.Linear(d_hidden, d_hidden, bias=False)
        nn.init.zeros_(self.out.weight)               # identity at init

    def forward_tokens(self, x: torch.Tensor,
                       maturity: float | None = None) -> torch.Tensor:
        h = x
        for blk in self.blocks:
            h = blk(h)
        h = self.out(self.norm(h.float()).to(h.dtype))
        m_eff = 1.0 if maturity is None else max(float(maturity), 0.05)
        return x + m_eff * h


__all__ = ["ExpertChoiceRouter", "LanguageExpert",
           "gather_tokens", "scatter_add_tokens"]
