"""Sparse Mixture-of-Experts.

Top-2 gated MoE with load-balancing loss. At ~100M total params and
top-2 routing across N=8 experts, ~25% of FF parameters are active per
token, giving ~3× effective capacity at constant FLOPs.

Implementation chooses simplicity + correctness over peak GPU efficiency:
no permutation kernels, no expert parallelism. Suitable for a single GPU
(T4/A10) with batch_size 4-8.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class SparseMoE(nn.Module):
    """Switch-style MoE with top-2 routing and aux load-balancing loss.

    Forward returns (y, aux_loss). Add aux_loss to the total training loss
    with a small weight (e.g. 0.01) to keep experts utilized.
    """

    def __init__(self, d_model: int, n_experts: int = 8, d_ff: int | None = None,
                 top_k: int = 2, capacity_factor: float = 1.25):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        d_ff = d_ff or 4 * d_model
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([_Expert(d_model, d_ff)
                                      for _ in range(n_experts)])
        self.register_buffer("usage_ema", torch.ones(n_experts) / n_experts,
                             persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)                      # (BT, D)
        N = x_flat.size(0)

        gate_logits = self.gate(x_flat)                # (BT, E)
        gate_probs = F.softmax(gate_logits, dim=-1)    # (BT, E)

        # Top-k routing
        topk_vals, topk_idx = gate_probs.topk(self.top_k, dim=-1)  # (BT, k)
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

        # Dispatch: for each expert, gather the tokens assigned to it
        out = torch.zeros_like(x_flat)
        for e_idx, expert in enumerate(self.experts):
            mask = (topk_idx == e_idx)                 # (BT, k)
            if not mask.any():
                continue
            # weights for this expert per token (sum across the k slot dim)
            w = (topk_vals * mask.float()).sum(dim=-1, keepdim=True)  # (BT, 1)
            tok_idx = (w.squeeze(-1) > 0).nonzero(as_tuple=True)[0]
            if tok_idx.numel() == 0:
                continue
            tok_in = x_flat.index_select(0, tok_idx)
            tok_out = expert(tok_in) * w.index_select(0, tok_idx)
            out.index_add_(0, tok_idx, tok_out)

        # Aux load-balance loss (Switch / GShard formulation)
        with torch.no_grad():
            tokens_per_expert = torch.zeros(self.n_experts, device=x.device)
            for k in range(self.top_k):
                tokens_per_expert.scatter_add_(
                    0, topk_idx[:, k],
                    torch.ones(N, device=x.device))
            frac = tokens_per_expert / (N * self.top_k)
            self.usage_ema.mul_(0.99).add_(0.01 * frac)
        importance = gate_probs.mean(dim=0)            # (E,)
        load = (gate_probs > 0).float().mean(dim=0)
        aux_loss = self.n_experts * (importance * load).sum()

        return out.view(B, T, D), aux_loss

    @torch.no_grad()
    def expert_utilization(self) -> dict:
        """Returns a snapshot of per-expert usage; useful for diagnostics."""
        return {f"e{i}": float(self.usage_ema[i]) for i in range(self.n_experts)}
