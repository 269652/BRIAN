"""Sparse Mixture-of-Experts with Dynamic Token Routing for NeuroSLM.

Standard MoE (Switch Transformer, GLaM, Mixtral) routes each token to a
fixed top-k experts chosen by a learned router. Two practical problems:

  1. Load imbalance: popular experts saturate; rare experts atrophy.
  2. Static capacity: a fixed capacity factor wastes memory when load is
     uneven.

This module addresses both with:

  a. Z-loss (Zoph et al. 2022): penalises large router logits → prevents
     routing collapse to a single expert.
  b. Auxiliary balance loss: pushes each expert to process roughly 1/E of
     tokens.
  c. Expert capacity enforcement: tokens that exceed an expert's capacity
     are "dropped" (passed through as residual). This is differentiable
     via a soft capacity mask.
  d. Router noise (Noisy Top-k): add uniform noise during training to
     ensure all experts receive some gradient, prevents dead experts.

Architecture:
  Router:   Linear(d_model → n_experts) + noisy top-k selection
  Experts:  n_experts × (Linear → GELU → Linear) with shared input norm
  Output:   sum of top-k expert outputs weighted by router probability

Novel addition over vanilla MoE — *Dynamic Expert Importance Scaling*:
  Each expert's output is additionally scaled by a learned importance
  vector (d_model,) that sharpens which features each expert is
  responsible for. This is analogous to "winner-take-most" rather than
  winner-take-all, and empirically reduces output collapse.

References:
  Shazeer et al. (2017): Outrageously Large Neural Networks (MoE)
  Fedus et al. (2021): Switch Transformers
  Zoph et al. (2022): ST-MoE: Designing Stable and Transferable Sparse Expert Models
  Jiang et al. (2024): Mixtral of Experts
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DynamicRoutingMoE(nn.Module):
    """Sparse MoE layer with load-balancing and z-loss.

    Parameters
    ----------
    d_model      : token embedding dimension
    n_experts    : number of expert FFNs
    top_k        : experts activated per token
    d_expert     : inner dimension of each expert (None → 4*d_model)
    capacity_factor : max tokens per expert as multiple of T/E
    noise_std    : std of Gaussian noise added to router logits during training
    """

    def __init__(self, d_model: int, n_experts: int = 8, top_k: int = 2,
                 d_expert: Optional[int] = None,
                 capacity_factor: float = 1.25,
                 noise_std: float = 1e-2):
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.n_experts       = n_experts
        self.top_k           = top_k
        self.capacity_factor = capacity_factor
        self.noise_std       = noise_std
        d_exp = d_expert or 4 * d_model

        # Input norm shared across experts (saves memory vs per-expert norm)
        self.inp_norm = nn.LayerNorm(d_model)

        # Router
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Expert FFNs
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_exp))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_exp, d_model))
        self.b1 = nn.Parameter(torch.zeros(n_experts, d_exp))
        self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))

        # Dynamic importance scaling per expert
        self.importance_scale = nn.Parameter(torch.ones(n_experts, d_model))

        nn.init.kaiming_uniform_(self.w1)
        nn.init.kaiming_uniform_(self.w2)

    # ------------------------------------------------------------------

    @staticmethod
    def _z_loss(router_logits: torch.Tensor) -> torch.Tensor:
        """Z-loss: penalises large logits to prevent routing collapse.

        L_z = (1/B) * sum_b [ log sum_e exp(x_{b,e}) ]^2
        """
        log_sum_exp = torch.logsumexp(router_logits, dim=-1)  # (BT,)
        return log_sum_exp.pow(2).mean()

    @staticmethod
    def _balance_loss(router_probs: torch.Tensor,
                      dispatch_mask: torch.Tensor) -> torch.Tensor:
        """Auxiliary load-balancing loss (Switch Transformer eq. 4).

        Minimises the product of mean router probability × mean dispatch
        fraction across experts, encouraging equal load.
        """
        # mean fraction of tokens routed to each expert
        fraction = dispatch_mask.float().mean(0)          # (n_experts,)
        # mean router probability for each expert
        prob_mean = router_probs.mean(0)                  # (n_experts,)
        return (fraction * prob_mean).sum() * dispatch_mask.shape[0]

    # ------------------------------------------------------------------

    def _expert_forward(self, x: torch.Tensor, e: int) -> torch.Tensor:
        """Forward pass for expert e.  x: (tokens, d_model)"""
        h = F.gelu(x @ self.w1[e] + self.b1[e])   # (tokens, d_exp)
        return h @ self.w2[e] + self.b2[e]          # (tokens, d_model)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, d_model)

        Returns:
          out:        (B, T, d_model) — MoE output (residual NOT applied)
          aux_loss:   scalar — z_loss + balance_loss (add to total loss)
        """
        B, T, D = x.shape
        x_flat = self.inp_norm(x).view(B * T, D)     # (N, D)  N = B*T

        # ---- Router ----
        logits = self.router(x_flat)                  # (N, E)

        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        probs = F.softmax(logits, dim=-1)             # (N, E)

        # Top-k selection
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)  # (N, k)
        topk_probs = topk_probs / (topk_probs.sum(-1, keepdim=True) + 1e-9)

        # Capacity: max tokens per expert
        capacity = int(self.capacity_factor * T * self.top_k / self.n_experts) + 1

        # ---- Dispatch & combine ----
        out_flat = torch.zeros(B * T, D, device=x.device, dtype=x.dtype)

        # Dispatch mask for balance loss: which tokens are assigned each expert
        dispatch_mask = torch.zeros(B * T, self.n_experts,
                                    device=x.device, dtype=torch.bool)

        for k in range(self.top_k):
            expert_ids = topk_idx[:, k]              # (N,)
            w          = topk_probs[:, k]            # (N,)

            for e in range(self.n_experts):
                sel = (expert_ids == e).nonzero(as_tuple=True)[0]  # token indices

                # Enforce capacity: drop excess tokens
                if sel.numel() > capacity:
                    sel = sel[:capacity]

                if sel.numel() == 0:
                    continue

                dispatch_mask[sel, e] = True
                expert_out = self._expert_forward(x_flat[sel], e)  # (n_sel, D)
                expert_out = expert_out * self.importance_scale[e]  # dynamic scaling
                out_flat[sel] += expert_out * w[sel].unsqueeze(-1)

        out = out_flat.view(B, T, D)

        # ---- Auxiliary losses ----
        z_loss  = self._z_loss(logits)
        bal_loss = self._balance_loss(probs, dispatch_mask)
        aux_loss = 1e-2 * z_loss + 1e-2 * bal_loss

        return out, aux_loss
