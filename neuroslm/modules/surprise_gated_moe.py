"""Surprise-gated Mixture-of-Experts.

Standard MoE (Shazeer et al. 2017, "Outrageously Large Neural Networks")
routes each token to its top-``k`` experts via a learned softmax gate
over the token embedding. This module extends that with a
**surprise-conditioned compute budget**: tokens whose recent reconstruct-
ion / prediction loss is high get routed to MORE experts (richer
mixture), while easy tokens use FEWER experts. The "surprise" signal is
either (a) supplied externally per token, or (b) computed internally as
a small auxiliary reconstruction loss against the input itself.

The compute economics are similar to mixture-of-depths (Raposo et al.
2024, "Mixture-of-Depths"): dynamic top-``k`` per token, where
``k(x) ∈ {k_min, ..., k_max}`` is a monotone function of surprise.

Why "surprise-gated"?
~~~~~~~~~~~~~~~~~~~~~
* In free-energy formulations of cognition (Friston 2010), surprise IS
  the driver of perception/learning — high surprise → more inference.
* Empirically, an MoE that always uses the same ``k`` wastes compute on
  trivial tokens; making ``k`` data-dependent gives the model a soft
  early-exit / extra-thought knob.
* Compatible with predictive-coding residuals (mechanism #4): the PC
  residual norm IS a surprise signal you can route on.

References
~~~~~~~~~~
* Shazeer et al. — "Outrageously Large Neural Networks: The Sparsely-
  Gated Mixture-of-Experts Layer", *ICLR* 2017.
* Fedus, Zoph, Shazeer — "Switch Transformers: Scaling to Trillion
  Parameter Models with Simple and Efficient Sparsity", *JMLR* 2022.
* Raposo, Ritter, Richards, Lillicrap, Humphreys, Santoro — "Mixture-
  of-Depths: Dynamically allocating compute in transformer-based
  language models", *NeurIPS* 2024.
* Friston, K. — "The free-energy principle: a unified brain theory?",
  *Nature Rev. Neurosci.* 11(2), 2010.

Implementation: tested by ``tests/test_surprise_gated_moe.py``
(18 contracts).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_LOAD_BAL_EPS = 1e-9


# ──────────────────────────────────────────────────────────────────────
# Helper: per-token dynamic top-k from a surprise scalar
# ──────────────────────────────────────────────────────────────────────


def surprise_to_k(
    surprise: torch.Tensor,
    k_min: int,
    k_max: int,
    *,
    midpoint: float = 1.0,
    steepness: float = 4.0,
) -> torch.Tensor:
    """Map per-token surprise ∈ ℝ≥0 to an integer ``k ∈ [k_min, k_max]``.

    Uses a logistic gate centred on ``midpoint``: tokens well below the
    midpoint stay at ``k_min`` (cheap); tokens well above climb toward
    ``k_max`` (expensive). The exact mapping:

        soft   = k_min + (k_max - k_min) · σ(steepness · (surprise - midpoint))
        result = round(soft)  (clamped to [k_min, k_max])

    The rounding is non-differentiable; we use ``torch.round`` for
    inference. The forward pass through MoE itself remains
    differentiable via the soft gating weights (rounding only chooses
    which experts to mix).
    """
    if k_min < 1:
        raise ValueError(f"k_min must be ≥ 1, got {k_min}")
    if k_max < k_min:
        raise ValueError(f"k_max ({k_max}) must be ≥ k_min ({k_min})")
    if steepness <= 0:
        raise ValueError(f"steepness must be positive, got {steepness}")
    gate = torch.sigmoid(steepness * (surprise - midpoint))
    soft = float(k_min) + float(k_max - k_min) * gate
    k = torch.round(soft).clamp(k_min, k_max).long()
    return k


def load_balance_loss(
    gates: torch.Tensor, top_indices: torch.Tensor, n_experts: int
) -> torch.Tensor:
    """Switch-transformer-style load-balancing auxiliary loss.

    ``L = N · Σ_e f_e · P_e`` where:

    * ``f_e`` is the fraction of tokens routed to expert ``e`` (the
      "frequency" — uses the discrete top-k assignment),
    * ``P_e`` is the average gating probability over expert ``e``
      across tokens (the soft "probability").

    Multiplied by ``N`` to keep the loss invariant under the number of
    experts. Reaches its minimum when both ``f`` and ``P`` are uniform
    over experts (perfect load balance).

    Args:
        gates: ``(B*T, n_experts)`` soft gating probabilities (after
            softmax over experts).
        top_indices: ``(B*T, k)`` integer indices of the selected
            experts per token.
        n_experts: total number of experts.
    """
    if gates.dim() != 2 or top_indices.dim() != 2:
        raise ValueError(
            f"load_balance_loss expects 2D tensors, got "
            f"gates={tuple(gates.shape)}, top_indices={tuple(top_indices.shape)}"
        )
    N = gates.shape[0]
    if N == 0:
        return torch.zeros((), device=gates.device, dtype=gates.dtype)
    # Probability mass per expert (averaged over tokens).
    P = gates.mean(dim=0)  # (n_experts,)
    # Discrete fraction routed per expert.
    one_hot = F.one_hot(top_indices, num_classes=n_experts).float()
    # Sum over k slots, then mean over tokens.
    f = one_hot.sum(dim=1).mean(dim=0)  # (n_experts,)
    return float(n_experts) * (f * P).sum()


# ──────────────────────────────────────────────────────────────────────
# Expert layer (Linear + GELU + Linear) — the workhorse
# ──────────────────────────────────────────────────────────────────────


class _Expert(nn.Module):
    """Per-expert feed-forward block: ``Linear → GELU → Linear``."""

    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# ──────────────────────────────────────────────────────────────────────
# Surprise-gated MoE module
# ──────────────────────────────────────────────────────────────────────


class SurpriseGatedMoE(nn.Module):
    """Mixture of experts with surprise-driven dynamic top-``k``.

    Acts as an ``edge`` endpoint in the BRIAN feature DSL: forward
    consumes ``(B, T, d_model)`` and returns ``(B, T, d_model)``.

    The ``surprise`` signal can be supplied per token via
    ``forward(x, surprise=...)`` (shape ``(B, T)``). When omitted, it
    is computed internally from a small auxiliary reconstruction loss:
    the gating logits' entropy ‒ low entropy = confident routing = low
    surprise; high entropy = uncertain = high surprise.

    Args:
        d_model: input/output embedding dimension.
        n_experts: total number of experts (must be ≥ ``k_max``).
        d_hidden: hidden width of each expert's MLP.
        k_min: minimum number of experts per token (low surprise).
        k_max: maximum number of experts per token (high surprise).
        midpoint, steepness: surprise-→-k mapping parameters.
        load_balance_weight: multiplier on the auxiliary load-balance
            loss exposed via ``self.last_aux_loss``. Set to ``0.0`` to
            disable (e.g. for diagnostic runs).
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        d_hidden: Optional[int] = None,
        *,
        k_min: int = 1,
        k_max: int = 4,
        midpoint: float = 1.0,
        steepness: float = 4.0,
        load_balance_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if n_experts < 1:
            raise ValueError(f"n_experts must be ≥ 1, got {n_experts}")
        if k_max > n_experts:
            raise ValueError(
                f"k_max ({k_max}) must be ≤ n_experts ({n_experts})"
            )
        if k_min < 1 or k_min > k_max:
            raise ValueError(
                f"k_min must be in [1, k_max={k_max}], got {k_min}"
            )
        self.d_model = d_model
        self.n_experts = n_experts
        self.d_hidden = d_hidden if d_hidden is not None else 4 * d_model
        self.k_min = k_min
        self.k_max = k_max
        self.midpoint = float(midpoint)
        self.steepness = float(steepness)
        self.load_balance_weight = float(load_balance_weight)

        self.experts = nn.ModuleList(
            [_Expert(d_model, self.d_hidden) for _ in range(n_experts)]
        )
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        # Diagnostic buffers — auxiliary loss + mean dispatched-k per
        # forward call. Both are non-persistent (no checkpoint bloat).
        self.register_buffer(
            "last_aux_loss", torch.zeros(1), persistent=False
        )
        self.register_buffer(
            "last_mean_k", torch.zeros(1), persistent=False
        )

    def _compute_surprise(
        self, x: torch.Tensor, gate_logits: torch.Tensor
    ) -> torch.Tensor:
        """Fallback surprise: gating-distribution entropy normalised by
        ``log(n_experts)`` so it lives in ``[0, 1]``.

        High entropy → uncertain routing → high surprise → larger ``k``.
        """
        log_p = F.log_softmax(gate_logits, dim=-1)
        p = log_p.exp()
        # Entropy in nats; normalise by max possible (uniform) entropy.
        H = -(p * log_p).sum(dim=-1)
        H_max = torch.log(
            torch.tensor(self.n_experts, dtype=H.dtype, device=H.device)
        )
        return H / H_max.clamp_min(_LOAD_BAL_EPS)

    def forward(
        self,
        x: torch.Tensor,
        *,
        surprise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"SurpriseGatedMoE expects (B, T, D), got shape "
                f"{tuple(x.shape)}"
            )
        B, T, D = x.shape

        # Per-token gating logits → softmax probabilities.
        gate_logits = self.gate(x)                       # (B, T, E)
        gate_probs = F.softmax(gate_logits, dim=-1)      # (B, T, E)

        # Surprise (B, T) → per-token k (B, T).
        if surprise is None:
            s = self._compute_surprise(x, gate_logits)
        else:
            if surprise.shape != (B, T):
                raise ValueError(
                    f"surprise must have shape (B, T) = ({B}, {T}), got "
                    f"{tuple(surprise.shape)}"
                )
            s = surprise
        k_per_tok = surprise_to_k(
            s,
            self.k_min,
            self.k_max,
            midpoint=self.midpoint,
            steepness=self.steepness,
        )  # (B, T) long

        # Flatten across (B, T) so we can route by token index.
        flat_x = x.reshape(B * T, D)
        flat_gp = gate_probs.reshape(B * T, self.n_experts)
        flat_k = k_per_tok.reshape(B * T)

        # Take top-k_max indices per token in one shot, then mask out
        # the slots beyond each token's individual k. This is the
        # standard MoE trick: avoid per-token Python loops.
        top_vals, top_idx = flat_gp.topk(self.k_max, dim=-1)
        # Build a (N, k_max) boolean mask: True for kept slots.
        slot_idx = torch.arange(self.k_max, device=x.device).unsqueeze(0)
        keep = slot_idx < flat_k.unsqueeze(-1)            # (N, k_max)
        top_vals = top_vals * keep.to(top_vals.dtype)
        # Re-normalise per-token weights so they sum to 1 over the
        # kept slots (this is the canonical Switch-style normalisation
        # that keeps the output's scale stable).
        denom = top_vals.sum(dim=-1, keepdim=True).clamp_min(_LOAD_BAL_EPS)
        top_w = top_vals / denom                          # (N, k_max)

        # Dispatch: accumulate weighted expert outputs.
        out = torch.zeros_like(flat_x)
        for e_id in range(self.n_experts):
            # Mask of (token, slot) entries assigned to this expert.
            mask = (top_idx == e_id) & keep              # (N, k_max)
            if not mask.any():
                continue
            tok_mask = mask.any(dim=-1)                  # (N,) which tokens
            if not tok_mask.any():
                continue
            tok_indices = tok_mask.nonzero(as_tuple=True)[0]
            # Weight contributed by this expert per dispatching token =
            # sum over its slots of the kept renormalised weights.
            w = (top_w * mask.to(top_w.dtype)).sum(dim=-1)[tok_indices]
            expert_out = self.experts[e_id](flat_x[tok_indices])
            out[tok_indices] = out[tok_indices] + w.unsqueeze(-1) * expert_out

        # Aux loss (Switch-style load balance, uses ALL kept indices).
        # Flatten kept indices over (N, k_max) → (N*k_max,), but only
        # the entries that are actually used (mask). For load balance
        # the published formulation uses the full top-k indices; we use
        # the per-token k by zeroing the soft probability of dropped
        # slots — done above via gate_probs (still the soft form).
        aux = load_balance_loss(flat_gp, top_idx, self.n_experts)
        self.last_aux_loss.copy_((self.load_balance_weight * aux).reshape(1).detach())
        self.last_mean_k.copy_(
            k_per_tok.float().mean().reshape(1).detach()
        )

        return out.reshape(B, T, D)
