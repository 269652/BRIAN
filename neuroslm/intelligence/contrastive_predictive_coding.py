"""Contrastive Predictive Coding (CPC) for NeuroSLM.

Standard next-token prediction (LM loss) maps the last hidden state to a
target token distribution.  This is a *generative* objective: generate the
exact next token.

CPC (Oord et al. 2018) uses a *discriminative* objective instead:
  Given context z_t, predict whether z_{t+k} is the TRUE future or a
  RANDOM negative sample from the same batch.

  InfoNCE loss:
    L = -E[ log( exp(z_t · W_k · z_{t+k} / τ) /
                 Σ_j exp(z_t · W_k · z_j / τ) ) ]

Why this is stronger than MSE prediction:
  1. The model doesn't need to predict the *exact* future representation.
     It only needs to capture the information in z_t that is predictive of
     z_{t+k} — throwing away everything that isn't predictive (noise).
  2. The contrastive objective forces the representation to be *maximally
     informative* about the future: representations that look alike when
     they're genuinely predictable, different when they're not.
  3. Works across multiple future steps k=1,2,...,K simultaneously, giving
     multi-scale temporal structure.

Formally, CPC maximises the *mutual information* between context and future:
  I(z_t; z_{t+k}) >= log(N) - L_CPC
where N = number of negatives. With N=64 negatives, this provides a lower
bound on 4 bits of mutual information per prediction step.

This is equivalent to learning a world model, but with a signal that is:
  - More robust to representation scale (contrastive, not MSE)
  - More informative (mutual information bound, not next-token entropy)
  - Better at capturing multi-step dependencies

In NeuroSLM, CPC is applied to the GWS slot representations: predict future
GWS states from current ones. This is the cognitive equivalent of the
hippocampal "pre-play" and "forward sweep" phenomena.

References:
  Oord et al. (2018): Representation Learning with Contrastive Predictive Coding
  Gutmann & Hyvarinen (2010): Noise-Contrastive Estimation
  Poole et al. (2019): On Variational Bounds of Mutual Information
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class CPCPredictor(nn.Module):
    """CPC predictor for one future step k.

    Maps current representation to predicted future via a learned
    bilinear transformation W_k.
    """

    def __init__(self, d_model: int, d_pred: int):
        super().__init__()
        # Bilinear form: context_proj · future_proj^T  (more expressive than dot)
        self.context_proj = nn.Linear(d_model, d_pred, bias=False)
        self.future_proj  = nn.Linear(d_model, d_pred, bias=False)
        nn.init.orthogonal_(self.context_proj.weight)
        nn.init.orthogonal_(self.future_proj.weight)

    def score(self, context: torch.Tensor,
              future: torch.Tensor) -> torch.Tensor:
        """
        context: (B, d_model)  — current z_t
        future:  (B or N, d_model) — z_{t+k} or N negatives

        Returns: (B,) or (B, N) scores
        """
        c = self.context_proj(context)   # (B, d_pred)
        f = self.future_proj(future)     # (B or N, d_pred)
        return torch.einsum("bd,nd->bn", c, f) if f.dim() == 2 and f.size(0) != c.size(0) \
               else (c * f).sum(-1)


class ContrastivePredictiveCoding(nn.Module):
    """Multi-step CPC with InfoNCE loss.

    Parameters
    ----------
    d_model    : representation dimension
    d_pred     : bilinear predictor dimension (can be smaller than d_model)
    max_steps  : maximum prediction horizon (K future steps)
    n_negatives: negatives per positive sample (from same batch)
    temperature: InfoNCE temperature τ (lower = harder, sharper gradients)
    """

    def __init__(self, d_model: int,
                 d_pred: int = 128,
                 max_steps: int = 5,
                 n_negatives: int = 32,
                 temperature: float = 0.07):
        super().__init__()
        self.d_model     = d_model
        self.max_steps   = max_steps
        self.n_negatives = n_negatives
        self.temperature = temperature

        # One predictor per future step
        self.predictors = nn.ModuleList([
            CPCPredictor(d_model, d_pred)
            for _ in range(max_steps)
        ])

        # Context encoder (optional: slightly transforms context for prediction)
        self.context_enc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Learned per-step loss weights (farther = weaker signal)
        self.step_weights = nn.Parameter(
            torch.exp(-torch.arange(max_steps).float() * 0.3)
        )

    # ------------------------------------------------------------------

    def _infonce_loss(self, context: torch.Tensor,
                      positives: torch.Tensor,
                      predictor: CPCPredictor) -> torch.Tensor:
        """InfoNCE for one (context, positive, negatives) set.

        context:   (B, d)
        positives: (B, d)  — true future representations

        Negatives are drawn from the OTHER samples in the batch
        (common in CPC: the future of one sample is a negative for another).
        """
        B = context.shape[0]

        # Score positives: dot(context_proj_i, future_proj_i)
        pos_scores = predictor.score(context, positives)  # (B,) if same-sample

        # Score all pairs: context_i vs all positives_j
        # This gives (B, B) matrix where diagonal = positive scores
        c_proj = predictor.context_proj(context)    # (B, d_pred)
        f_proj = predictor.future_proj(positives)   # (B, d_pred)
        logits = torch.mm(c_proj, f_proj.t()) / self.temperature  # (B, B)

        # InfoNCE: for each i, class label = i (diagonal)
        labels = torch.arange(B, device=context.device)
        loss   = F.cross_entropy(logits, labels)

        # Additional metric: top-1 accuracy (for logging)
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean()

        return loss, acc

    # ------------------------------------------------------------------

    def forward(self, representations: torch.Tensor,
                compute_loss: bool = True
               ) -> Tuple[torch.Tensor, dict]:
        """
        representations: (B, T, d_model) — sequence of representations
                         e.g. GWS slots over time, language hidden states

        Returns:
          total_loss:  scalar — weighted InfoNCE across all steps
          metrics:     dict — per-step loss and accuracy for logging
        """
        B, T, D = representations.shape

        if not compute_loss:
            return torch.tensor(0.0, device=representations.device), {}

        # Encode context
        ctx = self.context_enc(representations)   # (B, T, D)

        total_loss = torch.tensor(0.0, device=representations.device)
        metrics    = {}
        weights    = F.softmax(self.step_weights[:min(self.max_steps, T - 1)], dim=0)

        n_steps = min(self.max_steps, T - 1)
        for k in range(n_steps):
            stride = k + 1
            if T - stride <= 0:
                break

            # Context from positions 0..T-stride-1
            context_seq = ctx[:, :T - stride, :]         # (B, T-k, D)
            # Future from positions stride..T-1
            future_seq  = representations[:, stride:, :] # (B, T-k, D)

            # Flatten: treat each (batch, time) position as an independent sample
            c_flat = context_seq.reshape(-1, D)   # (B*(T-k), D)
            f_flat = future_seq.reshape(-1, D)    # (B*(T-k), D)

            # Subsample if too many samples
            n_samples = min(c_flat.size(0), 256)
            if c_flat.size(0) > n_samples:
                idx = torch.randperm(c_flat.size(0), device=c_flat.device)[:n_samples]
                c_flat = c_flat[idx]
                f_flat = f_flat[idx]

            loss_k, acc_k = self._infonce_loss(c_flat, f_flat, self.predictors[k])
            total_loss = total_loss + weights[k] * loss_k
            metrics[f"cpc_loss_k{stride}"] = loss_k.item()
            metrics[f"cpc_acc_k{stride}"]  = acc_k.item()

        return total_loss, metrics
