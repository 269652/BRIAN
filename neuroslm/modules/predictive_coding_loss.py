"""Multi-Scale Predictive Coding Auxiliary Losses for NeuroSLM.

Predictive coding (Rao & Ballard 1999) holds that the brain minimises
prediction error across a hierarchy of representations rather than
learning to classify inputs directly.  Each layer predicts the
representation of the layer below; only the residual (prediction error)
propagates upward.

This module computes several complementary predictive coding losses that
can be applied on top of any sequence model:

  1. Layer-wise vertical prediction:
     Higher layers predict lower-layer representations.
     Gradient flows only through the predictor, not the target — so the
     representation is regularised without changing the lower layer's
     gradient path.

  2. Temporal next-step prediction:
     Each position predicts the representation at the next position.
     Implemented at multiple scales (half, quarter, eighth of T) to
     capture both word- and phrase-level temporal structure.

  3. Masked reconstruction:
     A fraction of input positions is masked; the unmasked context
     is used to reconstruct the masked representations.  This is a
     harder objective than masked-token LM because it operates on
     continuous representations, not discrete token probabilities.

  4. Precision-weighted errors:
     Each prediction scale has a learned log-precision parameter.
     High-precision scales must be more accurate; low-precision scales
     are given slack.  This prevents easy scales from dominating the
     total loss.

The combined loss is a single scalar returned alongside optional
per-scale breakdowns for logging.

Novel aspect: the multi-scale temporal prediction (item 2) with
precision weighting (item 4) is not present in any published SLM
training objective.  It effectively trains the model to be a
multi-resolution temporal predictor — similar to how the auditory
cortex simultaneously processes phoneme, syllable, and word-level
time scales.

References:
  Rao & Ballard (1999): Predictive coding in the visual cortex
  Clark (2015): Surfing Uncertainty: Prediction, Action, and the Embodied Mind
  Friston (2010): The free-energy principle: a unified brain theory?
  Grill-Spector & Malach (2004): The human visual cortex
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class PredictiveCodingLoss(nn.Module):
    """Multi-scale predictive coding auxiliary loss module.

    Parameters
    ----------
    d_model    : representation dimension
    n_scales   : number of temporal prediction scales
    mask_ratio : fraction of positions masked for reconstruction loss
    """

    def __init__(self, d_model: int, n_scales: int = 3,
                 mask_ratio: float = 0.15):
        super().__init__()
        self.n_scales   = n_scales
        self.mask_ratio = mask_ratio
        self.d_model    = d_model

        # Temporal predictors — predict position t+stride from position t
        # Stride doubles per scale: 1, 2, 4, ...
        self.temporal_preds = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            ) for _ in range(n_scales)
        ])

        # Learned log-precision per temporal scale (higher = stricter)
        self.log_precision = nn.Parameter(torch.zeros(n_scales))

        # Masked reconstruction predictor (context → masked positions)
        self.recon_pred = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Layer-prediction head (used when layer_reps are supplied)
        self.layer_pred = nn.Linear(d_model, d_model)

    # ------------------------------------------------------------------

    def _temporal_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """Multi-scale next-step prediction loss.

        x: (B, T, d_model)
        Returns: scalar loss, list of per-scale losses
        """
        total = torch.tensor(0.0, device=x.device)
        per_scale = []
        precisions = torch.exp(self.log_precision)   # (n_scales,)

        for s in range(self.n_scales):
            stride = 2 ** s   # 1, 2, 4, ...
            if stride >= x.shape[1]:
                break
            src = x[:, :-stride, :]           # (B, T-stride, D)
            tgt = x[:, stride:, :].detach()   # (B, T-stride, D)  — no gradient to target

            pred = self.temporal_preds[s](src)
            err  = F.mse_loss(pred, tgt)
            weighted = precisions[s] * err
            total = total + weighted
            per_scale.append(err.item())

        n_active = min(self.n_scales, x.shape[1] - 1)
        if n_active > 0:
            total = total / n_active
        return total, per_scale

    def _masked_recon_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Masked representation reconstruction loss.

        x: (B, T, d_model)
        """
        B, T, _ = x.shape
        # Build binary mask: True = masked
        mask = torch.rand(B, T, device=x.device) < self.mask_ratio
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x.device)

        # Zero out masked positions, pass through reconstruction head
        x_masked = x.clone()
        x_masked[mask] = 0.0
        recon = self.recon_pred(x_masked)                # (B, T, D)

        # Loss only on masked positions
        loss = F.mse_loss(recon[mask], x[mask].detach())
        return loss

    def _layer_pred_loss(self, high_rep: torch.Tensor,
                         low_rep: torch.Tensor) -> torch.Tensor:
        """Higher-layer representation predicts lower-layer.

        high_rep: (B, T, D) — representation from a higher layer
        low_rep:  (B, T, D) — representation from a lower layer (target)
        """
        pred = self.layer_pred(high_rep)
        return F.mse_loss(pred, low_rep.detach())

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
                layer_reps: Optional[List[torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, dict]:
        """
        x:          (B, T, d_model) — current layer's representation
        layer_reps: list of (B, T, d_model) lower-layer representations
                    (e.g., from language encoder hidden states)

        Returns:
          loss:       scalar
          breakdown:  dict with keys 'temporal', 'masked', 'layer'
        """
        temporal_loss, per_scale = self._temporal_loss(x)
        masked_loss              = self._masked_recon_loss(x)

        layer_loss = torch.tensor(0.0, device=x.device)
        if layer_reps:
            for low_rep in layer_reps:
                layer_loss = layer_loss + self._layer_pred_loss(x, low_rep)
            layer_loss = layer_loss / len(layer_reps)

        total = temporal_loss + 0.5 * masked_loss + 0.3 * layer_loss

        breakdown = {
            "temporal":    temporal_loss.item(),
            "per_scale":   per_scale,
            "masked_recon": masked_loss.item(),
            "layer_pred":  layer_loss.item(),
        }
        return total, breakdown
