"""Predictive Coding Trunk (PCT) — top-down generative trunk update rule.

This module changes the trunk's *gradient flow mechanics* from
shortcut-friendly residual addition to free-energy minimization with
top-down generative predictors. It is the architectural mechanism behind
the proposal in arch/predictive-coding-trunk.

Distinction from existing PredictiveCodingHead
----------------------------------------------
The codebase already has `PredictiveCodingHead` (neuro_attention.py) and
`PredictiveCodingLoss` (predictive_coding_loss.py). Both run BOTTOM-UP:
layer n predicts layer n+1, providing deep supervision. That gives every
layer a local gradient but does NOT change information flow direction or
generalization properties.

PCT runs TOP-DOWN: layer n+1 must predict layer n.

This single direction flip is the crux. The deeper layer is forced to be
a generative inverse of the shallower layer's representation. Properties:

  • Causal/invariant features — noise and surface statistics cannot be
    predicted from semantic context, so they get pushed to a single noise
    channel that doesn't propagate.
  • Free-energy = upper bound on negative log model evidence (Friston
    2005-2010). Minimizing it is a generalization objective, not a
    training-set objective.
  • Sparse coding by construction — well-predicted features carry no
    error signal, so no representational capacity is "spent" on them.
  • Flatter loss landscapes (Salvatori et al. 2023, NeurIPS) — empirically
    equivalent to SAM without SAM's 2× backward cost.

References
----------
Rao & Ballard (1999), Nature Neuroscience — Predictive coding in V1.
Friston (2010), Nature Reviews Neuroscience — Free-energy principle.
Whittington & Bogacz (2017), Neural Computation — PC approximates BP.
Millidge, Tschantz, Buckley (2020), arXiv:2006.04182 — PC on arbitrary
    computation graphs (transformers included).
Salvatori et al. (2023), NeurIPS — Reverse differentiation via PC,
    flatter minima.
Song et al. (2024), ICLR — Prospective configuration, invariance.

Implementation notes
--------------------
Two PCT variants are provided, gated by `cfg.pct_mode`:

  "loss_only"   — forward path is unchanged; only the free-energy loss is
                  added. Cheapest (~+10 % FLOPs), zero changes to module
                  injections. Recommended starting point.

  "feedback"    — the previous layer's prediction error is projected back
                  into the next layer's forward update. Closer to true
                  iterated PC, ~+30 % FLOPs.

Both modes use top-down predictors and the same free-energy loss; only
the forward coupling differs.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import RMSNorm


class TopDownPredictor(nn.Module):
    """Predict layer n's representation from layer n+1's representation.

    g_n : h_{n+1} -> h_n_pred

    The predictor is a small 2-layer MLP with a learned scale. It is
    initialized so that g_n(h) ≈ h at step 0, which means the initial
    free-energy is dominated by the genuine inter-layer difference (the
    "natural surprise" of one block of computation) rather than by random
    predictor noise — the predictor then learns to absorb that natural
    structure, leaving only the un-generalizable residue.
    """

    def __init__(self, dim: int, hidden_mult: float = 0.5,
                 init_identity: bool = True):
        super().__init__()
        hidden = max(8, int(dim * hidden_mult))
        self.norm = RMSNorm(dim)
        self.fc1  = nn.Linear(dim, hidden, bias=False)
        self.fc2  = nn.Linear(hidden, dim, bias=False)
        # Per-channel learnable log-precision (higher = stricter target).
        # Initialized to zero (precision = 1) so all channels weighted equally
        # at start; the model can then learn which channels are predictable.
        self.log_precision = nn.Parameter(torch.zeros(dim))

        if init_identity:
            # Start as ~identity: g_n(h) ≈ h. fc2 zero-init means the MLP
            # contributes zero at step 0; the skip path (added in forward)
            # provides the identity.
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.fc2.weight)
        else:
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)

    def forward(self, h_upper: torch.Tensor) -> torch.Tensor:
        """h_upper: (B, T, dim) — representation from layer n+1.
        Returns prediction of layer n's representation, same shape."""
        z = self.norm(h_upper.float()).to(h_upper.dtype)
        z = self.fc2(F.silu(self.fc1(z)))
        # Identity skip + MLP correction (zero-init MLP means start = identity)
        return h_upper + z

    def precision_weights(self) -> torch.Tensor:
        """Channel-wise positive precision (B-broadcastable)."""
        # softplus to keep precision positive but unbounded above
        return F.softplus(self.log_precision) + 1e-3


def free_energy_loss(layer_states: list[torch.Tensor],
                     predictors: nn.ModuleList,
                     ignore_index_mask: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, dict]:
    """Compute layer-wise top-down free-energy across the trunk.

    Parameters
    ----------
    layer_states : list of (B, T, D) tensors, length L+1, where layer_states[n]
                   is the representation BEFORE block n (so layer_states[0]
                   is the embedding output and layer_states[-1] is the final
                   trunk output).
    predictors   : nn.ModuleList of length L of TopDownPredictor, where
                   predictors[n] predicts layer_states[n] from layer_states[n+1].
    ignore_index_mask : optional (B, T) bool tensor — positions to exclude
                   from the loss (e.g. pad tokens). True = include.

    Returns
    -------
    L_FE       : scalar free-energy loss, mean over (active positions × layers).
    breakdown  : dict with per-layer scalar errors (detached, for logging).
    """
    assert len(layer_states) == len(predictors) + 1, (
        f"PCT shape mismatch: {len(layer_states)} states vs "
        f"{len(predictors)} predictors")

    total = torch.zeros((), device=layer_states[0].device,
                        dtype=layer_states[0].dtype)
    per_layer = []

    for n, g in enumerate(predictors):
        h_lower = layer_states[n]       # target (will be DETACHED — only the
                                        # PREDICTOR trains on this, not the
                                        # lower layer; lower-layer pressure
                                        # comes via the block n forward path
                                        # of the LM loss).
        h_upper = layer_states[n + 1]   # input to predictor (gradients flow
                                        # back through the predictor into
                                        # h_upper -> shapes deeper rep to be
                                        # a generative model of shallower one)
        p_n = g(h_upper)                # (B, T, D) top-down prediction
        e_n = h_lower.detach() - p_n    # (B, T, D) prediction error

        # Channel-wise precision weighting (learned)
        prec = g.precision_weights().to(e_n.dtype)              # (D,)
        weighted = (e_n.pow(2) * prec.view(1, 1, -1))           # (B, T, D)

        if ignore_index_mask is not None:
            m = ignore_index_mask.unsqueeze(-1).to(weighted.dtype)
            denom = (m.sum() * weighted.size(-1)).clamp(min=1.0)
            layer_loss = (weighted * m).sum() / denom
        else:
            layer_loss = weighted.mean()

        total = total + layer_loss
        per_layer.append(float(layer_loss.detach().item()))

    # Mean across layers so the magnitude is comparable across model depths
    total = total / max(1, len(predictors))

    breakdown = {
        "fe_per_layer": per_layer,
        "fe_mean":      float(total.detach().item()),
    }
    return total, breakdown


class PredictiveCodingTrunk(nn.Module):
    """Container for the top-down predictors used by LanguageCortex.

    Holds one TopDownPredictor per adjacent layer pair plus a single
    forward-feedback projection (used when `mode == "feedback"`).

    Parameters
    ----------
    dim          : trunk hidden dim (d_hidden)
    n_layers     : number of trunk blocks (predictors = n_layers - 1 by
                   default; +1 if `include_embedding_predictor=True` to
                   also predict the embedding from the first block output).
    mode         : "loss_only" or "feedback"
    feedback_alpha : scaling for the feedforward error injection (only used
                   when mode == "feedback"). 0.0 disables.
    hidden_mult  : predictor hidden-dim multiplier (0.5 = small)
    include_embedding_predictor : add an extra predictor for h_0 (embedding)
                   from h_1 (first block out). Adds slight FLOPs but gives
                   embeddings direct generative pressure.
    """

    VALID_MODES = ("loss_only", "feedback")

    def __init__(self, dim: int, n_layers: int,
                 mode: str = "loss_only",
                 feedback_alpha: float = 0.05,
                 hidden_mult: float = 0.5,
                 include_embedding_predictor: bool = True):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"PCT mode must be in {self.VALID_MODES}, got {mode}")
        self.mode = mode
        self.feedback_alpha = float(feedback_alpha)
        self.dim = int(dim)
        self.n_layers = int(n_layers)

        # One predictor per adjacent layer pair. With include_embedding_predictor,
        # predictors[0] predicts h_0 from h_1; predictors[1] predicts h_1 from
        # h_2; ...; predictors[n_layers-1] predicts h_{n_layers-1} from h_{n_layers}.
        # Without it, predictors[0] predicts h_1 from h_2; ...
        n_preds = n_layers if include_embedding_predictor else max(0, n_layers - 1)
        self.include_embedding_predictor = bool(include_embedding_predictor)
        self.predictors = nn.ModuleList([
            TopDownPredictor(dim, hidden_mult=hidden_mult, init_identity=True)
            for _ in range(n_preds)
        ])

        # Feedforward error projection: maps the prediction error e_{n-1} at
        # the previous layer back into a small additive correction for the
        # NEXT block's input. Zero-init so the model behaves identically to
        # standard residual at step 0 (no discontinuity at training start).
        if mode == "feedback":
            self.error_proj = nn.Linear(dim, dim, bias=False)
            nn.init.zeros_(self.error_proj.weight)
        else:
            self.error_proj = None

    def make_error_for_layer(self, h_lower: torch.Tensor,
                             h_upper: torch.Tensor,
                             pred_idx: int) -> torch.Tensor:
        """Compute prediction error e_n = h_lower - g_n(h_upper).

        Used in `feedback` mode to inject the error into the next block.
        Returns the error tensor; does NOT detach h_lower (the caller decides).
        """
        if pred_idx < 0 or pred_idx >= len(self.predictors):
            return torch.zeros_like(h_lower)
        p = self.predictors[pred_idx](h_upper)
        return h_lower - p

    def feedforward_correction(self, error: torch.Tensor) -> torch.Tensor:
        """Project the previous-layer error into the next block's input.
        Returns zero-shaped tensor when error_proj is disabled."""
        if self.error_proj is None or self.feedback_alpha == 0.0:
            return torch.zeros_like(error)
        return self.feedback_alpha * self.error_proj(error)

    def compute_loss(self, layer_states: list[torch.Tensor],
                     ignore_index_mask: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, dict]:
        """Top-down free-energy loss across all collected layer states.

        `layer_states` must align with self.predictors:
          if include_embedding_predictor:
              layer_states = [h_0 (embedding), h_1, ..., h_{n_layers}]
              len = n_layers + 1
          else:
              layer_states = [h_1, h_2, ..., h_{n_layers}]
              len = n_layers
        """
        if len(self.predictors) == 0:
            zero = torch.zeros((), device=layer_states[0].device,
                               dtype=layer_states[0].dtype)
            return zero, {"fe_per_layer": [], "fe_mean": 0.0}
        return free_energy_loss(layer_states, self.predictors,
                                ignore_index_mask=ignore_index_mask)
