"""Predictive-Dropout — information-aware regularization driven by PCT errors.

Standard nn.Dropout is information-blind: it picks a channel to zero out
uniformly at random. Half the time it drops a channel that carries
genuine signal; half the time it drops one that was redundant anyway.

Predictive-dropout uses the Predictive Coding Trunk's per-channel
prediction error `e = h_n − g_n(h_{n+1})` as a per-channel
information-quality signal:

  • Low |e_c|² → channel c is well-predicted from the layer above →
    it carries little NEW information → safe to drop.
  • High |e_c|² → channel c is genuinely surprising → must keep.

This is the information-bottleneck idea (Tishby 2000) made concrete by
the free-energy decomposition of PCT: features the network can already
predict generatively are exactly the features it doesn't need to attend
to in the forward pass.

Theoretical link
----------------
In a properly-trained PCT layer, the per-channel precision-weighted
prediction error is an estimator of the channel's contribution to the
free-energy. Channels with low contribution carry low-entropy /
low-surprise / high-redundancy content (the three are equivalent under
Gaussian assumptions). Standard regularization theory says the
generalization gap is bounded by the mutual information I(X; T)
between input and representation; dropping low-entropy channels lowers
I(X; T) without losing predictive content. The bound therefore
tightens — which is precisely the OOD generalization claim we're
testing on synthesis-v1.

References
----------
Tishby, Pereira & Bialek (2000) — Information Bottleneck.
Gal & Ghahramani (2016) — A Theoretically Grounded Application of Dropout
    in Recurrent Neural Networks (concrete dropout).
Strouse et al. (2017) — Drop-Bottleneck.
Salvatori et al. (2023, NeurIPS) — PC networks find flatter minima
    (related: the channels predictive-dropout targets ARE the channels
    that flatten in a converged PC net).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class PredictiveDropout(nn.Module):
    """Channel-wise dropout whose per-channel keep probability is computed
    from the PCT prediction error.

    Parameters
    ----------
    base_keep : minimum keep probability for the LEAST-informative channel
                (default 0.5 — even predictable channels stay 50% of the
                time, so the model doesn't catastrophically lose any
                channel it might learn to use later).
    beta      : sigmoid sharpness for the info_score → keep_prob mapping.
                Higher = sharper transition between drop-frequently and
                keep-always. Default 4.0.
    per_token : if True, sample the mask independently per token (B*T*D
                random draws); if False, sample per-batch (one mask of
                shape (1, 1, D) tiled across the whole batch). Per-batch
                is cheaper and the original concrete-dropout convention.

    forward(h, error)
    -----------------
    h     : (B, T, D) hidden state
    error : (B, T, D) PCT prediction error tensor, DETACHED — gradients
            do NOT flow back through the mask into the error path
            (otherwise the predictor could shape the mask, which would
            create a perverse incentive to make errors small everywhere
            so the dropout is uniform).
    Returns the masked hidden state, same shape, with inverse-keep
    scaling so the expected magnitude is preserved (standard dropout
    convention).
    """

    def __init__(self, base_keep: float = 0.5, beta: float = 4.0,
                 per_token: bool = False):
        super().__init__()
        if not (0.0 < base_keep <= 1.0):
            raise ValueError(f"base_keep must be in (0, 1], got {base_keep}")
        self.base_keep = float(base_keep)
        self.beta = float(beta)
        self.per_token = bool(per_token)

    def _info_score(self, error: torch.Tensor) -> torch.Tensor:
        """Per-channel info score in [0, 1]. Channels with above-mean
        per-channel error magnitude get high score (keep more); below-
        mean ones get low score (drop more)."""
        # |e|² per channel, averaged over batch+positions → (D,)
        err_mag = error.detach().pow(2).mean(dim=(0, 1))
        # Normalize so the score is a relative ranking, not an absolute
        # magnitude — this is key for robustness across layers with
        # different error scales. Use the median as the pivot (more
        # robust than mean to outliers).
        pivot = err_mag.median()
        scale = err_mag.std().clamp(min=1e-6)
        return torch.sigmoid(self.beta * (err_mag - pivot) / scale)  # (D,)

    def forward(self, h: torch.Tensor,
                error: torch.Tensor) -> torch.Tensor:
        # Inference: no dropout (standard dropout convention).
        if not self.training:
            return h
        if error.shape != h.shape:
            # In rare cases (e.g. last layer with no top-down predictor),
            # caller may pass an empty/different-shaped error; act as
            # a no-op to be safe.
            return h

        info = self._info_score(error)                       # (D,)
        keep_prob = self.base_keep + (1.0 - self.base_keep) * info   # (D,)
        # Numerical floor / ceiling
        keep_prob = keep_prob.clamp(min=0.01, max=1.0)

        if self.per_token:
            mask = (torch.rand_like(h) < keep_prob.view(1, 1, -1)).to(h.dtype)
        else:
            # One mask per batch (B, 1, D), broadcast over T. Cheaper.
            B = h.size(0)
            sample = torch.rand(B, 1, h.size(-1),
                                device=h.device, dtype=h.dtype)
            mask = (sample < keep_prob.view(1, 1, -1)).to(h.dtype)

        # Inverse-keep scaling preserves expected magnitude.
        return h * mask / keep_prob.view(1, 1, -1).clamp(min=1e-3)

    def extra_repr(self) -> str:
        return (f"base_keep={self.base_keep}, beta={self.beta}, "
                f"per_token={self.per_token}")
