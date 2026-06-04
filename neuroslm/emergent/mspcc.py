# -*- coding: utf-8 -*-
"""HPB Phase 3 — Multi-Scale Predictive Coding Cascade (MSPCC).

A generalisation of the single-waist VBB free-energy at the bowtie
into a per-layer cascade. For each adjacent layer pair (ℓ, ℓ+1)
in the trunk, contribute one MDRV-VBB term:

.. math::

    r_\\ell    &= \\big\\lVert h_{\\ell+1} - W_\\ell\\,\\hat h_\\ell\\big\\rVert^2 \\\\
    \\text{KL}_\\ell &= \\tfrac12 \\sum_d (\\sigma_d^2 + \\mu_d^2 - 1 - \\log \\sigma_d^2) \\\\
    \\text{PEC}_\\ell &= -\\eta \\cdot \\tfrac12 \\, \\mathbb{E}[\\log \\sigma^2] \\\\
    \\mathcal{L}_\\ell &= \\beta_\\ell \\cdot r_\\ell - \\log \\beta_\\ell
                          + \\alpha\\,\\text{KL}_\\ell + \\text{PEC}_\\ell \\\\
    \\mathcal{L}_{\\text{MSPCC}} &= \\sum_{\\ell=0}^{L-2} \\lambda_\\ell \\, \\mathcal{L}_\\ell

with the **layer weights** geometrically decaying away from the deepest
pair (ℓ = L-2 → L-1):

.. math::
    \\lambda_\\ell = \\lambda_0 \\cdot \\text{decay}^{(L-1)-\\ell}

so that the bowtie waist (deepest) dominates the cascade and the
shallowest pair contributes ``λ_0 · decay^{L-1}``.

MDRV stabilisers
----------------
Free-bits, β-ceiling and PEC are applied **per layer pair** using the
same scalars as the single-waist VBB (``vbb_alpha``, ``vbb_free_bits``,
``vbb_log_beta_max``, ``vbb_entropy_eta``). No new hyperparameters.

The β scalar is per-pair and learnable (one parameter per pair), but
this helper computes a closed-form ``β_ℓ = 1 / (2·detached_r_ℓ)``
as a Lagrange-multiplier surrogate when no learnable params are
provided — avoiding the need to plumb β params into a stateless math
helper. The harness owns the proper ``log_beta`` parameters and
threads them in when wiring this up against the live trunk.

Reference
---------
* Sønderby et al. 2016 — Ladder VAE: hierarchical IB per latent.
* Whittington & Bogacz 2017 — predictive coding ≡ backprop.
* Felleman & Van Essen 1991 — cortical hierarchy narrows with depth.
"""
from __future__ import annotations
from typing import List, Optional

import math
import torch
import torch.nn.functional as F


def mspcc_layer_weights(num_layers: int,
                        base_weight: float,
                        layer_weight_decay: float) -> List[float]:
    """Return ``[λ_0, λ_1, …, λ_{L-1}]`` per the schedule above.

    Note we return ``L`` weights even though the cascade has ``L-1``
    pairs — callers index by the *predictor* layer ``ℓ`` (i.e. the
    lower index of each pair). The trailing weight is unused.
    """
    if num_layers <= 0:
        return []
    weights: List[float] = []
    for ell in range(num_layers):
        # Distance from the deepest pair: ℓ=L-1 (deepest) → power 0.
        power = (num_layers - 1) - ell
        weights.append(base_weight * (layer_weight_decay ** power))
    return weights


def _vbb_layer_term(h_low: torch.Tensor,
                    h_high: torch.Tensor,
                    alpha: float,
                    free_bits: float,
                    log_beta_max: float,
                    entropy_eta: float) -> torch.Tensor:
    """Compute one layer-pair's free-energy term.

    No learned predictor W here — we use the identity predictor
    ``ŝ = μ``. This is the cheapest non-trivial choice and matches the
    Whittington–Bogacz "predictive coding ≡ backprop with identity
    feedback" limit. The harness can later promote individual layer
    pairs to a learned W_ℓ; that's a Phase-6 refinement.

    The encoder is a per-element diagonal Gaussian whose log-variance
    is parameter-free: we use ``log σ²_d = log Var(h_low_d)`` from the
    batch itself. This is a closed-form posterior at the level of a
    Bayesian Empirical-Bayes diagonal Gaussian (Casella 1985) — it
    pulls log σ² up when the layer's activations are noisy and down
    when they're confident. No new optimised parameters.
    """
    # μ = h_low; sample-Var across batch+time is the diagonal Gaussian
    # log-variance. detach() so the gradient does not flow into σ via
    # the variance estimate — the σ path is a side channel.
    mu = h_low
    with torch.no_grad():
        var = mu.var(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-6)
        log_var = var.log().expand_as(mu).clamp(-12.0, 4.0)
    sigma = (0.5 * log_var).exp()

    # Reparameterised sample.
    eps = torch.randn_like(mu)
    mu_sample = mu + sigma * eps

    # Identity predictor: ŝ = μ_sample. Residual against h_high.
    residual = (h_high - mu_sample).pow(2).mean()

    # β: learnable scalar would live in the harness; here we use the
    # closed-form Lagrange equilibrium β* = 1 / (2·detached_r) with
    # a softplus-bounded floor. ALSO clamp via log_beta_max for MDRV.
    with torch.no_grad():
        r_det = residual.detach().clamp_min(1e-6)
        beta_star = (1.0 / (2.0 * r_det)).clamp_max(math.exp(log_beta_max)
                                                    if log_beta_max > 0 else 1e6)
    beta = beta_star + 0.0   # tracks but no grad — keeps β term in the
                             # objective without a learnable parameter.
    log_beta = torch.log(beta + 1e-6)

    # KL per dim. The harness VBB applies clamp(min=free_bits) per-dim.
    kl_per_dim = 0.5 * (log_var.exp() + mu.pow(2) - 1.0 - log_var)
    if free_bits > 0.0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)
    kl = kl_per_dim.mean()

    # PEC: -η · ½ E[log σ²]. As σ→0, log σ² → -∞ ⇒ PEC → +∞.
    pec = 0.0
    if entropy_eta > 0.0:
        pec = -entropy_eta * 0.5 * log_var.mean()

    # Free energy.
    return beta * residual - log_beta + alpha * kl + pec


def compute_mspcc_loss(layer_outputs: List[torch.Tensor],
                       base_weight: float,
                       layer_weight_decay: float,
                       alpha: float,
                       free_bits: float,
                       log_beta_max: float,
                       entropy_eta: float
                       ) -> Optional[torch.Tensor]:
    """Multi-Scale Predictive Coding Cascade loss.

    Parameters
    ----------
    layer_outputs
        List of ``L`` tensors of shape ``(B, T, D_ℓ)`` from the trunk.
        For shape mismatches between ``D_ℓ`` and ``D_{ℓ+1}`` we project
        the lower layer up via a fixed (mean-pooled) projection. In
        practice the DSLLanguageCortex uses constant D across layers
        so this is a no-op.
    base_weight, layer_weight_decay
        Geometric schedule parameters; see :func:`mspcc_layer_weights`.
    alpha, free_bits, log_beta_max, entropy_eta
        VBB / MDRV stabiliser hyperparameters, shared across layers.

    Returns
    -------
    torch.Tensor (scalar) or ``None``
        ``None`` when there are no adjacent pairs (L ≤ 1).
    """
    L = len(layer_outputs)
    if L < 2:
        return None
    weights = mspcc_layer_weights(L, base_weight, layer_weight_decay)
    # Sum over adjacent pairs (ℓ, ℓ+1); the lower-layer index ℓ is the
    # predictor, so we weight by weights[ℓ]. The deepest pair (ℓ=L-2)
    # gets weights[L-2] which equals base_weight · decay^1 = decay×λ_0;
    # NOTE: the deepest **predictor** is ℓ=L-2, so its λ equals
    # base_weight × decay^1. The very-top layer (ℓ=L-1) is a target
    # only and its weight (weights[L-1] = base_weight) is unused by
    # the cascade. The geometric schedule is computed accordingly.
    total = None
    device = layer_outputs[0].device
    dtype = layer_outputs[0].dtype
    for ell in range(L - 1):
        h_low = layer_outputs[ell]
        h_high = layer_outputs[ell + 1]
        # Bring high-layer to low-layer's dim if they differ.
        if h_low.shape[-1] != h_high.shape[-1]:
            # Fixed (non-parametric) average-pool to align dims.
            target_d = h_low.shape[-1]
            high_d = h_high.shape[-1]
            if high_d > target_d:
                # downsample average-pool over the last dim
                pool = high_d // target_d
                if pool * target_d == high_d:
                    h_high_aligned = h_high.reshape(*h_high.shape[:-1],
                                                    target_d, pool).mean(-1)
                else:
                    h_high_aligned = h_high[..., :target_d]
            else:
                # upsample by zero-pad
                pad = target_d - high_d
                h_high_aligned = F.pad(h_high, (0, pad))
        else:
            h_high_aligned = h_high
        term = _vbb_layer_term(h_low, h_high_aligned, alpha,
                               free_bits, log_beta_max, entropy_eta)
        w = float(weights[ell])
        # Convert to tensor on the right device for the multiplication.
        w_t = torch.as_tensor(w, device=device, dtype=dtype)
        contribution = w_t * term
        total = contribution if total is None else total + contribution
    return total
