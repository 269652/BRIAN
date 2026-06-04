# -*- coding: utf-8 -*-
r"""HPB Phase 4 — Hyperbolic Bowtie Waist (HBW).

A minimal, dependency-free implementation of the Poincaré-ball math
required to swap the Euclidean VBB posterior for a *wrapped Gaussian*
on :math:`\mathbb B^d_c`, the Poincaré ball of curvature :math:`-c`.

Only the operations the HBW actually needs are implemented:

* :func:`mobius_add` — Möbius gyrovector addition (the group law of
  the Poincaré ball).
* :func:`exp_map_zero`, :func:`log_map_zero` — exponential and
  logarithmic maps at the origin (we never need other base-points
  because the VBB is parameterised in the tangent space at ``o``).
* :func:`wrapped_normal_kl` — KL divergence between a wrapped Gaussian
  :math:`\mathcal W(\mu, \sigma^2 I)` and the unit-variance wrapped
  Normal at the origin :math:`\mathcal W(0, I)`. This is

  .. math::

      \mathrm{KL}\bigl[q \,\|\, \mathcal W(0,I)\bigr]
      = \underbrace{\tfrac12 \sum_d \bigl(\sigma_d^2 + \mu_d^2 - 1
                                          - \log \sigma_d^2\bigr)}_{
                                          \text{Euclidean part}}
      \;-\;
      \underbrace{(d-1)\,\log\!\frac{\sinh(\sqrt c\,\lVert\mu\rVert)}{
                                       \sqrt c\,\lVert\mu\rVert}}_{
                                       \text{Jacobian correction}}

  where the Jacobian correction comes from the change-of-variables of
  the exp-map at the origin (Nagano, Yamaguchi, Fujita & Koyama 2019,
  "A wrapped normal distribution on hyperbolic space for gradient-
  based learning", ICML 2019, eq. 11). Note we subtract because the
  ``log|det J|`` is added to the *entropy* of the wrapped Gaussian;
  in KL = E[log q − log p], the entropy is subtracted.

  The correction term :math:`\log(\sinh x / x)` is **non-negative** for
  :math:`x \ge 0` (sinh x ≥ x for x ≥ 0), but it appears with a
  **negative** sign in the original Nagano derivation because the
  wrapped Gaussian's log-density is bigger than the Euclidean one by
  this same factor; so log q − log p picks up a positive contribution.
  Our convention follows the wrapped-Gaussian KL of Skopek et al. 2020
  ("Mixed-curvature variational autoencoders", ICLR 2020, eq. 12),
  which is the form most LM-VAE papers cite.

  In words: the hyperbolic KL is **larger** than the Euclidean KL at
  any non-zero ``μ``, because the wrapped Gaussian has higher entropy
  in tangent-space coordinates than the unit Gaussian on the ball.
  This makes σ-collapse *harder* — which is exactly the MDRV objective.

Free-bits
---------
Per-dim free-bits clamping is applied to the **Euclidean** part of the
KL (per-dim ``½(σ²+μ²−1−logσ²)`` clamped to ``[δ, ∞)``), then the
scalar Jacobian correction is added. This composes cleanly with the
existing MDRV stabilisers.

Why the Poincaré ball
---------------------
Hyperbolic space has constant-radius volume growth
:math:`V(r) \propto e^{(d-1)\sqrt c\, r}` — exponentially many neighbours
within distance ``r``. This is the natural metric for trees, which are
the natural representation of language (syntactic, lexical, narrative
hierarchies). Sala et al. 2018 prove that an :math:`N`-leaf tree
embeds in :math:`\mathbb H^2` with distortion :math:`1+\epsilon`,
whereas Euclidean space needs :math:`\Theta(\log N)` dimensions for the
same distortion. The bowtie waist is the narrowest cross-section of
the network, so the metric mismatch matters most there.

Numerical hygiene
-----------------
We project all inputs into the ball boundary with a safety margin
``1 - eps`` to keep ``mobius_add`` and the conformal factor stable.
``c`` is parametrised as a *scalar* (homogeneous curvature) — making
``c`` per-dim or layer-wise is a Phase-6 refinement.
"""
from __future__ import annotations
from typing import Union

import math
import torch


_EPS = 1e-6


def _proj_to_ball(x: torch.Tensor, c: Union[torch.Tensor, float],
                  margin: float = 1.0 - 1e-3) -> torch.Tensor:
    """Project ``x`` into the open ball ``‖x‖ < margin / √c``.

    This guards :func:`mobius_add` against the boundary singularity
    where the conformal factor diverges.
    """
    c_t = torch.as_tensor(c, device=x.device, dtype=x.dtype)
    radius = margin / (c_t.sqrt() + _EPS)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    scale = torch.where(norm > radius, radius / norm,
                        torch.ones_like(norm))
    return x * scale


def mobius_add(x: torch.Tensor, y: torch.Tensor,
               c: Union[torch.Tensor, float]) -> torch.Tensor:
    r"""Möbius addition on the Poincaré ball of curvature :math:`-c`.

    .. math::

        x \oplus_c y = \frac{(1 + 2c\langle x,y\rangle
                              + c\,\lVert y\rVert^2)\,x
                            + (1 - c\,\lVert x\rVert^2)\,y}{
                            1 + 2c\langle x,y\rangle
                              + c^2\,\lVert x\rVert^2\,\lVert y\rVert^2}

    At ``c → 0`` this reduces to ``x + y`` (Euclidean addition).
    At the origin ``x = 0`` it returns ``y`` (identity).
    """
    c_t = torch.as_tensor(c, device=x.device, dtype=x.dtype)
    x = _proj_to_ball(x, c_t)
    y = _proj_to_ball(y, c_t)
    x2 = x.pow(2).sum(dim=-1, keepdim=True)         # (..., 1)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c_t * xy + c_t * y2) * x \
          + (1.0 - c_t * x2) * y
    den = 1.0 + 2.0 * c_t * xy + (c_t ** 2) * x2 * y2
    out = num / den.clamp_min(_EPS)
    return _proj_to_ball(out, c_t)


def exp_map_zero(v: torch.Tensor,
                 c: Union[torch.Tensor, float]) -> torch.Tensor:
    r"""Exponential map at the origin: tangent → ball.

    .. math::
        \exp^c_o(v) = \tanh(\sqrt c\,\lVert v\rVert)
                      \cdot \frac{v}{\sqrt c\,\lVert v\rVert}

    At ``c → 0`` this reduces to the identity.
    """
    c_t = torch.as_tensor(c, device=v.device, dtype=v.dtype)
    sqrt_c = c_t.sqrt().clamp_min(_EPS)
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    scale = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
    out = scale * v
    return _proj_to_ball(out, c_t)


def log_map_zero(x: torch.Tensor,
                 c: Union[torch.Tensor, float]) -> torch.Tensor:
    r"""Logarithmic map at the origin: ball → tangent.

    .. math::
        \log^c_o(x) = \mathrm{arctanh}(\sqrt c\,\lVert x\rVert)
                      \cdot \frac{x}{\sqrt c\,\lVert x\rVert}

    Inverse of :func:`exp_map_zero` for ``‖x‖ < 1/√c``.
    """
    c_t = torch.as_tensor(c, device=x.device, dtype=x.dtype)
    sqrt_c = c_t.sqrt().clamp_min(_EPS)
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    # arctanh saturates at ‖x‖ → 1/√c; clamp the argument to avoid inf.
    arg = (sqrt_c * x_norm).clamp(max=1.0 - 1e-5)
    scale = torch.atanh(arg) / (sqrt_c * x_norm)
    return scale * x


def wrapped_normal_kl(mu: torch.Tensor,
                      log_var: torch.Tensor,
                      c: Union[torch.Tensor, float],
                      free_bits: float = 0.0) -> torch.Tensor:
    r"""KL[ wrapped Normal(μ, σ²I) || wrapped Normal(0, I) ] on the
    Poincaré ball of curvature :math:`-c`.

    Closed-form:

    .. math::

        \mathrm{KL} = \underbrace{\tfrac12 \sum_d
                                  (\sigma_d^2 + \mu_d^2 - 1 - \log \sigma_d^2)}_{
                                  \text{Euclidean part (free-bits clamped)}}
                      \;+\;
                      \underbrace{(d-1)\,\log\!\frac{\sinh(\sqrt c\,\lVert\mu\rVert)}{
                                                    \sqrt c\,\lVert\mu\rVert}}_{
                                                    \text{wrapped Jacobian correction}}

    The Jacobian correction is :math:`\ge 0` because :math:`\sinh x \ge x`
    for :math:`x \ge 0`, so the hyperbolic KL strictly upper-bounds the
    Euclidean one for any non-zero ``μ`` — making σ-collapse strictly
    harder.

    Parameters
    ----------
    mu : (B, ..., D)
        Posterior mean (tangent-space at origin).
    log_var : (B, ..., D)
        Per-dimension posterior log-variance.
    c : scalar tensor or float
        Curvature parameter (> 0 for hyperbolic; → 0 ⇒ Euclidean).
    free_bits : float
        Per-dimension free-bits floor on the Euclidean KL. 0 ⇒ off.

    Returns
    -------
    torch.Tensor (scalar)
        Mean KL across batch / time / dim.
    """
    c_t = torch.as_tensor(c, device=mu.device, dtype=mu.dtype)
    # ── Euclidean part, free-bits clamped per-dim ─────────────────────
    kl_per_dim = 0.5 * (log_var.exp() + mu.pow(2) - 1.0 - log_var)
    if free_bits > 0.0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)
    kl_euclid = kl_per_dim.mean()

    # ── Jacobian correction ───────────────────────────────────────────
    # The correction is ZERO at c=0 and ZERO at μ=0 — both edge cases.
    # We use log(sinh x / x) = log(sinhc(x)) which is well-defined and
    # zero at x=0 (sinhc(0)=1).
    D = mu.shape[-1]
    if D <= 1:
        return kl_euclid
    sqrt_c = c_t.sqrt().clamp_min(_EPS)
    # ‖μ‖ over the last dim. Other dims (batch, time) are kept.
    mu_norm = mu.norm(dim=-1).clamp_min(_EPS)
    x = sqrt_c * mu_norm
    # log(sinh x / x) — use logarithm-of-sinhc that is numerically
    # stable. For small x: log(sinh x / x) ≈ x²/6.  For large x: ≈ x − log(2x).
    # We compute it directly with a safety branch.
    small = x.abs() < 1e-3
    log_sinhc_small = (x.pow(2) / 6.0) - (x.pow(4) / 180.0)
    # For larger x use the direct formula. Guard sinh(x) overflow at
    # x > ~700 (won't happen for c=1 and ‖μ‖ inside the ball).
    safe_x = x.clamp(max=80.0)
    log_sinhc_large = torch.log(torch.sinh(safe_x) / safe_x.clamp_min(_EPS))
    log_sinhc = torch.where(small, log_sinhc_small, log_sinhc_large)
    correction = (D - 1) * log_sinhc.mean()
    # As c → 0 the correction vanishes because x → 0 ⇒ log(sinhc(x)) → 0.

    return kl_euclid + correction
