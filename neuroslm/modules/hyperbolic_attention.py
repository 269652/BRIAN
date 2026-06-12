"""Poincaré-disc hyperbolic multi-head attention.

Implements the Ganea / Bécigneul / Hofmann formulation (NeurIPS 2018,
"Hyperbolic Neural Networks") on the Poincaré ball model with
configurable curvature ``c > 0``.

Geometric primitives (all expressed in PyTorch with autograd support):

* ``project_to_ball(x, c)`` — clip onto the open ball of radius
  ``1/sqrt(c)``.
* ``mobius_add(x, y, c)`` — Möbius vector addition (Ungar 2005).
* ``mobius_neg(x)`` — additive inverse: just ``-x``.
* ``poincare_distance(x, y, c)`` — closed-form hyperbolic distance.
* ``expmap0(v, c)`` / ``logmap0(x, c)`` — exp/log maps at the origin
  identifying the tangent space at 0 with ``R^n``.

The ``HyperbolicMultiHeadAttention`` module then:

1. linearly projects ``x`` to Euclidean Q, K, V;
2. maps Q and K into the ball via ``expmap0``;
3. computes scores as ``-d_hyp(Q, K) / sqrt(head_dim)``;
4. softmaxes over keys and returns ``softmax(scores) @ V`` (V stays
   Euclidean — standard Ganea practice, much cheaper and just as
   expressive in the LM setting).

Numerical stability is the main risk on the Poincaré ball; every
norm-bearing operation clamps appropriately and is exercised by the
contract tests in ``tests/test_hyperbolic_attention.py``.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Numerical safety margin: nothing in the ball is allowed to exceed
# ``1/sqrt(c) - BALL_EPS``. Tight enough that artanh stays well-defined,
# loose enough that gradients don't blow up.
BALL_EPS = 1e-5
# Floor for division-by-norm operations; tiny but nonzero so a
# zero vector has well-defined direction (= 0 after the multiplier).
NORM_EPS = 1e-15
# Clamp for artanh argument: keeps log((1+x)/(1-x)) finite.
ARTANH_CLAMP = 1.0 - 1e-7


# ──────────────────────────────────────────────────────────────────────
# Scalar helpers
# ──────────────────────────────────────────────────────────────────────


def _safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
    """L2 norm with a floor that keeps the autograd derivative finite.

    ``torch.norm`` has an ill-defined gradient at exactly 0; computing
    ``sqrt(sum_sq + eps)`` is the standard fix and is what every
    hyperbolic-NN library (geoopt, hyperlib, …) does.
    """
    sq = (x * x).sum(dim=dim, keepdim=keepdim)
    return torch.sqrt(sq.clamp_min(NORM_EPS))


def _artanh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic tangent with a clamp so it never sees ±1."""
    x = x.clamp(-ARTANH_CLAMP, ARTANH_CLAMP)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


# ──────────────────────────────────────────────────────────────────────
# Manifold operations
# ──────────────────────────────────────────────────────────────────────


def project_to_ball(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Project ``x`` onto the open Poincaré ball of radius ``1/sqrt(c)``.

    Vectors with norm below the boundary are returned unchanged; vectors
    on or outside the boundary are rescaled to lie at radius
    ``1/sqrt(c) - BALL_EPS``.
    """
    sqrt_c = math.sqrt(c)
    max_norm = (1.0 / sqrt_c) - BALL_EPS
    norm = _safe_norm(x)
    cond = norm > max_norm
    scaled = x / norm * max_norm
    return torch.where(cond, scaled, x)


def mobius_neg(x: torch.Tensor) -> torch.Tensor:
    """Additive inverse on the Poincaré ball: just ``-x``.

    (The Möbius group's inverse for ``x ⊕_c y`` w.r.t. y is ``-x ⊕_c ·``
    in general, but the additive inverse of ``x`` is ``-x``; together
    these give ``x ⊕_c (-x) = 0``.)
    """
    return -x


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Möbius vector addition (Ungar 2005, eq. 3.10).

    .. math::
        x \\oplus_c y =
        \\frac{(1 + 2c\\langle x,y\\rangle + c\\|y\\|^2)\\,x
               + (1 - c\\|x\\|^2)\\,y}
              {1 + 2c\\langle x,y\\rangle + c^2 \\|x\\|^2 \\|y\\|^2}

    Non-commutative in general but satisfies left identity / inverse,
    which is what attention scoring requires.
    """
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c**2) * x2 * y2
    # The denominator is provably > 0 for x, y in the open ball, but
    # numerical noise on the boundary can drift it to zero.
    return num / den.clamp_min(NORM_EPS)


def poincare_distance(
    x: torch.Tensor, y: torch.Tensor, c: float = 1.0
) -> torch.Tensor:
    """Geodesic distance between two points in the Poincaré ball.

    .. math::
        d_c(x, y) = \\frac{2}{\\sqrt{c}}\\, \\operatorname{artanh}
                    \\bigl(\\sqrt{c}\\,\\|-x \\oplus_c y\\|\\bigr)

    Symmetric, non-negative, zero iff ``x == y``. Returns a tensor with
    the last dim of x/y removed (standard distance broadcasting).
    """
    sqrt_c = math.sqrt(c)
    diff = mobius_add(mobius_neg(x), y, c=c)
    diff_norm = _safe_norm(diff, dim=-1, keepdim=False)
    return (2.0 / sqrt_c) * _artanh(sqrt_c * diff_norm)


def expmap0(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Exponential map at the origin: tangent space ``T_0 D^n_c`` → ball.

    .. math::
        \\exp_0^c(v) = \\tanh(\\sqrt{c}\\,\\|v\\|)\\, \\frac{v}{\\sqrt{c}\\,\\|v\\|}

    Maps the zero tangent vector to the origin (in the ball) and large
    tangent vectors arbitrarily close to (but never reaching) the
    boundary.
    """
    sqrt_c = math.sqrt(c)
    v_norm = _safe_norm(v)  # (..., 1)
    coeff = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
    out = coeff * v
    return project_to_ball(out, c=c)


def logmap0(y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Logarithm map at the origin: ball → tangent space ``T_0 D^n_c``.

    Inverse of :func:`expmap0`. Identifies the tangent space at 0 with
    ``R^n`` so the result is a plain Euclidean vector.
    """
    sqrt_c = math.sqrt(c)
    y_norm = _safe_norm(y)
    coeff = _artanh(sqrt_c * y_norm) / (sqrt_c * y_norm)
    return coeff * y


# ──────────────────────────────────────────────────────────────────────
# Multi-head attention with hyperbolic similarity
# ──────────────────────────────────────────────────────────────────────


class HyperbolicMultiHeadAttention(nn.Module):
    """Multi-head attention whose Q/K live on the Poincaré ball.

    Args:
        d_model: model embedding dimension.
        n_heads: number of attention heads (``d_model`` must be divisible
            by ``n_heads``).
        c: positive curvature of the Poincaré ball. ``c=1`` is the unit
            ball (canonical choice).
        dropout: attention-weight dropout probability.
        bias: whether the Q/K/V/out linear layers carry bias terms.
        return_weights: if ``True``, ``forward`` returns
            ``(output, attn_weights)`` instead of just ``output``. Useful
            for diagnostics + the contract tests.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        c: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
        return_weights: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        if c <= 0.0:
            raise ValueError(f"curvature c must be > 0, got {c}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.c = float(c)
        self.return_weights = bool(return_weights)

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Pre-computed scale: standard transformer practice but we use
        # hyperbolic distance instead of dot product, so the natural
        # scale is sqrt(head_dim) (same dimensionality argument).
        self._inv_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``(B, T, d_model)`` input.
            attn_mask: optional ``(T, T)`` or ``(B, 1, T, T)`` mask;
                positions where ``mask == True`` are excluded from
                softmax (i.e. set to ``-inf`` before softmax).

        Returns:
            ``(B, T, d_model)`` output tensor, or
            ``(output, attn_weights)`` if ``return_weights=True`` where
            ``attn_weights`` has shape ``(B, H, T_q, T_k)``.
        """
        B, T, _ = x.shape
        H, D_h = self.n_heads, self.head_dim

        # Euclidean projections, reshape to per-head
        q = self.q_proj(x).view(B, T, H, D_h).transpose(1, 2)  # (B, H, T, D_h)
        k = self.k_proj(x).view(B, T, H, D_h).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D_h).transpose(1, 2)

        # Map Q and K into the Poincaré ball
        q_ball = expmap0(q, c=self.c)
        k_ball = expmap0(k, c=self.c)

        # Pairwise hyperbolic distance: (B, H, T_q, T_k)
        # Broadcast: q_ball.unsqueeze(-2) is (B, H, T_q, 1, D_h);
        #            k_ball.unsqueeze(-3) is (B, H, 1, T_k, D_h).
        dist = poincare_distance(
            q_ball.unsqueeze(-2), k_ball.unsqueeze(-3), c=self.c
        )

        # Attention logits: closer in hyperbolic space → higher score.
        scores = -dist * self._inv_scale

        if attn_mask is not None:
            # True = masked out (don't attend). Broadcast to (B, H, T, T).
            scores = scores.masked_fill(attn_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # Euclidean weighted sum on V — standard Ganea practice.
        out = torch.matmul(weights, v)  # (B, H, T, D_h)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)

        if self.return_weights:
            return out, weights
        return out
