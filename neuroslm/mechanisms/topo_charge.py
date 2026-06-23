# -*- coding: utf-8 -*-
"""Pontryagin / Hopfion-lite topological-charge diagnostic.

Theory.
    A trained attention head's per-token output, projected onto S^2,
    is a discrete map from token-position to the unit sphere.
    Hand-wavily, the head is a function n_h : T -> S^2.

    A continuous map T^2 -> S^2 has a Pontryagin (skyrmion) charge
    Q = (1 / 4*pi) * integral of n . (d_x n  cross  d_y n) dx dy
    which classifies homotopy classes in pi_2(S^2) = Z. The discrete
    analogue on a triangulated surface (Berg-Lueschner 1981) sums
    signed spherical-triangle solid angles:

        Q = (1 / 4*pi) * sum_triangles  Omega(n_a, n_b, n_c)

    where Omega is given by van Oosterom-Strang (1983):

        tan(Omega / 2) =
            (n_a . (n_b x n_c)) / (1 + n_a.n_b + n_b.n_c + n_c.n_a)
        Omega = 2 * atan2(numerator, denominator).

    The atan2 form is numerically stable across the entire sphere
    (the acos-based form blows up near the antipodal boundary).

    Hopfion-lite (a.k.a. inter-layer decorrelation):
        eps_ortho = sum_{l=0}^{L-2}  mean( 1 - n_{l+1} . n_l )
    is the cheap inter-layer orientation-decorrelation diagnostic
    that survives the verifier descope from the full Hopf-integral
    (which would require FFT-Poisson on a 3-torus -- out of scope).

Scope.
    This module ONLY exposes diagnostics + an optional soft penalty.
    It does not modify the trunk's forward pass; the harness drains
    it after the LM step. Per CLAUDE.md sec 14 the penalty is
    structurally zero only when both alpha=0 AND gamma=0, NOT via an
    early-return / dead-branch.

References.
    Berg, B. & Lueschner, M. (1981). Definition and statistical
        distributions of a topological number in the lattice O(3)
        sigma-model. Nuclear Physics B 190 (3): 412-424.
    van Oosterom, A. & Strang, J. (1983). The solid angle of a plane
        triangle. IEEE Trans. Biomed. Eng. 30 (2): 125-126.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Geometry helpers
# ============================================================================


def solid_angle(
    n_a: torch.Tensor,
    n_b: torch.Tensor,
    n_c: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Signed spherical-triangle solid angle via van Oosterom-Strang.

    Args:
        n_a, n_b, n_c: broadcastable tensors with last dim 3; rows
            should lie on the unit sphere (not enforced -- caller is
            expected to F.normalize).
        eps: small constant guarding the atan2 denominator from
            becoming pathologically small. atan2 itself is well-
            defined at (0, 0), but a tiny atan2(num, eps) for huge
            num still rounds to +- pi correctly; this clamp just
            avoids the (0, 0) degenerate case where the sign is
            indeterminate.

    Returns:
        Tensor with shape = broadcast(n_a, n_b, n_c)[:-1]. The result
        is in radians and signed by triangle orientation (right-hand
        rule on the sphere).
    """
    # n_b x n_c
    cross_bc = torch.cross(n_b, n_c, dim=-1)
    numerator = (n_a * cross_bc).sum(dim=-1)
    denominator = (
        1.0
        + (n_a * n_b).sum(dim=-1)
        + (n_b * n_c).sum(dim=-1)
        + (n_c * n_a).sum(dim=-1)
    )
    # atan2 returns in (-pi, pi]; the formula gives Omega/2, so the
    # full range is (-2*pi, 2*pi] -- correct for the signed solid
    # angle of a non-degenerate spherical triangle.
    # Guard the (num=0, denom=0) degenerate case by clamping denom
    # magnitude in that corner only.
    safe_denom = torch.where(
        (numerator.abs() < eps) & (denominator.abs() < eps),
        torch.full_like(denominator, 1.0),
        denominator,
    )
    half_omega = torch.atan2(numerator, safe_denom)
    return 2.0 * half_omega


def berg_luscher_q(n: torch.Tensor) -> torch.Tensor:
    """Per-triangle signed winding contribution along the T axis.

    Args:
        n: Tensor[..., T, 3]. The last axis must be on the unit
           sphere (caller normalises).

    Returns:
        Tensor[..., (T - 2)]. Entry [..., t] is
            solid_angle(n[..., t,   :],
                        n[..., t+1, :],
                        n[..., t+2, :]) / (4 * pi).
        The signed sum over the last axis is the discrete winding
        Q_h of the (T -> S^2) sliding-triangle map.

        For T < 3 the output has zero elements along the last axis
        (no IndexError; autograd still flows).
    """
    T = n.shape[-2]
    if T < 3:
        # Zero-triangle case. Build an output tensor with the right
        # leading dims and the same dtype/device so callers can sum
        # and backward without special-casing.
        out_shape = n.shape[:-2] + (0,)
        return n.new_zeros(out_shape, requires_grad=n.requires_grad)
    n_a = n[..., :-2, :]
    n_b = n[..., 1:-1, :]
    n_c = n[..., 2:, :]
    omega = solid_angle(n_a, n_b, n_c)
    return omega / (4.0 * math.pi)


def hopfion_eps_ortho(n_per_layer: List[torch.Tensor]) -> torch.Tensor:
    """Sum over consecutive layer pairs of mean( 1 - n_{l+1} . n_l ).

    Args:
        n_per_layer: list of unit-sphere tensors [..., 3], one per
            layer. All shapes must match.

    Returns:
        Scalar tensor. Zero when L <= 1 (no pairs) or when every
        consecutive pair is aligned; 2 * (L - 1) when alternation
        is antipodal everywhere.
    """
    if len(n_per_layer) <= 1:
        # Build a scalar zero with the right dtype / device that
        # retains grad if the input tensors do.
        ref = n_per_layer[0] if n_per_layer else None
        if ref is None:
            return torch.zeros(())
        return (ref.sum() * 0.0)
    total = None
    for prev, curr in zip(n_per_layer[:-1], n_per_layer[1:]):
        dot = (prev * curr).sum(dim=-1)          # (..., T)
        pair = (1.0 - dot).mean()
        total = pair if total is None else total + pair
    return total


# ============================================================================
# TopoChargeDiagnostic module
# ============================================================================


class TopoChargeDiagnostic(nn.Module):
    """Project per-layer attention outputs to S^2 and accumulate Q_h
    and eps_ortho diagnostics.

    Construction.
        Linear(head_dim, 3) with weight init zeros and bias init
        [1, 0, 0]. The deterministic bias guarantees the projection
        produces a unit vector even at step 0 with no training
        signal (closes review FIX 8: zero-bias + zero-weight ->
        F.normalize(0) = 0 breaks the unit-norm invariant).

    Forward.
        Takes a list of per-layer attention outputs of shape
        (B, H, T, head_dim). Returns a dict:
            { "Q_h":       Tensor(B, H),  -- summed over layers + tris
              "eps_ortho": Tensor()  }    -- scalar.

    Penalty.
        penalty(Q_target, alpha, gamma) returns
            alpha * mean((Q_h - Q_target)**2) + gamma * eps_ortho.
        Default mode is alpha = gamma = 0 (DIAGNOSTIC ONLY). Per
        CLAUDE.md sec 14 the zero-contribution path is structural,
        not an early return; the multiplications are real.
    """

    def __init__(
        self,
        head_dim: int,
        bias_init: Optional[torch.Tensor] = None,
        weight_init_std: float = 0.02,
        normalize_eps: float = 1e-8,
    ):
        super().__init__()
        self.head_dim = int(head_dim)
        self.normalize_eps = float(normalize_eps)
        self.proj = nn.Linear(self.head_dim, 3, bias=True)
        # Small-random weight + a fixed unit-vector bias. The bias
        # guarantees that even on zero input the projection produces
        # a non-zero pre-normalize vector (closes review FIX 8:
        # zero-weight + zero-bias would give F.normalize(0) = 0,
        # breaking the unit-norm invariant). The non-zero weight
        # ensures the projection actually depends on the attention
        # output -- a zero-weight init would make Q_h independent of
        # the input (decorative, §14-banned).
        with torch.no_grad():
            nn.init.normal_(self.proj.weight, mean=0.0, std=weight_init_std)
            if bias_init is None:
                bias_init = torch.tensor([1.0, 0.0, 0.0])
            self.proj.bias.copy_(bias_init.to(self.proj.bias.dtype))
        # Cached forward outputs for pop_metrics() and penalty().
        # Stored as plain tensors (not Parameters / buffers) so they
        # don't enter state_dict.
        self._last_Q_h: Optional[torch.Tensor] = None
        self._last_eps_ortho: Optional[torch.Tensor] = None

    # -- internal --------------------------------------------------------

    def _project_to_S2(self, attn_out: torch.Tensor) -> torch.Tensor:
        """Project (..., head_dim) to S^2 via (..., 3) linear + L2."""
        v = self.proj(attn_out)
        return F.normalize(v, dim=-1, eps=self.normalize_eps)

    # -- public ----------------------------------------------------------

    def forward(
        self, attn_per_layer: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute Q_h (per-head winding) and eps_ortho (inter-layer
        decorrelation) from a list of per-layer attention outputs.

        Each entry must have shape (B, H, T, head_dim) with the SAME
        (B, H, T, head_dim) across layers.
        """
        if not attn_per_layer:
            raise ValueError(
                "TopoChargeDiagnostic.forward needs at least one "
                "per-layer attention output"
            )
        # Project every layer; collect normalised sphere coordinates.
        n_per_layer = [self._project_to_S2(a) for a in attn_per_layer]
        # Per-layer per-head winding contribution; sum over layers
        # AND over the sliding-window triangles, giving a (B, H)
        # diagnostic.
        Q_h = None
        for n in n_per_layer:
            per_tri = berg_luscher_q(n)         # (B, H, T-2) or (B, H, 0)
            layer_Q = per_tri.sum(dim=-1)       # (B, H)
            Q_h = layer_Q if Q_h is None else Q_h + layer_Q
        # Inter-layer decorrelation.
        eps_ortho = hopfion_eps_ortho(n_per_layer)
        self._last_Q_h = Q_h
        self._last_eps_ortho = eps_ortho
        return {"Q_h": Q_h, "eps_ortho": eps_ortho}

    def pop_metrics(self) -> Dict[str, torch.Tensor]:
        """Return the most recent forward's diagnostics. Does NOT
        clear the cache -- repeated calls return the same tensor."""
        if self._last_Q_h is None or self._last_eps_ortho is None:
            raise RuntimeError(
                "TopoChargeDiagnostic.pop_metrics called before any "
                "forward pass"
            )
        return {
            "Q_h": self._last_Q_h,
            "eps_ortho": self._last_eps_ortho,
        }

    def penalty(
        self,
        Q_target: float = 0.0,
        alpha: float = 0.0,
        gamma: float = 0.0,
    ) -> torch.Tensor:
        """alpha * mean((Q_h - Q_target)^2) + gamma * eps_ortho.

        Returns a scalar tensor. When alpha = gamma = 0 the result is
        exact zero with the right dtype / device. The multiplications
        are real (no early return), so a stub that returns junk Q_h
        will not silently pass.
        """
        if self._last_Q_h is None or self._last_eps_ortho is None:
            raise RuntimeError(
                "TopoChargeDiagnostic.penalty called before any "
                "forward pass"
            )
        Q_h = self._last_Q_h
        eps_ortho = self._last_eps_ortho
        target = Q_h.new_full(Q_h.shape, float(Q_target))
        q_term = ((Q_h - target) ** 2).mean()
        ortho_term = eps_ortho
        return float(alpha) * q_term + float(gamma) * ortho_term
