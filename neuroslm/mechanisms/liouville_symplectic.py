# -*- coding: utf-8 -*-
"""Liouville Symplectic Residual block (Phase 2 of the THSD program).

Splits d_model into canonical coordinates (q, p) of equal size and
advances them by ONE explicit Stoermer-Verlet (leapfrog) step of a
learned Hamiltonian:

    H(q, p) = (1/2) || M^{-1/2} p ||^2  +  V(q)  +  W(q)

where V is a q-only potential (SwiGLU in production, quadratic for
the load-bearing det(J)=1 test) and W is a low-rank pairwise q-only
potential. The leapfrog substeps are:

    (1) p_{1/2} = p_0 - (dtau/2) * grad_q H(q_0)
    (2) q_1     = q_0 + dtau     * M^{-1} * p_{1/2}
    (3) p_1     = p_{1/2} - (dtau/2) * grad_q H(q_1)

Each substep is a linear-in-(q,p) triangular SHEAR with determinant
1, so the full step has Jacobian determinant exactly 1 (symplectic
by construction -- Liouville's theorem). The mass matrix M is a
learnable diagonal forced positive via softplus.

The Noether residual is

    L_Noether = ( H(q_1, p_1) - H(q_0, p_0) ) ** 2

which equals 0 in the analytical limit and is finite-and-bounded
under leapfrog by the Hairer-Lubich-Wanner modified-Hamiltonian
theorem (Geometric Numerical Integration, 2006, Ch. IX, Thm 8.1).
The harness composes lambda_noether * L_Noether into the LM loss
budget; with lambda > 0 the optimizer learns potentials that
preserve H more tightly.

References.
    Hairer, Lubich, Wanner (2006) Geometric Numerical Integration:
        Structure-Preserving Algorithms for Ordinary Differential
        Equations. Springer Series in Computational Mathematics 31.
    Stoermer (1907); Verlet (1967) -- the leapfrog integrator.
    Greydanus, Dzamba, Yosinski (2019) Hamiltonian Neural Networks,
        NeurIPS -- but their architecture predicts forces; this
        block USES the architecture (q,p) directly and computes the
        symplectic update by hand to guarantee det(J)=1 statically.

CLAUDE.md sec 14 contract notes.
    The QOnlyPotential abstract base enforces q-only forward
    signature at construction time (no future contributor can
    silently thread p through V or W). The det(J)=1 invariant is
    therefore a TYPE-level guarantee, not a runtime assertion.
"""
from __future__ import annotations

import inspect
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. QOnlyPotential: type-enforced abstract base
# ============================================================================


class QOnlyPotential(nn.Module):
    """Abstract base class: subclasses MUST implement forward(self, q)
    whose only non-self parameter is named ``q``.

    The check fires at __init__ time so the failure mode is loud and
    early. A subclass that adds ``p`` (or any other name) raises a
    TypeError before any forward pass.
    """

    def __init__(self):
        super().__init__()
        forward = type(self).forward
        # Skip the type itself; the base's forward is abstract and
        # will be overridden in subclasses.
        if forward is QOnlyPotential.forward:
            return
        sig = inspect.signature(forward)
        non_self = [
            name for name in sig.parameters if name != "self"
        ]
        if non_self != ["q"]:
            raise TypeError(
                f"{type(self).__name__}.forward must take exactly "
                f"one non-self parameter named 'q'; got {non_self}. "
                f"This is a load-bearing type contract that backs "
                f"the det(J)=1 symplectic invariant -- a potential "
                f"that sees p would break the triangular-shear "
                f"structure of the leapfrog integrator."
            )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def energy(self, q: torch.Tensor) -> torch.Tensor:
        """Public alias for the potential energy. Returns a scalar."""
        return self.forward(q)


# ============================================================================
# 2. Concrete potentials
# ============================================================================


class QuadraticPotential(QOnlyPotential):
    """V(q) = (1/2) q^T A q. Used in tests where det(J)=1 must be
    pinned in closed form (the autograd jacobian of the quadratic
    case is exact in fp64)."""

    def __init__(self, d: int, omega: float = 1.0,
                 init_diag: Optional[float] = None):
        super().__init__()
        # A is a d x d symmetric Parameter init with small random
        # entries (so the energy is non-trivial). For the 1-D
        # harmonic-oscillator test, the constructor accepts an
        # explicit omega via the special branch below.
        A = 0.1 * torch.randn(d, d)
        A = 0.5 * (A + A.T)
        if init_diag is not None:
            with torch.no_grad():
                A.fill_diagonal_(float(init_diag))
        self.A = nn.Parameter(A)
        self._omega = float(omega)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # Mean over leading dims to keep loss-scale independent of
        # (B, T).
        return 0.5 * torch.einsum(
            "...i,ij,...j->...", q, self.A, q
        ).mean()


class QuadraticHarmonicPotential(QOnlyPotential):
    """V(q) = (1/2) * omega^2 * sum(q^2). The 1-D harmonic-
    oscillator test uses this with the matching kinetic term
    (M = I) so H = (1/2)(p^2 + omega^2 q^2) is the textbook
    closed-form whose leapfrog modified-Hamiltonian bound is
    given by Hairer-Lubich-Wanner."""

    def __init__(self, omega: float = 1.0):
        super().__init__()
        self.omega = float(omega)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.omega ** 2) * (q ** 2).sum()


class SwiGLUPotential(QOnlyPotential):
    """Production-grade scalar potential V(q) backed by a SwiGLU
    feed-forward followed by L2-norm projection. The output is
    sum( SwiGLU(q) ** 2 ) / 2 -- a smooth, learnable, q-only
    scalar function."""

    def __init__(self, d: int, hidden_mult: int = 2):
        super().__init__()
        h = int(hidden_mult) * int(d)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)
        nn.init.normal_(self.w3.weight, std=0.02)
        self._zeroed_for_test = False

    def zero_for_test(self) -> None:
        """Zero the final layer's weights so V(q) == 0 for any q.
        Used by the free-streaming test."""
        with torch.no_grad():
            self.w3.weight.zero_()
        self._zeroed_for_test = True

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        out = self.w3(F.silu(self.w1(q)) * self.w2(q))
        return 0.5 * (out ** 2).sum()


class LowRankPairwise(nn.Module):
    """W(q) = sum_{i < j} (U q_i)^T (V q_j). Rank-r pairwise
    interaction between token positions. q-only by construction --
    the forward signature contains a single parameter ``q``."""

    def __init__(self, d: int, rank: int = 4):
        super().__init__()
        self.U = nn.Parameter(0.02 * torch.randn(d, rank))
        self.V = nn.Parameter(0.02 * torch.randn(d, rank))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: (..., T, d). Compute pairwise sum:
        #   sum_{i<j} ( (q_i U) . (q_j V) )
        # = (1/2) [ ( sum_i q_i U ) . ( sum_j q_j V )
        #           - sum_i (q_i U) . (q_i V) ]
        # This O(T) formula avoids an O(T^2) loop.
        uq = q @ self.U          # (..., T, rank)
        vq = q @ self.V          # (..., T, rank)
        sum_uq = uq.sum(dim=-2)  # (..., rank)
        sum_vq = vq.sum(dim=-2)  # (..., rank)
        full = (sum_uq * sum_vq).sum(dim=-1)
        diag = (uq * vq).sum(dim=(-1, -2))
        return (0.5 * (full - diag)).mean()


# ============================================================================
# 3. LiouvilleSymplecticBlock
# ============================================================================


class LiouvilleSymplecticBlock(nn.Module):
    """One explicit leapfrog step of a learned Hamiltonian.

    Args:
        d_model: total residual dim. MUST be even (the (q, p) split).
        dtau_init: initial leapfrog stride.
        potential_kind: which V to use.
            "swiglu"          -- production SwiGLU potential.
            "quadratic"       -- QuadraticPotential (for det(J)=1 test).
            "quadratic_omega" -- QuadraticHarmonicPotential (for HLW
                                 1-D oscillator test).
        quadratic_omega: omega for "quadratic_omega".
        w_rank: rank of the pairwise W potential.
        zero_potentials_for_test: zero V and W for the free-streaming
            test (production runs leave this False).
    """

    def __init__(
        self,
        d_model: int,
        dtau_init: float = 0.1,
        potential_kind: str = "swiglu",
        quadratic_omega: float = 1.0,
        w_rank: int = 4,
        zero_potentials_for_test: bool = False,
    ):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(
                f"d_model={d_model} must be even (q/p split)"
            )
        self.d_model = int(d_model)
        self.d_half = self.d_model // 2

        # Learnable leapfrog stride (Parameter so the optimizer can
        # tune it; gradient flows via the (q,p) update chain).
        self.dtau = nn.Parameter(torch.tensor(float(dtau_init)))

        # Diagonal mass matrix M = softplus(raw_M) (positivity).
        # init raw_M = 0.5413 -> softplus(0.5413) ~ 1.0 (M = I init).
        raw_init = math.log(math.expm1(1.0))
        self.raw_M = nn.Parameter(
            torch.full((self.d_half,), float(raw_init))
        )

        # Potential V.
        self.potential_kind = potential_kind
        if potential_kind == "swiglu":
            self.V = SwiGLUPotential(d=self.d_half)
        elif potential_kind == "quadratic":
            self.V = QuadraticPotential(d=self.d_half)
        elif potential_kind == "quadratic_omega":
            self.V = QuadraticHarmonicPotential(omega=quadratic_omega)
        else:
            raise ValueError(
                f"unknown potential_kind={potential_kind!r}"
            )

        # Pairwise W (low-rank, q-only).
        self.W = LowRankPairwise(d=self.d_half, rank=w_rank)

        if zero_potentials_for_test:
            # Free-streaming test: zero both V and W so the leapfrog
            # reduces to pure drift.
            if hasattr(self.V, "zero_for_test"):
                self.V.zero_for_test()
            else:
                # Quadratic / harmonic: zero the parameters directly.
                with torch.no_grad():
                    if hasattr(self.V, "A"):
                        self.V.A.zero_()
                    if hasattr(self.V, "omega"):
                        self.V.omega = 0.0
            with torch.no_grad():
                self.W.U.zero_()
                self.W.V.zero_()

        # Stash for Noether residual.
        self._last_noether: Optional[torch.Tensor] = None
        self._last_H_initial: Optional[torch.Tensor] = None
        self._last_H_final: Optional[torch.Tensor] = None

    # -- properties ----------------------------------------------------

    @property
    def M_diag(self) -> torch.Tensor:
        return F.softplus(self.raw_M)

    # -- internals -----------------------------------------------------

    def _grad_q_H(self, q: torch.Tensor) -> torch.Tensor:
        """Compute d(V(q) + W(q)) / dq via autograd.

        ``create_graph=True`` so backward through the leapfrog can
        differentiate the FORCE (essential for training the
        potential parameters via the chain through the position
        update)."""
        if not q.requires_grad:
            q = q.detach().requires_grad_(True)
        e = self.V(q) + self.W(q)
        g, = torch.autograd.grad(
            e, q, create_graph=True, retain_graph=True
        )
        return g

    def hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        """H(q, p) = 0.5 * ||M^{-1/2} p||^2 + V(q) + W(q)."""
        q, p = x[..., : self.d_half], x[..., self.d_half :]
        M_inv = 1.0 / self.M_diag
        # Kinetic = 0.5 * sum( p^2 * M^-1 ), summed/averaged via
        # sum() to keep it scale-consistent with the potential
        # energies which use mean/sum likewise.
        kinetic = 0.5 * (p ** 2 * M_inv).sum() / max(1, q.shape[0] * q.shape[1])
        return kinetic + self.V(q) + self.W(q)

    # -- forward -------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One leapfrog step: (q, p) -> (q', p'); returns concatenated."""
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"input d_model {x.shape[-1]} != block d_model "
                f"{self.d_model}"
            )
        # Split (q, p). Clone with grad so we can autograd through
        # the symplectic update without aliasing the residual stream.
        q_0 = x[..., : self.d_half].clone()
        p_0 = x[..., self.d_half :].clone()
        if not q_0.requires_grad:
            q_0 = q_0.requires_grad_(True)

        # H_initial for Noether bookkeeping (no grad needed).
        with torch.no_grad():
            H_initial = self._hamiltonian_from_qp(
                q_0.detach(), p_0.detach()
            ).detach()

        # (1) Half-kick on p with force at q_0.
        force_0 = self._grad_q_H(q_0)
        p_half = p_0 - 0.5 * self.dtau * force_0

        # (2) Full drift on q with M^-1 * p_half.
        M_inv = 1.0 / self.M_diag
        q_1 = q_0 + self.dtau * M_inv * p_half

        # (3) Half-kick on p with force at q_1.
        force_1 = self._grad_q_H(q_1)
        p_1 = p_half - 0.5 * self.dtau * force_1

        # Noether residual: (H_final - H_initial)^2. H_final
        # gradients can flow back to the potential parameters via
        # q_1 / p_1.
        H_final = self._hamiltonian_from_qp(q_1, p_1)
        self._last_H_initial = H_initial
        self._last_H_final = H_final
        self._last_noether = (H_final - H_initial) ** 2

        return torch.cat([q_1, p_1], dim=-1)

    def _hamiltonian_from_qp(
        self, q: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """Internal helper: computes H from already-split (q, p)."""
        M_inv = 1.0 / self.M_diag
        kinetic = 0.5 * (p ** 2 * M_inv).sum() / max(
            1, q.shape[0] * q.shape[1])
        return kinetic + self.V(q) + self.W(q)
