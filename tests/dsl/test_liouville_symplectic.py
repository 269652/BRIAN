# -*- coding: utf-8 -*-
"""RED tests for the Liouville Symplectic Residual block (Phase 2).

Splits d_model into canonical coordinates (q, p) of equal size and
evolves them across each layer via ONE explicit leapfrog (Stoermer-
Verlet) step of a learned Hamiltonian

    H_ell(q, p) = 0.5 * ||M_ell^{-0.5} p||^2 + V_ell(q) + W_ell(q)

where V is a q-only potential (SwiGLU in production, quadratic for
the load-bearing det(J)=1 test) and W is a low-rank pairwise q-only
potential. The two key invariants:

  (I-1) det(J) == 1 EXACTLY (symplectic by construction).
  (I-2) The leapfrog of a quadratic Hamiltonian H = 0.5*p^2 + 0.5*omega^2 q^2
        has bounded H oscillation amplitude proportional to (omega*dtau)^2 / 24
        for second-order leapfrog (Hairer-Lubich-Wanner, "Geometric
        Numerical Integration", 2006). No secular drift.

The Noether residual L_Noether = (H_L - H_0)^2 is the diagnostic the
harness composes; it is identically zero for purely-quadratic V at
the analytical leapfrog timestep (machine-eps).

CLAUDE.md sec 14 contract-strength guards baked in:
  - QOnlyPotential is a TYPE: its forward() signature MUST accept only
    `q`; a future contributor cannot silently thread `p` through it.
  - det(J)=1 is tested in fp64 against a closed-form quadratic V; the
    autograd-jacobian of the full (q,p) -> (q',p') map must equal one
    to atol 1e-10 (fp32 atol is far too loose for this invariant).
  - The Hamiltonian-bounded-oscillation test uses the HLW closed-form
    upper bound, not a vibe-based "trend doesn't drift" heuristic.
  - Free-streaming limit uses the REAL SwiGLU potential with zeroed
    final-layer weights (not a mock), exercising the production
    code path.
"""
from __future__ import annotations

import inspect
import math
import pytest
import torch
import torch.nn as nn

from neuroslm.mechanisms.liouville_symplectic import (
    QOnlyPotential,
    QuadraticPotential,
    SwiGLUPotential,
    LowRankPairwise,
    LiouvilleSymplecticBlock,
)


# ============================================================================
# 1. QOnlyPotential type contract
# ============================================================================


class TestQOnlyPotentialType:
    def test_forward_signature_rejects_p(self):
        """A QOnlyPotential subclass whose forward accepts a parameter
        named other than `q` MUST raise at construction.

        This pins the type-level guarantee that backs det(J)=1: if
        the potential could see p, the leapfrog substeps would no
        longer be triangular shears and det(J) would generically
        differ from 1.
        """
        class BadPotential(QOnlyPotential):
            def forward(self, q, p):  # noqa: ARG002  -- intentional
                return q.sum()

        with pytest.raises((AssertionError, TypeError, ValueError),
                           match=r"q"):
            BadPotential()

    def test_subclass_with_only_q_constructs(self):
        """Reference subclass: QuadraticPotential takes only q."""
        pot = QuadraticPotential(d=4)
        sig = inspect.signature(pot.forward).parameters
        assert list(sig.keys()) == ["q"]


# ============================================================================
# 2. QuadraticPotential math
# ============================================================================


class TestQuadraticPotential:
    def test_energy_matches_half_qTAq(self):
        torch.manual_seed(0)
        d = 6
        pot = QuadraticPotential(d=d)
        q = torch.randn(2, 3, d, dtype=torch.float64)
        pot = pot.double()
        e_module = pot.energy(q)
        # Hand-computed reference: 0.5 * q^T A q, summed over leading
        # dims, averaged per the module's convention.
        A = pot.A.detach()
        manual = 0.5 * torch.einsum("...i,ij,...j->...", q, A, q).mean()
        torch.testing.assert_close(
            e_module, manual, atol=1e-10, rtol=1e-10
        )

    def test_grad_q_matches_Aq(self):
        torch.manual_seed(0)
        d = 4
        pot = QuadraticPotential(d=d).double()
        q = torch.randn(1, 1, d, dtype=torch.float64, requires_grad=True)
        e = pot.energy(q)
        g, = torch.autograd.grad(e, q, create_graph=False)
        # d(0.5 q^T A q)/dq = (A + A^T)/2 * q = symmetric A * q.
        A = pot.A.detach()
        A_sym = 0.5 * (A + A.T)
        manual = torch.einsum("ij,...j->...i", A_sym, q.detach())
        # Module divides energy by leading dims (mean), so the
        # gradient is also rescaled. Account for that.
        n_lead = q.shape[0] * q.shape[1]
        torch.testing.assert_close(
            g * n_lead, manual, atol=1e-10, rtol=1e-10
        )


# ============================================================================
# 3. LiouvilleSymplecticBlock: free streaming, leapfrog, det(J)=1
# ============================================================================


class TestLiouvilleSymplecticFreeStreaming:
    def test_V_W_zero_is_pure_drift(self):
        """With V == 0 (SwiGLU with last layer zeroed) and W == 0,
        the leapfrog reduces to a single drift:
            p_new = p   (no force)
            q_new = q + dtau * M^-1 * p
        Pin both invariants.
        """
        torch.manual_seed(0)
        d_model = 8
        blk = LiouvilleSymplecticBlock(
            d_model=d_model, dtau_init=0.1,
            zero_potentials_for_test=True,  # zero out V, W
        )
        x = torch.randn(2, 3, d_model)
        x_new = blk(x)
        # Recover (q, p) halves.
        q_old, p_old = x[..., : d_model // 2], x[..., d_model // 2 :]
        q_new, p_new = (
            x_new[..., : d_model // 2],
            x_new[..., d_model // 2 :],
        )
        # p unchanged (no force).
        torch.testing.assert_close(p_new, p_old, atol=1e-6, rtol=1e-6)
        # q drifts by dtau * M^-1 * p, where M is the learnable diag.
        M_inv = 1.0 / blk.M_diag.detach()
        dtau = blk.dtau.detach()
        expected_q = q_old + dtau * M_inv * p_old
        torch.testing.assert_close(
            q_new, expected_q, atol=1e-5, rtol=1e-5
        )


class TestLiouvilleSymplecticDetJacobian:
    """The load-bearing invariant: each leapfrog substep is a
    triangular shear with det = 1, and the composition is det = 1.

    Pinned in fp64 against a pure-quadratic V (no SwiGLU non-
    linearity) so the autograd Jacobian is exact.
    """

    def test_det_jacobian_is_one_fp64(self):
        torch.manual_seed(0)
        d_model = 6     # small enough for explicit Jacobian
        blk = LiouvilleSymplecticBlock(
            d_model=d_model, dtau_init=0.05,
            potential_kind="quadratic",
        )
        blk = blk.double()
        x = torch.randn(1, 1, d_model, dtype=torch.float64)

        def f(x_in: torch.Tensor) -> torch.Tensor:
            return blk(x_in.unsqueeze(0).unsqueeze(0)).squeeze()

        # 6 x 6 Jacobian. d_model is small so this is cheap.
        J = torch.autograd.functional.jacobian(f, x.squeeze())
        det = torch.linalg.det(J)
        assert abs(det.item() - 1.0) < 1e-9, (
            f"leapfrog must be symplectic (det J = 1); got det={det.item()}"
        )


class TestLiouvilleSymplecticHairerLubichWanner:
    """For a 1-D harmonic oscillator H = 0.5*p^2 + 0.5*omega^2*q^2
    with M = I, the second-order Stoermer-Verlet leapfrog gives a
    modified-Hamiltonian conservation with H oscillation bound
        max H - min H  <  C * (omega * dtau)^2 * H_0

    over an exponentially long time horizon (Hairer-Lubich-Wanner
    2006, Ch. IX, Thm 8.1). For omega * dtau = 0.1, the bound is
    ~0.005 * H_0 -- comfortably exceeded by a numerical-Euler-style
    drift, so this test catches "leapfrog implemented but with the
    wrong half-kick / drift / half-kick order."
    """

    def test_H_bounded_over_100_steps_pure_quadratic(self):
        torch.manual_seed(0)
        d_model = 2     # 1-d q + 1-d p
        omega = 1.0
        dtau = 0.1
        blk = LiouvilleSymplecticBlock(
            d_model=d_model, dtau_init=dtau,
            potential_kind="quadratic_omega",
            quadratic_omega=omega,
        ).double()
        # Initial (q, p) = (1, 0): H_0 = 0.5 * omega^2 * 1.
        x = torch.zeros(1, 1, d_model, dtype=torch.float64)
        x[..., 0] = 1.0   # q = 1
        H_trace = []
        for _ in range(100):
            H_trace.append(blk.hamiltonian(x).item())
            x = blk(x)
        # Final H reading.
        H_trace.append(blk.hamiltonian(x).item())
        H_0 = H_trace[0]
        H_amp = max(H_trace) - min(H_trace)
        # HLW bound: ~ (omega * dtau)^2 / 24 * H_0 -> ~4e-4 * H_0.
        # Generous 10x slack still well below the 0.05 * H_0 a
        # broken integrator would produce.
        bound = (omega * dtau) ** 2 / 24.0 * 10.0 * H_0
        assert H_amp < bound, (
            f"H oscillation {H_amp:.4f} exceeds HLW bound {bound:.4f}; "
            f"leapfrog kicks may be in the wrong order"
        )


# ============================================================================
# 4. Mass matrix positivity
# ============================================================================


class TestMassMatrix:
    def test_M_is_positive_for_any_raw_value(self):
        """M_diag = softplus(raw_M) > 0 for all raw_M, including
        large-negative values. Pins the kinetic-energy term's
        positive-definiteness invariant."""
        blk = LiouvilleSymplecticBlock(d_model=8)
        with torch.no_grad():
            blk.raw_M.fill_(-10.0)
        assert (blk.M_diag > 0).all().item()


# ============================================================================
# 5. Noether residual: (H_final - H_initial)^2
# ============================================================================


class TestNoetherResidual:
    def test_noether_equals_squared_H_diff(self):
        torch.manual_seed(0)
        d_model = 4
        blk = LiouvilleSymplecticBlock(
            d_model=d_model, dtau_init=0.1,
            potential_kind="quadratic",
        ).double()
        x = torch.randn(1, 1, d_model, dtype=torch.float64)
        H_initial = blk.hamiltonian(x)
        x_new = blk(x)
        H_final = blk.hamiltonian(x_new)
        expected = (H_final - H_initial) ** 2
        torch.testing.assert_close(
            blk._last_noether, expected,
            atol=1e-12, rtol=1e-12,
        )


# ============================================================================
# 6. Gradient flow: finite-diff vs autograd on a learnable
# ============================================================================


class TestGradFlow:
    def test_fd_vs_autograd_on_dtau(self):
        """The block's dtau Parameter must receive REAL gradient
        (computed by FD on a chosen target) matching autograd
        within atol=1e-3. Pins that backward isn't decorative."""
        torch.manual_seed(0)
        d_model = 4
        blk = LiouvilleSymplecticBlock(
            d_model=d_model, dtau_init=0.1,
            potential_kind="quadratic",
        ).double()
        x = torch.randn(1, 2, d_model, dtype=torch.float64)

        def loss_at(perturbation: float) -> float:
            with torch.no_grad():
                blk.dtau.add_(perturbation)
            try:
                out = blk(x)
                v = (out ** 2).sum().item()
            finally:
                with torch.no_grad():
                    blk.dtau.sub_(perturbation)
            return v

        eps = 1e-5
        fd = (loss_at(eps) - loss_at(-eps)) / (2 * eps)

        # Autograd.
        blk.dtau.grad = None
        out = blk(x)
        (out ** 2).sum().backward()
        ag = blk.dtau.grad.item()

        assert math.isclose(fd, ag, rel_tol=5e-2, abs_tol=1e-3), (
            f"finite-diff grad on dtau {fd:.5e} differs from "
            f"autograd grad {ag:.5e}"
        )


# ============================================================================
# 7. Input contract: odd d_model raises
# ============================================================================


class TestInputContract:
    def test_odd_d_model_raises(self):
        with pytest.raises((AssertionError, ValueError),
                           match=r"d_model"):
            LiouvilleSymplecticBlock(d_model=7)

    def test_T_equals_1_does_not_crash(self):
        """Boundary case: single token sequence must produce same-
        shape output and finite Noether residual."""
        blk = LiouvilleSymplecticBlock(
            d_model=8, potential_kind="quadratic",
        )
        x = torch.randn(2, 1, 8)
        out = blk(x)
        assert out.shape == x.shape
        assert torch.isfinite(blk._last_noether).all()


# ============================================================================
# 8. LowRankPairwise q-only construction
# ============================================================================


class TestLowRankPairwise:
    def test_low_rank_q_only(self):
        sig = inspect.signature(LowRankPairwise.forward).parameters
        # Excluding 'self'.
        non_self = [n for n in sig if n != "self"]
        assert non_self == ["q"], (
            f"LowRankPairwise.forward must be q-only; got {non_self}"
        )

    def test_low_rank_output_shape(self):
        torch.manual_seed(0)
        d = 5
        W = LowRankPairwise(d=d, rank=2)
        q = torch.randn(2, 4, d)
        out = W(q)
        # Scalar potential averaged or summed; non-shape-only:
        # require it changes with input.
        out_a = W(q)
        out_b = W(q + 0.5)
        assert out_a.item() != out_b.item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
