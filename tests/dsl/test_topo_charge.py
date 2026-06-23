# -*- coding: utf-8 -*-
"""RED tests for the Pontryagin / Hopfion-lite topological-charge diagnostic
(see C:/Users/morrossl/.claude/plans/lexical-petting-gizmo.md, Phase 1).

This file pins mathematical contracts, not shapes. Per CLAUDE.md §14, a
shape-only test is a stub-smell. Every test below either:
- exercises an analytical limit (uniform n -> Q=0; orthogonal octant -> Q=1/8;
  antipodal layers -> eps_ortho = 2(L-1))
- pins a named invariant (orientation flip -> sign flip; signed angles sum
  to a winding number along a closed loop)
- compares autograd grad to a finite-difference grad (atol 1e-3) on a real
  learnable, not `.grad > 0`

Public API under test (to be implemented in
neuroslm/mechanisms/topo_charge.py):

  solid_angle(n_a, n_b, n_c) -> Tensor
      Signed spherical-triangle solid angle via van Oosterom-Strang
      atan2 formulation. Inputs broadcastable Tensor[..., 3] on S^2.

  berg_luscher_q(n) -> Tensor[..., (T-2)]
      Per-triangle signed solid angle divided by 4*pi, summed via
      sliding window (n_t, n_{t+1}, n_{t+2}) for t=0..T-3.

  hopfion_eps_ortho(n_per_layer) -> Tensor[scalar]
      Mean over (B,H,T) of (1 - n_{l+1} . n_l) summed over l.

  TopoChargeDiagnostic(nn.Module)
      Projects per-layer attention outputs (Tensor[B,H,T,head_dim]) to
      S^2 via Linear(head_dim, 3) with bias init [1,0,0] (closes
      review FIX 8), exposes .Q_h, .eps_ortho via .pop_metrics(), and
      .penalty(Q_target, alpha, gamma).
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from neuroslm.mechanisms.topo_charge import (
    solid_angle,
    berg_luscher_q,
    hopfion_eps_ortho,
    TopoChargeDiagnostic,
)


# ════════════════════════════════════════════════════════════════════════════
# 1. solid_angle (van Oosterom-Strang signed spherical-triangle area)
# ════════════════════════════════════════════════════════════════════════════


class TestSolidAngle:
    """Pin the signed spherical triangle formula."""

    def test_three_identical_points_gives_zero(self):
        """A degenerate triangle has zero solid angle. Closes the
        zero-init failure mode (review risk D)."""
        n = torch.tensor([1.0, 0.0, 0.0]) / math.sqrt(1.0)
        omega = solid_angle(n, n, n)
        assert omega.abs().item() < 1e-6

    def test_two_identical_points_gives_zero(self):
        """Triangle with a repeated vertex is degenerate."""
        n_a = torch.tensor([1.0, 0.0, 0.0])
        n_b = torch.tensor([0.0, 1.0, 0.0])
        omega = solid_angle(n_a, n_a, n_b)
        assert omega.abs().item() < 1e-6

    def test_orthogonal_octant_equals_pi_over_two(self):
        """The three positive basis vectors form a triangle bounding
        one octant of S^2. Solid angle = pi/2."""
        e_x = torch.tensor([1.0, 0.0, 0.0])
        e_y = torch.tensor([0.0, 1.0, 0.0])
        e_z = torch.tensor([0.0, 0.0, 1.0])
        omega = solid_angle(e_x, e_y, e_z)
        # 4*pi sr total, 8 octants -> pi/2 per octant.
        torch.testing.assert_close(
            omega, torch.tensor(math.pi / 2), atol=1e-5, rtol=1e-5
        )

    def test_orientation_swap_flips_sign(self):
        """Swapping the last two vertices reverses the triangle's
        normal. Solid angle must flip sign."""
        e_x = torch.tensor([1.0, 0.0, 0.0])
        e_y = torch.tensor([0.0, 1.0, 0.0])
        e_z = torch.tensor([0.0, 0.0, 1.0])
        omega_pos = solid_angle(e_x, e_y, e_z)
        omega_neg = solid_angle(e_x, e_z, e_y)
        torch.testing.assert_close(
            omega_neg, -omega_pos, atol=1e-5, rtol=1e-5
        )

    def test_broadcasts_over_batch(self):
        """Input (B, 3) -> Output (B,). One batch element per row."""
        e_x = torch.tensor([1.0, 0.0, 0.0])
        e_y = torch.tensor([0.0, 1.0, 0.0])
        e_z = torch.tensor([0.0, 0.0, 1.0])
        n_a = torch.stack([e_x, e_y, e_z])           # (3, 3)
        n_b = torch.stack([e_y, e_z, e_x])
        n_c = torch.stack([e_z, e_x, e_y])
        omega = solid_angle(n_a, n_b, n_c)
        assert omega.shape == (3,)
        # Cyclic permutations all bound the SAME octant with the SAME
        # orientation (right-handed) -> all three triangles must give
        # the same pi/2.
        torch.testing.assert_close(
            omega, torch.full((3,), math.pi / 2), atol=1e-5, rtol=1e-5
        )

    def test_antipodal_third_vertex_is_finite(self):
        """When n_c ~= -n_a the denominator
        1 + a.b + b.c + c.a -> 1 + a.b - a.b - 1 = 0 in the limit
        n_b = (something orthogonal). atan2 must handle this without
        producing NaN / Inf."""
        n_a = torch.tensor([1.0, 0.0, 0.0])
        n_b = torch.tensor([0.0, 1.0, 0.0])
        # near-antipodal third vertex
        n_c = torch.tensor([-0.9999, 0.0, 0.01]) / torch.tensor(
            [-0.9999, 0.0, 0.01]
        ).norm()
        omega = solid_angle(n_a, n_b, n_c)
        assert torch.isfinite(omega).all()


# ════════════════════════════════════════════════════════════════════════════
# 2. berg_luscher_q (sliding-triangle discrete winding)
# ════════════════════════════════════════════════════════════════════════════


class TestBergLuscherQ:
    """Pin Q = (1 / 4*pi) * sum_t omega(n_t, n_{t+1}, n_{t+2})."""

    def test_uniform_sequence_gives_zero(self):
        """All identical n -> all triangles degenerate -> Q = 0."""
        B, H, T = 2, 4, 8
        n = torch.zeros(B, H, T, 3)
        n[..., 0] = 1.0   # all = (1, 0, 0)
        q_per_triangle = berg_luscher_q(n)
        assert q_per_triangle.shape == (B, H, T - 2)
        assert q_per_triangle.abs().max().item() < 1e-6

    def test_T_equals_2_no_triangles(self):
        """T<3 yields zero triangles. Shape must reflect this and
        gradient flow must remain well-defined (no IndexError)."""
        n = torch.zeros(1, 1, 2, 3, requires_grad=True)
        n.data[..., 0] = 1.0
        q = berg_luscher_q(n)
        assert q.shape == (1, 1, 0)
        # Backward must work even when the tensor has zero elements
        # along one axis.
        q.sum().backward()

    def test_T_equals_3_single_triangle_matches_solid_angle(self):
        """One triangle -> the per-triangle entry equals
        solid_angle(n_0, n_1, n_2) / (4*pi) exactly."""
        e_x = torch.tensor([1.0, 0.0, 0.0])
        e_y = torch.tensor([0.0, 1.0, 0.0])
        e_z = torch.tensor([0.0, 0.0, 1.0])
        n = torch.stack([e_x, e_y, e_z]).unsqueeze(0).unsqueeze(0)
        # Shape: (1, 1, 3, 3)
        q = berg_luscher_q(n)
        assert q.shape == (1, 1, 1)
        expected = math.pi / 2 / (4 * math.pi)
        torch.testing.assert_close(
            q.squeeze(), torch.tensor(expected),
            atol=1e-5, rtol=1e-5,
        )

    def test_reverse_sequence_negates_winding(self):
        """A reversed sequence traces the same triangles with
        opposite orientation -> winding flips sign."""
        torch.manual_seed(7)
        n = torch.randn(1, 2, 6, 3)
        n = nn.functional.normalize(n, dim=-1)
        q_fwd = berg_luscher_q(n).sum(dim=-1)
        q_rev = berg_luscher_q(n.flip(dims=(-2,))).sum(dim=-1)
        torch.testing.assert_close(
            q_rev, -q_fwd, atol=1e-5, rtol=1e-5
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. hopfion_eps_ortho (inter-layer orientation decorrelation)
# ════════════════════════════════════════════════════════════════════════════


class TestHopfionEpsOrtho:
    """Pin sum_{l=0}^{L-2} mean( 1 - n_{l+1} . n_l )."""

    def test_aligned_layers_gives_zero(self):
        """All layers identical -> each pair contributes 0."""
        L, B, H, T = 4, 2, 3, 5
        n_one = torch.zeros(B, H, T, 3)
        n_one[..., 0] = 1.0
        eps = hopfion_eps_ortho([n_one for _ in range(L)])
        assert eps.abs().item() < 1e-6

    def test_antipodal_alternation_max(self):
        """Layers alternate n, -n, n, -n -> each pair contributes
        mean(1 - (-1)) = 2. Total = 2 * (L - 1)."""
        L, B, H, T = 4, 2, 3, 5
        n = torch.zeros(B, H, T, 3)
        n[..., 0] = 1.0
        layers = [n if (i % 2 == 0) else -n for i in range(L)]
        eps = hopfion_eps_ortho(layers)
        expected = 2.0 * (L - 1)
        torch.testing.assert_close(
            eps, torch.tensor(expected), atol=1e-6, rtol=1e-6
        )

    def test_single_layer_gives_zero(self):
        """L=1 -> no pairs -> eps = 0 (must not IndexError)."""
        n = torch.zeros(1, 1, 4, 3)
        n[..., 0] = 1.0
        eps = hopfion_eps_ortho([n])
        assert eps.abs().item() < 1e-6


# ════════════════════════════════════════════════════════════════════════════
# 4. TopoChargeDiagnostic module
# ════════════════════════════════════════════════════════════════════════════


def _make_attn_outputs(L=3, B=2, H=4, T=6, head_dim=8, seed=0):
    """Synthetic per-layer attention output: (B, H, T, head_dim)."""
    torch.manual_seed(seed)
    return [torch.randn(B, H, T, head_dim) for _ in range(L)]


class TestTopoChargeDiagnostic:
    """End-to-end module contracts."""

    def test_projection_bias_init_gives_unit_norm_at_zero_input(self):
        """Closes review FIX 8: zero-init weights + zero-input would
        produce n = F.normalize(0) = 0, breaking the unit-norm contract.
        Bias init [1, 0, 0] guarantees n = (1, 0, 0) deterministically."""
        diag = TopoChargeDiagnostic(head_dim=8)
        x_zero = torch.zeros(2, 4, 6, 8)
        n = diag._project_to_S2(x_zero)        # internal helper
        norms = n.norm(dim=-1)
        torch.testing.assert_close(
            norms, torch.ones_like(norms),
            atol=1e-5, rtol=1e-5,
        )
        # The constant direction is the e_x basis (the bias init).
        torch.testing.assert_close(
            n[..., 0], torch.ones(2, 4, 6),
            atol=1e-5, rtol=1e-5,
        )

    def test_forward_emits_Q_h_and_eps_ortho(self):
        """forward(attn_per_layer) -> dict with both diagnostics,
        both finite, both differentiable."""
        diag = TopoChargeDiagnostic(head_dim=8)
        attn = _make_attn_outputs(L=3, B=2, H=4, T=6, head_dim=8)
        out = diag(attn)
        assert "Q_h" in out and "eps_ortho" in out
        # Q_h shape is (B, H) -- summed over layers and triangles.
        assert out["Q_h"].shape == (2, 4)
        assert out["eps_ortho"].dim() == 0
        assert torch.isfinite(out["Q_h"]).all()
        assert torch.isfinite(out["eps_ortho"]).all()

    def test_diagnostic_only_default_zero_loss(self):
        """penalty(alpha=0, gamma=0, Q_target=0) -> exact zero tensor.
        Diagnostic-only mode must add ZERO to the loss budget."""
        diag = TopoChargeDiagnostic(head_dim=8)
        attn = _make_attn_outputs()
        diag(attn)
        loss = diag.penalty(Q_target=0.0, alpha=0.0, gamma=0.0)
        assert loss.dim() == 0
        assert torch.equal(loss, torch.zeros((), dtype=loss.dtype))

    def test_pop_metrics_changes_with_input(self):
        """Two distinct inputs -> two distinct Q_h. Rules out a
        constant-returning stub (closes review FIX 11)."""
        diag = TopoChargeDiagnostic(head_dim=8)
        attn_a = _make_attn_outputs(seed=1)
        attn_b = _make_attn_outputs(seed=2)
        diag(attn_a)
        Q_a = diag.pop_metrics()["Q_h"].clone()
        diag(attn_b)
        Q_b = diag.pop_metrics()["Q_h"].clone()
        # Distinct random inputs at the same seed-shape must produce
        # distinct charges (within numerical safety margin).
        assert (Q_a - Q_b).abs().max().item() > 1e-4

    def test_penalty_nonzero_alpha_propagates_gradient(self):
        """With alpha > 0, penalty.backward() must put grad on the
        projection weight. Pins that Q_h actually depends on the
        projection learnable, not just on the input."""
        diag = TopoChargeDiagnostic(head_dim=8)
        # Use a deterministic seed for the attention outputs so this
        # test is reproducible.
        attn = _make_attn_outputs(seed=3)
        diag(attn)
        loss = diag.penalty(Q_target=0.0, alpha=0.5, gamma=0.0)
        loss.backward()
        w = diag.proj.weight.grad
        assert w is not None and w.abs().sum().item() > 1e-6

    def test_finite_difference_grad_matches_autograd(self):
        """Pin gradient correctness, not just non-zero-ness.
        Closes review FIX 2: `.grad > 0` is a wire-connection check;
        FD-vs-autograd is a real contract."""
        diag = TopoChargeDiagnostic(head_dim=4)
        attn = _make_attn_outputs(L=2, B=1, H=1, T=5, head_dim=4, seed=4)

        # Index the component to perturb.
        i, j = 0, 0

        def loss_at(perturbation: float) -> float:
            with torch.no_grad():
                diag.proj.weight[i, j] += perturbation
            try:
                diag(attn)
                v = diag.penalty(
                    Q_target=0.0, alpha=0.7, gamma=0.0
                ).detach().item()
            finally:
                with torch.no_grad():
                    diag.proj.weight[i, j] -= perturbation
            return v

        eps = 1e-3
        fd = (loss_at(eps) - loss_at(-eps)) / (2 * eps)

        # Autograd grad at the same point.
        diag.proj.weight.grad = None
        diag(attn)
        diag.penalty(Q_target=0.0, alpha=0.7, gamma=0.0).backward()
        ag = diag.proj.weight.grad[i, j].item()

        assert math.isclose(fd, ag, rel_tol=5e-2, abs_tol=1e-3), (
            f"finite-diff grad {fd} differs from autograd {ag}"
        )

    def test_module_forward_invokes_berg_luscher(self, monkeypatch):
        """Closes review FIX 10: the module must actually call the
        helper, not bypass it with a stub that returns zeros."""
        from neuroslm.mechanisms import topo_charge as topo_mod
        calls = []
        real_q = topo_mod.berg_luscher_q

        def spy(n):
            calls.append(n.shape)
            return real_q(n)

        monkeypatch.setattr(topo_mod, "berg_luscher_q", spy)
        diag = TopoChargeDiagnostic(head_dim=8)
        diag(_make_attn_outputs(L=2, B=1, H=2, T=5, head_dim=8))
        assert len(calls) >= 1, (
            "TopoChargeDiagnostic.forward must call berg_luscher_q "
            "at least once per invocation"
        )

    def test_T_equals_2_no_crash(self):
        """Boundary: T=2 means zero triangles per layer. Q_h must
        be (B, H) zeros, finite, and the module must not raise."""
        diag = TopoChargeDiagnostic(head_dim=8)
        attn = [torch.randn(1, 2, 2, 8) for _ in range(2)]  # T=2
        out = diag(attn)
        assert out["Q_h"].shape == (1, 2)
        assert torch.allclose(
            out["Q_h"], torch.zeros_like(out["Q_h"])
        )
        # eps_ortho still well-defined (n is rank-1 along T but
        # still has 3-axis content from the projection).
        assert torch.isfinite(out["eps_ortho"])

    def test_single_layer_eps_ortho_zero(self):
        """L=1 input -> no inter-layer pair -> eps_ortho = 0."""
        diag = TopoChargeDiagnostic(head_dim=8)
        out = diag([torch.randn(1, 2, 5, 8)])
        assert out["eps_ortho"].abs().item() < 1e-6
        # Q_h still computed for the single layer.
        assert out["Q_h"].shape == (1, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
