"""Contracts for Poincaré-disc hyperbolic multi-head attention.

Pins the math of ``neuroslm/modules/hyperbolic_attention.py`` against:

* the Poincaré ball model with curvature ``c > 0`` (ball of radius
  ``1/sqrt(c)``),
* the Möbius gyrovector operations (Ungar 2005; Ganea, Bécigneul,
  Hofmann, "Hyperbolic Neural Networks", NeurIPS 2018),
* hyperbolic distance
  ``d(x,y) = (2/sqrt(c)) * artanh(sqrt(c) * ||-x ⊕_c y||)``,
* attention scores given by ``-d_hyp(Q_i, K_j)`` so that nearer keys
  attend more.

Per CLAUDE.md §14 every test here pins a mathematical invariant, not
just an output shape.
"""

from __future__ import annotations

import math

import pytest
import torch

from neuroslm.modules.hyperbolic_attention import (
    HyperbolicMultiHeadAttention,
    expmap0,
    logmap0,
    mobius_add,
    mobius_neg,
    poincare_distance,
    project_to_ball,
)


# ──────────────────────────────────────────────────────────────────────
# 1. Projection onto the open ball
# ──────────────────────────────────────────────────────────────────────


class TestProjectToBall:
    def test_random_oversized_input_lands_strictly_inside_unit_ball(self):
        torch.manual_seed(42)
        x = torch.randn(64, 16) * 10.0  # blatantly outside the unit ball
        x_proj = project_to_ball(x, c=1.0)
        norms = x_proj.norm(dim=-1)
        assert torch.all(norms < 1.0), (
            f"projection failed: max norm {norms.max().item()} >= 1.0"
        )

    def test_input_already_well_inside_ball_is_preserved(self):
        torch.manual_seed(0)
        x = torch.randn(8, 8) * 0.01  # tiny, certainly inside
        x_proj = project_to_ball(x, c=1.0)
        assert torch.allclose(x_proj, x, atol=1e-6)

    def test_supports_nonunit_curvature(self):
        """For curvature c>0 the ball has radius 1/sqrt(c)."""
        torch.manual_seed(1)
        x = torch.randn(32, 16) * 5.0
        c = 4.0
        radius = 1.0 / math.sqrt(c)  # = 0.5
        x_proj = project_to_ball(x, c=c)
        norms = x_proj.norm(dim=-1)
        assert torch.all(norms < radius), (
            f"max norm {norms.max().item()} >= radius {radius}"
        )

    def test_no_nans_on_pathological_inputs(self):
        """Inputs that are zero, exactly on the boundary, and huge must
        all produce finite outputs (numerical-stability contract)."""
        x = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],  # exactly on the boundary
                [1e6, -1e6, 1e6],  # absurdly outside
            ]
        )
        x_proj = project_to_ball(x, c=1.0)
        assert torch.isfinite(x_proj).all()
        assert torch.all(x_proj.norm(dim=-1) < 1.0)


# ──────────────────────────────────────────────────────────────────────
# 2. Möbius vector addition
# ──────────────────────────────────────────────────────────────────────


class TestMobiusAddition:
    def test_adding_zero_on_the_left_is_identity(self):
        x = torch.tensor([[0.1, 0.2, 0.3]])
        zero = torch.zeros_like(x)
        out = mobius_add(zero, x, c=1.0)
        assert torch.allclose(out, x, atol=1e-6)

    def test_adding_zero_on_the_right_is_identity(self):
        x = torch.tensor([[0.1, -0.2, 0.05]])
        zero = torch.zeros_like(x)
        out = mobius_add(x, zero, c=1.0)
        assert torch.allclose(out, x, atol=1e-6)

    def test_left_cancellation_recovers_zero(self):
        """Defining inverse via ``-x``: ``x ⊕_c (-x) == 0``."""
        torch.manual_seed(7)
        x = torch.randn(5, 8) * 0.1  # well inside ball
        out = mobius_add(x, mobius_neg(x), c=1.0)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


# ──────────────────────────────────────────────────────────────────────
# 3. exp / log maps at origin
# ──────────────────────────────────────────────────────────────────────


class TestExpLogMapsAtOrigin:
    def test_logmap0_inverts_expmap0(self):
        """``logmap0(expmap0(v)) == v`` for v in the tangent space at 0."""
        torch.manual_seed(2)
        v = torch.randn(6, 8) * 0.3
        x = expmap0(v, c=1.0)
        v_back = logmap0(x, c=1.0)
        assert torch.allclose(v_back, v, atol=1e-5)

    def test_expmap0_image_lies_inside_ball(self):
        torch.manual_seed(3)
        v = torch.randn(16, 16) * 4.0  # large tangent vectors → near boundary
        x = expmap0(v, c=1.0)
        assert torch.all(x.norm(dim=-1) < 1.0)
        assert torch.isfinite(x).all()


# ──────────────────────────────────────────────────────────────────────
# 4. Poincaré distance — invariants + closed-form check
# ──────────────────────────────────────────────────────────────────────


class TestPoincareDistance:
    def test_self_distance_is_zero(self):
        torch.manual_seed(3)
        x = torch.randn(10, 16) * 0.1
        d = poincare_distance(x, x, c=1.0)
        assert torch.all(d.abs() < 1e-5), (
            f"max |d(x,x)| = {d.abs().max().item()}"
        )

    def test_distance_is_symmetric(self):
        torch.manual_seed(5)
        x = torch.randn(4, 8) * 0.2
        y = torch.randn(4, 8) * 0.2
        d_xy = poincare_distance(x, y, c=1.0)
        d_yx = poincare_distance(y, x, c=1.0)
        assert torch.allclose(d_xy, d_yx, atol=1e-5)

    def test_distance_is_non_negative(self):
        torch.manual_seed(8)
        x = torch.randn(20, 8) * 0.2
        y = torch.randn(20, 8) * 0.2
        d = poincare_distance(x, y, c=1.0)
        assert torch.all(d >= -1e-6)  # tiny slack for fp roundoff

    def test_distance_from_origin_matches_closed_form(self):
        """For c=1, ``d(0, y) = 2 * artanh(||y||)``."""
        torch.manual_seed(11)
        y = torch.randn(8, 16) * 0.1  # small, keep artanh well-defined
        zero = torch.zeros_like(y)
        d = poincare_distance(zero, y, c=1.0)
        expected = 2.0 * torch.atanh(y.norm(dim=-1))
        assert torch.allclose(d, expected, atol=1e-5)

    def test_distance_supports_broadcasting_for_attention(self):
        """For attention we need pairwise distances between two sets of
        vectors; the function must broadcast cleanly."""
        torch.manual_seed(19)
        q = torch.randn(2, 3, 8) * 0.1  # (B, T_q, D)
        k = torch.randn(2, 5, 8) * 0.1  # (B, T_k, D)
        # Add singleton dims so broadcast yields (B, T_q, T_k)
        d = poincare_distance(q.unsqueeze(2), k.unsqueeze(1), c=1.0)
        assert d.shape == (2, 3, 5)
        assert torch.isfinite(d).all()


# ──────────────────────────────────────────────────────────────────────
# 5. Full attention module
# ──────────────────────────────────────────────────────────────────────


class TestHyperbolicMultiHeadAttention:
    def test_forward_runs_on_cpu_without_nans(self):
        torch.manual_seed(0)
        mha = HyperbolicMultiHeadAttention(d_model=32, n_heads=4, c=1.0)
        x = torch.randn(2, 7, 32)
        out = mha(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_gradient_flows_through_attention(self):
        """Autograd must reach every trainable parameter AND the input."""
        torch.manual_seed(17)
        mha = HyperbolicMultiHeadAttention(d_model=16, n_heads=2, c=1.0)
        x = torch.randn(1, 5, 16, requires_grad=True)
        out = mha(x)
        loss = out.sum()
        loss.backward()
        for name, p in mha.named_parameters():
            if not p.requires_grad:
                continue
            assert p.grad is not None, f"{name} has no grad"
            assert torch.isfinite(p.grad).all(), f"{name} grad has NaN/Inf"
            assert p.grad.abs().sum() > 0, f"{name} grad is identically zero"
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_attention_weights_concentrate_on_nearby_keys(self):
        """The mechanism's whole point: small hyperbolic distance must
        translate into large attention weight.

        Construct a query and two keys: ``k_near`` is hyperbolically close
        to the query, ``k_far`` is far. The query's attention weight on
        ``k_near`` must exceed its weight on ``k_far``.
        """
        torch.manual_seed(23)
        mha = HyperbolicMultiHeadAttention(
            d_model=8, n_heads=1, c=1.0, return_weights=True
        )

        # Use a 3-token sequence (q=tok0, k_near=tok1, k_far=tok2). We
        # craft the input so that, AFTER the linear Q/K projections + the
        # exp-map onto the ball, the query lies near the near-key and far
        # from the far-key. Easiest robust way to guarantee that is to
        # freeze Q/K projection to identity and choose raw vectors.
        with torch.no_grad():
            mha.q_proj.weight.copy_(torch.eye(8))
            mha.k_proj.weight.copy_(torch.eye(8))
            if mha.q_proj.bias is not None:
                mha.q_proj.bias.zero_()
            if mha.k_proj.bias is not None:
                mha.k_proj.bias.zero_()

        q_raw = torch.tensor([0.20, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        near = torch.tensor([0.22, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        far = torch.tensor([-0.40, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        x = torch.stack([q_raw, near, far], dim=0).unsqueeze(0)  # (1, 3, 8)

        _, weights = mha(x)
        # weights: (B, H, T_q, T_k); look at token-0's attention over
        # tokens 1 (near) vs 2 (far).
        w_near = weights[0, 0, 0, 1].item()
        w_far = weights[0, 0, 0, 2].item()
        assert w_near > w_far, (
            f"hyperbolic attention failed to prefer the near key: "
            f"w_near={w_near:.4f} w_far={w_far:.4f}"
        )

    def test_output_preserves_finite_norms_at_higher_curvature(self):
        """Smaller ball (larger curvature) must not break the forward."""
        torch.manual_seed(29)
        mha = HyperbolicMultiHeadAttention(d_model=16, n_heads=4, c=4.0)
        x = torch.randn(3, 11, 16)
        out = mha(x)
        assert torch.isfinite(out).all()
        # Output is a Euclidean weighted combination of V — should not
        # explode catastrophically just because the ball shrank.
        assert out.abs().max().item() < 1e3
