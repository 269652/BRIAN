"""Math contracts for surprise-gated MoE (mechanism #5).

Pins:
  * surprise_to_k: monotone, bounded in [k_min, k_max], midpoint behaviour.
  * load_balance_loss: zero when uniform, positive when imbalanced,
    invariant of vocab-size scale.
  * SurpriseGatedMoE: shape preservation, surprise actually shifts
    mean_k, load-balance loss exposed, parameters get gradients.
  * Validation: bad inits raise.
"""

from __future__ import annotations

import pytest
import torch

from neuroslm.modules.surprise_gated_moe import (
    SurpriseGatedMoE,
    load_balance_loss,
    surprise_to_k,
)


# ──────────────────────────────────────────────────────────────────────
# 1. surprise_to_k mapping
# ──────────────────────────────────────────────────────────────────────


class TestSurpriseToK:
    def test_zero_surprise_yields_k_min(self):
        s = torch.tensor([0.0, 0.0, 0.0])
        k = surprise_to_k(s, k_min=1, k_max=8, midpoint=5.0, steepness=10.0)
        # With midpoint=5 and steepness=10, σ(10·(0-5)) ≈ 0 → k = k_min.
        assert (k == 1).all()

    def test_very_high_surprise_yields_k_max(self):
        s = torch.tensor([100.0, 100.0])
        k = surprise_to_k(s, k_min=1, k_max=8, midpoint=1.0, steepness=2.0)
        assert (k == 8).all()

    def test_midpoint_yields_midpoint_k(self):
        s = torch.tensor([2.0])
        # σ(steep·(2-2)) = 0.5 → k = (k_min + k_max)/2.
        k = surprise_to_k(s, k_min=2, k_max=6, midpoint=2.0, steepness=4.0)
        # (2 + 6)/2 = 4 → rounds to 4.
        assert k.item() == 4

    def test_monotonic_in_surprise(self):
        s = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0])
        k = surprise_to_k(s, k_min=1, k_max=8, midpoint=1.0, steepness=2.0)
        # Sorted by construction → k should be non-decreasing.
        assert (k[1:] >= k[:-1]).all()

    def test_bad_k_min_raises(self):
        with pytest.raises(ValueError, match="k_min"):
            surprise_to_k(torch.tensor([1.0]), k_min=0, k_max=4)

    def test_k_max_lt_k_min_raises(self):
        with pytest.raises(ValueError, match="k_max"):
            surprise_to_k(torch.tensor([1.0]), k_min=4, k_max=2)

    def test_bad_steepness_raises(self):
        with pytest.raises(ValueError, match="steepness"):
            surprise_to_k(torch.tensor([1.0]), k_min=1, k_max=4, steepness=0.0)


# ──────────────────────────────────────────────────────────────────────
# 2. load_balance_loss
# ──────────────────────────────────────────────────────────────────────


class TestLoadBalanceLoss:
    def test_uniform_routing_yields_one(self):
        """Perfectly uniform gating + uniform routing → loss = N · Σ (1/N)² = 1."""
        N = 8
        gates = torch.full((1024, N), 1.0 / N)
        # Top-1 routing distributed uniformly.
        top_idx = torch.arange(1024).reshape(-1, 1) % N
        loss = load_balance_loss(gates, top_idx, n_experts=N)
        # Each f_e = 1/N (uniform routing), each P_e = 1/N → loss = N · N · (1/N²) = 1.
        assert abs(loss.item() - 1.0) < 1e-3

    def test_concentrated_routing_increases_loss(self):
        """Routing everything to expert 0 should give a high loss."""
        N = 4
        gates = torch.zeros(100, N)
        gates[:, 0] = 1.0
        top_idx = torch.zeros(100, 1, dtype=torch.long)
        loss = load_balance_loss(gates, top_idx, n_experts=N)
        # f_0=1, others 0; P_0=1, others 0 → loss = N · 1 · 1 = N.
        assert abs(loss.item() - float(N)) < 1e-4

    def test_empty_input_returns_zero(self):
        gates = torch.empty(0, 4)
        top_idx = torch.empty(0, 1, dtype=torch.long)
        loss = load_balance_loss(gates, top_idx, n_experts=4)
        assert loss.item() == 0.0

    def test_bad_input_shape_raises(self):
        with pytest.raises(ValueError, match="2D tensors"):
            load_balance_loss(torch.randn(4), torch.zeros(4, dtype=torch.long), n_experts=2)


# ──────────────────────────────────────────────────────────────────────
# 3. SurpriseGatedMoE module
# ──────────────────────────────────────────────────────────────────────


class TestSurpriseGatedMoEModule:
    def test_forward_preserves_shape(self):
        mod = SurpriseGatedMoE(d_model=16, n_experts=4, k_min=1, k_max=2)
        x = torch.randn(2, 5, 16)
        y = mod(x)
        assert y.shape == x.shape

    def test_high_surprise_increases_mean_k(self):
        torch.manual_seed(0)
        mod = SurpriseGatedMoE(
            d_model=16, n_experts=4, k_min=1, k_max=4,
            midpoint=1.0, steepness=4.0,
        )
        x = torch.randn(1, 10, 16)
        # Low surprise everywhere.
        _ = mod(x, surprise=torch.zeros(1, 10))
        k_lo = mod.last_mean_k.item()
        # High surprise everywhere.
        _ = mod(x, surprise=torch.full((1, 10), 10.0))
        k_hi = mod.last_mean_k.item()
        assert k_hi > k_lo, (
            f"high surprise didn't increase mean_k: low={k_lo:.2f}, "
            f"high={k_hi:.2f}"
        )

    def test_internal_surprise_works_when_omitted(self):
        torch.manual_seed(1)
        mod = SurpriseGatedMoE(d_model=16, n_experts=4, k_min=1, k_max=3)
        x = torch.randn(2, 4, 16)
        y = mod(x)  # no explicit surprise
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        # mean_k must be in [k_min, k_max].
        k = mod.last_mean_k.item()
        assert 1.0 <= k <= 3.0

    def test_aux_loss_exposed(self):
        mod = SurpriseGatedMoE(d_model=16, n_experts=4)
        x = torch.randn(2, 4, 16)
        _ = mod(x)
        # Must be non-negative and finite.
        aux = mod.last_aux_loss.item()
        assert aux >= 0.0
        assert torch.isfinite(mod.last_aux_loss).all()

    def test_parameters_receive_gradients(self):
        torch.manual_seed(2)
        mod = SurpriseGatedMoE(d_model=8, n_experts=4, k_min=1, k_max=2)
        x = torch.randn(1, 6, 8, requires_grad=True)
        loss = mod(x).pow(2).sum()
        loss.backward()
        # Gate must always get grads (every expert weighted softly).
        assert mod.gate.weight.grad is not None
        assert mod.gate.weight.grad.abs().sum() > 0
        # Every expert that was hit by at least one token must get
        # grads. With k=2 over 6 tokens and 4 experts, at least 3
        # experts will be active in expectation; require ≥1.
        active_experts = sum(
            1 for e in mod.experts
            if e.fc1.weight.grad is not None
            and e.fc1.weight.grad.abs().sum() > 0
        )
        assert active_experts >= 1, "no experts received gradients"

    def test_wrong_input_dim_raises(self):
        mod = SurpriseGatedMoE(d_model=16, n_experts=4)
        with pytest.raises(ValueError, match=r"\(B, T, D\)"):
            mod(torch.randn(2, 16))

    def test_bad_surprise_shape_raises(self):
        mod = SurpriseGatedMoE(d_model=16, n_experts=4)
        x = torch.randn(2, 4, 16)
        with pytest.raises(ValueError, match=r"\(B, T\)"):
            mod(x, surprise=torch.randn(2, 4, 16))

    def test_k_max_gt_n_experts_raises(self):
        with pytest.raises(ValueError, match="k_max"):
            SurpriseGatedMoE(d_model=16, n_experts=4, k_min=1, k_max=8)

    def test_zero_n_experts_raises(self):
        with pytest.raises(ValueError, match="n_experts"):
            SurpriseGatedMoE(d_model=16, n_experts=0)
