# -*- coding: utf-8 -*-
"""RED tests for GIF-5: Attention Head Diversity Loss.

Mathematical specification:
    L_div = (2 / H(H-1)) Σ_{i<j} max(0, cos(q_i, q_j))²

where q_i ∈ R^{T·d_head} is head i's query vector (flattened over the
sequence dimension). Only positive cosine similarity is penalized —
anti-correlated heads are left alone. Squaring gives stronger gradient
when heads converge.

Gap-reactive weighting:
    w_div = w₀ · log(1 + gap_ratio / target)

Same Riemannian log-softening as GIF-1/GIF-3.  Automatically lifts off
zero when the gap exceeds target, compresses unbounded gap ratios.
"""
import math
import pytest
import torch

from neuroslm.emergent.gif import GIFController
from neuroslm.dsl import nn_ops


# ══════════════════════════════════════════════════════════════════════
# 1. GIFController.head_diversity_weight property
# ══════════════════════════════════════════════════════════════════════

class TestHeadDiversityWeight:
    """Pin the gap-reactive weighting formula for diversity loss."""

    def _make_ctrl(self, attn_div_weight=0.01, target=1.5):
        return GIFController(
            enabled=True, adaptive=True,
            target_gap_ratio=target,
            attn_div_weight=attn_div_weight,
        )

    def test_zero_when_no_gap_data(self):
        """Before any OOD eval, gap_ratio=0 → w_div=0."""
        ctrl = self._make_ctrl()
        assert ctrl.head_diversity_weight == pytest.approx(0.0)

    def test_zero_when_weight_is_zero(self):
        """attn_div_weight=0 disables the mechanism."""
        ctrl = self._make_ctrl(attn_div_weight=0.0)
        ctrl._last_gap_ratio = 3.0
        assert ctrl.head_diversity_weight == pytest.approx(0.0)

    def test_zero_when_not_adaptive(self):
        """Non-adaptive GIF → diversity weight always 0."""
        ctrl = GIFController(
            enabled=True, adaptive=False,
            attn_div_weight=0.01,
            target_gap_ratio=1.5,
        )
        ctrl._last_gap_ratio = 5.0
        assert ctrl.head_diversity_weight == pytest.approx(0.0)

    def test_log_softening_at_target(self):
        """gap=target → w = w₀·log(1 + 1) = w₀·ln(2)."""
        ctrl = self._make_ctrl(attn_div_weight=0.01, target=1.5)
        ctrl._last_gap_ratio = 1.5
        expected = 0.01 * math.log(1 + 1.5 / 1.5)
        assert ctrl.head_diversity_weight == pytest.approx(expected, rel=1e-6)

    def test_log_softening_at_double_target(self):
        """gap=2×target → w = w₀·log(1 + 2)."""
        ctrl = self._make_ctrl(attn_div_weight=0.01, target=1.5)
        ctrl._last_gap_ratio = 3.0
        expected = 0.01 * math.log(1 + 3.0 / 1.5)
        assert ctrl.head_diversity_weight == pytest.approx(expected, rel=1e-6)

    def test_monotonically_increasing_with_gap(self):
        """As gap increases, w_div increases."""
        ctrl = self._make_ctrl(attn_div_weight=0.01, target=1.5)
        weights = []
        for gap in [1.5, 2.0, 3.0, 5.0, 10.0]:
            ctrl._last_gap_ratio = gap
            weights.append(ctrl.head_diversity_weight)
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1]

    def test_sublinear_growth(self):
        """Log-softening makes growth sublinear (compresses large gaps)."""
        ctrl = self._make_ctrl(attn_div_weight=0.01, target=1.5)
        ctrl._last_gap_ratio = 3.0
        w_at_3 = ctrl.head_diversity_weight
        ctrl._last_gap_ratio = 6.0
        w_at_6 = ctrl.head_diversity_weight
        # Doubling the gap should NOT double the weight (sublinear)
        assert w_at_6 < 2 * w_at_3

    def test_from_config_parses_attn_div_weight(self):
        """from_config extracts attn_div_weight from gif dict."""
        class FakeCfg:
            gif = {
                "enabled": True,
                "adaptive": True,
                "target_gap_ratio": 1.5,
                "attn_div_weight": 0.02,
            }
        ctrl = GIFController.from_config(FakeCfg())
        assert ctrl.attn_div_weight == pytest.approx(0.02)

    def test_from_config_defaults_to_zero(self):
        """When attn_div_weight is absent, defaults to 0 (disabled)."""
        class FakeCfg:
            gif = {
                "enabled": True,
                "adaptive": True,
                "target_gap_ratio": 1.5,
            }
        ctrl = GIFController.from_config(FakeCfg())
        assert ctrl.attn_div_weight == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════
# 2. nn_ops diversity stash mechanism
# ══════════════════════════════════════════════════════════════════════

class TestDiversityStash:
    """Verify that nn_ops stashes Q tensors when activated."""

    def test_stash_is_none_by_default(self):
        """No stashing when _DIVERSITY_STASH is None."""
        assert nn_ops._DIVERSITY_STASH is None

    def test_stash_collects_q_tensors(self):
        """When list is set, causal_self_attention appends Q."""
        torch.manual_seed(42)
        B, T, D = 2, 16, 64
        n_heads, n_kv_heads = 4, 4
        x = torch.randn(B, T, D)
        Wq = torch.randn(D, D)
        Wkv = torch.randn(2 * n_kv_heads * (D // n_heads), D)
        Wo = torch.randn(D, D)

        nn_ops._DIVERSITY_STASH = []
        try:
            nn_ops.causal_self_attention(
                x, Wq, Wkv, Wo, n_heads, n_kv_heads, max_ctx=T
            )
            assert len(nn_ops._DIVERSITY_STASH) == 1
            q = nn_ops._DIVERSITY_STASH[0]
            assert q.shape == (B, n_heads, T, D // n_heads)
        finally:
            nn_ops._DIVERSITY_STASH = None

    def test_stash_accumulates_across_layers(self):
        """Multiple calls append multiple Q tensors (one per layer)."""
        torch.manual_seed(42)
        B, T, D = 2, 16, 64
        n_heads, n_kv_heads = 4, 4
        x = torch.randn(B, T, D)
        Wq = torch.randn(D, D)
        Wkv = torch.randn(2 * n_kv_heads * (D // n_heads), D)
        Wo = torch.randn(D, D)

        nn_ops._DIVERSITY_STASH = []
        try:
            for _ in range(3):  # simulate 3 layers
                nn_ops.causal_self_attention(
                    x, Wq, Wkv, Wo, n_heads, n_kv_heads, max_ctx=T
                )
            assert len(nn_ops._DIVERSITY_STASH) == 3
        finally:
            nn_ops._DIVERSITY_STASH = None

    def test_stash_preserves_grad(self):
        """Stashed Q tensors retain gradient connectivity."""
        torch.manual_seed(42)
        B, T, D = 2, 16, 64
        n_heads, n_kv_heads = 4, 4
        x = torch.randn(B, T, D, requires_grad=True)
        Wq = torch.randn(D, D)
        Wkv = torch.randn(2 * n_kv_heads * (D // n_heads), D)
        Wo = torch.randn(D, D)

        nn_ops._DIVERSITY_STASH = []
        try:
            nn_ops.causal_self_attention(
                x, Wq, Wkv, Wo, n_heads, n_kv_heads, max_ctx=T
            )
            q = nn_ops._DIVERSITY_STASH[0]
            assert q.requires_grad
        finally:
            nn_ops._DIVERSITY_STASH = None


# ══════════════════════════════════════════════════════════════════════
# 3. Diversity loss computation: compute_head_diversity_loss
# ══════════════════════════════════════════════════════════════════════

class TestComputeHeadDiversityLoss:
    """Pin the mathematical formula: L = (2/H(H-1)) Σ_{i<j} max(0,cos)²."""

    def test_zero_for_orthogonal_heads(self):
        """When heads produce orthogonal queries, L_div = 0."""
        from neuroslm.emergent.gif import compute_head_diversity_loss
        # 4 heads, T=8, d_head=4 — make orthogonal via identity blocks
        H, T, d_head = 4, 8, 4
        q = torch.zeros(1, H, T, d_head)
        # Each head projects only one d_head dimension → orthogonal
        for h in range(H):
            q[0, h, :, h % d_head] = 1.0
        loss = compute_head_diversity_loss([q])
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_maximal_for_identical_heads(self):
        """When all heads are identical, L_div = 1.0 (all pairs cos=1)."""
        from neuroslm.emergent.gif import compute_head_diversity_loss
        H, T, d_head = 4, 8, 4
        single = torch.randn(1, 1, T, d_head)
        q = single.expand(1, H, T, d_head).contiguous()
        loss = compute_head_diversity_loss([q])
        # All pairs have cos=1, max(0,1)²=1, mean over pairs = 1.0
        assert loss.item() == pytest.approx(1.0, rel=1e-4)

    def test_intermediate_for_partial_overlap(self):
        """Partially overlapping heads → 0 < L_div < 1."""
        from neuroslm.emergent.gif import compute_head_diversity_loss
        H, T, d_head = 4, 8, 4
        torch.manual_seed(7)
        q = torch.randn(1, H, T, d_head)
        loss = compute_head_diversity_loss([q])
        assert 0.0 < loss.item() < 1.0

    def test_ignores_anti_correlated_heads(self):
        """Negatively correlated heads (cos < 0) should contribute 0."""
        from neuroslm.emergent.gif import compute_head_diversity_loss
        H, T, d_head = 2, 8, 4
        base = torch.randn(1, 1, T, d_head)
        q = torch.cat([base, -base], dim=1)  # cos = -1
        loss = compute_head_diversity_loss([q])
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_averages_across_layers(self):
        """With multiple layers, loss is averaged across layers."""
        from neuroslm.emergent.gif import compute_head_diversity_loss
        H, T, d_head = 4, 8, 4
        # Layer 1: identical heads → loss = 1.0
        q1 = torch.randn(1, 1, T, d_head).expand(1, H, T, d_head).contiguous()
        # Layer 2: orthogonal heads → loss = 0.0
        q2 = torch.zeros(1, H, T, d_head)
        for h in range(H):
            q2[0, h, :, h % d_head] = 1.0
        loss = compute_head_diversity_loss([q1, q2])
        # Mean of [1.0, 0.0] = 0.5
        assert loss.item() == pytest.approx(0.5, rel=1e-3)

    def test_empty_stash_returns_zero(self):
        """Empty Q list → zero loss (no-op)."""
        from neuroslm.emergent.gif import compute_head_diversity_loss
        loss = compute_head_diversity_loss([])
        assert loss.item() == pytest.approx(0.0)

    def test_single_head_returns_zero(self):
        """With H=1, no pairs exist → loss = 0."""
        from neuroslm.emergent.gif import compute_head_diversity_loss
        q = torch.randn(1, 1, 8, 4)  # 1 head
        loss = compute_head_diversity_loss([q])
        assert loss.item() == pytest.approx(0.0)

    def test_grad_flows_through_loss(self):
        """Loss is differentiable w.r.t. Q."""
        from neuroslm.emergent.gif import compute_head_diversity_loss
        q = torch.randn(1, 4, 8, 4, requires_grad=True)
        loss = compute_head_diversity_loss([q])
        loss.backward()
        assert q.grad is not None
        assert q.grad.abs().sum() > 0

    def test_batch_invariant(self):
        """Loss should be the same regardless of batch size."""
        from neuroslm.emergent.gif import compute_head_diversity_loss
        torch.manual_seed(42)
        # Same data replicated across batch dim
        q_single = torch.randn(1, 4, 8, 4)
        q_batch = q_single.expand(4, -1, -1, -1).contiguous()
        l1 = compute_head_diversity_loss([q_single])
        l2 = compute_head_diversity_loss([q_batch])
        assert l1.item() == pytest.approx(l2.item(), rel=1e-4)


# ══════════════════════════════════════════════════════════════════════
# 4. Integration: arch.neuro → GIFController → live weight
# ══════════════════════════════════════════════════════════════════════

class TestSmolLMDiversityConfig:
    """Pin: SmolLM arch declares attention diversity weight."""

    def test_arch_has_attn_div_weight(self):
        from pathlib import Path
        from neuroslm.dsl.training_config import load_training_config_from_arch
        arch_root = Path(__file__).resolve().parents[2] / "architectures" / "SmolLM"
        if not (arch_root / "arch.neuro").is_file():
            pytest.skip("SmolLM arch not present")
        cfg = load_training_config_from_arch(arch_root)
        ctrl = GIFController.from_config(cfg)
        assert ctrl.attn_div_weight > 0, (
            "SmolLM GIF must have attn_div_weight > 0 — check arch.neuro"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
