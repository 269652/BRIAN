"""Math contracts for RoPE-on-a-Torus.

Pins the algebraic + geometric properties required by §14:

  * Period schedules return the expected lengths and orderings.
  * cos/sin tables wrap modulo each slice's period (the "torus" claim).
  * Rotation preserves norms (it's an O(2) action on each slice).
  * Relative position emerges: <RoPE(q, m), RoPE(k, n)> depends only
    on (n - m) inside one slice's period.
  * The RoPETorus module matches the functional implementation.
  * Learnable periods receive gradients.

References: Su et al., "RoFormer" (Neurocomputing 2024).
"""

from __future__ import annotations

import math

import pytest
import torch

from neuroslm.modules.rope_torus import (
    RoPETorus,
    apply_rope_torus,
    build_torus_cos_sin,
    build_torus_periods,
)


# ──────────────────────────────────────────────────────────────────────
# 1. Period schedule shapes + orderings
# ──────────────────────────────────────────────────────────────────────


class TestPeriodSchedules:
    def test_geometric_returns_right_length(self):
        p = build_torus_periods(8, base=10000.0, schedule="geometric")
        assert p.shape == (8,)

    def test_geometric_is_monotonically_non_decreasing(self):
        p = build_torus_periods(16, base=10000.0, schedule="geometric")
        # Classical RoPE: low-index slices rotate fast (short period),
        # high-index slices rotate slow (long period).
        diffs = p[1:] - p[:-1]
        assert (diffs >= -1e-5).all(), (
            "geometric schedule must be non-decreasing in slice index"
        )

    def test_linear_is_uniformly_spaced(self):
        p = build_torus_periods(8, base=100.0, schedule="linear")
        # Differences between consecutive periods are constant.
        diffs = p[1:] - p[:-1]
        spread = diffs.max() - diffs.min()
        assert spread < 1e-5

    def test_harmonic_decreases(self):
        p = build_torus_periods(8, base=100.0, schedule="harmonic")
        diffs = p[1:] - p[:-1]
        assert (diffs < 0).all(), "harmonic schedule must decrease"

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="unknown schedule"):
            build_torus_periods(8, schedule="banana")

    def test_zero_pairs_raises(self):
        with pytest.raises(ValueError, match="positive"):
            build_torus_periods(0)


# ──────────────────────────────────────────────────────────────────────
# 2. cos/sin tables — the torus wrap-around
# ──────────────────────────────────────────────────────────────────────


class TestCosSinTables:
    def test_table_shapes(self):
        periods = build_torus_periods(4, schedule="linear", base=8.0)
        cos, sin = build_torus_cos_sin(seq_len=10, periods=periods)
        assert cos.shape == (10, 4)
        assert sin.shape == (10, 4)

    def test_position_zero_is_identity_rotation(self):
        periods = build_torus_periods(4, schedule="geometric")
        cos, sin = build_torus_cos_sin(seq_len=5, periods=periods)
        # θ(0) = 0 → cos=1, sin=0 for every slice.
        assert torch.allclose(cos[0], torch.ones(4))
        assert torch.allclose(sin[0], torch.zeros(4), atol=1e-6)

    def test_wraps_modulo_period(self):
        # Pick a small integer period so the wrap is obvious.
        periods = torch.tensor([4.0, 8.0])
        cos, sin = build_torus_cos_sin(seq_len=16, periods=periods)
        # Slice 0: period 4 → cos at position 0 == cos at position 4.
        assert torch.allclose(cos[0, 0], cos[4, 0], atol=1e-6)
        assert torch.allclose(sin[0, 0], sin[4, 0], atol=1e-6)
        # Slice 1: period 8 → cos at position 0 == cos at position 8.
        assert torch.allclose(cos[0, 1], cos[8, 1], atol=1e-6)
        assert torch.allclose(sin[0, 1], sin[8, 1], atol=1e-6)

    def test_quarter_period_is_quarter_turn(self):
        periods = torch.tensor([4.0])
        cos, sin = build_torus_cos_sin(seq_len=4, periods=periods)
        # At p=1, θ = 2π·(1/4) = π/2 → cos=0, sin=1.
        assert cos[1, 0].abs() < 1e-6
        assert (sin[1, 0] - 1.0).abs() < 1e-6


# ──────────────────────────────────────────────────────────────────────
# 3. apply_rope_torus — geometric properties
# ──────────────────────────────────────────────────────────────────────


class TestApplyRoPETorus:
    def test_preserves_norm(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, 8)  # (B, T, D)
        periods = build_torus_periods(4, schedule="geometric")
        cos, sin = build_torus_cos_sin(5, periods)
        y = apply_rope_torus(x, cos, sin)
        # Each 2D slice is rotated by an O(2) action → preserves the
        # per-pair norm, hence the whole vector's norm.
        n_x = x.norm(dim=-1)
        n_y = y.norm(dim=-1)
        assert torch.allclose(n_x, n_y, atol=1e-5)

    def test_position_zero_is_identity(self):
        torch.manual_seed(1)
        x = torch.randn(1, 3, 16)
        periods = build_torus_periods(8, schedule="geometric")
        cos, sin = build_torus_cos_sin(3, periods)
        y = apply_rope_torus(x, cos, sin)
        # Token 0 is unrotated (θ=0).
        assert torch.allclose(y[:, 0], x[:, 0], atol=1e-6)

    def test_relative_position_inner_product(self):
        """Classical RoPE relative-position property, restricted to
        one slice: <RoPE(q, m), RoPE(k, n)> on slice j depends only on
        (n - m) inside that slice's period."""
        torch.manual_seed(2)
        q = torch.randn(1, 1, 2)  # single slice
        k = torch.randn(1, 1, 2)
        periods = torch.tensor([16.0])
        # Build a long table so we can pick arbitrary positions.
        cos, sin = build_torus_cos_sin(32, periods)

        def rotated(x, p):
            c = cos[p : p + 1]
            s = sin[p : p + 1]
            return apply_rope_torus(x, c, s)

        ip_a = (rotated(q, 3) * rotated(k, 7)).sum(dim=-1)
        ip_b = (rotated(q, 10) * rotated(k, 14)).sum(dim=-1)
        # Both have relative shift = 4; inner products must agree.
        assert torch.allclose(ip_a, ip_b, atol=1e-5)

    def test_odd_dim_raises(self):
        x = torch.randn(1, 4, 5)
        periods = build_torus_periods(2)
        cos, sin = build_torus_cos_sin(4, periods)
        with pytest.raises(ValueError, match="even"):
            apply_rope_torus(x, cos, sin)

    def test_works_on_4d_per_head_shape(self):
        # (B, H, T, D_h) — the multi-head attention layout.
        x = torch.randn(2, 4, 6, 8)
        periods = build_torus_periods(4)
        cos, sin = build_torus_cos_sin(6, periods)
        y = apply_rope_torus(x, cos, sin)
        assert y.shape == x.shape
        assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


# ──────────────────────────────────────────────────────────────────────
# 4. RoPETorus nn.Module wrapper
# ──────────────────────────────────────────────────────────────────────


class TestRoPETorusModule:
    def test_module_forward_matches_functional(self):
        torch.manual_seed(3)
        mod = RoPETorus(d_model=8, max_seq_len=10, schedule="linear", base=16.0)
        x = torch.randn(2, 7, 8)
        y_mod = mod(x)
        # Reproduce via the functional API.
        periods = build_torus_periods(4, base=16.0, schedule="linear")
        cos, sin = build_torus_cos_sin(7, periods)
        y_fn = apply_rope_torus(x, cos, sin)
        assert torch.allclose(y_mod, y_fn, atol=1e-5)

    def test_seq_len_exceeds_cache_recomputes_correctly(self):
        mod = RoPETorus(d_model=4, max_seq_len=4)
        x = torch.randn(1, 12, 4)  # exceeds cache
        y = mod(x)
        assert y.shape == x.shape
        # And norm preservation still holds.
        assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)

    def test_learnable_periods_have_gradient(self):
        mod = RoPETorus(d_model=8, max_seq_len=10, learnable_periods=True)
        x = torch.randn(1, 5, 8, requires_grad=True)
        y = mod(x).sum()
        y.backward()
        assert mod.log_periods.grad is not None
        assert mod.log_periods.grad.abs().sum() > 0

    def test_fixed_periods_create_no_parameters(self):
        mod = RoPETorus(d_model=8, max_seq_len=10, learnable_periods=False)
        # No nn.Parameter at all — just buffers.
        assert sum(p.numel() for p in mod.parameters()) == 0
