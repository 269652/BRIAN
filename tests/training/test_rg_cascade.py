# -*- coding: utf-8 -*-
"""TDD: RenormalizationGroupCascade — Kolmogorov turbulence-inspired multi-scale attention.

Mathematical contracts verified:
  1. Coarse-graining: mean pooling over blocks of size 2^g (no information gain)
  2. Fluctuation extraction: δH = H - upsample(H_coarse) has zero block-mean
  3. Perfect reconstruction: upsample(H_coarse) + δH == H  (algebraic identity)
  4. Kolmogorov coupling: λ_g ∝ 2^{-5g/6} (5/3-law from turbulence theory)
  5. Output shape: same as input (B, T, d)
  6. Padding: handles T not divisible by block_size via truncation
  7. RG cascade forward: combines all scales (sum of all scale-weighted residuals)

Run:  brian test tests/training/test_rg_cascade.py
"""
from __future__ import annotations

import math
import torch
import pytest


# ── import target ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rg_cls():
    from neuroslm.emergent.semantic_turbulence import RenormalizationGroupCascade
    return RenormalizationGroupCascade


# ── helpers ───────────────────────────────────────────────────────────────


def _ones_signal(B=2, T=16, d=32, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, T, d)


# ── Kolmogorov coupling ───────────────────────────────────────────────────


class TestKolmogorovCoupling:
    """λ_g must follow the 5/3-law: λ_g = λ_0 · 2^{-5g/6}."""

    def test_kolmogorov_ratios(self, rg_cls):
        n_groups = 4
        rg = rg_cls(d_model=32, n_groups=n_groups, kolmogorov_init=True)
        lambdas = rg.kolmogorov_lambdas()  # returns list of n_groups floats
        assert len(lambdas) == n_groups
        expected_ratio = 2 ** (-5 / 6)
        for g in range(1, n_groups):
            ratio = lambdas[g] / lambdas[g - 1]
            assert abs(ratio - expected_ratio) < 1e-5, (
                f"λ_{g}/λ_{g-1} = {ratio:.6f}, expected {expected_ratio:.6f}"
            )

    def test_lambdas_are_positive(self, rg_cls):
        rg = rg_cls(d_model=32, n_groups=3, kolmogorov_init=True)
        for lam in rg.kolmogorov_lambdas():
            assert lam > 0

    def test_uniform_init_when_kolmogorov_false(self, rg_cls):
        rg = rg_cls(d_model=32, n_groups=3, kolmogorov_init=False)
        lambdas = rg.kolmogorov_lambdas()
        # All equal (or close) when Kolmogorov disabled
        assert max(lambdas) - min(lambdas) < 1e-5


# ── Coarse-graining ───────────────────────────────────────────────────────


class TestCoarseGraining:
    """coarse_grain(H, block_size) must return block-level mean."""

    def test_output_shape(self, rg_cls):
        rg = rg_cls(d_model=32, n_groups=3)
        H = torch.randn(2, 16, 32)
        H_c = rg.coarse_grain(H, block_size=2)
        assert H_c.shape == (2, 8, 32)

    def test_mean_semantics(self, rg_cls):
        rg = rg_cls(d_model=32, n_groups=3)
        H = torch.zeros(1, 4, 8)
        H[0, 0] = 1.0
        H[0, 1] = 3.0
        # block 0 = tokens [0,1], mean = 2.0
        H_c = rg.coarse_grain(H, block_size=2)
        assert H_c.shape == (1, 2, 8)
        assert abs(H_c[0, 0, :].mean().item() - 2.0) < 1e-5

    def test_energy_non_expansion(self, rg_cls):
        """‖H_coarse‖_F ≤ ‖H‖_F (coarse-graining can only lose energy)."""
        rg = rg_cls(d_model=32, n_groups=3)
        H = _ones_signal(T=16)
        H_c = rg.coarse_grain(H, block_size=2)
        H_c_up = H_c.repeat_interleave(2, dim=1)
        assert H_c_up.norm().item() <= H.norm().item() + 1e-5

    def test_padding_truncates(self, rg_cls):
        """T=15 with block_size=4 → 3 complete blocks (T_eff=12)."""
        rg = rg_cls(d_model=32, n_groups=3)
        H = torch.randn(1, 15, 32)
        H_c = rg.coarse_grain(H, block_size=4)
        assert H_c.shape == (1, 3, 32)


# ── Fluctuation extraction ────────────────────────────────────────────────


class TestFluctuationExtraction:
    """extract_fluctuations(H, H_coarse) must return δH = H - upsample(H_coarse)."""

    def test_zero_mean_within_blocks(self, rg_cls):
        """Block-mean of δH should be zero by construction."""
        rg = rg_cls(d_model=32, n_groups=3)
        H = _ones_signal(T=16)
        H_c = rg.coarse_grain(H, block_size=2)
        dH, H_repeat = rg.extract_fluctuations(H, H_c, block_size=2)
        # Reshape to (B, n_blocks, block_size, d) and check block-mean ≈ 0
        dH_blocks = dH.unfold(1, 2, 2)  # (B, 8, d, 2)
        block_means = dH_blocks.mean(dim=-1)  # (B, 8, d)
        assert block_means.abs().max().item() < 1e-5

    def test_perfect_reconstruction(self, rg_cls):
        """H_repeat + δH must equal the original H (exact for complete blocks)."""
        rg = rg_cls(d_model=32, n_groups=3)
        H = _ones_signal(T=16)
        H_c = rg.coarse_grain(H, block_size=2)
        dH, H_repeat = rg.extract_fluctuations(H, H_c, block_size=2)
        reconstructed = H_repeat + dH
        torch.testing.assert_close(reconstructed, H, rtol=1e-5, atol=1e-5)

    def test_shapes_consistent(self, rg_cls):
        rg = rg_cls(d_model=32, n_groups=3)
        H = _ones_signal(B=2, T=16, d=32)
        H_c = rg.coarse_grain(H, block_size=4)
        dH, H_repeat = rg.extract_fluctuations(H, H_c, block_size=4)
        assert dH.shape == H_repeat.shape == (2, 16, 32)


# ── Full forward pass ─────────────────────────────────────────────────────


class TestRGCascadeForward:
    """Forward pass must return same shape as input and be differentiable."""

    def test_output_shape(self, rg_cls):
        rg = rg_cls(d_model=32, n_groups=3)
        H = _ones_signal(B=2, T=16, d=32)
        out = rg(H)
        assert out.shape == H.shape, f"Expected {H.shape}, got {out.shape}"

    def test_differentiable(self, rg_cls):
        rg = rg_cls(d_model=32, n_groups=3, kolmogorov_init=True)
        H = _ones_signal(B=2, T=16, d=32).requires_grad_(True)
        out = rg(H)
        loss = out.sum()
        loss.backward()
        assert H.grad is not None

    def test_output_not_identical_to_input(self, rg_cls):
        """RG cascade must modify the representation (not be identity)."""
        torch.manual_seed(42)
        rg = rg_cls(d_model=32, n_groups=3, kolmogorov_init=True)
        # Initialize with non-zero weights
        for p in rg.parameters():
            torch.nn.init.normal_(p, std=0.1)
        H = _ones_signal(B=2, T=16, d=32)
        out = rg(H)
        assert not torch.allclose(out, H, atol=1e-6), (
            "RG cascade output was identical to input — cascade has no effect"
        )

    def test_consistent_across_groups(self, rg_cls):
        """n_groups=1 and n_groups=3 produce different enrichments."""
        H = _ones_signal(B=2, T=16, d=32, seed=7)
        rg1 = rg_cls(d_model=32, n_groups=1)
        rg3 = rg_cls(d_model=32, n_groups=3)
        for p in rg1.parameters():
            torch.nn.init.normal_(p, std=0.1)
        for p in rg3.parameters():
            torch.nn.init.normal_(p, std=0.1)
        out1 = rg1(H)
        out3 = rg3(H)
        assert not torch.allclose(out1, out3, atol=1e-6)

    def test_no_nan_inf(self, rg_cls):
        rg = rg_cls(d_model=64, n_groups=4, kolmogorov_init=True)
        H = _ones_signal(B=4, T=32, d=64)
        out = rg(H)
        assert torch.isfinite(out).all(), "RG cascade produced NaN/Inf"

    def test_t_not_power_of_two(self, rg_cls):
        """T=13 (not divisible by 2^3=8) must not crash."""
        rg = rg_cls(d_model=32, n_groups=3)
        H = torch.randn(1, 13, 32)
        out = rg(H)
        assert out.shape[0] == 1
        assert out.shape[2] == 32
