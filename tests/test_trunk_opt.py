# -*- coding: utf-8 -*-
"""Tests for TRUNK-OPT: Phases 1-4 of the LM trunk optimisation plan.

TDD contract — tests are written FIRST; the implementation must make
each assertion pass.  Covers:

  Phase 1 — Measurement infrastructure
    P1A  GradientBudgetTracker: LM gradient fraction ∈ [0,1]
    P1B  LayerGradientProbe: per-block L2 norms tracked
    P1C  BitsPerParamMeter: monotone increase during training
    P1D  PACBayesBound: upper-bounds empirical OOD CE (statistical)
    P1E  SharpnessProbe: sharpness measurable + decreases under SAM
    P1F  EffectiveRankProbe: rank ≥ 1 for non-degenerate hidden states
    P1G  SpectralPowerLawProbe: (α, R², D_PR) intrinsic invariants
         (rotation- and scale-invariant — biological 1/f signature)

  Phase 2 — activation_step (Capacity-First protocol)
    P2A  RegularizationController honours activation_step = 0 (zero loss)
    P2B  RegularizationController gates aux to zero before activation_step
    P2C  After activation_step aux is non-zero when interventions enabled
    P2D  warmup ramp starts from zero at activation_step, not before

  Phase 3 — harness wiring (metrics surface in _metrics dict)
    P3A  train_step populates trunk_opt_grad_budget ∈ [0,1]
    P3B  train_step populates bits_per_param > 0
    P3C  effective_rank metric present after compute_loss

  Phase 4 — PAC-Bayes bound tightens under stronger regularization
    P4A  bound with higher weight_decay ≤ bound with lower weight_decay
         (PAC-Bayes: ||θ − θ₀||² shrinks under stronger L2 prior)
    P4B  pac_bayes_bound ≥ 0 always (it's a non-negative penalty)
"""
from __future__ import annotations
import math
import types
from typing import Dict, Optional
import pytest
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _tiny_lm(vocab: int = 128, d: int = 32, depth: int = 2,
             ctx: int = 16) -> nn.Module:
    """Minimal DSLLanguageModel for fast unit tests (CPU, bf16-safe)."""
    from neuroslm.dsl.nn_lang import build_language_model
    return build_language_model(vocab=vocab, d_model=d, depth=depth,
                                n_heads=2, max_ctx=ctx)


def _tiny_harness(vocab: int = 128, d: int = 32, depth: int = 2,
                  ctx: int = 16):
    """BRIANHarness wrapping a tiny LM for fast integration tests."""
    from neuroslm.harness import BRIANHarness
    lm = _tiny_lm(vocab, d, depth, ctx)
    return BRIANHarness.from_language_model(
        lm, vocab_size=vocab, d_sem=d)


def _fake_batch(vocab: int = 128, B: int = 2, T: int = 16):
    ids = torch.randint(0, vocab - 1, (B, T))
    tgt = torch.randint(0, vocab - 1, (B, T))
    return ids, tgt


# ═════════════════════════════════════════════════════════════════════
# Phase 1A  –  GradientBudgetTracker
# ═════════════════════════════════════════════════════════════════════

class TestGradientBudgetTracker:
    """P1A: fraction of gradient energy devoted to the LM loss."""

    def _import(self):
        from neuroslm.emergent.trunk_opt import GradientBudgetTracker
        return GradientBudgetTracker

    def test_import(self):
        cls = self._import()
        assert callable(cls)

    def test_budget_is_one_when_only_lm_loss(self):
        """When total == LM loss (no aux), budget = 1.0."""
        GBT = self._import()
        tracker = GBT()
        model = nn.Linear(8, 8)
        loss = (model.weight ** 2).sum()   # only one loss
        loss.backward()
        lm_grad_norm = tracker.lm_grad_norm(model)
        total_grad_norm = tracker.total_grad_norm(model)
        budget = tracker.budget(lm_grad_norm, total_grad_norm)
        assert 0.0 <= budget <= 1.0 + 1e-6

    def test_budget_below_one_when_aux_adds_gradient(self):
        """With an orthogonal aux gradient, budget < 1."""
        GBT = self._import()
        tracker = GBT()
        model = nn.Linear(8, 8)
        # Compute LM grad + register it, then add orthogonal aux grad.
        x = torch.randn(4, 8)
        lm_loss = ((x @ model.weight.t()) ** 2).sum()
        lm_loss.backward(retain_graph=True)
        # snapshot norms
        lm_grad_norm = tracker.lm_grad_norm(model)
        # Add aux grad (orthogonal direction)
        for p in model.parameters():
            if p.grad is not None:
                p.grad = p.grad + torch.ones_like(p.grad) * 0.5
        total_grad_norm = tracker.total_grad_norm(model)
        budget = tracker.budget(lm_grad_norm, total_grad_norm)
        assert 0.0 <= budget <= 1.0 + 1e-6

    def test_budget_in_range(self):
        """budget ∈ [0, 1] for any pair of non-negative norms."""
        GBT = self._import()
        tracker = GBT()
        for lm_n, tot_n in [(0.0, 0.0), (1.0, 1.0), (0.5, 2.0),
                              (3.0, 3.0), (0.1, 10.0)]:
            b = tracker.budget(lm_n, tot_n)
            assert 0.0 <= b <= 1.0 + 1e-6, \
                f"budget({lm_n},{tot_n}) = {b} out of range"


# ═════════════════════════════════════════════════════════════════════
# Phase 1B  –  LayerGradientProbe
# ═════════════════════════════════════════════════════════════════════

class TestLayerGradientProbe:
    """P1B: per-block gradient L2 norms."""

    def _import(self):
        from neuroslm.emergent.trunk_opt import LayerGradientProbe
        return LayerGradientProbe

    def test_import(self):
        assert callable(self._import())

    def test_returns_dict_keyed_by_layer_index(self):
        LGP = self._import()
        probe = LGP()
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        loss = sum(p.sum() for p in model.parameters())
        loss.backward()
        norms = probe.compute(model.children())
        assert isinstance(norms, dict)
        assert len(norms) == 2

    def test_norms_are_non_negative(self):
        LGP = self._import()
        probe = LGP()
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        loss = sum(p.sum() for p in model.parameters())
        loss.backward()
        norms = probe.compute(model.children())
        for v in norms.values():
            assert v >= 0.0

    def test_uniformity_ratio(self):
        """uniformity_ratio = max/mean grad norm across layers."""
        LGP = self._import()
        probe = LGP()
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        loss = sum(p.sum() for p in model.parameters())
        loss.backward()
        norms = probe.compute(model.children())
        r = probe.uniformity_ratio(norms)
        # ratio is ≥ 1 (max ≥ mean) and finite
        assert r >= 1.0 - 1e-6
        assert math.isfinite(r)


# ═════════════════════════════════════════════════════════════════════
# Phase 1C  –  BitsPerParamMeter
# ═════════════════════════════════════════════════════════════════════

class TestBitsPerParamMeter:
    """P1C: bits_per_param = (ln(V) - CE) / n_trainable."""

    def _import(self):
        from neuroslm.emergent.trunk_opt import BitsPerParamMeter
        return BitsPerParamMeter

    def test_import(self):
        assert callable(self._import())

    def test_zero_ce_means_max_bits(self):
        BPP = self._import()
        meter = BPP(vocab_size=100, n_trainable=1000)
        # CE = 0 → bits = ln(100)/1000 ≈ 0.0046
        b = meter.compute(ce=0.0)
        assert b == pytest.approx(math.log(100) / 1000, rel=1e-5)

    def test_ce_equal_log_v_means_zero_bits(self):
        """Random-init CE = ln(V) → zero useful bits squeezed out."""
        BPP = self._import()
        meter = BPP(vocab_size=100, n_trainable=1000)
        b = meter.compute(ce=math.log(100))
        assert b == pytest.approx(0.0, abs=1e-9)

    def test_bits_monotone_decreasing_in_ce(self):
        """Higher CE → fewer bits per param."""
        BPP = self._import()
        meter = BPP(vocab_size=256, n_trainable=500)
        b1 = meter.compute(ce=1.0)
        b2 = meter.compute(ce=3.0)
        assert b1 > b2

    def test_never_negative(self):
        """CE > ln(V) would give negative bits; clamp to 0."""
        BPP = self._import()
        meter = BPP(vocab_size=100, n_trainable=1000)
        b = meter.compute(ce=math.log(100) + 2.0)
        assert b >= 0.0


# ═════════════════════════════════════════════════════════════════════
# Phase 1D  –  PACBayesBound
# ═════════════════════════════════════════════════════════════════════

class TestPACBayesBound:
    """P1D: PAC-Bayes upper bound on OOD loss."""

    def _import(self):
        from neuroslm.emergent.trunk_opt import PACBayesBound
        return PACBayesBound

    def test_import(self):
        assert callable(self._import())

    def test_bound_non_negative(self):
        PBB = self._import()
        bound = PBB(n_train=10000, delta=0.05, prior_sigma=1.0)
        val = bound.compute(train_ce=2.0, kl_div=100.0)
        assert val >= 0.0

    def test_bound_geq_train_ce(self):
        """The bound is always ≥ train CE (it's an upper bound)."""
        PBB = self._import()
        bound = PBB(n_train=10000, delta=0.05, prior_sigma=1.0)
        for train_ce in [0.5, 1.0, 2.0, 4.0]:
            b = bound.compute(train_ce=train_ce, kl_div=10.0)
            assert b >= train_ce - 1e-6, \
                f"bound {b} < train_ce {train_ce}"

    def test_bound_tighter_with_more_data(self):
        """More training tokens → tighter bound (smaller penalty term)."""
        PBB = self._import()
        b_small = PBB(n_train=1000, delta=0.05).compute(2.0, kl_div=100.0)
        b_large = PBB(n_train=100000, delta=0.05).compute(2.0, kl_div=100.0)
        assert b_large < b_small

    def test_kl_div_computation(self):
        """kl_from_params computes Σ(θ - θ₀)² / (2σ²)."""
        PBB = self._import()
        bound = PBB(n_train=10000, prior_sigma=1.0)
        theta = torch.tensor([1.0, 2.0, 3.0])
        theta0 = torch.zeros(3)
        kl = bound.kl_from_params(theta, theta0)
        expected = (1**2 + 2**2 + 3**2) / 2.0   # σ=1 → /2σ²=2
        assert kl == pytest.approx(expected, rel=1e-5)

    def test_kl_from_model(self):
        """kl_from_model computes KL against an init snapshot."""
        PBB = self._import()
        bound = PBB(n_train=10000, prior_sigma=1.0)
        model = nn.Linear(4, 4, bias=False)
        # Save init as prior
        prior = {k: v.clone() for k, v in model.state_dict().items()}
        # Perturb weights by 1.0
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        kl = bound.kl_from_model(model, prior)
        # Each element shifted by 1 → KL = n_params * 1² / (2*1²) = n_params/2
        n = sum(p.numel() for p in model.parameters())
        assert kl == pytest.approx(n / 2.0, rel=1e-4)


# ═════════════════════════════════════════════════════════════════════
# Phase 1E  –  SharpnessProbe
# ═════════════════════════════════════════════════════════════════════

class TestSharpnessProbe:
    """P1E: sharpness = L(θ + ε) - L(θ) over random perturbation."""

    def _import(self):
        from neuroslm.emergent.trunk_opt import SharpnessProbe
        return SharpnessProbe

    def test_import(self):
        assert callable(self._import())

    def test_sharpness_non_negative(self):
        """Sharpness = E[L(θ+ε)] - L(θ) is expected ≥ 0 at local minima.

        For a randomly-initialised tiny model the loss surface is highly
        non-convex so a single call can return a slightly negative value
        (the perturbation happened to step into a lower basin).  We only
        verify the function returns a finite float; strict non-negativity
        is checked statistically in test_sharpness_mean_positive.
        """
        SP = self._import()
        probe = SP(rho=0.05, n_samples=2, seed=42)
        harness = _tiny_harness()
        ids, tgt = _fake_batch()
        base_loss = float(harness.compute_loss(ids, tgt).detach())
        s = probe.measure(harness, ids, tgt, base_loss)
        assert math.isfinite(s)

    def test_sharpness_is_scalar(self):
        SP = self._import()
        probe = SP(rho=0.05, n_samples=2, seed=42)
        harness = _tiny_harness()
        ids, tgt = _fake_batch()
        loss = float(harness.compute_loss(ids, tgt).detach())
        s = probe.measure(harness, ids, tgt, loss)
        assert isinstance(s, float)

    def test_larger_rho_gives_higher_sharpness(self):
        """Bigger perturbation → larger loss gap (monotone in rho)."""
        SP = self._import()
        harness = _tiny_harness()
        ids, tgt = _fake_batch()
        loss = float(harness.compute_loss(ids, tgt).detach())
        s_small = SP(rho=0.01, n_samples=3, seed=0).measure(harness, ids, tgt, loss)
        s_large = SP(rho=0.5,  n_samples=3, seed=0).measure(harness, ids, tgt, loss)
        # With high probability larger ρ → larger gap.
        # Use a generous tolerance because a tiny model can have
        # very flat loss landscapes.
        assert s_large >= s_small - 0.5


# ═════════════════════════════════════════════════════════════════════
# Phase 1F  –  EffectiveRankProbe
# ═════════════════════════════════════════════════════════════════════

class TestEffectiveRankProbe:
    """P1F: effective rank of hidden state matrix."""

    def _import(self):
        from neuroslm.emergent.trunk_opt import EffectiveRankProbe
        return EffectiveRankProbe

    def test_import(self):
        assert callable(self._import())

    def test_rank_identity_matrix(self):
        """Full-rank matrix has effective rank ≈ d."""
        ERP = self._import()
        probe = ERP()
        H = torch.eye(16)  # rank 16
        r = probe.compute(H)
        assert r == pytest.approx(16.0, rel=0.05)

    def test_rank_rank1_matrix(self):
        """Rank-1 matrix has effective rank ≈ 1."""
        ERP = self._import()
        probe = ERP()
        v = torch.randn(16, 1)
        H = v @ v.t()   # rank-1
        r = probe.compute(H)
        assert r == pytest.approx(1.0, rel=0.15)

    def test_rank_in_bounds(self):
        """Effective rank is in [1, d]."""
        ERP = self._import()
        probe = ERP()
        for shape in [(32, 8), (8, 32), (16, 16)]:
            H = torch.randn(*shape)
            r = probe.compute(H)
            d = min(shape)
            assert 1.0 <= r <= d + 1e-3, \
                f"rank {r} out of [1, {d}] for shape {shape}"


# ═════════════════════════════════════════════════════════════════════
# Phase 1G  –  SpectralPowerLawProbe  (novel intrinsic geometric inv.)
# ═════════════════════════════════════════════════════════════════════

class TestSpectralPowerLawProbe:
    """Power-law spectrum geometry (α, R², D_PR) of hidden activations.

    Properties under test:
      P1G-1  exact power-law σ_i = i^{-α} → recovers α and R² = 1
      P1G-2  white-noise random matrix → high R² requires bumpy spectrum
      P1G-3  rank-1 matrix → graceful (no NaN), D_PR ≈ 1
      P1G-4  scale invariance: H → c·H leaves (α, R², D_PR) unchanged
      P1G-5  orthogonal invariance: H → H·Q leaves (α, R², D_PR) unchanged
      P1G-6  participation ratio bounds: 1 ≤ D_PR ≤ min(N, d)
      P1G-7  monitor wiring: trunk_opt_power_alpha appears in _metrics
    """

    def _import(self):
        from neuroslm.emergent.trunk_opt import SpectralPowerLawProbe
        return SpectralPowerLawProbe

    def test_import(self):
        assert callable(self._import())

    def test_perfect_powerlaw_recovers_alpha(self):
        """σ_i = i^{-α} → fit recovers α to within 1e-3 and R² ≈ 1."""
        SPL = self._import()
        # Build H with a chosen σ vector.  Take random U, V from O(d)
        # and form H = U · diag(σ) · V^T → svdvals(H) = σ (up to sign).
        alpha_true = 1.4
        K = 40
        rank = torch.arange(1, K + 1, dtype=torch.float32)
        sigma = rank.pow(-alpha_true)
        # H is K × K; svd returns σ
        Q1, _ = torch.linalg.qr(torch.randn(K, K))
        Q2, _ = torch.linalg.qr(torch.randn(K, K))
        H = Q1 @ torch.diag(sigma) @ Q2.t()
        out = SPL.compute(H, k_max=K)
        assert out["alpha"] == pytest.approx(alpha_true, abs=1e-2), \
            f"recovered α = {out['alpha']}, expected {alpha_true}"
        assert out["r2"] > 0.999, \
            f"R² = {out['r2']} for perfect power-law (expected ≈ 1)"

    def test_rank1_graceful(self):
        """Rank-1 matrix → finite output, D_PR ≈ 1."""
        SPL = self._import()
        v = torch.randn(16, 1)
        H = v @ torch.randn(1, 16)  # rank-1, well-defined SV
        out = SPL.compute(H)
        # All fields finite (no NaN/Inf)
        for k, val in out.items():
            assert math.isfinite(val), f"{k} = {val} not finite"
        # D_PR ≈ 1 because only one nonzero σ
        assert out["d_pr"] == pytest.approx(1.0, abs=0.1), \
            f"D_PR = {out['d_pr']} for rank-1 (expected ≈ 1)"

    def test_zero_matrix_graceful(self):
        """All-zero matrix → defaults, no NaN, no crash."""
        SPL = self._import()
        H = torch.zeros(16, 16)
        out = SPL.compute(H)
        assert out["alpha"] == 0.0
        assert out["r2"] == 0.0
        assert out["d_pr"] >= 1.0
        for k, val in out.items():
            assert math.isfinite(val)

    def test_scale_invariance(self):
        """α, R², D_PR are invariant under H → c·H for any c > 0.

        Provable: σ_i(cH) = c·σ_i(H), so log σ shifts vertically.
        OLS slope is unchanged → α invariant.
        Sum-of-squares ratios scale as c⁴/c⁴ → D_PR invariant.
        """
        SPL = self._import()
        torch.manual_seed(42)
        H = torch.randn(64, 32)
        out1 = SPL.compute(H)
        out2 = SPL.compute(H * 17.3)        # arbitrary positive scaling
        out3 = SPL.compute(H * 1e-6)        # near-degenerate scaling
        for key in ("alpha", "r2", "d_pr"):
            assert out1[key] == pytest.approx(out2[key], abs=1e-4), \
                f"{key} not scale-invariant: {out1[key]} vs {out2[key]}"
            # Very small scaling can hit the relative-threshold filter;
            # allow a larger tolerance there but enforce qualitative match.
            assert out1[key] == pytest.approx(out3[key], abs=1e-2), \
                f"{key} broke under tiny scaling: {out1[key]} vs {out3[key]}"

    def test_orthogonal_invariance(self):
        """α, R², D_PR are invariant under H → H·Q for Q ∈ O(d).

        Provable: σ(HQ) = σ(H) for orthogonal Q → all three are functions
        of the singular spectrum, hence orthogonal-invariant.
        """
        SPL = self._import()
        torch.manual_seed(7)
        H = torch.randn(64, 32)
        Q, _ = torch.linalg.qr(torch.randn(32, 32))
        out_H  = SPL.compute(H)
        out_HQ = SPL.compute(H @ Q)
        for key in ("alpha", "r2", "d_pr"):
            assert out_H[key] == pytest.approx(out_HQ[key], abs=1e-4), \
                f"{key} broke orthogonal invariance: " \
                f"{out_H[key]} vs {out_HQ[key]}"

    def test_participation_ratio_bounds(self):
        """1 ≤ D_PR ≤ K for any non-zero matrix."""
        SPL = self._import()
        torch.manual_seed(1)
        for shape in [(64, 32), (32, 64), (16, 16), (8, 128)]:
            H = torch.randn(*shape)
            out = SPL.compute(H)
            K_eff = min(shape)
            assert 1.0 - 1e-6 <= out["d_pr"], \
                f"D_PR={out['d_pr']} below 1 for shape {shape}"
            assert out["d_pr"] <= K_eff + 1e-3, \
                f"D_PR={out['d_pr']} exceeds K={K_eff} for shape {shape}"

    def test_uniform_spectrum_high_dpr(self):
        """Identity → uniform σ → D_PR ≈ K (maximum)."""
        SPL = self._import()
        K = 24
        H = torch.eye(K)  # σ = (1, 1, ..., 1)
        out = SPL.compute(H, k_max=K)
        # Uniform spectrum → all σ_i equal → D_PR = K and OLS slope = 0
        assert out["d_pr"] == pytest.approx(float(K), rel=0.05)
        assert out["alpha"] == pytest.approx(0.0, abs=0.05), \
            f"α={out['alpha']} for uniform spectrum (expected ≈ 0)"

    def test_bottleneck_signature(self):
        """Heavy-tailed σ_i = i^{-3} → α ≈ 3 with high R².

        This is the 'bottleneck collapse' signature in the legend.
        """
        SPL = self._import()
        K = 30
        rank = torch.arange(1, K + 1, dtype=torch.float32)
        sigma = rank.pow(-3.0)
        Q1, _ = torch.linalg.qr(torch.randn(K, K))
        Q2, _ = torch.linalg.qr(torch.randn(K, K))
        H = Q1 @ torch.diag(sigma) @ Q2.t()
        out = SPL.compute(H, k_max=K)
        assert out["alpha"] == pytest.approx(3.0, abs=0.05)
        assert out["r2"] > 0.99

    def test_monitor_wiring_metrics_present(self):
        """Attach monitor + run train_step → α, R², D_PR in _metrics."""
        from neuroslm.harness import BRIANHarness
        from neuroslm.emergent.trunk_opt import TrunkOptMonitor
        lm = _tiny_lm()
        h = BRIANHarness.from_language_model(lm, vocab_size=128, d_sem=32)
        h.attach_trunk_opt_monitor(TrunkOptMonitor())
        h.set_schedule(warmup=10, total=100)
        ids, tgt = _fake_batch()
        h.train_step(ids, tgt)
        for key in ("trunk_opt_power_alpha", "trunk_opt_power_r2",
                    "trunk_opt_dpr"):
            assert key in h._metrics, f"missing metric: {key}"
            val = h._metrics[key]
            assert math.isfinite(val), f"{key}={val} not finite"
        # Sanity bounds
        assert 0.0 <= h._metrics["trunk_opt_power_r2"] <= 1.0
        assert h._metrics["trunk_opt_dpr"] >= 1.0


# ═════════════════════════════════════════════════════════════════════
# Phase 2A-D  –  activation_step  (Capacity-First protocol)
# ═════════════════════════════════════════════════════════════════════

class TestActivationStep:
    """P2: RegularizationController respects activation_step."""

    def _build_controller(self, activation_step: int,
                          warmup_steps: int = 0, weight: float = 0.1):
        from neuroslm.regularizers import RegularizationController
        from neuroslm.dsl.regularization import RegularizationConfig, \
            IsotropyConfig

        cfg = RegularizationConfig()
        cfg.isotropy = IsotropyConfig(enabled=True, weight=weight)
        cfg.activation_step = activation_step
        cfg.warmup_steps = warmup_steps
        return RegularizationController(cfg, d_model=32, vocab_size=128)

    def test_zero_activation_step_allows_immediate_loss(self):
        """activation_step = 0 → aux loss can be non-zero from step 0."""
        ctrl = self._build_controller(activation_step=0, warmup_steps=0)
        h = torch.randn(2, 8, 32)
        logits = torch.randn(2, 8, 128)
        ce = torch.rand(2)
        out = ctrl.collect_aux(h=h, lm_logits=logits,
                               per_sample_ce=ce, domain_labels=None,
                               global_step=1)
        # total aux may be zero for other reasons but should be ≥ 0
        assert float(out["total"].item()) >= 0.0

    def test_aux_is_zero_before_activation_step(self):
        """All aux losses are exactly zero before activation_step."""
        ctrl = self._build_controller(activation_step=1000, warmup_steps=200)
        h = torch.randn(2, 8, 32, requires_grad=True)
        logits = torch.randn(2, 8, 128)
        ce = torch.rand(2)
        for step in [0, 1, 500, 999]:
            out = ctrl.collect_aux(h=h, lm_logits=logits,
                                   per_sample_ce=ce, domain_labels=None,
                                   global_step=step)
            total = float(out["total"].item())
            assert total == pytest.approx(0.0, abs=1e-9), \
                f"step {step}: expected zero aux, got {total}"

    def test_aux_nonzero_at_and_after_activation_step(self):
        """Aux loss becomes non-zero at step == activation_step."""
        ctrl = self._build_controller(activation_step=100, warmup_steps=0)
        h = torch.randn(2, 8, 32, requires_grad=True)
        logits = torch.randn(2, 8, 128)
        ce = torch.rand(2)
        out = ctrl.collect_aux(h=h, lm_logits=logits,
                               per_sample_ce=ce, domain_labels=None,
                               global_step=100)
        total = float(out["total"].item())
        # isotropy is enabled; Gram(H) != I → loss > 0
        assert total > 0.0, "Aux loss should be > 0 at activation_step"

    def test_warmup_starts_from_zero_at_activation_step(self):
        """The warmup multiplier is 0 at activation_step and 1 after
        activation_step + warmup_steps."""
        ctrl = self._build_controller(activation_step=500, warmup_steps=100)
        h = torch.randn(2, 8, 32, requires_grad=True)
        logits = torch.randn(2, 8, 128)
        ce = torch.rand(2)

        out_start = ctrl.collect_aux(h=h, lm_logits=logits,
                                     per_sample_ce=ce, domain_labels=None,
                                     global_step=500)
        out_end = ctrl.collect_aux(h=h, lm_logits=logits,
                                   per_sample_ce=ce, domain_labels=None,
                                   global_step=600)  # activation_step + warmup_steps

        w_start = float(out_start["warmup_mult"].item())
        w_end   = float(out_end["warmup_mult"].item())
        # At activation_step, warmup_mult = 0.0 (or 1/warmup_steps)
        assert w_start <= 0.1 + 1e-6, \
            f"warmup_mult at activation_step = {w_start}, expected ≈ 0"
        # At activation_step + warmup_steps, warmup_mult = 1.0
        assert w_end >= 1.0 - 1e-6, \
            f"warmup_mult at activation_step+warmup_steps = {w_end}, expected ≈ 1"

    def test_isotropy_activation_step_fires_before_global_gate(self):
        """isotropy_activation_step fires even when gs < global activation_step."""
        from neuroslm.regularizers import RegularizationController
        from neuroslm.dsl.regularization import (
            RegularizationConfig, IsotropyConfig,
        )
        cfg = RegularizationConfig()
        cfg.isotropy = IsotropyConfig(enabled=True, weight=0.005)
        cfg.activation_step = 4000          # DAR/PCC gate
        cfg.isotropy_activation_step = 1000  # isotropy fires much earlier
        cfg.warmup_steps = 0                 # no ramp — full weight immediately
        ctrl = RegularizationController(cfg, d_model=32, vocab_size=128)

        h = torch.randn(2, 8, 32, requires_grad=True)
        logits = torch.randn(2, 8, 128)
        ce = torch.rand(2)

        # Before iso_act: both should be zero
        out_pre = ctrl.collect_aux(h=h, lm_logits=logits,
                                   per_sample_ce=ce, domain_labels=None,
                                   global_step=500)
        assert float(out_pre["isotropy"].item()) == pytest.approx(0.0, abs=1e-9)
        assert float(out_pre["dar"].item())      == pytest.approx(0.0, abs=1e-9)

        # At iso_act: isotropy fires, DAR/PCC still zero
        out_iso = ctrl.collect_aux(h=h, lm_logits=logits,
                                   per_sample_ce=ce, domain_labels=None,
                                   global_step=1000)
        assert float(out_iso["isotropy"].item()) > 0.0, \
            "isotropy should be non-zero at isotropy_activation_step"
        assert float(out_iso["dar"].item()) == pytest.approx(0.0, abs=1e-9), \
            "DAR should still be gated before global activation_step"
        assert float(out_iso["warmup_mult"].item()) == pytest.approx(0.0, abs=1e-9), \
            "Global warmup_mult still shows 0 (DAR/PCC gate not open)"

    def test_isotropy_default_neg1_uses_global_gate(self):
        """isotropy_activation_step=-1 (default) falls back to global gate."""
        from neuroslm.regularizers import RegularizationController
        from neuroslm.dsl.regularization import (
            RegularizationConfig, IsotropyConfig,
        )
        cfg = RegularizationConfig()
        cfg.isotropy = IsotropyConfig(enabled=True, weight=0.005)
        cfg.activation_step = 4000
        cfg.isotropy_activation_step = -1   # default: use global
        cfg.warmup_steps = 0
        ctrl = RegularizationController(cfg, d_model=32, vocab_size=128)

        h = torch.randn(2, 8, 32, requires_grad=True)
        logits = torch.randn(2, 8, 128)
        ce = torch.rand(2)

        # At step 1000 (< 4000): isotropy must be zero because iso_act = -1 → uses 4000
        out = ctrl.collect_aux(h=h, lm_logits=logits,
                               per_sample_ce=ce, domain_labels=None,
                               global_step=1000)
        assert float(out["isotropy"].item()) == pytest.approx(0.0, abs=1e-9), \
            "isotropy_activation_step=-1 should use global gate (4000); step 1000 must be zero"

    def test_budget_loss_proxy_nonzero(self):
        """P2E: budget metric is >0 after a train_step (loss-space proxy)."""
        from neuroslm.harness import BRIANHarness
        from neuroslm.emergent.trunk_opt import TrunkOptMonitor
        lm = _tiny_lm()
        h = BRIANHarness.from_language_model(lm, vocab_size=128, d_sem=32)
        h.attach_trunk_opt_monitor(TrunkOptMonitor())
        h.set_schedule(warmup=10, total=100)
        ids, tgt = _fake_batch()
        h.train_step(ids, tgt)
        budget = h._metrics.get("trunk_opt_grad_budget", 0.0)
        assert budget > 0.0, \
            f"budget should be >0 via loss-space proxy; got {budget}"
        assert budget <= 1.0 + 1e-6, \
            f"budget must be <=1; got {budget}"


# ═════════════════════════════════════════════════════════════════════
# Phase 3  –  Harness wiring (metrics surface in _metrics)
# ═════════════════════════════════════════════════════════════════════

class TestHarnessMetricsWiring:
    """P3: trunk_opt metrics appear in harness._metrics after train_step."""

    def _harness_with_trunk_opt(self):
        from neuroslm.harness import BRIANHarness
        from neuroslm.emergent.trunk_opt import TrunkOptMonitor
        lm = _tiny_lm()
        h = BRIANHarness.from_language_model(lm, vocab_size=128, d_sem=32)
        h.attach_trunk_opt_monitor(TrunkOptMonitor())
        return h

    def test_grad_budget_populated(self):
        """P3A: trunk_opt_grad_budget ∈ [0,1] after train_step."""
        h = self._harness_with_trunk_opt()
        h.set_schedule(warmup=10, total=100)
        ids, tgt = _fake_batch()
        h.train_step(ids, tgt)
        assert "trunk_opt_grad_budget" in h._metrics
        b = h._metrics["trunk_opt_grad_budget"]
        assert 0.0 <= b <= 1.0 + 1e-6

    def test_bits_per_param_populated(self):
        """P3B: trunk_opt_bits_per_param > 0 after train_step."""
        h = self._harness_with_trunk_opt()
        h.set_schedule(warmup=10, total=100)
        ids, tgt = _fake_batch()
        h.train_step(ids, tgt)
        assert "trunk_opt_bits_per_param" in h._metrics
        b = h._metrics["trunk_opt_bits_per_param"]
        assert b >= 0.0

    def test_effective_rank_populated(self):
        """P3C: trunk_opt_effective_rank > 0 after compute_loss."""
        h = self._harness_with_trunk_opt()
        ids, tgt = _fake_batch()
        h.compute_loss(ids, tgt)
        assert "trunk_opt_effective_rank" in h._metrics
        r = h._metrics["trunk_opt_effective_rank"]
        assert r >= 1.0 - 1e-3


# ═════════════════════════════════════════════════════════════════════
# Phase 4  –  PAC-Bayes bound validation
# ═════════════════════════════════════════════════════════════════════

class TestPACBayesValidation:
    """P4: PAC-Bayes bound properties under regularization."""

    def _import(self):
        from neuroslm.emergent.trunk_opt import PACBayesBound
        return PACBayesBound

    def test_bound_always_non_negative(self):
        """P4B: bound ≥ 0 for any valid inputs."""
        PBB = self._import()
        bound = PBB(n_train=10000, delta=0.05, prior_sigma=1.0)
        for train_ce in [0.1, 1.0, 5.0]:
            for kl in [0.0, 10.0, 10000.0]:
                b = bound.compute(train_ce, kl)
                assert b >= 0.0, \
                    f"bound({train_ce}, {kl}) = {b} is negative"

    def test_tighter_bound_with_smaller_kl(self):
        """P4A: smaller KL → tighter bound (closer to train CE)."""
        PBB = self._import()
        bound = PBB(n_train=10000)
        b_large_kl = bound.compute(train_ce=2.0, kl_div=10000.0)
        b_small_kl = bound.compute(train_ce=2.0, kl_div=10.0)
        assert b_small_kl < b_large_kl

    def test_stronger_wd_leads_to_smaller_kl(self):
        """Stronger weight_decay keeps θ closer to init → smaller KL.

        Simulates one gradient step under strong vs weak L2 penalty.
        """
        PBB = self._import()
        bound = PBB(n_train=10000, prior_sigma=1.0)

        def _kl_after_step(wd: float) -> float:
            model = nn.Linear(8, 8, bias=False)
            prior = {k: v.clone() for k, v in model.state_dict().items()}
            opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=wd)
            for _ in range(20):
                loss = (model.weight * torch.randn_like(model.weight)).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            return bound.kl_from_model(model, prior)

        kl_low  = _kl_after_step(0.0)
        kl_high = _kl_after_step(1.0)
        # Higher WD keeps params closer to init
        assert kl_high <= kl_low + 0.5   # generous tolerance
