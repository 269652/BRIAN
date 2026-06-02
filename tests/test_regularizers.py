# -*- coding: utf-8 -*-
"""Numerical-equivalence tests for the five OOD interventions.

Each test asserts that the PyTorch module produces a value matching the
canonical equation declared in
`architectures/rcc_bowtie/lib/regularizers.neuro`.

The tests cover (1) module forward output, (2) gradient flow, and (3)
disabled-path no-op behavior so the controller is safe to leave wired
in production builds even when the corresponding flag is false.
"""
from __future__ import annotations
import math

import pytest
import torch
import torch.nn.functional as F

from neuroslm.dsl.regularization import (
    DARConfig, PCCConfig, IsotropyConfig, CMDConfig, AdaptiveMixtureConfig,
    RegularizationConfig,
)
from neuroslm.regularizers import (
    DARReweighter,
    PCCLoss,
    IsotropyLoss,
    CMDLoss,
    AdaptiveMixtureController,
    RegularizationController,
    GradientReversal,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


# ══════════════════════════════════════════════════════════════════════
# Intervention A — DARReweighter
# ══════════════════════════════════════════════════════════════════════

class TestGradientReversal:
    def test_forward_identity(self):
        _seed()
        x = torch.randn(4, 8, requires_grad=True)
        y = GradientReversal.apply(x, 1.0)
        # Forward must be exact identity
        assert torch.equal(y, x)

    def test_backward_sign_flip(self):
        _seed()
        x = torch.randn(4, 8, requires_grad=True)
        alpha = 0.3
        y = GradientReversal.apply(x, alpha)
        # Apply a unit upstream gradient
        y.sum().backward()
        # GRL flips the sign and scales by alpha
        assert torch.allclose(x.grad, -alpha * torch.ones_like(x))


class TestDARReweighter:
    def _make(self, enabled: bool = True, d: int = 16, lam: float = 1.0):
        cfg = DARConfig(enabled=enabled, lam=lam, hidden=8, grl_alpha=0.5)
        return DARReweighter(cfg, d_model=d), cfg

    def test_disabled_returns_zero(self):
        m, _ = self._make(enabled=False)
        h = torch.randn(4, 16)
        labels = torch.tensor([0, 1, 0, 1])
        per_sample_ce = torch.tensor([1.0, 2.0, 1.5, 0.5])
        out = m(h, per_sample_ce, labels)
        assert out["weighted_ce"].item() == pytest.approx(per_sample_ce.mean().item())
        assert out["disc_loss"].item() == 0.0
        assert out["total_aux"].item() == 0.0

    def test_disabled_when_labels_none(self):
        m, _ = self._make(enabled=True)
        h = torch.randn(4, 16)
        per_sample_ce = torch.tensor([1.0, 2.0, 1.5, 0.5])
        out = m(h, per_sample_ce, domain_labels=None)
        # No labels → falls back to mean CE, zero aux
        assert out["weighted_ce"].item() == pytest.approx(per_sample_ce.mean().item())
        assert out["disc_loss"].item() == 0.0

    def test_reweighting_lifts_minority(self):
        """When minority loss > 0, weighted mean > unweighted mean."""
        _seed()
        m, _ = self._make(enabled=True, lam=2.0)
        h = torch.randn(8, 16)
        # 6 majority (low loss), 2 minority (high loss)
        labels = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
        per_sample_ce = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 3.0, 3.0])
        out = m(h, per_sample_ce, labels)
        # The minority class gets upweighted via exp(lam * L * minority_mask)
        assert out["weighted_ce"].item() > per_sample_ce.mean().item()

    def test_discriminator_loss_is_bce(self):
        """disc_loss must equal BCE(sigmoid(GRL(h)·W), labels)."""
        _seed()
        m, _ = self._make(enabled=True)
        h = torch.randn(4, 16)
        labels = torch.tensor([0, 1, 0, 1]).float()
        per_sample_ce = torch.zeros(4)
        out = m(h, per_sample_ce, labels.long())
        # Recompute the BCE the discriminator should produce
        # via the same path (GRL doesn't change forward values).
        with torch.no_grad():
            logits = m.discriminator(h).squeeze(-1)
            expected = F.binary_cross_entropy_with_logits(logits, labels)
        assert out["disc_loss"].item() == pytest.approx(expected.item(), rel=1e-5)

    def test_gradient_flows_to_h_and_discriminator(self):
        _seed()
        m, _ = self._make(enabled=True)
        # The final layer is zero-init by design (no step-0 perturbation
        # of the trunk). Perturb it so the gradient test exercises real
        # flow rather than the deliberate zero-init guard.
        with torch.no_grad():
            for p in m.discriminator.parameters():
                p.add_(0.1 * torch.randn_like(p))
        h = torch.randn(4, 16, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1])
        per_sample_ce = torch.ones(4)
        out = m(h, per_sample_ce, labels)
        out["total_aux"].backward()
        assert h.grad is not None
        assert (h.grad.abs() > 0).any()
        # All discriminator params should have non-zero grad
        for p in m.discriminator.parameters():
            assert p.grad is not None
            assert (p.grad.abs() > 0).any()


# ══════════════════════════════════════════════════════════════════════
# Intervention B — PCCLoss
# ══════════════════════════════════════════════════════════════════════

class TestPCCLoss:
    def _make(self, enabled: bool = True, k: int = 2, neg: int = 8, d: int = 16):
        cfg = PCCConfig(enabled=enabled, k=k, n_negatives=neg, tau=0.1, layers=[])
        return PCCLoss(cfg, d_model=d), cfg

    def test_disabled_returns_zero(self):
        m, _ = self._make(enabled=False)
        h = torch.randn(2, 8, 16)
        loss = m(h)
        assert loss.item() == 0.0

    def test_loss_is_positive_when_random(self):
        _seed()
        m, _ = self._make(enabled=True, k=2)
        # Pre-fill negatives buffer
        for _ in range(3):
            with torch.no_grad():
                m(torch.randn(2, 8, 16))
        h = torch.randn(2, 8, 16)
        loss = m(h)
        # InfoNCE on random vectors with τ=0.1: pos and neg logits both
        # ~N(0, 1/τ²), so loss = -E[pos] + E[logsumexp(...)] is bounded
        # by O(M / τ). Just verify positivity + sanity upper bound.
        assert loss.item() > 0.0
        assert loss.item() < 100.0

    def test_loss_is_small_when_anchor_matches_positive(self):
        """If we hand-craft anchor ≈ positive and negatives orthogonal,
        InfoNCE should be very small (positive dominates)."""
        _seed()
        m, _ = self._make(enabled=True, k=1, neg=4, d=8)
        # Make a strongly self-consistent sequence: h[:, t+1] ≈ h[:, t]
        T = 4
        base = torch.randn(1, 1, 8)
        h = base.repeat(1, T, 1) + 0.001 * torch.randn(1, T, 8)
        # Fill negative buffer with orthogonal-ish noise
        for _ in range(2):
            with torch.no_grad():
                m(torch.randn(1, T, 8) * 0.001 + 100.0)
        loss = m(h)
        # With aligned positives and far negatives, loss should be small
        assert loss.item() < 0.5

    def test_gradient_flow(self):
        _seed()
        m, _ = self._make(enabled=True, k=2)
        for _ in range(2):
            with torch.no_grad():
                m(torch.randn(2, 8, 16))
        h = torch.randn(2, 8, 16, requires_grad=True)
        loss = m(h)
        loss.backward()
        assert h.grad is not None
        assert (h.grad.abs() > 0).any()


# ══════════════════════════════════════════════════════════════════════
# Intervention C — IsotropyLoss
# ══════════════════════════════════════════════════════════════════════

class TestIsotropyLoss:
    def _make(self, enabled: bool = True, d: int = 8, buf: int = 64,
              dist: str = "frobenius"):
        cfg = IsotropyConfig(enabled=enabled, weight=1.0, buffer=buf,
                             distance=dist)
        return IsotropyLoss(cfg, d_model=d), cfg

    def test_disabled_returns_zero(self):
        m, _ = self._make(enabled=False)
        h = torch.randn(2, 8, 8)
        assert m(h).item() == 0.0

    def test_isotropic_input_has_low_loss(self):
        """Whitened (truly isotropic) embeddings give Gram ≈ I → loss ≈ 0."""
        _seed()
        m, _ = self._make(enabled=True, d=8, buf=2048)
        # Generate a large batch of standard normal vectors — Gram → I
        big = torch.randn(2048, 8)
        h = big.reshape(2, 1024, 8)
        loss = m(h)
        # E[||G - I||_F^2 / d^2] is small for d=8 with N=2048
        assert loss.item() < 0.05

    def test_anisotropic_input_has_high_loss(self):
        """Embeddings concentrated on one axis → far from I."""
        _seed()
        m, _ = self._make(enabled=True, d=8, buf=2048)
        # All tokens point along axis 0
        h = torch.zeros(2, 256, 8)
        h[..., 0] = 1.0
        loss = m(h)
        # G = e_0 e_0^T → ||G - I||_F^2 = (1-1)^2 + (d-1)*1 = d - 1 = 7
        # Divided by d^2 = 64 → 7/64 ≈ 0.109
        assert loss.item() > 0.05

    def test_matches_equation_frobenius(self):
        """Direct numerical check against ||G - I||_F^2 / d^2."""
        _seed()
        d = 8
        m, _ = self._make(enabled=True, d=d, buf=512)
        # Single forward to fill buffer
        h = torch.randn(1, 64, d)
        loss = m(h)
        # Recompute expected from the controller's own internal buffer
        H = m.get_buffer_view()
        N = H.shape[0]
        G = (H.t() @ H) / max(1, N)
        I = torch.eye(d, device=G.device)
        expected = ((G - I) ** 2).sum() / (d * d)
        assert loss.item() == pytest.approx(expected.item(), rel=1e-4)

    def test_gradient_flows_to_h(self):
        _seed()
        m, _ = self._make(enabled=True, d=8)
        h = torch.randn(2, 16, 8, requires_grad=True)
        loss = m(h)
        loss.backward()
        assert h.grad is not None
        assert (h.grad.abs() > 0).any()


# ══════════════════════════════════════════════════════════════════════
# Intervention D — CMDLoss
# ══════════════════════════════════════════════════════════════════════

class TestCMDLoss:
    def _make(self, enabled: bool = True, vocab: int = 32, d: int = 16,
              div: str = "jsd"):
        cfg = CMDConfig(enabled=enabled, weight=1.0, divergence=div,
                        heads=["lm", "narrative"])
        return CMDLoss(cfg, d_model=d, vocab_size=vocab), cfg

    def test_disabled_returns_zero(self):
        m, _ = self._make(enabled=False)
        h = torch.randn(2, 4, 16)
        lm_logits = torch.randn(2, 4, 32)
        assert m(h, lm_logits).item() == 0.0

    def test_jsd_zero_when_distributions_identical(self):
        """If narrative head outputs same logits as lm, JSD = 0."""
        _seed()
        m, _ = self._make(enabled=True)
        h = torch.randn(2, 4, 16)
        # Make narrative head match the lm path: build identity-like read-out
        # then compute lm_logits = narrative_logits manually
        with torch.no_grad():
            narr_logits = m.head(h)
        loss = m(h, narr_logits)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_jsd_is_bounded_by_ln2(self):
        """JSD ∈ [0, ln 2]."""
        _seed()
        m, _ = self._make(enabled=True)
        h = torch.randn(2, 4, 16)
        # Make lm_logits a delta on one token, narrative a delta on another
        lm_logits = torch.zeros(2, 4, 32)
        lm_logits[..., 0] = 100.0
        loss = m(h, lm_logits)
        # weight=1 → loss ≤ ln 2
        assert loss.item() <= math.log(2.0) + 1e-3

    def test_kl_sym_path(self):
        _seed()
        m, _ = self._make(enabled=True, div="kl_sym")
        h = torch.randn(2, 4, 16)
        lm_logits = torch.randn(2, 4, 32)
        loss = m(h, lm_logits)
        assert loss.item() >= 0.0

    def test_l1_path(self):
        _seed()
        m, _ = self._make(enabled=True, div="l1")
        h = torch.randn(2, 4, 16)
        lm_logits = torch.randn(2, 4, 32)
        loss = m(h, lm_logits)
        # L1 between two distributions over 32 classes ≤ 2.0
        assert 0.0 <= loss.item() <= 2.0

    def test_gradient_flows_to_h_and_head(self):
        _seed()
        m, _ = self._make(enabled=True)
        h = torch.randn(2, 4, 16, requires_grad=True)
        lm_logits = torch.randn(2, 4, 32)
        loss = m(h, lm_logits)
        loss.backward()
        assert h.grad is not None
        for p in m.head.parameters():
            assert p.grad is not None


# ══════════════════════════════════════════════════════════════════════
# Intervention E — AdaptiveMixtureController
# ══════════════════════════════════════════════════════════════════════

class TestAdaptiveMixtureController:
    def _make(self, enabled: bool = True, target_H: float = 4.5,
              gamma: float = 2.0):
        cfg = AdaptiveMixtureConfig(
            enabled=enabled, target_entropy=target_H,
            probe_interval=1, gamma=gamma, min_ratio=0.05, max_ratio=0.95)
        return AdaptiveMixtureController(cfg, initial_ratio=0.6), cfg

    def test_disabled_keeps_ratio_constant(self):
        m, _ = self._make(enabled=False)
        # Even with absurd entropy, ratio must not change
        m.observe_logits(torch.zeros(2, 4, 1024))  # very low entropy
        assert m.ratio() == 0.6

    def test_low_entropy_shrinks_ratio(self):
        """H_t < H_target → ratio decreases."""
        m, _ = self._make(enabled=True, target_H=5.0, gamma=2.0)
        initial = m.ratio()
        # Logits with very low entropy
        logits = torch.zeros(2, 4, 256)
        logits[..., 0] = 100.0
        m.observe_logits(logits)
        assert m.ratio() < initial

    def test_high_entropy_grows_ratio(self):
        """H_t > H_target → ratio increases (toward max)."""
        m, _ = self._make(enabled=True, target_H=1.0, gamma=2.0)
        initial = m.ratio()
        # Uniform logits → max entropy = ln(vocab)
        logits = torch.zeros(2, 4, 256)
        m.observe_logits(logits)
        assert m.ratio() > initial

    def test_ratio_clipped_to_bounds(self):
        m, _ = self._make(enabled=True, target_H=10.0, gamma=4.0)
        # Collapsed logits, big gain → ratio should hit min_ratio
        for _ in range(20):
            logits = torch.zeros(1, 1, 32)
            logits[..., 0] = 100.0
            m.observe_logits(logits)
        assert m.ratio() == pytest.approx(0.05)

    def test_probe_interval_throttles_updates(self):
        cfg = AdaptiveMixtureConfig(
            enabled=True, target_entropy=10.0, probe_interval=5,
            gamma=2.0, min_ratio=0.05, max_ratio=0.95)
        m = AdaptiveMixtureController(cfg, initial_ratio=0.6)
        initial = m.ratio()
        # First 4 observations should NOT update the ratio
        for _ in range(4):
            m.observe_logits(torch.zeros(1, 1, 32))
        assert m.ratio() == initial
        # 5th observation triggers update
        m.observe_logits(torch.zeros(1, 1, 32))
        assert m.ratio() != initial


# ══════════════════════════════════════════════════════════════════════
# Top-level RegularizationController
# ══════════════════════════════════════════════════════════════════════

class TestRegularizationController:
    def test_all_disabled_zero_aux(self):
        cfg = RegularizationConfig()
        ctrl = RegularizationController(cfg, d_model=16, vocab_size=32)
        h = torch.randn(2, 4, 16)
        lm_logits = torch.randn(2, 4, 32)
        per_sample_ce = torch.ones(2)
        out = ctrl.collect_aux(h, lm_logits, per_sample_ce,
                                domain_labels=None)
        assert out["total"].item() == 0.0

    def test_isotropy_only(self):
        cfg = RegularizationConfig()
        cfg.isotropy.enabled = True
        cfg.isotropy.weight = 1.0
        ctrl = RegularizationController(cfg, d_model=8, vocab_size=32)
        # Anisotropic input
        h = torch.zeros(2, 16, 8)
        h[..., 0] = 1.0
        out = ctrl.collect_aux(h, torch.randn(2, 16, 32), torch.ones(2),
                                domain_labels=None)
        assert out["total"].item() > 0.0
        assert out["isotropy"].item() > 0.0

    def test_combined_interventions_sum(self):
        cfg = RegularizationConfig()
        cfg.isotropy.enabled = True
        cfg.cmd.enabled = True
        cfg.cmd.weight = 0.5
        ctrl = RegularizationController(cfg, d_model=8, vocab_size=32)
        h = torch.randn(2, 4, 8)
        lm_logits = torch.randn(2, 4, 32)
        out = ctrl.collect_aux(h, lm_logits, torch.ones(2),
                                domain_labels=None)
        # Total should be at least the sum of the non-zero components
        components = (
            out["isotropy"].item() + out["cmd"].item())
        assert out["total"].item() == pytest.approx(components, rel=1e-5)
