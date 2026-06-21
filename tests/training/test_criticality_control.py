# -*- coding: utf-8 -*-
"""TDD: BranchingRatioMonitor — neural criticality at σ=1 (Beggs & Plenz 2003).

Mathematical contracts verified:
  1. σ = mean Frobenius Jacobian norm across tokens: σ = (1/T) Σ_t ‖∂h_{l+1,t}/∂h_{l,t}‖_F
  2. σ > 1 → supercritical; σ < 1 → subcritical; |σ-1| < ε → critical
  3. EMA tracking: σ_ema ← α·σ + (1-α)·σ_ema
  4. NT stress = (σ - 1)² injected into allostasis load
  5. Criticality loss = weight * (σ - target)²
  6. BranchingRatioMonitor.nt_signals() returns {'gaba': ..., 'ne': ..., 'da': ...}
     with monotonic relationships to σ distance from target
  7. σ ≈ 1 → da_reward fires (high DA signal)
  8. σ > 1 → gaba increases (inhibitory → dampen activity)
  9. σ < 1 → ne increases (excitatory → boost activity)

Run:  brian test tests/training/test_criticality_control.py
"""
from __future__ import annotations

import math
import torch
import pytest


@pytest.fixture(scope="module")
def monitor_cls():
    from neuroslm.emergent.semantic_turbulence import BranchingRatioMonitor
    return BranchingRatioMonitor


# ── σ measurement ─────────────────────────────────────────────────────────


class TestBranchingRatioMeasurement:
    """σ = mean Frobenius Jacobian norm through a linear layer."""

    def test_sigma_near_1_for_identity(self, monitor_cls):
        """Through identity transform, Jacobian = I, ‖I‖_F = √d → σ ≈ √d/d → 1 after normalisation."""
        monitor = monitor_cls(target=1.0, ema_alpha=0.05)
        d = 32
        h_prev = torch.randn(1, 8, d, requires_grad=True)
        # Identity mapping: h_next = h_prev
        h_next = h_prev + 0.0  # same tensor, identity Jacobian
        sigma = monitor.measure_sigma(h_prev, h_next)
        # Frobenius norm of d×d identity = √d; normalised per dimension = 1
        assert abs(sigma.item() - 1.0) < 0.15, (
            f"Expected σ≈1 for identity, got {sigma.item():.4f}"
        )

    def test_sigma_greater_1_for_expansion(self, monitor_cls):
        """Amplifying transform (h_next = 2*h_prev) → σ > 1."""
        monitor = monitor_cls(target=1.0, ema_alpha=0.05)
        h_prev = torch.randn(1, 8, 32, requires_grad=True)
        h_next = 2.0 * h_prev
        sigma = monitor.measure_sigma(h_prev, h_next)
        assert sigma.item() > 1.0, f"Expected σ>1 for 2x scaling, got {sigma.item():.4f}"

    def test_sigma_less_1_for_contraction(self, monitor_cls):
        """Contracting transform (h_next = 0.5*h_prev) → σ < 1."""
        monitor = monitor_cls(target=1.0, ema_alpha=0.05)
        h_prev = torch.randn(1, 8, 32, requires_grad=True)
        h_next = 0.5 * h_prev
        sigma = monitor.measure_sigma(h_prev, h_next)
        assert sigma.item() < 1.0, f"Expected σ<1 for 0.5x scaling, got {sigma.item():.4f}"

    def test_sigma_is_scalar(self, monitor_cls):
        monitor = monitor_cls(target=1.0, ema_alpha=0.05)
        h_prev = torch.randn(2, 16, 64, requires_grad=True)
        h_next = h_prev * 1.2
        sigma = monitor.measure_sigma(h_prev, h_next)
        assert sigma.ndim == 0 or sigma.numel() == 1

    def test_sigma_nonnegative(self, monitor_cls):
        monitor = monitor_cls(target=1.0, ema_alpha=0.05)
        h_prev = torch.randn(2, 8, 32, requires_grad=True)
        h_next = torch.randn(2, 8, 32)
        sigma = monitor.measure_sigma(h_prev, h_next)
        assert sigma.item() >= 0.0


# ── EMA tracking ──────────────────────────────────────────────────────────


class TestEMATracking:
    """σ_ema ← α·σ + (1-α)·σ_ema, starts at target (1.0)."""

    def test_ema_starts_at_target(self, monitor_cls):
        monitor = monitor_cls(target=1.0, ema_alpha=0.1)
        assert abs(monitor.sigma_ema - 1.0) < 1e-6

    def test_ema_updates(self, monitor_cls):
        monitor = monitor_cls(target=1.0, ema_alpha=0.5)
        monitor.update_ema(2.0)
        assert abs(monitor.sigma_ema - 1.5) < 1e-5  # 0.5*2 + 0.5*1

    def test_ema_converges_to_constant(self, monitor_cls):
        monitor = monitor_cls(target=1.0, ema_alpha=0.1)
        for _ in range(100):
            monitor.update_ema(3.0)
        assert abs(monitor.sigma_ema - 3.0) < 0.01


# ── NT signals ────────────────────────────────────────────────────────────


class TestNTSignals:
    """nt_signals() returns correct directional modulation."""

    def test_returns_dict_with_gaba_ne_da(self, monitor_cls):
        monitor = monitor_cls(target=1.0, ema_alpha=0.05)
        signals = monitor.nt_signals(sigma=1.0)
        assert "gaba" in signals
        assert "ne" in signals
        assert "da" in signals

    def test_critical_da_is_high(self, monitor_cls):
        """σ ≈ 1.0 (at target) → DA reward fires."""
        monitor = monitor_cls(target=1.0, ema_alpha=0.05, da_reward=0.1)
        signals_critical = monitor.nt_signals(sigma=1.0)
        signals_off = monitor.nt_signals(sigma=2.0)
        assert signals_critical["da"] > signals_off["da"], (
            f"DA at criticality ({signals_critical['da']:.4f}) should exceed "
            f"DA off-criticality ({signals_off['da']:.4f})"
        )

    def test_supercritical_gaba_is_high(self, monitor_cls):
        """σ > 1 → GABA (inhibitory) increases to damp activity."""
        monitor = monitor_cls(target=1.0, ema_alpha=0.05)
        signals_super = monitor.nt_signals(sigma=2.0)
        signals_critical = monitor.nt_signals(sigma=1.0)
        assert signals_super["gaba"] > signals_critical["gaba"], (
            f"GABA supercritical ({signals_super['gaba']:.4f}) should exceed "
            f"GABA critical ({signals_critical['gaba']:.4f})"
        )

    def test_subcritical_ne_is_high(self, monitor_cls):
        """σ < 1 → NE (norepinephrine/excitatory) increases to boost activity."""
        monitor = monitor_cls(target=1.0, ema_alpha=0.05)
        signals_sub = monitor.nt_signals(sigma=0.3)
        signals_critical = monitor.nt_signals(sigma=1.0)
        assert signals_sub["ne"] > signals_critical["ne"], (
            f"NE subcritical ({signals_sub['ne']:.4f}) should exceed "
            f"NE critical ({signals_critical['ne']:.4f})"
        )

    def test_signals_bounded(self, monitor_cls):
        """All NT signals should be in a reasonable positive range."""
        monitor = monitor_cls(target=1.0, ema_alpha=0.05)
        for sigma in [0.1, 0.5, 1.0, 1.5, 3.0]:
            signals = monitor.nt_signals(sigma=sigma)
            for nt, val in signals.items():
                assert val >= 0.0, f"{nt} signal negative at σ={sigma}: {val}"


# ── Criticality loss ──────────────────────────────────────────────────────


class TestCriticalityLoss:
    """Loss term = weight * (σ - target)²."""

    def test_criticality_loss_zero_at_target(self, monitor_cls):
        monitor = monitor_cls(target=1.0, ema_alpha=0.05, weight=0.01)
        loss = monitor.criticality_loss(sigma=1.0)
        assert abs(loss.item()) < 1e-8

    def test_criticality_loss_positive_elsewhere(self, monitor_cls):
        monitor = monitor_cls(target=1.0, ema_alpha=0.05, weight=0.01)
        loss = monitor.criticality_loss(sigma=2.0)
        assert loss.item() > 0.0

    def test_criticality_loss_is_differentiable(self, monitor_cls):
        monitor = monitor_cls(target=1.0, ema_alpha=0.05, weight=0.01)
        sigma_t = torch.tensor(1.5, requires_grad=True)
        loss = monitor.criticality_loss(sigma=sigma_t)
        loss.backward()
        assert sigma_t.grad is not None

    def test_weight_scales_loss(self, monitor_cls):
        m1 = monitor_cls(target=1.0, ema_alpha=0.05, weight=0.01)
        m2 = monitor_cls(target=1.0, ema_alpha=0.05, weight=0.1)
        l1 = m1.criticality_loss(sigma=2.0).item()
        l2 = m2.criticality_loss(sigma=2.0).item()
        assert abs(l2 / l1 - 10.0) < 1e-5

    def test_nt_stress_is_sigma_minus_target_squared(self, monitor_cls):
        """NT stress = (σ - 1)² added to allostasis load."""
        monitor = monitor_cls(target=1.0, ema_alpha=0.05, weight=1.0)
        for sigma in [0.5, 1.0, 1.5, 2.0]:
            expected = (sigma - 1.0) ** 2
            actual = monitor.nt_stress(sigma=sigma)
            assert abs(actual - expected) < 1e-6, (
                f"nt_stress({sigma}) = {actual:.6f}, expected {expected:.6f}"
            )
