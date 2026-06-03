# -*- coding: utf-8 -*-
"""TDD spec for the neuromechanically-stabilised AdaptiveMixtureController.

Background — 2026-06-03 (second run, after PR-A1)
─────────────────────────────────────────────────
The first PR-A1 fix corrected the controller direction ("balance" rule)
but exposed a deeper bug: the controller observes the entropy of
TRAINING-DATA logits, not a held-out prose probe. At init the LM
produces near-uniform logits over the 50257-vocab regardless of
input distribution, so H_t ≈ log(V) ≈ 10.5. With the corrected
"balance" rule and γ=2:

    gain = (H_target / H_t)^γ = (4.5/10.5)^2 = 0.184

First update: chat_ratio 0.60 × 0.184 = 0.11
Second update: 0.11 × 0.184 → clipped to min_ratio=0.10

The controller saturated to its floor in 20 steps BEFORE the LM had
formed any features. The model was then shocked from 60% chat to
90% FineWeb-Edu prose at step 100, with random-init weights, in the
critical early-training window. Loss plateaued at ppl ~3000.

Biological gain controllers (retinal adaptation, thalamic relay
neurons) avoid this failure mode via three mechanisms:

  1. STARTUP GRACE — no adaptation until input SNR is sufficient
     (controller_warmup_steps)
  2. SLEW-RATE LIMIT — bounded |Δratio| per update
     (max_step_delta)
  3. LOW-PASS FILTER on the input signal — EMA on the entropy
     observation (entropy_ema_alpha)

This file specifies all three.
"""
import pytest
import torch

from neuroslm.dsl.regularization import (
    AdaptiveMixtureConfig, parse_regularization_block)
from neuroslm.regularizers import AdaptiveMixtureController


def _logits_with_entropy(target_entropy: float, vocab: int = 256,
                          shape=(4, 8)) -> torch.Tensor:
    """Same helper as test_adaptive_mixture_direction."""
    lo, hi = 1.0 / vocab + 1e-6, 1.0 - 1e-6
    for _ in range(60):
        p = 0.5 * (lo + hi)
        q = (1.0 - p) / (vocab - 1)
        H = -(p * torch.log(torch.tensor(p)).item()
              + (vocab - 1) * q * torch.log(torch.tensor(q)).item())
        if H > target_entropy:
            lo = p
        else:
            hi = p
    p = 0.5 * (lo + hi)
    q = (1.0 - p) / (vocab - 1)
    log_p = torch.log(torch.tensor(p))
    log_q = torch.log(torch.tensor(q))
    base = torch.full((*shape, vocab), log_q.item())
    base[..., 0] = log_p.item()
    return base


# ── Config defaults ─────────────────────────────────────────────────

class TestConfigDefaults:
    """The dataclass defaults are intentionally *no-ops* so legacy
    callers and tests are unaffected. The production arch.neuro
    config opts in to the protective values (asserted separately in
    test_arch_neuro_controller_safety.py)."""

    def test_controller_warmup_steps_default_is_zero(self):
        cfg = AdaptiveMixtureConfig()
        assert cfg.controller_warmup_steps == 0

    def test_max_step_delta_default_is_one(self):
        cfg = AdaptiveMixtureConfig()
        assert cfg.max_step_delta == pytest.approx(1.0)

    def test_entropy_ema_alpha_default_is_one(self):
        cfg = AdaptiveMixtureConfig()
        assert cfg.entropy_ema_alpha == pytest.approx(1.0)


# ── DSL parsing ─────────────────────────────────────────────────────

class TestDSLParsing:
    def test_parser_accepts_neuromechanical_block(self):
        body = """adaptive_mixture: {
            enabled: true, direction: "balance",
            controller_warmup_steps: 1500, max_step_delta: 0.05,
            entropy_ema_alpha: 0.2
        }"""
        cfg = parse_regularization_block(body)
        m = cfg.adaptive_mixture
        assert m.controller_warmup_steps == 1500
        assert m.max_step_delta == pytest.approx(0.05)
        assert m.entropy_ema_alpha == pytest.approx(0.2)


# ── Property 1: Startup grace ───────────────────────────────────────

class TestControllerWarmup:
    def _make(self, warmup=2000, initial=0.60, target_H=4.5):
        cfg = AdaptiveMixtureConfig(
            enabled=True, target_entropy=target_H,
            probe_interval=1, gamma=2.0,
            min_ratio=0.10, max_ratio=0.80,
            direction="balance",
            controller_warmup_steps=warmup,
            max_step_delta=1.0,           # disable slew limit for this test
            entropy_ema_alpha=1.0,         # disable EMA for this test
        )
        return AdaptiveMixtureController(cfg, initial_ratio=initial)

    def test_no_update_before_warmup(self):
        """The controller must NOT change ratio during the warmup
        window, even if observations would otherwise drive it to the
        floor."""
        m = self._make(warmup=200, initial=0.60)
        # Feed 100 high-entropy observations (would normally slam to min)
        for _ in range(100):
            m.observe_logits(_logits_with_entropy(10.0))
        # Still inside warmup → ratio unchanged
        assert m.ratio() == pytest.approx(0.60)

    def test_updates_after_warmup(self):
        m = self._make(warmup=50, initial=0.60)
        # Run past warmup with high-entropy observations
        for _ in range(200):
            m.observe_logits(_logits_with_entropy(10.0))
        # After warmup, balance + high H → ratio shrinks
        assert m.ratio() < 0.60

    def test_warmup_zero_disables_grace(self):
        """warmup=0 falls back to immediate-update behaviour."""
        m = self._make(warmup=0, initial=0.60)
        m.observe_logits(_logits_with_entropy(10.0))
        assert m.ratio() < 0.60


# ── Property 2: Slew-rate limit ─────────────────────────────────────

class TestSlewRateLimit:
    def _make(self, max_delta=0.03, initial=0.60, target_H=4.5):
        cfg = AdaptiveMixtureConfig(
            enabled=True, target_entropy=target_H,
            probe_interval=1, gamma=2.0,
            min_ratio=0.05, max_ratio=0.95,
            direction="balance",
            controller_warmup_steps=0,    # disable grace for this test
            max_step_delta=max_delta,
            entropy_ema_alpha=1.0,         # disable EMA for this test
        )
        return AdaptiveMixtureController(cfg, initial_ratio=initial)

    def test_delta_bounded(self):
        """One update can never move the ratio by more than max_step_delta."""
        m = self._make(max_delta=0.03, initial=0.60, target_H=2.0)
        before = m.ratio()
        # Big imbalance: H_t=10 vs H_target=2 → gain = (2/10)^2 = 0.04
        # Without limiter: 0.60 × 0.04 = 0.024 → clip to 0.05 (delta ~0.55)
        m.observe_logits(_logits_with_entropy(10.0))
        after = m.ratio()
        assert abs(after - before) <= 0.03 + 1e-6, (
            f"Δratio = {after - before:.4f}, max allowed = 0.03"
        )

    def test_repeated_updates_accumulate(self):
        """N updates can move the ratio at most N × max_step_delta."""
        m = self._make(max_delta=0.05, initial=0.80, target_H=2.0)
        # 10 updates pushing toward min
        for _ in range(10):
            m.observe_logits(_logits_with_entropy(10.0))
        # Total deviation ≤ 10 × 0.05 = 0.50
        assert m.ratio() >= 0.80 - 0.50 - 1e-6
        # And strictly less than 0.80 (controller is active)
        assert m.ratio() < 0.80

    def test_no_limit_when_step_delta_one(self):
        """max_step_delta=1.0 effectively disables the limiter — one
        update reaches the unlimited target in a single step. (We
        verify the result matches what a limit-free controller would
        produce, not a specific value; vocab=256 caps H at log(256).)"""
        m_limited = self._make(max_delta=1.0, initial=0.80, target_H=2.0)
        m_unlim = self._make(max_delta=1.0, initial=0.80, target_H=2.0)
        m_limited.observe_logits(_logits_with_entropy(10.0))
        m_unlim.observe_logits(_logits_with_entropy(10.0))
        assert m_limited.ratio() == pytest.approx(m_unlim.ratio())
        # And the controller did move substantially in one shot.
        assert abs(m_limited.ratio() - 0.80) > 0.5


# ── Property 3: EMA on entropy observation ──────────────────────────

class TestEntropyEMA:
    def _make(self, alpha=0.1, initial=0.60, target_H=4.5):
        cfg = AdaptiveMixtureConfig(
            enabled=True, target_entropy=target_H,
            probe_interval=1, gamma=2.0,
            min_ratio=0.05, max_ratio=0.95,
            direction="balance",
            controller_warmup_steps=0,
            max_step_delta=1.0,
            entropy_ema_alpha=alpha,
        )
        return AdaptiveMixtureController(cfg, initial_ratio=initial)

    def test_ema_dampens_single_spike(self):
        """A single high-entropy spike, smoothed by EMA, should move
        the ratio LESS than a sustained high-entropy stream."""
        m_smooth = self._make(alpha=0.1)   # τ = 10 updates
        m_sharp = self._make(alpha=1.0)     # no smoothing
        # First feed both a baseline observation at target_H
        m_smooth.observe_logits(_logits_with_entropy(4.5))
        m_sharp.observe_logits(_logits_with_entropy(4.5))
        # Then ONE spike at H=8
        m_smooth.observe_logits(_logits_with_entropy(8.0))
        m_sharp.observe_logits(_logits_with_entropy(8.0))
        # The smoothed controller should have moved less
        delta_smooth = abs(m_smooth.ratio() - 0.60)
        delta_sharp = abs(m_sharp.ratio() - 0.60)
        assert delta_smooth < delta_sharp

    def test_ema_alpha_one_disables_smoothing(self):
        """alpha=1.0 uses the raw observation each step."""
        m = self._make(alpha=1.0, target_H=4.5)
        m.observe_logits(_logits_with_entropy(4.5))   # baseline
        r1 = m.ratio()
        m.observe_logits(_logits_with_entropy(10.0))  # big spike
        r2 = m.ratio()
        # Big change expected
        assert abs(r2 - r1) > 0.05


# ── Combined scenario: replay the 2026-06-03 failure mode ──────────

class TestRegressionScenario:
    """Reproduce the exact failure trace: random-init LM → high-entropy
    logits → controller would slam from 0.60 → 0.10 in 20 steps.

    With neuromechanical stabilisation (warmup=2000 + slew=0.03 +
    EMA=0.1) the ratio must NOT leave a sensible band during the
    first 1000 steps."""

    def test_first_1000_steps_stable(self):
        cfg = AdaptiveMixtureConfig(
            enabled=True, target_entropy=4.5,
            probe_interval=1, gamma=2.0,
            min_ratio=0.10, max_ratio=0.50,
            direction="balance",
            # Defaults from the new neuromechanical design
            controller_warmup_steps=2000,
            max_step_delta=0.03,
            entropy_ema_alpha=0.1,
        )
        m = AdaptiveMixtureController(cfg, initial_ratio=0.60)
        # 1000 observations of random-init-style logits (H ≈ 10.5)
        for _ in range(1000):
            m.observe_logits(_logits_with_entropy(10.5))
        # During the entire warmup window the ratio must stay at 0.60
        assert m.ratio() == pytest.approx(0.60), (
            f"controller fired during warmup: ratio={m.ratio():.4f}"
        )

    def test_post_warmup_with_slew_limit_bounded_drift(self):
        """After warmup, with the slew limit, the controller may
        adjust but the move is GRADUAL — it takes many probes to
        reach the floor instead of one bang-bang step."""
        cfg = AdaptiveMixtureConfig(
            enabled=True, target_entropy=4.5,
            probe_interval=1, gamma=2.0,
            min_ratio=0.10, max_ratio=0.50,
            direction="balance",
            controller_warmup_steps=100,
            max_step_delta=0.03,
            entropy_ema_alpha=0.1,
        )
        m = AdaptiveMixtureController(cfg, initial_ratio=0.40)
        # Pass warmup
        for _ in range(100):
            m.observe_logits(_logits_with_entropy(4.5))
        # After warmup the controller is at the initial 0.40.
        assert m.ratio() == pytest.approx(0.40)
        # 3 post-warmup probes of high-entropy: with max_step_delta=0.03
        # the ratio can drop at most 3*0.03=0.09 → still above floor.
        for _ in range(3):
            m.observe_logits(_logits_with_entropy(10.5))
        assert m.ratio() >= 0.40 - 3 * 0.03 - 1e-6
        assert m.ratio() > 0.10   # nowhere near the floor yet
        # Eventually (many probes) it does reach the floor.
        for _ in range(50):
            m.observe_logits(_logits_with_entropy(10.5))
        assert m.ratio() == pytest.approx(0.10)
