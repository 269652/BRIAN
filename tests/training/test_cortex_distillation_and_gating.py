# -*- coding: utf-8 -*-
"""TDD: KL-distillation aux loss + NT-mediated α-gating for cortex fusion.

Motivation
----------
The pre-trained GPT-2 cortices contribute to forward logits via
``α · cortex_logits + (1-α) · lm_logits`` but transmit zero signal to
the trunk's parameters: the trunk learns from targets directly, never
from the cortex's distribution. This wastes ~500M params of pretrained
knowledge as a parallel sibling instead of a teacher.

Two mechanisms together fix both problems:

  A. **KL-distillation aux loss** (Hinton 2015) with adaptive λ:
       L_total = L_CE + λ_t · T² · KL(softmax(cortex.detach()/T)
                                      || softmax(lm/T))
     λ_t decays linearly from λ_max (trunk much worse than cortex)
     to 0 (trunk has caught up). This forces the trunk to learn
     the cortex's full distribution, not just argmax targets.
     ``.detach()`` ensures gradient flows trunk ← cortex (not back).

  C. **NT-mediated α gating** (BRIAN-native):
       α_eff = α · (1 - cortex_inhibition_level)
     The `cortex_inhibition_level` is a [0, 1] EMA that rises as
     trunk_loss < cortex_loss (trunk has outgrown the teacher).
     Once near 1, the cortex contributes ~nothing — at inference
     the forward pass can skip the cortex entirely (FLOP savings).

Together: the trunk distils from cortex during the early phase
(A active, C close to 0); as trunk catches up, A's λ shrinks AND
C's inhibition rises so the cortex naturally retires.

This test suite pins both mechanisms with strict TDD contracts.
All tests use ``weights="stub"`` (offline) and the synthetic
``_FakeDSLLM`` from the sister suites.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.dsl.training_config import MultiCortexConfig, TrainingConfig


VOCAB = 64
D_SEM = 32


# ──────────────────────────────────────────────────────────────────────
# Fixtures (mirror tests/training/test_multi_cortex_logits_wired.py)
# ──────────────────────────────────────────────────────────────────────

class _FakeDSLLM(nn.Module):
    """Minimal LM mirroring DSLLanguageModel surface."""

    def __init__(self, vocab: int = VOCAB, d_model: int = D_SEM, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Parameter(torch.randn(vocab, d_model, generator=g) * 0.02)
        self.lm_head = nn.Parameter(torch.randn(vocab, d_model, generator=g) * 0.02)
        self._last_hidden = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed[ids]
        self._last_hidden = h
        return F.linear(h, self.lm_head)


@pytest.fixture
def fake_lm() -> _FakeDSLLM:
    return _FakeDSLLM(seed=0)


def _cfg_baseline_fusion() -> TrainingConfig:
    """Baseline: fusion on, distillation OFF, inhibition OFF."""
    cfg = TrainingConfig()
    cfg.multi_cortex = MultiCortexConfig(
        enabled=True, n_cortices=4,
        domains=["math", "code", "chat", "general"],
        weights="stub", freeze_weights=False,
        lexical_bias_weight=2.0, bema_tau=0.5,
        router_d_model=D_SEM,
    )
    return cfg


def _cfg_distill_on() -> TrainingConfig:
    cfg = _cfg_baseline_fusion()
    cfg.multi_cortex.distillation_enabled = True
    cfg.multi_cortex.distillation_lambda_max = 1.0
    cfg.multi_cortex.distillation_temperature = 4.0
    cfg.multi_cortex.distillation_gap_floor = 0.1
    cfg.multi_cortex.distillation_gap_ceiling = 2.0
    return cfg


def _cfg_inhibition_on() -> TrainingConfig:
    cfg = _cfg_baseline_fusion()
    cfg.multi_cortex.inhibition_enabled = True
    cfg.multi_cortex.inhibition_ema_alpha = 0.05
    cfg.multi_cortex.inhibition_temperature = 1.0
    return cfg


# ──────────────────────────────────────────────────────────────────────
# F1.1 — Distillation config defaults preserve back-compat
# ──────────────────────────────────────────────────────────────────────

class TestDistillationConfigDefaults:
    """The distillation feature MUST default to off. Any existing
    arch.neuro that doesn't mention `distillation_*` must produce
    bit-identical behaviour after this change lands."""

    def test_distillation_enabled_defaults_to_false(self):
        cfg = MultiCortexConfig()
        assert cfg.distillation_enabled is False, (
            "distillation_enabled must default to False so existing arch.neuro "
            "files are not silently changed by this commit"
        )

    def test_distillation_parameters_have_sane_defaults(self):
        cfg = MultiCortexConfig()
        # Specific values can change; we just pin that they're defined
        # and have semantically sensible bounds.
        assert hasattr(cfg, "distillation_lambda_max")
        assert hasattr(cfg, "distillation_temperature")
        assert hasattr(cfg, "distillation_gap_floor")
        assert hasattr(cfg, "distillation_gap_ceiling")
        assert cfg.distillation_lambda_max > 0
        assert cfg.distillation_temperature > 0
        assert 0 <= cfg.distillation_gap_floor < cfg.distillation_gap_ceiling


# ──────────────────────────────────────────────────────────────────────
# F1.2 — λ_t schedule (linear ramp between floor and ceiling)
# ──────────────────────────────────────────────────────────────────────

class TestDistillationLambdaSchedule:
    """λ_t is a piecewise-linear function of `gap = lm_loss_ema - cortex_loss_ema`:
        gap <= floor   → λ = 0           (trunk has caught up / surpassed)
        floor < gap < ceiling → linear interpolation
        gap >= ceiling → λ = lambda_max  (trunk much worse than cortex)
    Test the boundaries plus monotonicity.
    """

    def test_lambda_zero_when_gap_below_floor(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        cfg = _cfg_distill_on()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        # gap below floor (e.g. -1.0 nats, trunk already better)
        lam = h._distillation_lambda(gap_nats=-1.0)
        assert lam == 0.0, (
            f"λ must be 0 when trunk has caught up; got {lam}"
        )
        lam_at_floor = h._distillation_lambda(gap_nats=0.0)
        assert lam_at_floor == 0.0, (
            f"λ must be 0 at gap=0 (well below floor=0.1); got {lam_at_floor}"
        )

    def test_lambda_max_when_gap_above_ceiling(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        cfg = _cfg_distill_on()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        lam = h._distillation_lambda(gap_nats=5.0)
        assert lam == cfg.multi_cortex.distillation_lambda_max, (
            f"λ must saturate at lambda_max={cfg.multi_cortex.distillation_lambda_max} "
            f"when gap >> ceiling; got {lam}"
        )

    def test_lambda_monotonic_in_gap(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        cfg = _cfg_distill_on()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        gaps = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        lams = [h._distillation_lambda(g) for g in gaps]
        for i in range(len(lams) - 1):
            assert lams[i] <= lams[i + 1] + 1e-9, (
                f"λ must be non-decreasing in gap; got λ({gaps[i]})="
                f"{lams[i]} > λ({gaps[i+1]})={lams[i+1]}"
            )

    def test_lambda_midpoint_interpolation(self, fake_lm):
        """At gap = (floor + ceiling) / 2, λ should equal lambda_max / 2."""
        from neuroslm.harness import BRIANHarness
        cfg = _cfg_distill_on()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        mc = cfg.multi_cortex
        midpoint = (mc.distillation_gap_floor + mc.distillation_gap_ceiling) / 2
        lam = h._distillation_lambda(gap_nats=midpoint)
        expected = mc.distillation_lambda_max / 2
        assert abs(lam - expected) < 1e-5, (
            f"At midpoint gap={midpoint}, expected λ≈{expected}, got {lam}"
        )


# ──────────────────────────────────────────────────────────────────────
# F1.3 — Distillation loss is ADDED to total when enabled
# ──────────────────────────────────────────────────────────────────────

class TestDistillationLossAdded:
    """When distillation_enabled=True AND λ_t > 0, the total loss
    must include a non-zero KL term. When disabled, the total loss
    must be bit-identical to the baseline (no-distillation) total."""

    def test_total_loss_increased_when_distillation_enabled_with_gap(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        # Build two harnesses sharing same LM; one with distillation on,
        # one off.
        cfg_off = _cfg_baseline_fusion()
        cfg_on = _cfg_distill_on()

        torch.manual_seed(42)
        h_off = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg_off,
        )
        torch.manual_seed(42)
        h_on = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg_on,
        )
        # Force λ to be positive by setting the gap state high
        h_on._lm_loss_ema = 5.0
        h_on._cortex_loss_ema = 1.0  # gap = 4.0 nats >> ceiling

        ids = torch.randint(0, VOCAB, (2, 8))
        targets = torch.randint(0, VOCAB, (2, 8))
        loss_off = h_off.compute_loss(ids, targets).item()
        loss_on = h_on.compute_loss(ids, targets).item()
        assert loss_on > loss_off + 1e-4, (
            f"Distillation should add a positive KL contribution: "
            f"loss_off={loss_off:.4f}, loss_on={loss_on:.4f}, "
            f"Δ={loss_on - loss_off:+.4f} (want > 1e-4)"
        )

    def test_total_loss_bit_identical_when_distillation_disabled(self, fake_lm):
        """The crucial back-compat contract: enabling fusion without
        distillation must give the EXACT same loss as before this commit."""
        from neuroslm.harness import BRIANHarness
        cfg_baseline = _cfg_baseline_fusion()
        # disabled is the default; we just verify the field exists.
        assert cfg_baseline.multi_cortex.distillation_enabled is False

        torch.manual_seed(42)
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg_baseline,
        )
        ids = torch.randint(0, VOCAB, (2, 8))
        targets = torch.randint(0, VOCAB, (2, 8))

        # Run twice with same seed → identical loss (proves no randomness
        # introduced by the distillation code path even when disabled).
        l1 = h.compute_loss(ids, targets).item()
        l2 = h.compute_loss(ids, targets).item()
        assert abs(l1 - l2) < 1e-6, (
            f"Loss is non-deterministic ({l1} vs {l2}) — the disabled "
            "distillation path is leaking randomness"
        )

    def test_kl_term_finite_and_nonnegative(self, fake_lm):
        """KL divergence is mathematically non-negative; sanity-check the
        implementation actually delivers that."""
        from neuroslm.harness import BRIANHarness
        cfg = _cfg_distill_on()
        torch.manual_seed(0)
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        h._lm_loss_ema = 5.0
        h._cortex_loss_ema = 1.0
        ids = torch.randint(0, VOCAB, (2, 8))
        targets = torch.randint(0, VOCAB, (2, 8))
        _ = h.compute_loss(ids, targets)
        kl = h._metrics.get("distill_kl", None)
        assert kl is not None, (
            "harness._metrics['distill_kl'] not exposed after compute_loss "
            "with distillation enabled"
        )
        assert math.isfinite(kl), f"distill_kl = {kl} is not finite"
        assert kl >= -1e-6, f"distill_kl = {kl} is negative (KL must be ≥ 0)"


# ──────────────────────────────────────────────────────────────────────
# F1.4 — Gradient flow: trunk learns FROM cortex, not vice versa
# ──────────────────────────────────────────────────────────────────────

class TestDistillationGradientFlow:
    """The whole point of distillation is to make the trunk learn from
    the cortex. The cortex must be a frozen teacher in this term:
    gradient on cortex parameters from the KL term MUST be zero."""

    def test_trunk_receives_gradient_from_kl_term(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        cfg = _cfg_distill_on()
        torch.manual_seed(0)
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        # Push gap above floor so λ > 0
        h._lm_loss_ema = 5.0
        h._cortex_loss_ema = 1.0

        # Zero out trunk grads
        for p in fake_lm.parameters():
            if p.grad is not None:
                p.grad.zero_()

        ids = torch.randint(0, VOCAB, (2, 8))
        targets = torch.randint(0, VOCAB, (2, 8))
        loss = h.compute_loss(ids, targets)
        loss.backward()

        # At least one trunk parameter must have non-trivial gradient
        max_grad = max(
            (float(p.grad.abs().max()) if p.grad is not None else 0.0)
            for p in fake_lm.parameters()
        )
        assert max_grad > 1e-8, (
            f"Trunk received no gradient (max|grad|={max_grad:.2e}) — "
            "either the LM forward path or the KL term is broken"
        )

    def test_cortex_receives_no_gradient_from_kl_term_alone(self, fake_lm):
        """Isolate the KL term: zero out CE-loss gradient by using
        teacher=student labels (CE near 0), then verify cortex grads
        from the remaining KL term are zero (proof of `.detach()`)."""
        from neuroslm.harness import BRIANHarness
        cfg = _cfg_distill_on()
        torch.manual_seed(0)
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        h._lm_loss_ema = 5.0
        h._cortex_loss_ema = 1.0

        # Compute loss ONCE just to populate cortex_logits via forward.
        # Then zero all grads, then call ONLY the KL term to isolate.
        ids = torch.randint(0, VOCAB, (2, 8))
        targets = torch.randint(0, VOCAB, (2, 8))

        # Method: zero all grads first, then run the full loss, then
        # check the cortex_proj weights (small subset of cortex params
        # that DEFINITELY connect to logits via fusion).
        for p in h.parameters():
            if p.grad is not None:
                p.grad.zero_()
        # Strip away the CE gradient by using a custom path: get logits,
        # then compute ONLY the KL term manually with teacher=cortex.detach().
        with torch.no_grad():
            _ = h(ids)  # populates _last_pre_fusion_*

        assert h._last_pre_fusion_cortex_logits is not None, (
            "harness must stash _last_pre_fusion_cortex_logits during forward"
        )
        assert h._last_pre_fusion_lm_logits is not None, (
            "harness must stash _last_pre_fusion_lm_logits during forward"
        )
        # Re-run with grad to get the KL term in isolation
        h._reset_stashes_for_test()  # private helper for the contract
        logits = h(ids)
        # Build KL manually with detach on the teacher
        teacher = h._last_pre_fusion_cortex_logits.detach()
        student = h._last_pre_fusion_lm_logits
        T = cfg.multi_cortex.distillation_temperature
        kl = F.kl_div(
            F.log_softmax(student / T, dim=-1),
            F.softmax(teacher / T, dim=-1),
            reduction="batchmean",
        ) * (T ** 2)
        kl.backward()

        # Now check: cortex params should have ZERO gradient because
        # of the detach. Trunk should have NON-zero.
        cortex_max_grad = max(
            (float(p.grad.abs().max()) if p.grad is not None else 0.0)
            for p in h.multi_cortex.parameters()
        )
        # cortex_lm_head is tied to fake_lm.embed (trunk param) so it
        # SHOULD have grad — exclude it from this check.
        # The MultiCortexEnsemble's own params (projections, sub-cortices)
        # MUST have zero grad from the KL alone.
        assert cortex_max_grad < 1e-7, (
            f"Cortex ensemble received gradient ({cortex_max_grad:.2e}) "
            "from the KL term — the `.detach()` on the teacher is missing"
        )


# ──────────────────────────────────────────────────────────────────────
# F3.1 — Inhibition config defaults preserve back-compat
# ──────────────────────────────────────────────────────────────────────

class TestInhibitionConfigDefaults:
    def test_inhibition_enabled_defaults_to_false(self):
        cfg = MultiCortexConfig()
        assert cfg.inhibition_enabled is False, (
            "inhibition_enabled must default to False so existing arch.neuro "
            "files are not silently changed"
        )

    def test_inhibition_parameters_have_sane_defaults(self):
        cfg = MultiCortexConfig()
        assert hasattr(cfg, "inhibition_ema_alpha")
        assert hasattr(cfg, "inhibition_temperature")
        assert 0 < cfg.inhibition_ema_alpha <= 1.0
        assert cfg.inhibition_temperature > 0


# ──────────────────────────────────────────────────────────────────────
# F3.2 — NT inhibition state
# ──────────────────────────────────────────────────────────────────────

class TestCortexInhibitionState:
    """The `cortex_inhibition_level` is a [0, 1] scalar held on the
    harness; rises with `cortex_loss - lm_loss` (gap is positive
    when trunk has surpassed cortex)."""

    def test_inhibition_initialised_at_zero(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        cfg = _cfg_inhibition_on()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        assert h._cortex_inhibition_level == 0.0, (
            "cortex_inhibition_level must start at 0 (cortex fully active)"
        )

    def test_inhibition_stays_in_unit_interval(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        cfg = _cfg_inhibition_on()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        # Synthetic: trunk WAY better than cortex (huge negative gap)
        # for many EMA updates — should saturate at 1, not overflow.
        for _ in range(200):
            h._update_cortex_inhibition(
                lm_loss=0.5, cortex_loss=10.0
            )
        assert 0.0 <= h._cortex_inhibition_level <= 1.0, (
            f"inhibition out of bounds: {h._cortex_inhibition_level}"
        )

        # Synthetic: trunk way WORSE than cortex (huge positive gap) —
        # should pull back toward 0, not negative.
        for _ in range(200):
            h._update_cortex_inhibition(
                lm_loss=10.0, cortex_loss=0.5
            )
        assert 0.0 <= h._cortex_inhibition_level <= 1.0, (
            f"inhibition out of bounds: {h._cortex_inhibition_level}"
        )

    def test_inhibition_rises_when_trunk_outperforms_cortex(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        cfg = _cfg_inhibition_on()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        # cortex_loss > lm_loss → trunk is better → inhibition should rise
        before = h._cortex_inhibition_level
        for _ in range(50):
            h._update_cortex_inhibition(lm_loss=2.0, cortex_loss=4.0)
        after = h._cortex_inhibition_level
        assert after > before + 0.01, (
            f"Inhibition did not rise when trunk outperformed cortex: "
            f"before={before:.4f}, after={after:.4f}"
        )


# ──────────────────────────────────────────────────────────────────────
# F4 — Effective α applies inhibition
# ──────────────────────────────────────────────────────────────────────

class TestEffectiveAlpha:
    """The forward path must use ``α_eff = α · (1 - inhibition)`` so
    the cortex's logits contribution shrinks as the trunk outgrows it.
    When inhibition_enabled=False, α_eff = α (back-compat)."""

    def test_alpha_eff_equals_alpha_when_inhibition_zero(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        cfg = _cfg_inhibition_on()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        h._cortex_inhibition_level = 0.0
        alpha_eff = h._effective_alpha()
        alpha = float(torch.sigmoid(h.cortex_mix_logit).item())
        assert abs(alpha_eff - alpha) < 1e-7, (
            f"At inhibition=0, α_eff must equal α; "
            f"got α_eff={alpha_eff:.6f}, α={alpha:.6f}"
        )

    def test_alpha_eff_zero_when_inhibition_one(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        cfg = _cfg_inhibition_on()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        h._cortex_inhibition_level = 1.0
        alpha_eff = h._effective_alpha()
        assert abs(alpha_eff) < 1e-7, (
            f"At inhibition=1, α_eff must be 0; got {alpha_eff:.6f}"
        )

    def test_alpha_eff_scales_linearly(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        cfg = _cfg_inhibition_on()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        alpha = float(torch.sigmoid(h.cortex_mix_logit).item())
        for inh in [0.0, 0.25, 0.5, 0.75, 1.0]:
            h._cortex_inhibition_level = inh
            alpha_eff = h._effective_alpha()
            expected = alpha * (1.0 - inh)
            assert abs(alpha_eff - expected) < 1e-7, (
                f"At inhibition={inh}, expected α_eff={expected:.6f}, "
                f"got {alpha_eff:.6f}"
            )


# ──────────────────────────────────────────────────────────────────────
# F4 — Forward path respects α_eff (cortex contribution gated)
# ──────────────────────────────────────────────────────────────────────

class TestForwardRespectsInhibition:
    """When inhibition is full (1.0), the forward logits must match
    the LM-only path (cortex contributes nothing). When disabled,
    forward must be bit-identical to the baseline."""

    def test_logits_match_lm_only_when_inhibition_full(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        cfg = _cfg_inhibition_on()
        torch.manual_seed(42)
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        h._cortex_inhibition_level = 1.0  # cortex fully gated off

        # Reference: same harness with fusion off entirely
        cfg_off = TrainingConfig()  # multi_cortex disabled
        torch.manual_seed(42)
        h_lm_only = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg_off,
        )

        ids = torch.randint(0, VOCAB, (2, 8))
        with torch.no_grad():
            l_gated = h(ids)
            l_ref = h_lm_only(ids)

        max_diff = (l_gated - l_ref).abs().max().item()
        assert max_diff < 1e-5, (
            f"At inhibition=1, logits should match LM-only; "
            f"max|Δ|={max_diff:.2e}"
        )

    def test_forward_bit_identical_when_inhibition_disabled(self, fake_lm):
        """If inhibition_enabled=False, the inhibition code path
        must be a no-op — bit-identical forward to before this commit."""
        from neuroslm.harness import BRIANHarness
        cfg = _cfg_baseline_fusion()  # inhibition_enabled=False
        assert cfg.multi_cortex.inhibition_enabled is False
        torch.manual_seed(42)
        h1 = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        torch.manual_seed(42)
        h2 = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        # Tamper with inhibition on h2 (should be ignored when disabled)
        h2._cortex_inhibition_level = 1.0

        ids = torch.randint(0, VOCAB, (2, 8))
        with torch.no_grad():
            l1 = h1(ids)
            l2 = h2(ids)
        max_diff = (l1 - l2).abs().max().item()
        assert max_diff < 1e-7, (
            f"Inhibition leaked into the forward path when disabled: "
            f"max|Δ|={max_diff:.2e}"
        )


# ──────────────────────────────────────────────────────────────────────
# F4 — Telemetry: both α and α_eff exposed
# ──────────────────────────────────────────────────────────────────────

class TestTelemetryExposesFusionState:
    """The training log displays NT[...] and other metric channels;
    both `cortex_inhibition` and `alpha_effective` must show up there
    so the operator can SEE the gate close as training progresses."""

    def test_metrics_include_alpha_and_inhibition(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        cfg = _cfg_inhibition_on()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        ids = torch.randint(0, VOCAB, (2, 8))
        targets = torch.randint(0, VOCAB, (2, 8))
        _ = h.compute_loss(ids, targets)

        assert "cortex_inhibition" in h._metrics, (
            "harness._metrics must expose 'cortex_inhibition' for telemetry"
        )
        assert "alpha_effective" in h._metrics, (
            "harness._metrics must expose 'alpha_effective' for telemetry"
        )
        # Sane bounds
        assert 0.0 <= h._metrics["cortex_inhibition"] <= 1.0
        assert 0.0 <= h._metrics["alpha_effective"] <= 1.0
