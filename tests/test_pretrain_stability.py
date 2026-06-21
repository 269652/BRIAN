"""Strict-TDD regression tests for the three pretrain-stability fixes
diagnosed from the live Lightning training log on 2026-06-18
(job ln-20260618-163415-563b, full trajectory at logs/downloads/).

Three problems found:

  #1  GRADIENT SPIKES — 9 catastrophic spikes (gnorm > 1000) caused
      PPL regressions of 20–50 points each. Root cause: SmolLM has
      ``grad_clip = 1.0`` AND ``divisive_grad_c = 5.0`` but the
      harness logic treats them as mutually exclusive — when
      ``div_c > 0``, hard clip is skipped entirely. Divisive at c=5
      caps post-norm gradients at ~5.0 asymptote, which is too
      permissive for a model that's been observed to spike.

  #2  NFO α STUCK AT ZERO — ``alpha_init=0.0`` (ReZero pattern) is
      paired with ``nn.init.zeros_(read_out.weight)`` which creates a
      JOINT zero gradient deadlock::

          ∂L/∂α          = read_out(field) · ∂L/∂h  = 0 · ... = 0
          ∂L/∂read_out   = α · field · ∂L/∂h        = 0 · ... = 0

      Both can never lift off → the entire NFO block contributes
      zero gradient throughout training. Confirmed in log: 217/217
      samples report ``α = 0.000``.

  #3  DAR/PCC FIRE TOO LATE — SmolLM ``activation_step = 4000``
      means in a 10k-step run, the OOD interventions are disabled
      for the first 40% of training. Capacity-First protocol was
      well-meant, but train PPL plateaus around step 2500 so we're
      leaving 1500 steps of regularisation runway on the table.

The fixes:

  #1  Add ``TrainingConfig.grad_clip_after_divisive: bool`` (default
      False for backward compat) + ``divisive_then_hard_clip`` helper
      in gif7.py. Set True in SmolLM arch.

  #2  Change SmolLM nfo block: ``alpha_init: 0.0 → 0.01``. The
      forward pass is still baseline-identical at init because
      ``read_out.weight`` is zero — but now ∂L/∂read_out can flow.

  #3  Change SmolLM regularization block:
      ``activation_step: 4000 → 2000`` (== warmup_steps, so DAR/PCC
      activate at the end of warmup).

Run alone::

    pytest tests/test_pretrain_stability.py -v

Or one group::

    pytest tests/test_pretrain_stability.py::TestGradClipSafetyNet -v
    pytest tests/test_pretrain_stability.py::TestNFORezeroDeadlock -v
    pytest tests/test_pretrain_stability.py::TestDARPCCActivation -v
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOLLM_ARCH = REPO_ROOT / "architectures" / "SmolLM" / "arch.neuro"


# ──────────────────────────────────────────────────────────────────
# Group G — Gradient clipping safety net
# ──────────────────────────────────────────────────────────────────
#
# These tests pin the contract that ``divisive_grad_c`` and
# ``grad_clip`` COMPOSE when ``grad_clip_after_divisive=True`` and
# remain mutually exclusive when False (back-compat).

class TestGradClipSafetyNet:
    """Divisive grad norm + hard clip safety ceiling."""

    def _make_model_with_huge_grads(self, grad_value: float = 100.0
                                    ) -> nn.Module:
        """Linear layer with controllable gradient magnitude.

        gnorm = |grad_value| * sqrt(n_params).  For Linear(5,5) bias=False,
        that's |grad_value| * sqrt(25) = 5 * |grad_value|.
        """
        m = nn.Linear(5, 5, bias=False)
        m.weight.grad = torch.full_like(m.weight, grad_value)
        return m

    def test_G1_divisive_alone_bounds_at_c_asymptote(self):
        """REGRESSION: divisive-only mode (current behavior) caps
        post-norm gnorm at ≈ c as ||g|| → ∞. With grad_value=100,
        gnorm = 500, c = 5: post-norm ≈ 5 / sqrt(1 + 25/250000) ≈ 5.

        This test PROVES the bug: divisive at c=5 is far too
        permissive for a system that experiences true outlier
        gradients (the live log showed pre-norm gnorm=2511 with
        c=5 → post-norm ≈ 5.0, which is 5× the user's stated 1.0
        ceiling)."""
        m = self._make_model_with_huge_grads(grad_value=100.0)
        from neuroslm.emergent.gif7 import divisive_grad_normalize
        gnorm_pre, scale = divisive_grad_normalize(m.parameters(), c=5.0)
        # gnorm_pre = 100 * sqrt(25) = 500.0
        assert abs(gnorm_pre - 500.0) < 0.5, gnorm_pre
        # post-norm = 500 * scale ≈ 5.0 (the c value)
        post_norm = float(m.weight.grad.norm())
        assert 4.0 < post_norm < 5.5, (
            f"Divisive at c=5 should cap post-norm at ≈5, got {post_norm}")

    def test_G2_compose_helper_enforces_hard_ceiling(self):
        """When `divisive_then_hard_clip(div_c=5, hard_clip=1.0)` is
        called, post-norm gnorm must be ≤ 1.0 (the hard ceiling
        wins). This is the SAFETY NET that prevents the residual
        post-divisive 5.0-norm gradients from harming training."""
        # This import will fail until we add the helper — that's the
        # whole point of TDD.
        from neuroslm.emergent.gif7 import divisive_then_hard_clip
        m = self._make_model_with_huge_grads(grad_value=100.0)
        gnorm_pre, scale_div, gnorm_post = divisive_then_hard_clip(
            m.parameters(), div_c=5.0, hard_clip=1.0)
        assert abs(gnorm_pre - 500.0) < 0.5, gnorm_pre
        # After the chain, gradient norm must be <= 1.0 (hard ceiling)
        post_norm = float(m.weight.grad.norm())
        assert post_norm <= 1.0 + 1e-4, (
            f"Compose helper must enforce hard ceiling of 1.0, "
            f"got post-norm = {post_norm}")

    def test_G3_compose_helper_returns_three_metrics(self):
        """The compose helper returns ``(gnorm_pre, scale_div,
        gnorm_post)`` so monitors can log the full chain."""
        from neuroslm.emergent.gif7 import divisive_then_hard_clip
        m = self._make_model_with_huge_grads(grad_value=10.0)
        result = divisive_then_hard_clip(
            m.parameters(), div_c=5.0, hard_clip=1.0)
        assert isinstance(result, tuple) and len(result) == 3, (
            f"Expected 3-tuple (pre, scale_div, post), got {result!r}")
        gnorm_pre, scale_div, gnorm_post = result
        # gnorm_pre = 10 * sqrt(25) = 50
        assert abs(gnorm_pre - 50.0) < 0.5
        # scale_div = 5 / sqrt(25 + 2500) ≈ 0.0995
        assert 0.05 < scale_div < 0.15, scale_div
        # gnorm_post ≤ hard_clip
        assert gnorm_post <= 1.0 + 1e-4, gnorm_post

    def test_G4_compose_with_zero_hard_clip_is_divisive_only(self):
        """REGRESSION: if hard_clip <= 0, the helper degrades to
        plain divisive (preserves current behavior when user opts
        out of the safety net)."""
        from neuroslm.emergent.gif7 import divisive_then_hard_clip
        m = self._make_model_with_huge_grads(grad_value=100.0)
        gnorm_pre, scale_div, gnorm_post = divisive_then_hard_clip(
            m.parameters(), div_c=5.0, hard_clip=0.0)
        # Post-norm should be the divisive asymptote ≈ 5 (no hard clip)
        post = float(m.weight.grad.norm())
        assert 4.0 < post < 5.5, (
            f"hard_clip=0 should leave divisive-only behaviour, "
            f"got post-norm = {post}")

    def test_G5_compose_with_zero_div_c_is_hard_clip_only(self):
        """REGRESSION: if div_c <= 0, the helper degrades to plain
        hard clip (preserves legacy code paths that never used
        divisive)."""
        from neuroslm.emergent.gif7 import divisive_then_hard_clip
        m = self._make_model_with_huge_grads(grad_value=100.0)
        gnorm_pre, scale_div, gnorm_post = divisive_then_hard_clip(
            m.parameters(), div_c=0.0, hard_clip=1.0)
        # divisive disabled → scale_div should be 1.0
        assert abs(scale_div - 1.0) < 1e-6, (
            f"div_c=0 should produce scale_div=1.0, got {scale_div}")
        # Hard clip still enforces ceiling
        post = float(m.weight.grad.norm())
        assert post <= 1.0 + 1e-4, post

    def test_G6_training_config_has_grad_clip_after_divisive_field(self):
        """`TrainingConfig.grad_clip_after_divisive: bool = False`
        must be a queryable field with the right default. Default
        False preserves back-compat for every existing arch."""
        from neuroslm.dsl.training_config import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, "grad_clip_after_divisive"), (
            "TrainingConfig must expose grad_clip_after_divisive")
        assert cfg.grad_clip_after_divisive is False, (
            "Default must be False for back-compat with existing arches")

    def test_G7_dsl_parses_grad_clip_after_divisive(self):
        """`grad_clip_after_divisive: true` in the DSL training
        block must round-trip into TrainingConfig.grad_clip_after_divisive."""
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config(
            "grad_clip_after_divisive: true\ngrad_clip: 1.0\n"
            "divisive_grad_c: 5.0\n")
        assert cfg.grad_clip_after_divisive is True
        assert cfg.grad_clip == 1.0
        assert cfg.divisive_grad_c == 5.0

    def test_G8_smollm_arch_enables_grad_clip_after_divisive(self):
        """SmolLM (the active deploy arch) must opt into the safety
        net so the live training run benefits."""
        text = SMOLLM_ARCH.read_text(encoding="utf-8")
        assert re.search(r"grad_clip_after_divisive\s*:\s*true",
                         text, re.I), (
            "architectures/SmolLM/arch.neuro must include "
            "'grad_clip_after_divisive: true' in the training block")


# ──────────────────────────────────────────────────────────────────
# Group H — NFO ReZero deadlock
# ──────────────────────────────────────────────────────────────────
#
# These tests pin the contract that the NFO block can actually learn
# (gradient flows to its parameters under LM loss).

class TestNFORezeroDeadlock:
    """NFO α + read_out joint zero-init creates an unbreakable
    deadlock. With ``alpha_init > 0`` and zero-init read_out, the
    forward stays baseline-identical but gradient can now flow."""

    def _make_nfo(self, alpha_init: float):
        """Build a small NFO block with the given alpha_init."""
        from neuroslm.modules.neural_field_oscillator import (
            NeuralFieldOscillator, NFOConfig,
        )
        cfg = NFOConfig(n_osc=8, n_steps=1, alpha_init=alpha_init)
        return NeuralFieldOscillator(d_model=32, cfg=cfg)

    def _forward_backward(self, blk: nn.Module, B: int = 2, T: int = 4
                          ) -> None:
        """Run one forward + simple sum loss + backward through `blk`."""
        h = torch.randn(B, T, blk.d_model, requires_grad=True)
        y = blk(h)
        # Simple loss that depends on the full output. If the block
        # contributes 0 to y (deadlock), gradients to alpha + read_out
        # will be 0 too.
        loss = (y * torch.randn_like(y)).sum()
        loss.backward()

    def test_H1_legacy_zero_alpha_creates_zero_gradient_deadlock(self):
        """REGRESSION (proves the bug): With ``alpha_init=0`` AND
        zero-init read_out, both ``alpha.grad`` and
        ``read_out.weight.grad`` are exactly zero after a backward
        pass. This is the deadlock that froze NFO α at 0.000 for
        all 4340 logged steps."""
        blk = self._make_nfo(alpha_init=0.0)
        # Confirm read_out is zero-init (the partner condition)
        assert torch.all(blk.read_out.weight == 0.0), (
            "Test prereq: read_out.weight is zero-init")
        self._forward_backward(blk)
        # The deadlock: both α and read_out have zero gradient
        assert blk.alpha.grad is not None
        assert blk.read_out.weight.grad is not None
        assert torch.all(blk.alpha.grad == 0.0), (
            f"PROVES BUG: alpha.grad must be all-zero, "
            f"got max |grad| = {blk.alpha.grad.abs().max().item()}")
        assert torch.all(blk.read_out.weight.grad == 0.0), (
            f"PROVES BUG: read_out.weight.grad must be all-zero, "
            f"got max |grad| = {blk.read_out.weight.grad.abs().max().item()}")

    def test_H2_small_positive_alpha_unlocks_read_out_gradient(self):
        """With ``alpha_init=0.01`` and zero-init read_out, the
        gradient w.r.t. read_out.weight is non-zero — the block can
        finally learn. This is the FIX."""
        blk = self._make_nfo(alpha_init=0.01)
        assert torch.all(blk.read_out.weight == 0.0), (
            "Prereq: read_out.weight is still zero-init")
        self._forward_backward(blk)
        assert blk.read_out.weight.grad is not None
        max_grad = blk.read_out.weight.grad.abs().max().item()
        assert max_grad > 1e-8, (
            f"FIX must unlock learning: read_out.weight.grad must be "
            f"non-zero, got max |grad| = {max_grad}")

    def test_H3_small_positive_alpha_preserves_baseline_identity(self):
        """The CRITICAL contract: with read_out.weight=0, the block
        output equals the input regardless of ``alpha_init`` value.
        So setting ``alpha_init=0.01`` instead of 0.0 does NOT
        change initial behaviour — only the gradient topology."""
        torch.manual_seed(0)
        blk = self._make_nfo(alpha_init=0.01)
        h = torch.randn(2, 4, blk.d_model)
        with torch.no_grad():
            y = blk(h)
        # delta = alpha * read_out(field) = 0.01 * 0 = 0 → y == h
        assert torch.allclose(y, h, atol=1e-6), (
            f"Baseline identity broken: max |y - h| = "
            f"{(y - h).abs().max().item()} (expected ~0)")

    def test_H4_smollm_arch_has_nonzero_alpha_init_for_nfo(self):
        """SmolLM arch.neuro nfo block must have alpha_init > 0 so
        the live training run can learn the NFO oscillator field."""
        text = SMOLLM_ARCH.read_text(encoding="utf-8")
        # The nfo block spans 2 lines: find alpha_init within the
        # nfo {...} block.
        nfo_match = re.search(
            r"nfo\s*:\s*\{[^}]*alpha_init\s*:\s*([\d.eE+-]+)",
            text, re.DOTALL)
        assert nfo_match, (
            "Could not locate alpha_init within SmolLM nfo block")
        alpha = float(nfo_match.group(1))
        assert alpha > 0.0, (
            f"SmolLM nfo.alpha_init must be > 0 to escape the ReZero "
            f"deadlock (currently {alpha}). Recommended: 0.01.")


# ──────────────────────────────────────────────────────────────────
# Group I — DAR/PCC earlier activation
# ──────────────────────────────────────────────────────────────────
#
# These tests pin the contract that DAR/PCC are activated at the end
# of warmup (not 4× warmup later) so they actually fire for most of
# the training run.

class TestDARPCCActivation:
    """SmolLM's regularization.activation_step must be ≤ warmup_steps
    so DAR/PCC fire as soon as warmup completes, not 2000 steps
    later."""

    def _parse_reg_section(self, text: str) -> dict:
        """Pull `key: value` pairs from the SmolLM regularization
        block, robust to comments and inner braces."""
        # Find regularization { ... } body. Track brace depth.
        m = re.search(r"regularization\s*:\s*\{", text)
        assert m, "Could not find regularization block in SmolLM"
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[start:i - 1]
        # Strip comments
        body = re.sub(r"#[^\n]*", "", body)
        # Find top-level "key: value" pairs (skip nested braces).
        out = {}
        for line in body.splitlines():
            line = line.strip().rstrip(",").rstrip(";")
            # Skip lines that open a nested block
            if "{" in line:
                continue
            mm = re.match(r"([a-z_][a-z0-9_]*)\s*:\s*(-?[\d.]+)\s*$",
                          line, re.I)
            if mm:
                out[mm.group(1)] = float(mm.group(2))
        return out

    def test_I1_smollm_activation_step_le_warmup(self):
        """``activation_step`` must be ≤ ``warmup_steps`` so DAR/PCC
        fire as soon as the warmup ramp finishes. Previously
        ``activation_step=4000`` with ``warmup_steps=2000`` meant
        interventions were dead-gated for an extra 2000 steps."""
        text = SMOLLM_ARCH.read_text(encoding="utf-8")
        reg = self._parse_reg_section(text)
        assert "warmup_steps" in reg, reg
        assert "activation_step" in reg, reg
        warmup = int(reg["warmup_steps"])
        activation = int(reg["activation_step"])
        assert activation <= warmup, (
            f"activation_step ({activation}) must be ≤ warmup_steps "
            f"({warmup}) so DAR/PCC fire when warmup completes, not "
            f"after a 2nd dead window. Recommended: activation_step "
            f"= warmup_steps = 2000.")

    def test_I2_smollm_activation_step_le_2500(self):
        """Hard upper bound: in a 10k-step run we want DAR/PCC active
        for ≥75% of training, i.e. activation_step ≤ 2500."""
        text = SMOLLM_ARCH.read_text(encoding="utf-8")
        reg = self._parse_reg_section(text)
        activation = int(reg.get("activation_step", 0))
        assert activation <= 2500, (
            f"activation_step ({activation}) must be ≤ 2500 so "
            f"interventions get ≥75% runway in a 10k-step run.")

    def _parse_isotropy_weight(self, text: str) -> float:
        """Extract isotropy.weight from the regularization block."""
        m = re.search(r"isotropy\s*:\s*\{[^}]*weight\s*:\s*([\d.eE+-]+)",
                      text, re.DOTALL)
        assert m, "Could not locate isotropy.weight in SmolLM arch"
        return float(m.group(1))

    def _parse_bma_field(self, text: str, field: str) -> float:
        """Extract a top-level bma_<field> value from arch.neuro."""
        m = re.search(rf"{re.escape(field)}\s*:\s*([\d.eE+-]+)", text)
        assert m, f"Could not locate {field} in SmolLM arch"
        return float(m.group(1))

    def test_I4_isotropy_weight_sufficient_to_prevent_rank_collapse(self):
        """isotropy.weight must be ≥ 0.04 to meaningfully counteract rank collapse.

        2026-06-21: weight=0.005 was 10× too small — isotropy contribution at
        step 300 was ~0.0003 vs LM loss ~5.5 (ratio 0.005%). At ≥0.04 the
        contribution is ~0.003 and begins to resist erank→7 collapse.
        """
        text = SMOLLM_ARCH.read_text(encoding="utf-8")
        w = self._parse_isotropy_weight(text)
        assert w >= 0.04, (
            f"isotropy.weight ({w}) is too small to prevent rank collapse. "
            f"Minimum: 0.04 (10× the broken 0.005 baseline). "
            f"Erank collapsed 53→7 by step 300 with weight=0.005."
        )

    def test_I5_bma_ramp_end_tight_enough_for_early_rank_collapse(self):
        """bma_ramp_end must be ≤ 1500 so BMA reaches ≥20% weight by step 300.

        2026-06-21: bma_ramp_end=3000 meant only 10% weight at step 300,
        contributing 0.005 — far too small while erank collapses 53→7.
        At ramp_end=1500, step-300 weight is 20% (0.010), enough to register.
        """
        text = SMOLLM_ARCH.read_text(encoding="utf-8")
        ramp_end = self._parse_bma_field(text, "bma_ramp_end")
        assert ramp_end <= 1500, (
            f"bma_ramp_end ({ramp_end}) ramps BMA too slowly: at step 300 "
            f"only {100*300/ramp_end:.0f}% weight. Must be ≤1500 so BMA "
            f"reaches ≥20% by step 300 (before erank collapse completes)."
        )

    def test_I3_isotropy_activation_unchanged(self):
        """REGRESSION: isotropy fires before DAR/PCC (rank-collapse guard).

        2026-06-21: changed from 1000 → 0 so isotropy fires immediately.
        Erank collapsed by step 200-300 in the resumed run; isotropy must
        be active from step 0 to counter this.
        """
        text = SMOLLM_ARCH.read_text(encoding="utf-8")
        reg = self._parse_reg_section(text)
        iso_step = int(reg.get("isotropy_activation_step", -1))
        act_step = int(reg.get("activation_step", 0))
        # Isotropy must activate at or before DAR/PCC (rank-collapse guard)
        assert iso_step < act_step, (
            f"isotropy_activation_step ({iso_step}) must be < activation_step ({act_step}). "
            f"Isotropy guards rank-collapse and must fire before the main reg gate."
        )
        assert iso_step == 0, (
            f"isotropy_activation_step should be 0 (fires immediately to prevent "
            f"rank collapse in steps 0-300), got {iso_step}."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
