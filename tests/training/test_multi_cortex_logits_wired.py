# -*- coding: utf-8 -*-
"""TDD: the multitrunk's pretrained cortex weights must actually drive
the LM head.

Background
----------
Audit revealed that ``BRIANHarness._build_multi_cortex`` constructs the
``MultiCortexEnsemble`` (downloading the four GPT-2 family checkpoints
on first run) but the ensemble is then **never invoked anywhere in the
forward path** — only ``self.language_model(ids)`` reaches the logits.

Symptom: the operator's GPU run shows ``Loading weights: 100%`` four
times (292/76/148/148 safetensors), reports ``833.3M parameters``, then
trains at ``lm_loss ≈ 13.6``. The ``log(vocab=50257) ≈ 10.82`` baseline
sits *below* that value because a randomly-initialised ``nn.Linear`` LM
head produces "confidently wrong" logits on real tokens.

Root cause: the ensemble is dead weight in the graph. ~700M params (the
HF checkpoints) carry zero gradient and contribute zero signal to the
predicted distribution.

This suite pins five contracts that together force the cortex to drive
the loss:

  1. Enabling ``multi_cortex`` must change the logits — proves the
     ensemble output reaches the LM head.
  2. ``multi_cortex.parameters()`` must receive non-zero gradient on
     ``loss.backward()`` — proves it's in the autograd graph.
  3. A ``cortex_lm_head`` attribute must exist when fusion is enabled.
  4. The mixing weight must be a learnable parameter (not a constant)
     so the optimizer can decide how much to trust the cortex.
  5. Disabling ``multi_cortex`` (or setting ``fusion_mode="off"``) is
     back-compat — no cortex head, no logits change.

The tests use ``weights="stub"`` exclusively so they run offline.
"""
from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.dsl.training_config import MultiCortexConfig, TrainingConfig


VOCAB = 64
D_SEM = 32


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

class _FakeDSLLM(nn.Module):
    """A minimal language-model that mirrors the DSL's surface:

    * exposes ``self.embed`` as an ``nn.Parameter`` of shape
      ``(vocab, d_model)`` (the harness uses this for tied-weights
      initialisation of ``cortex_lm_head``).
    * exposes ``self._last_hidden`` after forward (PR2 regularizers
      contract).

    Deterministic init via the constructor's ``seed`` so tests can
    compare two harnesses sharing the same language model.
    """

    def __init__(self, vocab: int = VOCAB, d_model: int = D_SEM,
                 seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Parameter(torch.randn(vocab, d_model, generator=g) * 0.02)
        self.lm_head = nn.Parameter(torch.randn(vocab, d_model, generator=g) * 0.02)
        self._last_hidden = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed[ids]                       # (B, T, D)
        self._last_hidden = h
        return F.linear(h, self.lm_head)          # (B, T, V)


@pytest.fixture
def fake_lm() -> _FakeDSLLM:
    return _FakeDSLLM(seed=0)


@pytest.fixture
def disabled_cfg() -> TrainingConfig:
    """Legacy: multi_cortex.enabled=False (default)."""
    return TrainingConfig()


@pytest.fixture
def stub_fusion_cfg() -> TrainingConfig:
    """Production-shaped: multi_cortex enabled with stub weights.

    ``router_d_model=D_SEM`` so the ensemble's d_target matches the
    harness's d_sem (no dim-mismatch in the fusion path).
    """
    cfg = TrainingConfig()
    cfg.multi_cortex = MultiCortexConfig(
        enabled=True,
        n_cortices=4,
        domains=["math", "code", "chat", "general"],
        weights="stub",
        freeze_weights=False,    # need grads on cortex params for the
                                  # "receives gradient" test
        lexical_bias_weight=2.0,
        bema_tau=0.5,
        router_d_model=D_SEM,
    )
    return cfg


# ──────────────────────────────────────────────────────────────────────
# Contract 1 — enabling multi_cortex must change the logits
# ──────────────────────────────────────────────────────────────────────

class TestMultiCortexAffectsLogits:
    """The bug-pinning test: a multi_cortex that's enabled but doesn't
    contribute to logits is silently broken. Logits with vs without
    the cortex MUST differ on the same input."""

    def test_logits_differ_with_vs_without_multi_cortex(
        self, fake_lm, stub_fusion_cfg, disabled_cfg,
    ):
        from neuroslm.harness import BRIANHarness

        # Build two harnesses sharing the SAME underlying language_model
        # so any logits delta comes ONLY from the multi_cortex path.
        torch.manual_seed(42)
        h_on = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=stub_fusion_cfg,
        )
        torch.manual_seed(42)
        h_off = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=disabled_cfg,
        )

        ids = torch.randint(0, VOCAB, (2, 8))
        with torch.no_grad():
            logits_on = h_on(ids)
            logits_off = h_off(ids)

        # The shared language_model alone produces identical "off" logits,
        # so any delta when h_on is invoked proves the cortex is wired in.
        max_diff = (logits_on - logits_off).abs().max().item()
        assert max_diff > 1e-4, (
            f"Logits identical with vs without multi_cortex (max|Δ|={max_diff:.2e}). "
            "The ensemble's output is NOT reaching the LM head — "
            "pretrained cortex weights are dead weight in the graph."
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 2 — gradient flows into the ensemble
# ──────────────────────────────────────────────────────────────────────

class TestMultiCortexInGradientGraph:
    """Every parameter in the ensemble must receive a gradient from
    the LM loss. Zero gradient = parameter is orphaned from the loss."""

    def test_cortex_parameters_receive_nonzero_gradient(
        self, fake_lm, stub_fusion_cfg,
    ):
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=stub_fusion_cfg,
        )

        ids = torch.randint(0, VOCAB, (2, 8))
        targets = torch.randint(0, VOCAB, (2, 8))
        logits = h(ids)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB), targets.reshape(-1)
        )
        loss.backward()

        cortex_params = list(h.multi_cortex.parameters())
        assert cortex_params, "multi_cortex has no parameters at all"
        # Find at least one cortex param with a non-zero gradient.
        max_abs_grad = 0.0
        for p in cortex_params:
            if p.grad is not None:
                g = float(p.grad.abs().max().item())
                if g > max_abs_grad:
                    max_abs_grad = g
        assert max_abs_grad > 1e-8, (
            f"All cortex parameters have zero gradient (max|grad|={max_abs_grad:.2e}). "
            "The ensemble is detached from the loss graph — autograd never "
            "follows ids → multi_cortex → logits."
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 3 — cortex_lm_head exists and has the right shape
# ──────────────────────────────────────────────────────────────────────

class TestCortexLMHeadStructure:
    """When fusion is enabled, the harness must expose a head that
    maps cortex hidden → vocab logits. Shape must match (vocab, d_sem)
    so it can read from MultiCortexEnsemble outputs of (B, T, d_sem)."""

    def test_cortex_lm_head_attribute_exists_when_enabled(
        self, fake_lm, stub_fusion_cfg,
    ):
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=stub_fusion_cfg,
        )
        assert hasattr(h, "cortex_lm_head"), (
            "cortex_lm_head attribute missing — fusion path is unimplemented"
        )
        assert h.cortex_lm_head is not None, (
            "cortex_lm_head is None when multi_cortex is enabled"
        )

    def test_cortex_lm_head_maps_d_sem_to_vocab(
        self, fake_lm, stub_fusion_cfg,
    ):
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=stub_fusion_cfg,
        )
        dummy = torch.randn(2, 8, D_SEM)
        out = h.cortex_lm_head(dummy)
        assert out.shape == (2, 8, VOCAB), (
            f"cortex_lm_head output shape mismatch: got {tuple(out.shape)}, "
            f"expected (2, 8, {VOCAB}). Must consume (B, T, d_sem) and "
            f"produce (B, T, vocab)."
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 4 — fusion weight is a learnable parameter at meaningful init
# ──────────────────────────────────────────────────────────────────────

class TestFusionWeightLearnable:
    """The mixing coefficient α between LM logits and cortex logits
    must be a learnable parameter (so the optimizer can rebalance),
    and it must be initialised so the cortex contributes meaningfully
    from step 0 (else the pretrained weights still don't help on the
    first batch)."""

    def test_cortex_mix_logit_is_a_parameter(
        self, fake_lm, stub_fusion_cfg,
    ):
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=stub_fusion_cfg,
        )
        assert hasattr(h, "cortex_mix_logit"), (
            "cortex_mix_logit attribute missing"
        )
        assert isinstance(h.cortex_mix_logit, nn.Parameter), (
            f"cortex_mix_logit must be nn.Parameter, got "
            f"{type(h.cortex_mix_logit).__name__}"
        )
        assert h.cortex_mix_logit.requires_grad, (
            "cortex_mix_logit must have requires_grad=True so the "
            "optimizer can learn the fusion balance"
        )

    def test_initial_fusion_weight_is_meaningful(
        self, fake_lm, stub_fusion_cfg,
    ):
        """``sigmoid(cortex_mix_logit)`` at init must be ≥ 0.1 so the
        cortex actually contributes to the first batch's logits.
        Init at 0 (i.e., α=0.5) is the default; init at -∞ defeats the
        whole point of loading pretrained weights."""
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=stub_fusion_cfg,
        )
        alpha = float(torch.sigmoid(h.cortex_mix_logit).item())
        assert alpha >= 0.1, (
            f"Initial fusion weight α={alpha:.4f} is too small. The "
            "pretrained cortex won't contribute meaningfully at step 0. "
            "Init cortex_mix_logit at 0 (α=0.5) or higher."
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 5 — back-compat: disabled / fusion_mode="off" leaves logits alone
# ──────────────────────────────────────────────────────────────────────

class TestBackCompat:
    """When ``multi_cortex.enabled=False`` (or ``fusion_mode="off"``),
    the harness must not build any cortex head — preserving the legacy
    single-cortex behaviour bit-for-bit."""

    def test_disabled_cortex_has_no_head(self, fake_lm, disabled_cfg):
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=disabled_cfg,
        )
        assert getattr(h, "cortex_lm_head", None) is None, (
            "cortex_lm_head must be None when multi_cortex.enabled=False"
        )
        assert getattr(h, "cortex_mix_logit", None) is None, (
            "cortex_mix_logit must be None when multi_cortex.enabled=False"
        )

    def test_fusion_mode_off_keeps_ensemble_but_no_head(self, fake_lm):
        """``fusion_mode="off"`` lets the user keep the ensemble (for
        routing telemetry / aux objectives) without touching logits."""
        from neuroslm.harness import BRIANHarness
        cfg = TrainingConfig()
        cfg.multi_cortex = MultiCortexConfig(
            enabled=True,
            n_cortices=4,
            domains=["math", "code", "chat", "general"],
            weights="stub",
            router_d_model=D_SEM,
            fusion_mode="off",
        )
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        # Ensemble built — fusion head NOT built.
        assert h.multi_cortex is not None, (
            "multi_cortex must still be built when enabled=True even if "
            "fusion_mode='off' — the ensemble is still used for routing "
            "telemetry / auxiliary objectives."
        )
        assert getattr(h, "cortex_lm_head", None) is None, (
            "cortex_lm_head must NOT be built when fusion_mode='off'"
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 6 — tied weights init (pretrained-quality starting logits)
# ──────────────────────────────────────────────────────────────────────

class TestTiedWeightsInit:
    """For a sane initial loss, ``cortex_lm_head`` should be initialised
    from the language model's input embedding (the standard transformer
    tied-weights trick). This gives logits ``cortex_h @ embed^T`` which
    are geometrically aligned with the LM's token space instead of the
    Xavier-uniform noise that plain ``nn.Linear`` defaults to."""

    def test_cortex_lm_head_tied_to_language_model_embed(
        self, fake_lm, stub_fusion_cfg,
    ):
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=stub_fusion_cfg,
        )
        # The fake LM has `.embed` of shape (vocab, d_model). We expect
        # cortex_lm_head.weight to be tied to (i.e. share storage with) it.
        head_w = h.cortex_lm_head.weight
        lm_embed = fake_lm.embed
        # Same shape — required for tied weights.
        assert head_w.shape == lm_embed.shape, (
            f"cortex_lm_head.weight shape {tuple(head_w.shape)} != "
            f"language_model.embed shape {tuple(lm_embed.shape)}"
        )
        # Same values — they should be either the literal same tensor or
        # at least identical at init.
        assert torch.equal(head_w.detach(), lm_embed.detach()), (
            "cortex_lm_head.weight is not initialised from "
            "language_model.embed. Use tied weights at init so the "
            "cortex's initial logits are not random noise."
        )
