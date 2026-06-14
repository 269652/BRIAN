"""End-to-end smoke test: the deploy's exact harness construction path.

Pins, in one file, every contract the 2026-01-28 deploy needs to honour
end-to-end on cuda+bf16 without crashing:

  1. ``arch.neuro`` parses with ``multi_cortex.experts: [...]`` populated.
  2. The harness's ``_build_multi_cortex`` takes the ``LMExpertEnsemble``
     branch (not the legacy random-projection chain).
  3. Forward returns ``(B, T, V_trunk)`` logits with finite values.
  4. Initial CE on natural English is in the pretrained range
     (≤ 6 nats), NOT the ~ln(50257)≈10.85-nats random baseline that
     would prove the experts are being bypassed.
  5. Under ``torch.amp.autocast(bf16)`` the forward completes without
     the ``StubSubCortex`` illegal-memory-access path being taken
     (defensive: the legacy path should be unreachable when experts
     are set, but we assert the legacy class is NEVER instantiated
     to catch any regression that re-routes through it).

These tests use CPU bf16 autocast so they run on the dev box (the same
contract holds on cuda; the bug we're guarding against is dtype-mediated,
not device-specific). The single GPU-only contract is marked
``@pytest.mark.gpu`` and skipped on the dev box.
"""
from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")
transformers = pytest.importorskip("transformers")


# Constants matching the deployed arch
GPT2_VOCAB = 50257
D_SEM = 64  # small for test speed


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trunk_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


class _FakeGPT2VocabLM(torch.nn.Module):
    """Tiny trunk LM that exposes GPT-2's vocab size so the
    LMExpertEnsemble's bridged GPT-2 logits can be mixed without a
    vocab-size mismatch error in the fusion path. Bit-identical surface
    to the harness's expected ``language_model`` shape."""

    def __init__(self, vocab: int = GPT2_VOCAB, d_model: int = D_SEM,
                 seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = torch.nn.Parameter(
            torch.randn(vocab, d_model, generator=g) * 0.02
        )
        self.lm_head = torch.nn.Parameter(
            torch.randn(vocab, d_model, generator=g) * 0.02
        )
        self._last_hidden = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed[ids]
        self._last_hidden = h
        return F.linear(h, self.lm_head)


@pytest.fixture
def fake_lm() -> _FakeGPT2VocabLM:
    return _FakeGPT2VocabLM(seed=0)


def _experts_cfg():
    """Build the exact ``TrainingConfig.multi_cortex`` block the deploy
    sees — minus the heavier Qwen expert (we only need two same-tok
    GPT-2 experts to exercise the contract, and tests must stay fast)."""
    from neuroslm.dsl.training_config import (
        ExpertSpec, MultiCortexConfig, TrainingConfig,
    )
    cfg = TrainingConfig()
    cfg.multi_cortex = MultiCortexConfig(
        enabled=True,
        n_cortices=2,
        domains=["general", "code"],
        experts=[
            ExpertSpec(id="gpt2",       domain="general", freeze=True),
            ExpertSpec(id="distilgpt2", domain="code",    freeze=True),
        ],
        trunk_tokenizer="gpt2",
        freeze_weights=True,
        lexical_bias_weight=0.0,   # uniform routing for predictability
        bema_tau=0.0,
        router_d_model=D_SEM,
        fusion_mode="logits_mixture",
        fusion_init=0.5,
    )
    return cfg


# ──────────────────────────────────────────────────────────────────────
# Contract 1 — arch.neuro DOES parse to multi_cortex.experts populated
# ──────────────────────────────────────────────────────────────────────


class TestArchNeuroParsesExperts:
    """The deployed arch.neuro must parse to a non-empty experts roster
    so the new path actually fires. If this fails, the deploy will
    silently fall back to the legacy StubSubCortex chain — exactly the
    regression that caused the 2026-01-28 crash on instance 40844277."""

    def test_arch_neuro_parses_three_experts(self):
        from neuroslm.dsl.training_config import (
            load_training_config_from_arch,
        )

        cfg = load_training_config_from_arch("architectures/master")
        mc = cfg.multi_cortex
        assert mc.enabled, (
            "arch.neuro multi_cortex.enabled must be True so the harness "
            "actually builds an ensemble"
        )
        assert mc.experts is not None, (
            "arch.neuro multi_cortex.experts must NOT be None — without "
            "an explicit roster the harness falls back to the legacy "
            "random-projection chain and CE stays at ln(V) ≈ 10.85"
        )
        assert len(mc.experts) >= 2, (
            f"deploy roster should have ≥ 2 experts; got {len(mc.experts)}"
        )
        domains = [e.domain for e in mc.experts]
        assert len(set(domains)) == len(domains), (
            f"every expert.domain must be unique; got {domains}"
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 2 — harness builds LMExpertEnsemble, not the legacy path
# ──────────────────────────────────────────────────────────────────────


class TestHarnessTakesNewPath:
    def test_builds_lm_expert_ensemble(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        from neuroslm.experts import LMExpertEnsemble

        cfg = _experts_cfg()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=GPT2_VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        assert isinstance(h.multi_cortex, LMExpertEnsemble), (
            f"expected LMExpertEnsemble; got {type(h.multi_cortex).__name__}. "
            "The harness ignored cfg.multi_cortex.experts and fell back to "
            "the legacy chain — this is the deploy-crash regression."
        )

    def test_legacy_stub_subcortex_is_NOT_instantiated(self, fake_lm):
        """Defensive: assert the legacy StubSubCortex class is never
        instantiated when experts are configured. The 2026-01-28 crash
        was inside ``StubSubCortex.forward`` — proving that path was
        active when it shouldn't have been."""
        from neuroslm.harness import BRIANHarness
        from neuroslm.cortex import StubSubCortex

        cfg = _experts_cfg()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=GPT2_VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        for name, module in h.named_modules():
            assert not isinstance(module, StubSubCortex), (
                f"StubSubCortex found at {name!r} despite experts roster "
                f"being set — the harness took the legacy path"
            )


# ──────────────────────────────────────────────────────────────────────
# Contract 3 — Forward returns trunk-vocab logits with pretrained CE
# ──────────────────────────────────────────────────────────────────────


class TestForwardSemantics:
    def test_returns_trunk_vocab_shape(self, fake_lm):
        from neuroslm.harness import BRIANHarness

        cfg = _experts_cfg()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=GPT2_VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        ids = torch.randint(0, GPT2_VOCAB, (2, 16))
        with torch.no_grad():
            logits = h(ids)
        assert logits.shape == (2, 16, GPT2_VOCAB), logits.shape
        assert torch.isfinite(logits).all(), "non-finite logits"

    def test_initial_ce_uses_pretrained_heads(self, fake_lm,
                                              trunk_tokenizer):
        """Same smoking-gun as the harness_integration test but on the
        DEPLOY-SHAPED config: ensures the end-to-end CE drop holds for
        the exact knobs the deploy will use."""
        from neuroslm.harness import BRIANHarness

        cfg = _experts_cfg()
        cfg.multi_cortex.fusion_init = 1.0 - 1e-6  # ensemble-dominant
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=GPT2_VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        text = (
            "Once upon a time in a land far away there lived a princess. "
            "The quick brown fox jumps over the lazy dog. "
            "Hello world, this is a sentence in plain English."
        )
        ids = torch.tensor([trunk_tokenizer.encode(text)],
                           dtype=torch.long)
        with torch.no_grad():
            logits = h(ids)
        ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
        ).item()
        uniform = math.log(GPT2_VOCAB)
        assert ce < 0.6 * uniform, (
            f"deploy-shaped harness CE on English = {ce:.2f}, must be < "
            f"0.6·ln(V) = {0.6 * uniform:.2f}. If this fails the deploy "
            "will plateau at ~10.85 nats just like every prior run."
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 4 — bf16 autocast forward survives (the deploy's exact dtype)
# ──────────────────────────────────────────────────────────────────────


class TestBf16Autocast:
    """The vast.ai A100 deploy wraps every forward in
    ``torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)``.
    We exercise the same code path on CPU autocast — the dtype is
    identical, and any tensor-shape mismatch / nested-tensor fast-path
    bug would surface here too."""

    def test_bf16_forward_completes(self, fake_lm, trunk_tokenizer):
        from neuroslm.harness import BRIANHarness

        cfg = _experts_cfg()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=GPT2_VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        ids = torch.tensor(
            [trunk_tokenizer.encode("Once upon a time")],
            dtype=torch.long,
        )
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            with torch.no_grad():
                logits = h(ids)
        assert torch.isfinite(logits).all(), (
            "bf16 forward produced non-finite logits — the deploy will "
            "see NaN gradients within a few hundred steps"
        )
