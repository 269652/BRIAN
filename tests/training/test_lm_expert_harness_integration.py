"""End-to-end contract: ``multi_cortex.experts: [...]`` → ``LMExpertEnsemble``
wired into ``BRIANHarness``.

Architectural promise being pinned
==================================

The legacy ``multi_cortex.weights="gpt2"`` path produces:

    pretrained GPT-2 hidden (768)
        ──Linear(768→d_sem)──▶   [random Xavier]
        ──LayerNorm──▶            [band-aid for rogue-dim]
        ──Linear(d_sem→V_trunk)── [tied to RANDOM trunk embed at init]
        ──softmax──▶               CE ≈ ln(V)  (uniform baseline)

The new ``multi_cortex.experts: [...]`` path produces:

    pretrained GPT-2 logits (V_expert) ──VocabBridge──▶
        logits (V_trunk) ──softmax──▶  CE ≈ 3-5 nats (real LM)

This file enforces, at the harness level:

  1. When ``cfg.experts`` is set, ``harness.multi_cortex`` is an
     ``LMExpertEnsemble`` (not the legacy hidden-state ensemble).
  2. The random-projection chain (``cortex_lm_head``,
     ``cortex_pre_head_norm``) is NOT built — by-passed entirely.
  3. ``harness(ids)`` returns logits with shape ``(B, T, V_trunk)``.
  4. **Smoking gun**: initial CE on natural English << ``ln(V_trunk)``
     because the pretrained heads do real work from step 0.
  5. ``cortex_mix_logit`` is still built (fusion still mixes trunk
     logits with ensemble logits — the cortex isn't dead).
  6. Back-compat: the legacy ``weights="stub"`` path still uses the
     hidden-state ``MultiCortexEnsemble`` and still builds the
     projection chain (no behavioural drift for existing runs).
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
F = pytest.importorskip("torch.nn.functional")
transformers = pytest.importorskip("transformers")

from neuroslm.dsl.training_config import (  # noqa: E402
    ExpertSpec,
    MultiCortexConfig,
    TrainingConfig,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures — a fake LM with the GPT-2 vocab size so the ensemble's
# trunk-vocab logits are directly comparable to the LM's logits.
# ──────────────────────────────────────────────────────────────────────


GPT2_VOCAB = 50257  # gpt2.tokenizer.vocab_size
D_SEM = 64


class _FakeGPT2VocabLM(nn.Module):
    """Tiny LM that uses the GPT-2 vocab size so its logits can be
    mixed with the LMExpertEnsemble's bridged GPT-2 logits without a
    vocab-mismatch error.

    Surface matches the DSL transformer's: ``self.embed`` is an
    ``nn.Parameter[vocab, d_model]`` so the harness's tied-weights
    init path is happy (even though it's a no-op when the ensemble
    short-circuits the lm_head).
    """

    def __init__(self, vocab: int = GPT2_VOCAB, d_model: int = D_SEM,
                 seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Parameter(
            torch.randn(vocab, d_model, generator=g) * 0.02
        )
        self.lm_head = nn.Parameter(
            torch.randn(vocab, d_model, generator=g) * 0.02
        )
        self._last_hidden = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed[ids]                       # (B, T, D)
        self._last_hidden = h
        return F.linear(h, self.lm_head)          # (B, T, V)


@pytest.fixture(scope="module")
def trunk_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def fake_lm() -> _FakeGPT2VocabLM:
    return _FakeGPT2VocabLM(seed=0)


def _experts_cfg(trunk_tokenizer_name: str = "gpt2") -> TrainingConfig:
    """Build a TrainingConfig that uses the new ``experts: [...]`` path.

    Two small experts (`gpt2`, `distilgpt2`) — both share the GPT-2
    tokenizer so the VocabBridge is identity (no surface-string walk
    at construction time). Keeps the test under a few seconds even
    on CPU.
    """
    cfg = TrainingConfig()
    cfg.multi_cortex = MultiCortexConfig(
        enabled=True,
        # n_cortices / domains are auto-derived from `experts` by the
        # parser, but we set them here too so the dataclass round-trips
        # cleanly outside the parser.
        n_cortices=2,
        domains=["general", "code"],
        experts=[
            ExpertSpec(id="gpt2",        domain="general",   freeze=True),
            ExpertSpec(id="distilgpt2",  domain="code",      freeze=True),
        ],
        trunk_tokenizer=trunk_tokenizer_name,
        freeze_weights=True,
        lexical_bias_weight=0.0,        # uniform router for predictability
        bema_tau=0.0,
        router_d_model=D_SEM,
        # Logits are already in trunk-vocab space → fusion mixes them
        # directly.
        fusion_mode="logits_mixture",
        fusion_init=0.5,
    )
    return cfg


# ──────────────────────────────────────────────────────────────────────
# Contract 1 — experts: [...] builds an LMExpertEnsemble
# ──────────────────────────────────────────────────────────────────────


class TestEnsembleType:
    """When ``cfg.experts`` is set, the harness must build the
    *new* ensemble (``LMExpertEnsemble``) — never the legacy
    ``MultiCortexEnsemble`` hidden-state path."""

    def test_uses_lm_expert_ensemble(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        from neuroslm.experts import LMExpertEnsemble

        cfg = _experts_cfg()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=GPT2_VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        assert isinstance(h.multi_cortex, LMExpertEnsemble), (
            f"expected LMExpertEnsemble, got {type(h.multi_cortex)!r}; "
            "the harness ignored cfg.experts and fell back to the legacy path"
        )

    def test_random_projection_chain_is_bypassed(self, fake_lm):
        """The whole point of the new path is to remove the random
        ``cortex_lm_head`` and the ``cortex_pre_head_norm`` band-aid."""
        from neuroslm.harness import BRIANHarness

        cfg = _experts_cfg()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=GPT2_VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        assert h.cortex_lm_head is None, (
            "cortex_lm_head should be None on the experts path — "
            "the ensemble outputs trunk-vocab logits directly"
        )
        assert h.cortex_pre_head_norm is None, (
            "cortex_pre_head_norm should be None on the experts path — "
            "no random projection means no rogue-dim band-aid needed"
        )

    def test_fusion_mixing_scalar_still_built(self, fake_lm):
        """``cortex_mix_logit`` must still exist — the harness still
        learns how much to weight the ensemble vs. the trunk."""
        from neuroslm.harness import BRIANHarness

        cfg = _experts_cfg()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=GPT2_VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        assert h.cortex_mix_logit is not None
        assert isinstance(h.cortex_mix_logit, nn.Parameter)


# ──────────────────────────────────────────────────────────────────────
# Contract 2 — forward shape & smoking-gun CE
# ──────────────────────────────────────────────────────────────────────


class TestForwardShape:
    def test_returns_trunk_vocab_logits(self, fake_lm):
        from neuroslm.harness import BRIANHarness

        cfg = _experts_cfg()
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=GPT2_VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        ids = torch.randint(0, GPT2_VOCAB, (2, 16))
        with torch.no_grad():
            logits = h(ids)
        assert logits.shape == (2, 16, GPT2_VOCAB), (
            f"expected (2, 16, {GPT2_VOCAB}), got {tuple(logits.shape)}"
        )


class TestSmokingGunCE:
    """The single test that justifies this whole refactor.

    With the legacy `weights="gpt2"` path, initial CE on natural English
    sits at or above ``ln(50257) ≈ 10.82`` because the random Xavier
    projection chain converts pretrained features back into noise.

    With the new `experts: [...]` path, the pretrained heads produce
    logits directly in trunk-vocab space, so even at step 0 — without
    any training — CE on natural English is in the 3-5 nat range.

    Threshold: ``CE < 0.6 · ln(V)`` (≈ 6.5 nats). The actual value with
    GPT-2 small as the only domain weighted matter is typically ~3-4.
    We measure on the trunk-only forward path (α=1.0 in mix terms) to
    isolate the ensemble's contribution.
    """

    def test_initial_ce_uses_pretrained_heads(self, fake_lm, trunk_tokenizer):
        from neuroslm.harness import BRIANHarness

        cfg = _experts_cfg()
        # Slam fusion to "all-cortex" so the smoking-gun isolates the
        # ensemble; otherwise the fake LM's noise contaminates the CE.
        cfg.multi_cortex.fusion_init = 1.0 - 1e-6
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=GPT2_VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Once upon a time in a land far away there lived a princess."
        )
        ids = torch.tensor(
            [trunk_tokenizer.encode(text)], dtype=torch.long
        )
        with torch.no_grad():
            logits = h(ids)
        ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
        ).item()
        uniform = math.log(GPT2_VOCAB)
        assert ce < 0.6 * uniform, (
            f"harness CE on English ({ce:.2f}) >= 0.6 · ln(V) "
            f"({0.6 * uniform:.2f}); the experts ensemble is NOT "
            "contributing pretrained-quality logits through the harness "
            "fusion path — check `_build_multi_cortex` and the forward()"
            "fusion branch"
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 3 — back-compat: legacy weights="stub" still uses old path
# ──────────────────────────────────────────────────────────────────────


class TestLegacyPathUnchanged:
    """The legacy ``weights="stub"`` / ``weights="gpt2"`` configs that
    don't set ``experts`` must keep their old behaviour: hidden-state
    ``MultiCortexEnsemble`` + ``cortex_lm_head`` + ``cortex_pre_head_norm``."""

    def test_stub_path_still_builds_projection_chain(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        from neuroslm.cortex import MultiCortexEnsemble

        cfg = TrainingConfig()
        cfg.multi_cortex = MultiCortexConfig(
            enabled=True,
            n_cortices=4,
            domains=["math", "code", "chat", "general"],
            weights="stub",
            experts=None,           # ← legacy path, explicitly
            freeze_weights=False,
            lexical_bias_weight=2.0,
            bema_tau=0.5,
            router_d_model=D_SEM,
        )
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=GPT2_VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        assert isinstance(h.multi_cortex, MultiCortexEnsemble), (
            "legacy weights=stub must still build the hidden-state ensemble"
        )
        assert h.cortex_lm_head is not None, (
            "legacy path must still build cortex_lm_head (the random "
            "projection is part of its known contract)"
        )
        assert h.cortex_pre_head_norm is not None, (
            "legacy path must still build cortex_pre_head_norm (the "
            "rogue-dim band-aid was tied to this path)"
        )
