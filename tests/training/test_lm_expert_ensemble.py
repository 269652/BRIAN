"""Contracts for ``LMExpertEnsemble`` — the N-expert mixture-of-experts
that replaces ``MultiCortexEnsemble`` on the new path.

Architectural change vs. the legacy ensemble:
  * Inputs: ``ids: LongTensor[B, T]`` (trunk-vocab)
  * Outputs: ``logits: Tensor[B, T, V_trunk]`` — **router-weighted
    mixture of every expert's trunk-vocab logits**.
  * No hidden-state projection step, no ``cortex_lm_head``,
    no ``cortex_pre_head_norm``. Each expert's pretrained LM head
    is what produces logits.

The legacy ``MultiCortexEnsemble`` returned ``(B, T, d_target)`` hidden
states and required an extra random ``cortex_lm_head`` to reach vocab —
that's the random-projection chain we're killing.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
transformers = pytest.importorskip("transformers")

from neuroslm.cortex import ThalamicRouter, DomainLexicon  # noqa: E402
from neuroslm.experts import LMExpert, LMExpertEnsemble  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trunk_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def gpt2_experts(trunk_tokenizer):
    return [
        LMExpert(
            model_id="gpt2",
            domain="general",
            trunk_tokenizer=trunk_tokenizer,
            freeze=True,
        ),
        LMExpert(
            model_id="distilgpt2",
            domain="code",
            trunk_tokenizer=trunk_tokenizer,
            freeze=True,
        ),
    ]


@pytest.fixture(scope="module")
def two_expert_ensemble(gpt2_experts, trunk_tokenizer):
    router = ThalamicRouter(
        vocab_size=trunk_tokenizer.vocab_size,
        d_model=128,
        domains=["general", "code"],
        lexicon=DomainLexicon.empty(domains=["general", "code"]),
        lexical_bias_weight=0.0,  # Uniform routing for predictable test
        bema_tau=0.0,
    )
    return LMExpertEnsemble(experts=gpt2_experts, router=router)


# ──────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_n_experts(self, two_expert_ensemble):
        assert len(two_expert_ensemble.experts) == 2

    def test_domain_order(self, two_expert_ensemble):
        assert two_expert_ensemble.domains == ["general", "code"]

    def test_rejects_domain_mismatch(self, gpt2_experts, trunk_tokenizer):
        # Router knows "general" and "math" but experts have "general" and "code"
        bad_router = ThalamicRouter(
            vocab_size=trunk_tokenizer.vocab_size,
            d_model=128,
            domains=["general", "math"],
            lexicon=DomainLexicon.empty(domains=["general", "math"]),
        )
        with pytest.raises(ValueError, match="domain"):
            LMExpertEnsemble(experts=gpt2_experts, router=bad_router)

    def test_rejects_empty_experts(self, trunk_tokenizer):
        # ThalamicRouter requires n_cortices >= 2, so we can't build a
        # router with zero domains. Test the empty-experts guard
        # directly by constructing a real router and an empty list.
        router = ThalamicRouter(
            vocab_size=trunk_tokenizer.vocab_size,
            d_model=128,
            domains=["a", "b"],  # router-side OK
            lexicon=DomainLexicon.empty(domains=["a", "b"]),
        )
        with pytest.raises(ValueError, match="at least one"):
            LMExpertEnsemble(experts=[], router=router)


# ──────────────────────────────────────────────────────────────────────
# Forward shape & semantics
# ──────────────────────────────────────────────────────────────────────


class TestForward:
    def test_output_in_trunk_vocab_space(
        self, two_expert_ensemble, trunk_tokenizer
    ):
        ids = torch.randint(0, trunk_tokenizer.vocab_size, (2, 16))
        with torch.no_grad():
            logits = two_expert_ensemble(ids)
        assert logits.shape == (2, 16, trunk_tokenizer.vocab_size)

    def test_initial_ce_uses_pretrained_heads(
        self, two_expert_ensemble, trunk_tokenizer
    ):
        """**Smoking-gun test.** With the new path, ensemble CE on
        natural English at step 0 must be << ln(V) — the pretrained
        heads do real work."""
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Once upon a time in a land far away there lived a princess."
        )
        ids = torch.tensor(
            [trunk_tokenizer.encode(text)], dtype=torch.long
        )
        with torch.no_grad():
            logits = two_expert_ensemble(ids)
        ce = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
        ).item()
        uniform = math.log(trunk_tokenizer.vocab_size)
        assert ce < 0.6 * uniform, (
            f"ensemble CE ({ce:.2f}) must be << uniform ({uniform:.2f}); "
            f"if this fails, the ensemble is not actually using pretrained heads"
        )

    def test_router_weights_are_returned(
        self, two_expert_ensemble, trunk_tokenizer
    ):
        ids = torch.randint(0, trunk_tokenizer.vocab_size, (2, 8))
        with torch.no_grad():
            _ = two_expert_ensemble(ids)
        w = two_expert_ensemble.last_routing_weights
        assert w is not None
        assert w.shape == (2, 8, 2)  # (B, T, N)
        # Each row is a probability distribution (sums to 1 per token)
        sums = w.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), (
            f"router rows must sum to 1; got {sums}"
        )


# ──────────────────────────────────────────────────────────────────────
# Back-compat: legacy MultiCortexEnsemble path unchanged
# ──────────────────────────────────────────────────────────────────────


class TestLegacyEnsembleUnchanged:
    """Make sure adding LMExpertEnsemble doesn't break the existing
    hidden-state ensemble that some tests still use."""

    def test_legacy_import_still_works(self):
        from neuroslm.cortex import MultiCortexEnsemble  # noqa: F401

    def test_legacy_factory_still_works(self):
        from neuroslm.cortex import build_default_ensemble

        ens = build_default_ensemble(vocab=128, d_model=32)
        assert ens is not None
        ids = torch.randint(0, 128, (2, 8))
        with torch.no_grad():
            h = ens(ids)
        # Old contract: returns hidden states (B, T, d_target)
        assert h.shape == (2, 8, 32)
