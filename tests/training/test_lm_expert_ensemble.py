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


# ──────────────────────────────────────────────────────────────────────
# dtype cast in the fusion loop
# ──────────────────────────────────────────────────────────────────────


class TestExpertDtypeCast:
    """The _forward_same_tok docstring promises that fp32 expert logits
    are "cast back to the harness's expected dtype in the fusion path."
    This test pins that contract.

    At training scale (B=16, T=2048, V=50257) an fp32 expert logit
    tensor is 6.14 GiB.  Without the cast the ensemble allocates a
    second 6.14 GiB tensor for ``w_i * e_logits`` (bf16 × fp32 → fp32)
    → CUDA OOM on the A100.
    """

    def _make_mock_expert(self, vocab: int, domain: str,
                          out_dtype: torch.dtype) -> nn.Module:
        """A minimal expert stub: returns fixed fp32 logits."""
        class _MockExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.domain = domain
                self._out_dtype = out_dtype
                self._V = vocab

            def forward(self, ids):
                B, T = ids.shape
                # Return ones so the weighted sum is predictable.
                return torch.ones(B, T, self._V, dtype=self._out_dtype)

        return _MockExpert()

    def test_output_dtype_matches_router_dtype(self):
        """Ensemble output must be in the router weight dtype (bf16
        during training), not fp32 from the frozen experts."""
        V = 64
        d_model = 32

        expert_a = self._make_mock_expert(V, "a", torch.float32)
        expert_b = self._make_mock_expert(V, "b", torch.float32)

        router = ThalamicRouter(
            vocab_size=V,
            d_model=d_model,
            domains=["a", "b"],
            lexicon=DomainLexicon.empty(domains=["a", "b"]),
            lexical_bias_weight=0.0,
            bema_tau=0.0,
        ).to(torch.bfloat16)

        ens = LMExpertEnsemble(experts=[expert_a, expert_b], router=router)

        ids = torch.randint(0, V, (2, 8)).to(torch.long)
        with torch.no_grad():
            out = ens(ids)

        assert out.dtype == torch.bfloat16, (
            f"ensemble output should be bf16 (router dtype), got {out.dtype}; "
            f"fp32 expert logits must be cast before accumulation to avoid "
            f"allocating a duplicate fp32 (B,T,V) tensor in the fusion loop"
        )

    def test_weighted_sum_correct_after_cast(self):
        """Cast must not break the mathematical weighted sum.

        With uniform router weights (0.5, 0.5) and ones logits, the
        expected output is all-ones."""
        V = 32

        expert_a = self._make_mock_expert(V, "a", torch.float32)
        expert_b = self._make_mock_expert(V, "b", torch.float32)

        router = ThalamicRouter(
            vocab_size=V,
            d_model=16,
            domains=["a", "b"],
            lexicon=DomainLexicon.empty(domains=["a", "b"]),
            lexical_bias_weight=0.0,
            bema_tau=0.0,
        ).to(torch.bfloat16)

        ens = LMExpertEnsemble(experts=[expert_a, expert_b], router=router)

        ids = torch.randint(0, V, (1, 4))
        with torch.no_grad():
            out = ens(ids)

        # Sum of router weights = 1, both experts return ones.
        # Weighted sum = 1.0 * ones + 1.0 * ones weighted by (0.5, 0.5) = ones.
        assert out.shape == (1, 4, V)
        torch.testing.assert_close(
            out, torch.ones(1, 4, V, dtype=out.dtype),
            rtol=1e-2, atol=1e-2,
        )
