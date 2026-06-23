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
from unittest import mock

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

    def test_output_dtype_matches_training_compute_dtype(self):
        """Ensemble output must be in the AMP compute dtype (bf16 during
        training), not fp32 from the frozen experts.

        Note: the router's *parameter* dtype (fp32) is irrelevant — under
        autocast, nn.LayerNorm is promoted to fp32 regardless, so
        weights.dtype is fp32 even during bf16 training.  The ensemble reads
        the AMP dtype via torch.get_autocast_gpu_dtype() instead.
        """
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
        )  # fp32 params — the real training default (NOT .to(bf16))

        ens = LMExpertEnsemble(experts=[expert_a, expert_b], router=router)

        ids = torch.randint(0, V, (2, 8)).to(torch.long)
        with mock.patch.multiple(
            "torch",
            is_autocast_enabled=mock.Mock(return_value=True),
            get_autocast_gpu_dtype=mock.Mock(return_value=torch.bfloat16),
        ), torch.no_grad():
            out = ens(ids)

        assert out.dtype == torch.bfloat16, (
            f"ensemble output should be bf16 (AMP compute dtype), got {out.dtype}; "
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


# ──────────────────────────────────────────────────────────────────────
# Root-cause tests: LayerNorm fp32-promotion under bf16 autocast
# ──────────────────────────────────────────────────────────────────────


class TestFusionDtypeDetection:
    """Pin the REAL bug: nn.LayerNorm inside ThalamicRouter is promoted to
    fp32 by PyTorch's autocast policy (for numerical stability), so
    ``weights.dtype`` is **fp32** even during bf16 training.

    The original ``cast to weights.dtype`` fix was therefore a no-op:
    fp32 → fp32.  The correct fix reads the AMP compute dtype via
    ``torch.get_autocast_gpu_dtype()`` when ``torch.is_autocast_enabled()``.

    We simulate the real training scenario by:
      - keeping router parameters in *fp32* (the default, no `.to(bf16)`)
      - patching ``torch.is_autocast_enabled`` / ``torch.get_autocast_gpu_dtype``
        to mimic an active bf16 autocast context on CPU
      - asserting the ensemble output is bf16, not fp32

    The existing ``TestExpertDtypeCast`` tests pass a router explicitly moved
    to bf16 WITHOUT autocast; in that scenario ``weights.dtype`` is bf16 and
    the cast appears to work — but that scenario does NOT reproduce the
    actual training failure.
    """

    # ── shared mock helpers ────────────────────────────────────────────

    def _fp32_router(self, vocab: int, d_model: int, domains: list) -> "ThalamicRouter":
        """Router with parameters in fp32 — the real default at training time."""
        return ThalamicRouter(
            vocab_size=vocab,
            d_model=d_model,
            domains=domains,
            lexicon=DomainLexicon.empty(domains=domains),
            lexical_bias_weight=0.0,
            bema_tau=0.0,
        )  # note: no .to(bf16) — stays fp32

    def _make_fp32_expert(self, vocab: int, domain: str) -> "nn.Module":
        """Minimal expert stub that always returns fp32 ones."""
        class _E(nn.Module):
            def __init__(self):
                super().__init__()
                self.domain = domain

            def forward(self, ids):
                B, T = ids.shape
                return torch.ones(B, T, vocab, dtype=torch.float32)

        return _E()

    def _bf16_autocast_patches(self):
        """Context manager that patches torch to look like active bf16 autocast."""
        return mock.patch.multiple(
            "torch",
            is_autocast_enabled=mock.Mock(return_value=True),
            get_autocast_gpu_dtype=mock.Mock(return_value=torch.bfloat16),
        )

    # ── actual training scenario: fp32 router + mocked bf16 autocast ──

    def test_layernorm_promotes_to_fp32_means_weights_dtype_is_fp32(self):
        """Verify the premise: a fp32-param router outputs fp32 even when
        its output is all that is examined (no autocast mock here).

        This is the root cause: in training, the router's LayerNorm is
        promoted to fp32 by autocast, producing fp32 weights.  The original
        'cast to weights.dtype' fix then does nothing."""
        V, d = 64, 32
        router = self._fp32_router(V, d, ["a", "b"])
        ids = torch.randint(0, V, (2, 8))
        with torch.no_grad():
            weights = router(ids)
        # In the default fp32 case (and what autocast produces via LN promotion):
        assert weights.dtype == torch.float32, (
            "router with fp32 params must return fp32; if this fails the "
            "premise of this test class is wrong"
        )

    def test_autocast_dtype_used_over_weights_dtype(self):
        """Core regression test.

        With a fp32 router (weights.dtype = fp32) AND mocked bf16 autocast,
        the ensemble output must be bf16.

        Before the fix (using weights.dtype): output was fp32 → OOM.
        After the fix (using get_autocast_gpu_dtype()): output is bf16.
        """
        V, d = 64, 32
        expert_a = self._make_fp32_expert(V, "a")
        expert_b = self._make_fp32_expert(V, "b")
        router = self._fp32_router(V, d, ["a", "b"])
        ens = LMExpertEnsemble(experts=[expert_a, expert_b], router=router)

        ids = torch.randint(0, V, (2, 8))
        with self._bf16_autocast_patches(), torch.no_grad():
            out = ens(ids)

        assert out.dtype == torch.bfloat16, (
            f"under bf16 autocast + fp32 router, output must be bf16 (not "
            f"{out.dtype}); 'cast to weights.dtype' is broken because "
            f"LayerNorm promotes router output to fp32 under autocast"
        )

    def test_w_i_is_cast_before_multiply(self):
        """weights[..., i] (fp32 from LN-promoted router) must also be cast
        to _fuse_dtype before the multiply, not just e_logits.

        If only e_logits is cast (bf16) but w_i stays fp32, then
        w_i (fp32) × e_logits (bf16) → fp32 product = 6.14 GiB OOM.
        The fix casts BOTH by pre-casting `weights` before the loop.
        """
        V, d = 64, 32
        expert_a = self._make_fp32_expert(V, "a")
        expert_b = self._make_fp32_expert(V, "b")
        router = self._fp32_router(V, d, ["a", "b"])
        ens = LMExpertEnsemble(experts=[expert_a, expert_b], router=router)

        ids = torch.randint(0, V, (2, 4))
        with self._bf16_autocast_patches(), torch.no_grad():
            out = ens(ids)

        # If w_i were fp32 and e_logits bf16, the product would promote to fp32.
        assert out.dtype == torch.bfloat16, (
            f"w_i must be cast alongside e_logits; got output dtype {out.dtype}"
        )

    def test_large_vocab_no_fp32_leak_under_autocast(self):
        """At V=50257 (GPT-2 vocab), a fp32 (B,T,V) tensor is 6.14 GiB.
        Under mocked autocast the output must be bf16 to avoid that alloc."""
        V = 50257
        expert_a = self._make_fp32_expert(V, "a")
        router = ThalamicRouter(
            vocab_size=V, d_model=32, domains=["a", "b"],
            lexicon=DomainLexicon.empty(domains=["a", "b"]),
            lexical_bias_weight=0.0, bema_tau=0.0,
        )
        # Need 2 experts minimum for LMExpertEnsemble
        expert_b = self._make_fp32_expert(V, "b")
        ens = LMExpertEnsemble(experts=[expert_a, expert_b], router=router)

        ids = torch.randint(0, V, (1, 4))
        with self._bf16_autocast_patches(), torch.no_grad():
            out = ens(ids)

        assert out.dtype == torch.bfloat16, (
            f"at V=50257 under bf16 autocast, output must be bf16 to avoid "
            f"6.14 GiB fp32 alloc, got {out.dtype}"
        )
        assert out.shape == (1, 4, V)

    def test_no_autocast_keeps_fp32(self):
        """Without active autocast, expert fp32 logits stay fp32 — there is
        no precision change and no risk of unexpected down-casting."""
        V, d = 64, 32
        expert_a = self._make_fp32_expert(V, "a")
        expert_b = self._make_fp32_expert(V, "b")
        router = self._fp32_router(V, d, ["a", "b"])
        ens = LMExpertEnsemble(experts=[expert_a, expert_b], router=router)

        ids = torch.randint(0, V, (2, 8))
        # no autocast context — torch.is_autocast_enabled() is False
        with torch.no_grad():
            out = ens(ids)

        assert out.dtype == torch.float32, (
            f"without autocast, fp32 experts + fp32 router must produce fp32 "
            f"output (no unexpected down-cast), got {out.dtype}"
        )

    def test_fp16_autocast_uses_fp16(self):
        """get_autocast_gpu_dtype() → fp16 must also be handled (not hard-coded
        to bf16).  The logic must respect whatever AMP dtype is active."""
        V, d = 64, 32
        expert_a = self._make_fp32_expert(V, "a")
        expert_b = self._make_fp32_expert(V, "b")
        router = self._fp32_router(V, d, ["a", "b"])
        ens = LMExpertEnsemble(experts=[expert_a, expert_b], router=router)

        ids = torch.randint(0, V, (2, 4))
        with mock.patch.multiple(
            "torch",
            is_autocast_enabled=mock.Mock(return_value=True),
            get_autocast_gpu_dtype=mock.Mock(return_value=torch.float16),
        ), torch.no_grad():
            out = ens(ids)

        assert out.dtype == torch.float16, (
            f"under fp16 autocast, output must be fp16, got {out.dtype}"
        )

    def test_weighted_sum_numerically_correct_with_mocked_bf16_autocast(self):
        """Cast to bf16 must preserve the weighted sum.

        Both experts return ones.  Router with uniform weights → output ≈ ones
        (up to bf16 rounding)."""
        V, d = 128, 32
        expert_a = self._make_fp32_expert(V, "a")
        expert_b = self._make_fp32_expert(V, "b")
        router = self._fp32_router(V, d, ["a", "b"])
        ens = LMExpertEnsemble(experts=[expert_a, expert_b], router=router)

        ids = torch.randint(0, V, (1, 8))
        with self._bf16_autocast_patches(), torch.no_grad():
            out = ens(ids)

        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(
            out, torch.ones(1, 8, V, dtype=torch.bfloat16),
            rtol=1e-2, atol=1e-2,
        )

    def test_inplace_accumulation_correct_multiple_experts(self):
        """in-place out.add_(w_i * e_logits) must produce the same result
        as the naive out = out + w_i * e_logits for N > 2 experts.

        We use 3 experts all returning ones with roughly uniform routing
        and check the output is close to ones (weighted sum property).
        """
        V, d = 64, 32
        domains = ["a", "b", "c"]
        experts = [self._make_fp32_expert(V, dom) for dom in domains]
        router = ThalamicRouter(
            vocab_size=V, d_model=d, domains=domains,
            lexicon=DomainLexicon.empty(domains=domains),
            lexical_bias_weight=0.0, bema_tau=0.0,
        )
        ens = LMExpertEnsemble(experts=experts, router=router)

        ids = torch.randint(0, V, (2, 6))
        with self._bf16_autocast_patches(), torch.no_grad():
            out = ens(ids)

        assert out.dtype == torch.bfloat16
        assert out.shape == (2, 6, V)
        # All experts return ones; router weights sum to 1 per token → out ≈ ones
        torch.testing.assert_close(
            out, torch.ones(2, 6, V, dtype=torch.bfloat16),
            rtol=1e-2, atol=1e-2,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_real_cuda_bf16_autocast_no_fp32_product(self):
        """End-to-end on CUDA: under real torch.autocast the ensemble must
        produce bf16 output, not fp32."""
        V, d = 128, 32
        expert_a = self._make_fp32_expert(V, "a").cuda()
        expert_b = self._make_fp32_expert(V, "b").cuda()
        router = self._fp32_router(V, d, ["a", "b"]).cuda()
        ens = LMExpertEnsemble(experts=[expert_a, expert_b], router=router).cuda()

        ids = torch.randint(0, V, (2, 8)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            out = ens(ids)

        assert out.dtype == torch.bfloat16, (
            f"under real CUDA bf16 autocast + fp32-param router, output must "
            f"be bf16; got {out.dtype}"
        )


# ──────────────────────────────────────────────────────────────────────
# Root-cause tests: _forward_bridge fp32 output buffer under bf16 autocast
# ──────────────────────────────────────────────────────────────────────


class TestBridgeBufferDtype:
    """Pin that ``_forward_bridge`` allocates its output buffer in the AMP
    compute dtype (not always fp32).

    At B=16, T=2048, V=50257 an fp32 output buffer is 6.14 GiB per expert.
    Even if the fusion loop casts to bf16 afterwards, the peak VRAM during
    the cast is fp32 + bf16 simultaneously = 9.21 GiB → CUDA OOM on the
    A100 (which also holds frozen expert weights, optimizer states, and
    forward activations).

    The fix: read the active AMP dtype at the TOP of ``_forward_bridge``
    (same ``torch.is_autocast_enabled()`` / ``torch.get_autocast_gpu_dtype()``
    pattern as the fusion loop) and allocate ``out`` in that dtype.

    These tests use the real gpt2 ``LMExpert`` from the module fixture but
    force ``is_same_tokenizer = False`` so ``_forward_bridge`` is exercised.
    The vocab bridge for gpt2→gpt2 is identity (same tokenizer, 100% coverage)
    so this is cheap: one small LM forward per sample, identity bridge apply.
    """

    def _bf16_patches(self):
        return mock.patch.multiple(
            "torch",
            is_autocast_enabled=mock.Mock(return_value=True),
            get_autocast_gpu_dtype=mock.Mock(return_value=torch.bfloat16),
        )

    def test_bridge_output_is_bf16_under_bf16_autocast(self, gpt2_experts):
        """``_forward_bridge`` must return bf16 when the active AMP dtype is
        bf16.  Before the fix, it always returned fp32 regardless of autocast,
        causing a 6.14 GiB allocation + a 3.07 GiB cast copy = 9.21 GiB peak
        → OOM even on a 40 GB A100."""
        expert = gpt2_experts[0]
        ids = torch.randint(0, expert.vocab_size_trunk, (1, 16))
        orig = expert.is_same_tokenizer
        expert.is_same_tokenizer = False
        try:
            with self._bf16_patches(), torch.no_grad():
                out = expert._forward_bridge(ids)
        finally:
            expert.is_same_tokenizer = orig

        assert out.dtype == torch.bfloat16, (
            f"_forward_bridge must allocate output buffer in the AMP compute "
            f"dtype (bf16) to avoid a 6.14 GiB fp32 allocation; got {out.dtype}"
        )

    def test_bridge_output_is_fp32_without_autocast(self, gpt2_experts):
        """Without active autocast, ``_forward_bridge`` must return fp32 —
        no silent downcast when running eval / CPU inference."""
        expert = gpt2_experts[0]
        ids = torch.randint(0, expert.vocab_size_trunk, (1, 8))
        orig = expert.is_same_tokenizer
        expert.is_same_tokenizer = False
        try:
            with torch.no_grad():
                out = expert._forward_bridge(ids)
        finally:
            expert.is_same_tokenizer = orig

        assert out.dtype == torch.float32, (
            f"without autocast, _forward_bridge must return fp32 (no "
            f"unexpected downcast), got {out.dtype}"
        )

    def test_bridge_output_dtype_matches_fp16_autocast(self, gpt2_experts):
        """``_forward_bridge`` must respect fp16 AMP, not hard-code bf16."""
        expert = gpt2_experts[0]
        ids = torch.randint(0, expert.vocab_size_trunk, (1, 8))
        orig = expert.is_same_tokenizer
        expert.is_same_tokenizer = False
        try:
            with mock.patch.multiple(
                "torch",
                is_autocast_enabled=mock.Mock(return_value=True),
                get_autocast_gpu_dtype=mock.Mock(return_value=torch.float16),
            ), torch.no_grad():
                out = expert._forward_bridge(ids)
        finally:
            expert.is_same_tokenizer = orig

        assert out.dtype == torch.float16, (
            f"under fp16 autocast, _forward_bridge must return fp16; got {out.dtype}"
        )

    def test_bridge_content_unchanged_after_dtype_fix(self, gpt2_experts):
        """Switching the output buffer dtype must not change the VALUES —
        logits must be the same (within bf16 rounding) as the fp32 path."""
        expert = gpt2_experts[0]
        ids = torch.randint(0, expert.vocab_size_trunk, (1, 8))
        orig = expert.is_same_tokenizer
        expert.is_same_tokenizer = False
        try:
            with torch.no_grad():
                out_fp32 = expert._forward_bridge(ids)  # no autocast → fp32
            with self._bf16_patches(), torch.no_grad():
                out_bf16 = expert._forward_bridge(ids)  # bf16 autocast
        finally:
            expert.is_same_tokenizer = orig

        # Cast fp32 to bf16 for comparison — only bf16 precision should differ.
        torch.testing.assert_close(
            out_bf16, out_fp32.to(torch.bfloat16),
            rtol=1e-2, atol=1e-2,
        )


# ──────────────────────────────────────────────────────────────────────
# Stubs for TestExpertMaxContext — no real model loading, no network.
# ──────────────────────────────────────────────────────────────────────

_MAX_CTX_SMALL = 64   # stub expert context limit (simulates GPT-2 n_positions=1024)
_VOCAB_SMALL = 128


def _make_stub_lm(n_positions: int = _MAX_CTX_SMALL, vocab: int = _VOCAB_SMALL):
    """Minimal nn.Module stub with config.n_positions.

    Raises IndexError for T > n_positions, mirroring the exact failure mode
    GPT-2 exhibits when position embedding indices exceed n_positions=1024.
    """
    import types as _t
    class _StubLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _t.SimpleNamespace(n_positions=n_positions)
        def forward(self, input_ids, **kwargs):
            B, T = input_ids.shape
            if T > n_positions:
                raise IndexError(
                    f"position index {T - 1} is out of bounds for "
                    f"Embedding(num_embeddings={n_positions})"
                )
            return _t.SimpleNamespace(
                logits=torch.zeros(B, T, vocab, dtype=torch.float32)
            )
        def parameters(self): return iter([])
        def eval(self): return self
    return _StubLM()


def _make_stub_expert_same_tok(
    n_positions: int = _MAX_CTX_SMALL, vocab: int = _VOCAB_SMALL
) -> "LMExpert":
    """LMExpert with same-tokenizer path, bypassing __init__ (no model loading)."""
    expert = object.__new__(LMExpert)
    nn.Module.__init__(expert)
    expert.model_id = "stub"
    expert.domain = "general"
    expert.lm = _make_stub_lm(n_positions, vocab)
    expert.is_same_tokenizer = True
    expert.vocab_size_trunk = vocab
    expert.vocab_size_expert = vocab
    expert.last_alignment_coverage = None
    expert._expert_max_ctx = n_positions
    return expert


def _make_stub_expert_bridge(
    n_positions: int = _MAX_CTX_SMALL,
    vocab: int = _VOCAB_SMALL,
    t_trunk: int | None = None,
) -> "LMExpert":
    """LMExpert wired for bridge path, bypassing __init__.

    The expert tokenizer returns n_positions+20 tokens (more than the expert
    can handle), so without the _expert_max_ctx truncation the stub LM
    receives > n_positions tokens and raises IndexError.
    t_trunk is the trunk sequence length passed to _forward_bridge.
    Defaults to n_positions+5, which is > n_positions and triggers the bug.
    """
    if t_trunk is None:
        t_trunk = n_positions + 5
    n_expert_toks = n_positions + 20  # always > t_trunk

    expert = object.__new__(LMExpert)
    nn.Module.__init__(expert)
    expert.model_id = "stub"
    expert.domain = "general"
    expert.lm = _make_stub_lm(n_positions, vocab)
    expert.is_same_tokenizer = False
    expert.vocab_size_trunk = vocab
    expert.vocab_size_expert = vocab
    expert.last_alignment_coverage = None
    expert._expert_max_ctx = n_positions

    trunk_tok = mock.MagicMock()
    trunk_tok.decode.return_value = "word " * t_trunk
    # Trunk offsets: step-5 windows, width-4 — end-offsets 4, 9, 14, ...
    trunk_tok.return_value = {
        "input_ids": list(range(t_trunk)),
        "offset_mapping": [(i * 5, i * 5 + 4) for i in range(t_trunk)],
    }

    expert_tok = mock.MagicMock()
    # Expert offsets: step-4 windows, width-3 — end-offsets 3, 7, 11, ...
    # Deliberately non-overlapping with trunk end-offsets → all positions
    # abstain, simplifying the alignment check.
    expert_tok.return_value = {
        "input_ids": list(range(n_expert_toks)),
        "offset_mapping": [(i * 4, i * 4 + 3) for i in range(n_expert_toks)],
    }

    bridge = mock.MagicMock()
    bridge.is_identity = False
    bridge.apply.side_effect = lambda x: x.float()

    expert._trunk_tokenizer = trunk_tok
    expert._expert_tokenizer = expert_tok
    expert.vocab_bridge = bridge
    return expert


class TestExpertMaxContext:
    """LMExpert must clamp inputs at the expert model's context limit.

    All tests use stub models — no weights, no network, no GPU required.
    The stub LM raises IndexError for T > n_positions to confirm the fix
    actually prevents the position-embedding OOB crash that afflicted
    CodeGPT-small-py (GPT-2 arch, n_positions=1024) under trunk seq_len=2048.
    """

    def test_expert_max_ctx_attribute_set_on_init(self):
        """_expert_max_ctx must be derived from lm.config.n_positions in __init__."""
        stub_lm_obj = _make_stub_lm(n_positions=_MAX_CTX_SMALL)
        stub_tok = mock.MagicMock()
        stub_tok.name_or_path = "stub"
        stub_tok.vocab_size = _VOCAB_SMALL
        stub_bridge = mock.MagicMock()
        stub_bridge.is_identity = True
        stub_bridge.vocab_size_trunk = _VOCAB_SMALL
        stub_bridge.vocab_size_expert = _VOCAB_SMALL

        with (
            mock.patch("neuroslm.experts._load_lm_cached", return_value=stub_lm_obj),
            mock.patch("neuroslm.experts._load_tokenizer_cached", return_value=stub_tok),
            mock.patch("neuroslm.experts.VocabBridge.build", return_value=stub_bridge),
        ):
            # "stub/model" contains "/" → resolve_expert_alias returns it as-is
            # without a network call (rule 2: owner/repo form is always canonical)
            expert = LMExpert("stub/model", "general", trunk_tokenizer=stub_tok, freeze=True)

        assert hasattr(expert, '_expert_max_ctx'), (
            "LMExpert.__init__ must set self._expert_max_ctx"
        )
        assert expert._expert_max_ctx == _MAX_CTX_SMALL, (
            f"must derive _expert_max_ctx from config.n_positions; "
            f"expected {_MAX_CTX_SMALL}, got {expert._expert_max_ctx}"
        )

    def test_same_tok_doesnt_crash_when_t_exceeds_expert_ctx(self):
        """_forward_same_tok must not crash when T > _expert_max_ctx.

        The stub LM raises IndexError for T > n_positions; without the fix
        this test would fail with IndexError.
        """
        expert = _make_stub_expert_same_tok()
        T = _MAX_CTX_SMALL + 10
        ids = torch.randint(0, _VOCAB_SMALL, (1, T))
        with torch.no_grad():
            out = expert._forward_same_tok(ids)
        assert out.shape == (1, T, _VOCAB_SMALL), (
            f"must return full-T output, got {out.shape}"
        )

    def test_same_tok_abstains_beyond_expert_ctx(self):
        """Positions t >= _expert_max_ctx must have zero logits (abstain)."""
        expert = _make_stub_expert_same_tok()
        T = _MAX_CTX_SMALL + 10
        ids = torch.randint(0, _VOCAB_SMALL, (1, T))
        with torch.no_grad():
            out = expert._forward_same_tok(ids)
        assert out[:, _MAX_CTX_SMALL:, :].abs().sum().item() == 0.0, (
            "positions beyond _expert_max_ctx must be zero (abstain)"
        )

    def test_bridge_doesnt_crash_when_expert_retok_exceeds_ctx(self):
        """_forward_bridge must cap expert tokens at _expert_max_ctx before LM call.

        Stub tokenizer returns n_positions+20 tokens; T=n_positions+5. Without
        the cap, min(T, n_positions+20)=T tokens reach the stub LM → IndexError.
        """
        T = _MAX_CTX_SMALL + 5
        expert = _make_stub_expert_bridge(t_trunk=T)
        ids = torch.randint(0, _VOCAB_SMALL, (1, T))
        with torch.no_grad():
            out = expert._forward_bridge(ids)
        assert out.shape == (1, T, _VOCAB_SMALL), (
            f"bridge must return (1, {T}, V), got {out.shape}"
        )
