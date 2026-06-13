# -*- coding: utf-8 -*-
"""TDD spec — process-wide caching for LMExpert pretrained weights.

User requirement: ``LMExpert("gpt2", ...)`` should not pay the ~6-7s
``AutoModelForCausalLM.from_pretrained`` cost on every invocation.
Across the test suite the same model id is requested 6+ times in the
same process; sharing the loaded weights cuts ~30s off the suite.

Safety invariant: frozen experts (``freeze=True``, the default and the
production case) are stateless during forward — sharing the underlying
``nn.Module`` between ``LMExpert`` instances cannot leak gradients or
mutate weights. Unfrozen experts bypass the cache to preserve the
existing semantics.

Locked behaviour:

    1. First ``LMExpert(model_id="gpt2", freeze=True)`` loads from HF.
    2. Second call with the SAME ``model_id`` (any trunk_tokenizer,
       any domain) returns an expert whose ``.lm`` is the SAME object.
    3. Second call with ``freeze=False`` bypasses the cache (returns
       a fresh module so training mutations don't leak).
    4. Tokenizer caching mirrors the model cache.
    5. ``VocabBridge.build`` for the same ``(trunk_name, expert_name)``
       pair returns the SAME tensor on the second call.
"""
from __future__ import annotations

import time
import pytest


# Skip the whole file when ``transformers`` isn't installed (CI without
# the optional dep). The tests are pointless without real HF models.
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


class TestPretrainedLMCache:
    """``LMExpert(model_id, freeze=True)`` shares the underlying
    ``nn.Module`` across calls with the same ``model_id``."""

    def test_two_frozen_experts_share_lm_module(self, gpt2_tokenizer):
        from neuroslm.experts import LMExpert
        a = LMExpert("gpt2", domain="general", trunk_tokenizer=gpt2_tokenizer)
        b = LMExpert("gpt2", domain="reasoning", trunk_tokenizer=gpt2_tokenizer)
        # Frozen experts must share weights — `is` not `==`.
        assert a.lm is b.lm, (
            "Frozen LMExperts with the same model_id must share the "
            "underlying nn.Module to avoid reloading from HF."
        )

    def test_unfrozen_expert_bypasses_cache(self, gpt2_tokenizer):
        from neuroslm.experts import LMExpert
        cached = LMExpert("gpt2", domain="general",
                          trunk_tokenizer=gpt2_tokenizer)
        fresh = LMExpert("gpt2", domain="general",
                         trunk_tokenizer=gpt2_tokenizer, freeze=False)
        # Unfrozen experts must NOT share weights — training would
        # otherwise leak across them.
        assert fresh.lm is not cached.lm

    def test_second_construction_is_fast(self, gpt2_tokenizer):
        """The second LMExpert with the same model_id must complete
        in well under a second (cached weights, no HF round-trip)."""
        from neuroslm.experts import LMExpert
        # Warm the cache.
        _ = LMExpert("gpt2", domain="warm",
                     trunk_tokenizer=gpt2_tokenizer)
        # Now time a second call.
        t0 = time.perf_counter()
        _ = LMExpert("gpt2", domain="cold",
                     trunk_tokenizer=gpt2_tokenizer)
        dt = time.perf_counter() - t0
        # 0.5s is generous — typically ~10-50ms after warm-up.
        # The slow path takes 5-8s (full from_pretrained + state_dict load).
        assert dt < 0.5, (
            f"Second LMExpert construction took {dt:.2f}s — "
            f"cache is not active. Expected < 0.5s."
        )


class TestVocabBridgeCache:
    """``VocabBridge.build`` for the same ``(trunk, expert)`` returns
    the same instance — the cross-tokenizer build is O(V) Python loops
    and must never be re-paid in the same process."""

    def test_same_tokenizer_pair_returns_same_instance(self,
                                                       gpt2_tokenizer):
        from neuroslm.experts import VocabBridge
        a = VocabBridge.build(trunk_tokenizer=gpt2_tokenizer,
                              expert_tokenizer=gpt2_tokenizer)
        b = VocabBridge.build(trunk_tokenizer=gpt2_tokenizer,
                              expert_tokenizer=gpt2_tokenizer)
        assert a is b, (
            "VocabBridge.build must memoise on (trunk, expert) "
            "tokenizer name_or_path."
        )


class TestTokenizerCache:
    """``AutoTokenizer.from_pretrained`` is cheap but adds up across
    20+ test files. Cache it the same way."""

    def test_second_tokenizer_load_is_fast(self):
        """The cache lives on ``neuroslm.experts._load_tokenizer_cached``
        — calling it twice for the same model id must be near-instant."""
        from neuroslm.experts import _load_tokenizer_cached
        _ = _load_tokenizer_cached("gpt2")  # warm
        t0 = time.perf_counter()
        _ = _load_tokenizer_cached("gpt2")
        dt = time.perf_counter() - t0
        assert dt < 0.05, (
            f"Cached tokenizer load took {dt:.3f}s — cache inactive"
        )

    def test_two_calls_return_same_instance(self):
        from neuroslm.experts import _load_tokenizer_cached
        a = _load_tokenizer_cached("gpt2")
        b = _load_tokenizer_cached("gpt2")
        assert a is b
