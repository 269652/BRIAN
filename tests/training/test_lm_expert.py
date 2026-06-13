"""Contracts for ``neuroslm.experts.LMExpert`` — the new expert wrapper.

`LMExpert` replaces the old `GPT2SubCortex` + random `cortex_proj` +
random `cortex_lm_head` chain. It returns logits directly in TRUNK VOCAB
space, so the pretrained head of each HF causal LM actually drives the
loss from step 0.

Two paths:

  * **Fast path** (same tokenizer as trunk): just runs the HF LM and
    returns its logits. Zero per-step overhead beyond the LM forward.

  * **Bridge path** (different tokenizer): per-sample retokenisation +
    char-offset alignment + sparse vocab bridge mapping expert vocab
    ids back to trunk vocab ids.

These tests pin down:
  * construction + freeze contract
  * vocab-bridge correctness (gpt2 ↔ gpt2 is identity; gpt2 ↔ Qwen has
    real overlap; missing ids map to -1)
  * char-offset alignment is monotone and covers every trunk position
  * the **smoking-gun**: initial CE on real text matches the expert's
    own pretrained CE (≈ 3-5 nats for GPT-2 on natural English),
    NOT the random-projection baseline of ≈ ln(50257) = 10.82 nats

The cross-tok tests are marked `@pytest.mark.slow` since they download
Qwen weights (~1GB). Same-tok tests run on every pre-commit.
"""
from __future__ import annotations

import math
from typing import Optional

import pytest


# Skip the entire module on environments without torch/transformers.
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

# Import the modules under test AFTER the importorskip so they're not
# resolved at collection time on machines that lack the deps.
from neuroslm.experts import (  # noqa: E402
    LMExpert,
    VocabBridge,
    _align_by_char_offsets,
)


# ──────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trunk_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def gpt2_expert(trunk_tokenizer):
    """Fast-path expert: same tokenizer as the trunk."""
    return LMExpert(
        model_id="gpt2",
        domain="general",
        trunk_tokenizer=trunk_tokenizer,
        freeze=True,
    )


class TestConstruction:
    def test_construct_same_tokenizer(self, gpt2_expert, trunk_tokenizer):
        assert gpt2_expert.model_id == "gpt2"
        assert gpt2_expert.domain == "general"
        assert gpt2_expert.is_same_tokenizer is True
        assert gpt2_expert.vocab_size_expert == trunk_tokenizer.vocab_size
        assert gpt2_expert.vocab_size_trunk == trunk_tokenizer.vocab_size

    def test_freeze_disables_grad(self, gpt2_expert):
        params = list(gpt2_expert.parameters())
        assert len(params) > 0, "expert must have parameters"
        assert all(not p.requires_grad for p in params), (
            "freeze=True must zero requires_grad on every parameter"
        )

    def test_unfrozen_keeps_grad(self, trunk_tokenizer):
        e = LMExpert(
            model_id="gpt2",
            domain="general",
            trunk_tokenizer=trunk_tokenizer,
            freeze=False,
        )
        assert any(p.requires_grad for p in e.parameters())


# ──────────────────────────────────────────────────────────────────────
# Vocab bridge — same-tokenizer is identity
# ──────────────────────────────────────────────────────────────────────


class TestVocabBridgeSameTokenizer:
    def test_identity_mapping(self, trunk_tokenizer):
        b = VocabBridge.build(
            trunk_tokenizer=trunk_tokenizer,
            expert_tokenizer=trunk_tokenizer,
        )
        assert b.is_identity is True
        # Every trunk token maps to itself
        idx = torch.arange(trunk_tokenizer.vocab_size)
        mapped = b.trunk_to_expert[idx]
        assert torch.equal(mapped, idx)
        # Coverage is 100%
        assert b.coverage == 1.0

    def test_identity_apply_is_noop(self, trunk_tokenizer):
        """For same-tok, applying the bridge to expert logits must
        return them unchanged (no gather, no mask)."""
        b = VocabBridge.build(
            trunk_tokenizer=trunk_tokenizer,
            expert_tokenizer=trunk_tokenizer,
        )
        V = trunk_tokenizer.vocab_size
        logits = torch.randn(2, 5, V)
        out = b.apply(logits)
        assert torch.equal(out, logits)


# ──────────────────────────────────────────────────────────────────────
# Same-tokenizer fast path — the whole point of this exercise
# ──────────────────────────────────────────────────────────────────────


class TestFastPathLogits:
    def test_forward_shape(self, gpt2_expert, trunk_tokenizer):
        ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        with torch.no_grad():
            logits = gpt2_expert(ids)
        assert logits.shape == (1, 5, trunk_tokenizer.vocab_size)
        assert logits.dtype in (torch.float32, torch.float16, torch.bfloat16)

    def test_forward_no_grad_when_frozen(self, gpt2_expert):
        ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        with torch.no_grad():
            logits = gpt2_expert(ids)
        assert logits.requires_grad is False

    def test_smoking_gun_initial_ce_is_pretrained(
        self, gpt2_expert, trunk_tokenizer
    ):
        """**This is the central fix.**

        With the legacy random-projection chain, initial cortex CE on
        natural English was ≈ 10.85 nats (random-uniform baseline).
        With the new path that goes through GPT-2's own pretrained
        head, CE on the same text must be ≈ 3-5 nats — orders of
        magnitude better, available from step 0 with no training.
        """
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This sentence is in plain English. "
            "GPT-2 should predict it well above chance."
        )
        ids = torch.tensor(
            [trunk_tokenizer.encode(text)], dtype=torch.long
        )
        with torch.no_grad():
            logits = gpt2_expert(ids)
        # Next-token CE: predict ids[:, 1:] from logits[:, :-1]
        ce = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
        ).item()
        uniform_baseline = math.log(trunk_tokenizer.vocab_size)
        assert ce < 6.0, (
            f"GPT-2 initial CE on natural English should be < 6 nats; "
            f"got {ce:.2f} (uniform baseline = {uniform_baseline:.2f})"
        )
        # And critically: orders of magnitude better than uniform
        assert ce < 0.6 * uniform_baseline, (
            f"GPT-2 CE ({ce:.2f}) must be << uniform "
            f"({uniform_baseline:.2f}) — if this fails, the fast path "
            f"is not actually invoking the pretrained LM head"
        )


# ──────────────────────────────────────────────────────────────────────
# Char-offset alignment helper (the core of the bridge path)
# ──────────────────────────────────────────────────────────────────────


class TestCharOffsetAlignment:
    """Given two tokenisations of the same string, build an index map
    from trunk-positions to expert-positions whose char-offsets are
    closest. The map must be monotone (no time-travel) and cover every
    trunk position."""

    def test_identical_offsets_is_identity_map(self):
        # Both tokenisations have the same offsets — alignment is identity.
        trunk_offsets = [(0, 3), (3, 7), (7, 10)]
        expert_offsets = [(0, 3), (3, 7), (7, 10)]
        idx = _align_by_char_offsets(trunk_offsets, expert_offsets)
        assert list(idx) == [0, 1, 2]

    def test_expert_finer_grained(self):
        # Expert splits where trunk doesn't.
        trunk_offsets = [(0, 5), (5, 10)]              # 2 tokens
        expert_offsets = [(0, 3), (3, 5), (5, 8), (8, 10)]  # 4 tokens
        idx = _align_by_char_offsets(trunk_offsets, expert_offsets)
        # Trunk position 0 ends at char 5 → expert position whose end
        # is closest to 5 is index 1 (end=5).
        # Trunk position 1 ends at char 10 → expert position 3 (end=10).
        assert idx[0] == 1
        assert idx[1] == 3

    def test_expert_coarser_grained(self):
        # Expert merges where trunk splits — many trunk positions map to
        # the same expert position.
        trunk_offsets = [(0, 3), (3, 5), (5, 8), (8, 10)]
        expert_offsets = [(0, 5), (5, 10)]
        idx = _align_by_char_offsets(trunk_offsets, expert_offsets)
        # Trunk 0,1 (ends 3,5) both → expert 0 (end 5)
        # Trunk 2,3 (ends 8,10) both → expert 1 (end 10)
        assert idx[0] in (0,)
        assert idx[1] == 0
        assert idx[2] in (1,)
        assert idx[3] == 1

    def test_monotone_non_decreasing(self):
        trunk_offsets = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
        expert_offsets = [(0, 1), (1, 3), (3, 5), (5, 7), (7, 9), (9, 10)]
        idx = _align_by_char_offsets(trunk_offsets, expert_offsets)
        # Monotone non-decreasing — no time travel
        for i in range(1, len(idx)):
            assert idx[i] >= idx[i - 1], (
                f"alignment must be monotone; got {idx} (drop at i={i})"
            )

    def test_covers_every_trunk_position(self):
        trunk_offsets = [(0, 3), (3, 7), (7, 10), (10, 15)]
        expert_offsets = [(0, 5), (5, 10), (10, 15)]
        idx = _align_by_char_offsets(trunk_offsets, expert_offsets)
        assert len(idx) == len(trunk_offsets)
        for i in idx:
            assert 0 <= i < len(expert_offsets)


# ──────────────────────────────────────────────────────────────────────
# Cross-tokenizer bridge path (requires Qwen download — marked slow)
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def qwen_expert(trunk_tokenizer):
    pytest.importorskip("transformers")
    try:
        return LMExpert(
            model_id="Qwen/Qwen2.5-0.5B",
            domain="reasoning",
            trunk_tokenizer=trunk_tokenizer,
            freeze=True,
        )
    except Exception as exc:  # network / disk-space failures
        pytest.skip(f"could not load Qwen2.5-0.5B: {exc}")


@pytest.mark.slow
class TestBridgePath:
    def test_qwen_is_cross_tokenizer(self, qwen_expert, trunk_tokenizer):
        assert qwen_expert.is_same_tokenizer is False
        assert qwen_expert.vocab_size_expert > trunk_tokenizer.vocab_size

    def test_vocab_bridge_has_partial_coverage(self, qwen_expert):
        b = qwen_expert.vocab_bridge
        assert b.is_identity is False
        # Real overlap exists (BPE has lots of common subwords) but is
        # nowhere near 100%.
        assert 0.05 < b.coverage < 0.95, (
            f"expected partial overlap; got coverage={b.coverage:.3f}"
        )

    def test_bridge_path_returns_trunk_vocab_logits(
        self, qwen_expert, trunk_tokenizer
    ):
        text = "The capital of France is"
        ids = torch.tensor(
            [trunk_tokenizer.encode(text)], dtype=torch.long
        )
        with torch.no_grad():
            logits = qwen_expert(ids)
        # Output is in trunk vocab space
        assert logits.shape == (
            1, ids.shape[1], trunk_tokenizer.vocab_size
        )
        # Some logits are -inf (or large negative) where the bridge maps to -1
        finite_frac = torch.isfinite(logits).float().mean().item()
        assert finite_frac >= qwen_expert.vocab_bridge.coverage * 0.99, (
            f"finite fraction {finite_frac:.3f} should be ≥ bridge "
            f"coverage {qwen_expert.vocab_bridge.coverage:.3f}"
        )

    def test_bridge_path_better_than_uniform(
        self, qwen_expert, trunk_tokenizer
    ):
        """Even with partial coverage, Qwen should be much better than
        uniform on natural English. Loose threshold (just better than
        uniform) since the bridge introduces some loss."""
        text = (
            "Mathematics is the language of the universe. "
            "Theorems are proved by logical deduction from axioms. "
            "Numbers and equations describe physical reality."
        )
        ids = torch.tensor(
            [trunk_tokenizer.encode(text)], dtype=torch.long
        )
        with torch.no_grad():
            logits = qwen_expert(ids)
        # Replace -inf with very negative so cross_entropy doesn't NaN
        logits = torch.nan_to_num(logits, neginf=-1e4)
        ce = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
        ).item()
        uniform = math.log(trunk_tokenizer.vocab_size)
        assert ce < uniform * 0.8, (
            f"bridged Qwen should beat uniform by ≥ 20%; got "
            f"CE={ce:.2f} vs uniform={uniform:.2f}"
        )
