# -*- coding: utf-8 -*-
"""TDD: exact-end char-offset alignment for cross-tokenizer expert bridge.

Forensic motivation
───────────────────
After H22 (deploy 40952126) shipped SmolLM2 as the only expert and
regressed lm CE by +1.1 nats vs the gpt2-baseline run, we ran an
isolation experiment (``scripts/diagnose_bridge_ce.py``) on natural
English and discovered three things:

  1. **Vocab coverage is NOT the bottleneck.** Strict single-token-only
     mapping covers 73% of trunk vocab. Relaxing to ``len(eids) >= 1``
     (use first expert subtoken) bumps coverage to 99.99% — BUT IT
     MAKES CE WORSE by 0.8 nats. Reason: many trunk tokens share the
     same first expert subtoken (e.g. " general", " generate",
     " generation" all start with " gen"), so softmax over the bridged
     trunk logits dilutes correct mass across siblings: CE penalty ≈
     ln(sibling count).

  2. **Alignment SHIFT is the bottleneck.** The current
     ``_align_by_char_offsets`` returns ``smallest e such that
     expert_offsets[e].end >= trunk_offsets[t].end``. At positions
     where the two tokenisations don't share a boundary, the chosen
     expert position has end-offset STRICTLY GREATER than trunk's.
     Two problems compound:

     a. **Leakage**: the expert at position e has SEEN the trunk's
        target as part of its input prefix.
     b. **Wrong-horizon prediction**: the expert at position e
        predicts the next-token *after* its own end-offset, which
        is PAST trunk's prediction horizon. Using this as the
        distillation target for trunk_token[t+1] mislabels trunk.

  3. **Fixing alignment makes SmolLM2 BEAT gpt2.** On a 7-sentence
     English paragraph with gpt2 trunk + SmolLM2 expert:

        gpt2 own next-token CE      = 3.016 nats
        current bridge (smallest_ge) = 3.068 nats   (+0.05 vs gpt2)
        proposed bridge (exact)     = 2.798 nats   (-0.22 vs gpt2 !)

     Exact alignment loses the ~5% of positions that don't share a
     boundary — those positions abstain to uniform — but stops adding
     wrong-horizon noise on those 5%, AND on the 95% that DO align it
     gives a clean, correctly-horizoned distillation signal.

This file pins the contract for the fix:
  * New helper ``_align_by_char_offsets_exact`` that returns -1 at
    misaligned positions.
  * ``LMExpert._forward_bridge`` uses the exact helper.
  * ``LMExpert`` exposes ``last_alignment_coverage`` for telemetry.

The legacy ``_align_by_char_offsets`` stays in ``__all__`` for back-compat
but the bridge no longer uses it.
"""
from __future__ import annotations

import math
import os
from typing import List, Tuple

import pytest
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Unit tests for the NEW helper: _align_by_char_offsets_exact
# ──────────────────────────────────────────────────────────────────────


class TestAlignExactReturnsMinusOneOnMisalignment:
    """Whenever no expert position has end-offset EXACTLY equal to the
    trunk position's end-offset, the helper must return -1."""

    def test_minus_one_when_no_expert_end_matches(self):
        from neuroslm.experts import _align_by_char_offsets_exact
        # Trunk t=0 ends at 5; expert ends are at 3 and 7. No exact match.
        trunk_offsets = [(0, 5)]
        expert_offsets = [(0, 3), (3, 7)]
        idx = _align_by_char_offsets_exact(trunk_offsets, expert_offsets)
        assert idx == [-1]

    def test_exact_match_returns_correct_index(self):
        from neuroslm.experts import _align_by_char_offsets_exact
        # Trunk t=0 ends at 5; expert ends are at 3 and 5.
        trunk_offsets = [(0, 5)]
        expert_offsets = [(0, 3), (3, 5)]
        idx = _align_by_char_offsets_exact(trunk_offsets, expert_offsets)
        assert idx == [1]

    def test_mixed_aligned_and_misaligned(self):
        from neuroslm.experts import _align_by_char_offsets_exact
        # Trunk ends: 3, 5, 8, 10.  Expert ends: 3, 7, 10.
        # t=0 (end=3) → matches e=0 (end=3) → 0
        # t=1 (end=5) → NO match (next expert end is 7) → -1
        # t=2 (end=8) → NO match (next expert end is 10) → -1
        # t=3 (end=10) → matches e=2 (end=10) → 2
        trunk_offsets = [(0, 3), (3, 5), (5, 8), (8, 10)]
        expert_offsets = [(0, 3), (3, 7), (7, 10)]
        idx = _align_by_char_offsets_exact(trunk_offsets, expert_offsets)
        assert idx == [0, -1, -1, 2]

    def test_identical_offsets_is_identity(self):
        from neuroslm.experts import _align_by_char_offsets_exact
        trunk_offsets = [(0, 3), (3, 7), (7, 10)]
        expert_offsets = [(0, 3), (3, 7), (7, 10)]
        idx = _align_by_char_offsets_exact(trunk_offsets, expert_offsets)
        assert idx == [0, 1, 2]

    def test_expert_coarser_yields_minus_one_at_inner_splits(self):
        """When the expert merges several trunk tokens, the inner trunk
        boundaries have no exact expert match → -1.
        """
        from neuroslm.experts import _align_by_char_offsets_exact
        trunk_offsets = [(0, 3), (3, 5), (5, 8), (8, 10)]
        expert_offsets = [(0, 5), (5, 10)]
        idx = _align_by_char_offsets_exact(trunk_offsets, expert_offsets)
        # t=0 (end=3) → no exact match → -1
        # t=1 (end=5) → matches e=0 → 0
        # t=2 (end=8) → no exact match → -1
        # t=3 (end=10) → matches e=1 → 1
        assert idx == [-1, 0, -1, 1]

    def test_empty_expert_returns_all_minus_one(self):
        from neuroslm.experts import _align_by_char_offsets_exact
        trunk_offsets = [(0, 3), (3, 5)]
        expert_offsets: List[Tuple[int, int]] = []
        idx = _align_by_char_offsets_exact(trunk_offsets, expert_offsets)
        assert idx == [-1, -1]

    def test_empty_trunk_returns_empty(self):
        from neuroslm.experts import _align_by_char_offsets_exact
        idx = _align_by_char_offsets_exact([], [(0, 3)])
        assert idx == []

    def test_trunk_extends_past_last_expert(self):
        """If expert tokenisation ends earlier than trunk's, the trailing
        trunk positions have no possible exact match → -1.
        """
        from neuroslm.experts import _align_by_char_offsets_exact
        # Trunk goes up to char 15; expert only up to char 10.
        trunk_offsets = [(0, 5), (5, 10), (10, 15)]
        expert_offsets = [(0, 5), (5, 10)]
        idx = _align_by_char_offsets_exact(trunk_offsets, expert_offsets)
        assert idx == [0, 1, -1]


# ──────────────────────────────────────────────────────────────────────
# Helper is exported
# ──────────────────────────────────────────────────────────────────────


class TestHelperIsExported:
    def test_helper_is_in_dunder_all(self):
        import neuroslm.experts as exp
        assert "_align_by_char_offsets_exact" in exp.__all__, (
            "the new helper must be exported so other modules + tests can "
            "use it without reaching into private namespace"
        )

    def test_legacy_helper_still_in_dunder_all(self):
        """Don't break back-compat: the legacy ``_align_by_char_offsets``
        helper must remain exported."""
        import neuroslm.experts as exp
        assert "_align_by_char_offsets" in exp.__all__


# ──────────────────────────────────────────────────────────────────────
# _forward_bridge uses exact alignment + leaves misaligned slots uniform
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_cross_tok_bridge_setup(monkeypatch):
    """Build a synthetic LMExpert whose two tokenisers DON'T share
    boundaries — every other trunk position is misaligned.

    Reuses a single ``transformers.AutoTokenizer`` for both trunk and
    expert (so the test doesn't need a network), then forces
    ``is_same_tokenizer = False`` and rewrites the bridge to be the
    identity. The point is exercising the alignment/abstain machinery,
    not the vocab bridge.
    """
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    return tok


def _make_synthetic_lm_expert(tok, monkeypatch):
    """Create an LMExpert without invoking HF download — patches the
    cache + model lookup with stubs."""
    from neuroslm import experts as exp

    # Stub the LM so __init__ doesn't try to load anything from HF
    class FakeLM(torch.nn.Module):
        def __init__(self, v: int) -> None:
            super().__init__()
            self.v = v

        def eval(self):
            return self

        def parameters(self, recurse: bool = True):
            return iter([])

        def __call__(self, input_ids: torch.Tensor):
            # logits = the position index as a scalar broadcast across vocab,
            # offset by a base value so we can recognise per-position logits
            class _Out:
                pass
            out = _Out()
            B, T = input_ids.shape
            # Distinct per-position logits: logits[b, t, v] = t * 1000 + v
            base = torch.arange(T, dtype=torch.float32).view(1, T, 1)
            voc  = torch.arange(self.v, dtype=torch.float32).view(1, 1, self.v)
            out.logits = base * 1000.0 + voc
            return out

    fake_lm = FakeLM(v=tok.vocab_size)
    monkeypatch.setitem(exp._LM_CACHE, "gpt2", fake_lm)
    monkeypatch.setitem(exp._TOKENIZER_CACHE, "gpt2", tok)

    lm_expert = exp.LMExpert(
        model_id="gpt2",
        domain="test",
        trunk_tokenizer=tok,
        freeze=True,
    )
    return lm_expert


class TestForwardBridgeUsesExactAlignment:
    """Once ``_forward_bridge`` switches to ``_align_by_char_offsets_exact``,
    positions with no exact char-end match must be left at uniform
    (the zero-init output buffer post-softmax = uniform).
    """

    def test_misaligned_positions_are_uniform(
        self, fake_cross_tok_bridge_setup, monkeypatch
    ):
        """Construct trunk + expert offsets where alternate positions
        misalign; verify the bridge output at those positions is all
        zero (= uniform after softmax).
        """
        tok = fake_cross_tok_bridge_setup
        lm_expert = _make_synthetic_lm_expert(tok, monkeypatch)
        # Force the bridge path even though tokenizers are identical
        lm_expert.is_same_tokenizer = False
        # Use an identity vocab bridge so the gather is a no-op
        from neuroslm.experts import VocabBridge
        V = tok.vocab_size
        lm_expert.vocab_bridge = VocabBridge(
            trunk_to_expert=torch.arange(V, dtype=torch.long),
            is_identity=False,
            coverage=1.0,
            vocab_size_trunk=V,
            vocab_size_expert=V,
        )
        # Patch the trunk + expert tokenisers' __call__ with offset sequences
        # that misalign at every other position.
        T = 4
        # Trunk char-ends: 3, 5, 8, 10.  Expert char-ends: 3, 7, 10.
        trunk_offsets_fake = [(0, 3), (3, 5), (5, 8), (8, 10)]
        expert_offsets_fake = [(0, 3), (3, 7), (7, 10)]
        expert_ids_fake = [100, 200, 300]

        def _fake_trunk_tok(text, **kw):
            return {
                "input_ids": [0, 1, 2, 3],
                "offset_mapping": trunk_offsets_fake,
            }

        def _fake_expert_tok(text, **kw):
            return {
                "input_ids": expert_ids_fake,
                "offset_mapping": expert_offsets_fake,
            }

        # Patch instance attributes (not class) so the originals stay clean
        monkeypatch.setattr(lm_expert, "_trunk_tokenizer", _FakeTokenizer(_fake_trunk_tok))
        monkeypatch.setattr(lm_expert, "_expert_tokenizer", _FakeTokenizer(_fake_expert_tok))

        # Run the bridge
        ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        with torch.no_grad():
            out = lm_expert(ids)
        assert out.shape == (1, T, V)
        # Expected from align_exact: [0, -1, -1, 2]
        # → positions 0 and 3 are bridged from expert positions 0 and 2;
        #   positions 1 and 2 are uniform (all zero logits).
        # We verify by checking that the non-zero variance only exists at
        # the aligned positions.
        per_position_max = out.abs().amax(dim=-1).squeeze(0)
        # Aligned positions should have non-zero amax (because FakeLM uses
        # per-position logits)
        assert per_position_max[0] > 0.0, "pos 0 (exact match) should not be uniform"
        assert per_position_max[3] > 0.0, "pos 3 (exact match) should not be uniform"
        # Misaligned positions should be exactly zero (uniform)
        assert per_position_max[1] == 0.0, "pos 1 (misaligned) must be uniform"
        assert per_position_max[2] == 0.0, "pos 2 (misaligned) must be uniform"


class _FakeTokenizer:
    """Wraps a callable into something that looks tokenizer-like for the
    bridge path. Only the ``__call__`` + ``.decode`` surface used by
    ``_forward_bridge`` is supported.
    """
    def __init__(self, call_fn):
        self._call_fn = call_fn

    def __call__(self, text, **kw):
        return self._call_fn(text, **kw)

    def decode(self, ids, **kw):
        # Just produce a stable surface for any id sequence — used only
        # to seed the per-sample text in _forward_bridge.
        return "x" * (len(ids) * 3)


# ──────────────────────────────────────────────────────────────────────
# LMExpert exposes alignment-coverage telemetry
# ──────────────────────────────────────────────────────────────────────


class TestAlignmentCoverageTelemetry:
    """The harness will want to log per-expert alignment coverage during
    training (a leading indicator: low coverage = mostly-uniform
    distillation signal = expert is silenced)."""

    def test_default_is_none_before_first_forward(
        self, fake_cross_tok_bridge_setup, monkeypatch
    ):
        tok = fake_cross_tok_bridge_setup
        lm_expert = _make_synthetic_lm_expert(tok, monkeypatch)
        assert lm_expert.last_alignment_coverage is None

    def test_same_tok_expert_reports_1_0(
        self, fake_cross_tok_bridge_setup, monkeypatch
    ):
        """Same-tok experts are effectively 100% aligned (no bridge, no
        re-tokenisation). The telemetry value should be 1.0 after a
        forward pass.
        """
        tok = fake_cross_tok_bridge_setup
        lm_expert = _make_synthetic_lm_expert(tok, monkeypatch)
        assert lm_expert.is_same_tokenizer is True
        ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        with torch.no_grad():
            _ = lm_expert(ids)
        assert lm_expert.last_alignment_coverage == 1.0

    def test_cross_tok_expert_reports_fraction_aligned(
        self, fake_cross_tok_bridge_setup, monkeypatch
    ):
        """For the synthetic 4-trunk / 3-expert misaligned setup used
        above, only 2 of 4 trunk positions exact-align → coverage = 0.5.
        """
        tok = fake_cross_tok_bridge_setup
        lm_expert = _make_synthetic_lm_expert(tok, monkeypatch)
        lm_expert.is_same_tokenizer = False
        from neuroslm.experts import VocabBridge
        V = tok.vocab_size
        lm_expert.vocab_bridge = VocabBridge(
            trunk_to_expert=torch.arange(V, dtype=torch.long),
            is_identity=False,
            coverage=1.0,
            vocab_size_trunk=V,
            vocab_size_expert=V,
        )
        trunk_offsets_fake = [(0, 3), (3, 5), (5, 8), (8, 10)]
        expert_offsets_fake = [(0, 3), (3, 7), (7, 10)]
        monkeypatch.setattr(
            lm_expert, "_trunk_tokenizer",
            _FakeTokenizer(lambda text, **kw: {
                "input_ids": [0, 1, 2, 3],
                "offset_mapping": trunk_offsets_fake,
            }),
        )
        monkeypatch.setattr(
            lm_expert, "_expert_tokenizer",
            _FakeTokenizer(lambda text, **kw: {
                "input_ids": [100, 200, 300],
                "offset_mapping": expert_offsets_fake,
            }),
        )
        ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        with torch.no_grad():
            _ = lm_expert(ids)
        assert lm_expert.last_alignment_coverage == pytest.approx(0.5, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────
# Integration: real bridge CE measurably improves on natural English
# (slow — downloads SmolLM2; runs only when env var is set so CI is fast)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_BRIDGE_CE_SMOKE", "0") != "1",
    reason="set RUN_BRIDGE_CE_SMOKE=1 to download SmolLM2 + run the CE smoke test",
)
class TestExactAlignmentBeatsCurrentOnEnglish:
    """End-to-end: exact alignment must yield STRICTLY LOWER CE than
    the legacy smallest-ge alignment on natural English."""

    def test_smollm2_exact_align_ce_better_than_smallest_ge(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from neuroslm.experts import (
            VocabBridge,
            _align_by_char_offsets,        # legacy
            _align_by_char_offsets_exact,  # new
        )
        trunk_tok = AutoTokenizer.from_pretrained("gpt2")
        expert_tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
        expert_lm = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-360M", use_safetensors=True,
        ).eval()

        text = (
            "Mathematics is the language in which the universe is written. "
            "Theorems are proved by logical deduction from a small set of axioms."
        )
        trunk_enc = trunk_tok(text, add_special_tokens=False,
                              return_offsets_mapping=True)
        expert_enc = expert_tok(text, add_special_tokens=False,
                                return_offsets_mapping=True)
        trunk_ids = torch.tensor([trunk_enc["input_ids"]], dtype=torch.long)
        expert_ids = torch.tensor([expert_enc["input_ids"]], dtype=torch.long)
        T = trunk_ids.shape[1]
        targets = trunk_ids[0, 1:]

        with torch.no_grad():
            e_logits = expert_lm(input_ids=expert_ids).logits.squeeze(0)

        bridge = VocabBridge.build(trunk_tok, expert_tok)

        def bridge_with(align_fn):
            idx = align_fn(trunk_enc["offset_mapping"], expert_enc["offset_mapping"])
            V_t = trunk_tok.vocab_size
            out = torch.zeros((T, V_t), dtype=torch.float32)
            for t, e in enumerate(idx):
                if e < 0:
                    continue
                row = bridge.apply(e_logits[e:e+1])
                out[t] = row.squeeze(0)
            return out

        out_legacy = bridge_with(_align_by_char_offsets)
        out_exact  = bridge_with(_align_by_char_offsets_exact)

        ce_legacy = F.cross_entropy(out_legacy[:-1], targets).item()
        ce_exact  = F.cross_entropy(out_exact[:-1],  targets).item()

        assert ce_exact < ce_legacy, (
            f"exact alignment must beat legacy on English; "
            f"got exact={ce_exact:.3f} legacy={ce_legacy:.3f}"
        )
        # And meaningfully so — at least 0.1 nat improvement
        assert (ce_legacy - ce_exact) > 0.1, (
            f"expected CE improvement >0.1 nats, got "
            f"{ce_legacy - ce_exact:+.3f}"
        )
