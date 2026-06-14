"""Defensive contracts: ``LMExpert._forward_bridge`` cannot crash from
oversized expert tokenisations or autocast-dtype mismatches.

Root-cause record
=================

On the 2026-06-14 deploy (instance ``40921910``, A100 SXM4, bf16),
the run died inside the bridge path with::

    File "/workspace/brian/neuroslm/experts.py", line 548
        expert_logits = self.lm(input_ids=expert_input_ids).logits
    ...
    RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED
        when calling cublasLtMatmul ... m 768 n 600 k 768

Two compounding problems
------------------------

  1. **Expert tokenisation can exceed the trunk's seq_len.** A trunk
     batch of 512 random ids decodes (via the GPT-2 tokenizer) to ~936
     chars, which the CodeGPT-small-py tokenizer re-encodes to 694
     tokens. We feed those 694 tokens straight into HF GPT-2's
     ``addmm`` under bf16 autocast — well beyond what's been exercised
     by any local test, and beyond what the trunk's chunking has
     hardened.

  2. **Bf16 autocast leaks into the frozen HF expert.** The harness
     wraps every step in ``torch.amp.autocast(dtype=bfloat16)``. The
     legacy ``.bin``-loaded GPT-2 expert lives in fp32, but the
     autocast context downcasts its inputs (and the ``Conv1D``
     ``addmm`` weights too in unstable ways) → CUBLAS executes a
     malformed kernel and aborts.

Fix policy
==========

Two complementary safeguards, both required:

  * **Truncate ``expert_input_ids``** to the same length the trunk
     received (``T``). The bridge only needs to produce ``t_count
     <= T`` aligned logits anyway; the trailing expert tokens are
     never used.
  * **Disable autocast inside the expert forward** via
     ``torch.amp.autocast(enabled=False)``. Frozen experts run in
     their loaded dtype (fp32); downcast happens only when the
     bridged logits flow back into the harness fusion.

These are independent — either alone would have prevented the crash
on this exact deploy — but we apply both because either failure mode
can resurface in different configs.
"""
from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
transformers = pytest.importorskip("transformers")


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trunk_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


class _FakeBPETokenizer:
    """Minimal stand-in for a cross-vocab BPE tokenizer.

    Re-encodes the input text into ~1.4x more tokens than the trunk
    (mirrors the CodeGPT-small-py → 694-from-512 blow-up observed
    in the on-device crash). Returns offset-mappings so the bridge
    path's char-offset alignment can still run.
    """
    def __init__(self, vocab_size: int = 50000,
                 name_or_path: str = "fake-cross-tok"):
        self.vocab_size = vocab_size
        self.name_or_path = name_or_path

    def __call__(self, text: str, **kwargs):
        # Produce one token per character (extreme blow-up: 936 from
        # 512). Each "token" is just its char index for simplicity.
        ids = [i % self.vocab_size for i in range(len(text))]
        offsets = [(i, i + 1) for i in range(len(text))]
        return {
            "input_ids": ids,
            "offset_mapping": offsets,
        }

    def get_vocab(self):
        return {str(i): i for i in range(self.vocab_size)}

    def convert_ids_to_tokens(self, ids, **kwargs):
        return [str(i) for i in ids]

    def convert_tokens_to_ids(self, tokens):
        return [int(t) if str(t).isdigit() else 0 for t in tokens]


class _CountingFakeLM(nn.Module):
    """Wrapper that records the largest ``input_ids.shape[1]`` it ever
    sees. Used by the truncation tests."""
    def __init__(self, vocab: int, d_model: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.head = nn.Linear(d_model, vocab)
        self.config = type("C", (), {"vocab_size": vocab,
                                     "max_position_embeddings": 256})
        self.max_seen_T = 0
        self.dtypes_seen = []

    def forward(self, input_ids=None, **_):
        self.max_seen_T = max(self.max_seen_T, input_ids.shape[1])
        h = self.embed(input_ids)
        self.dtypes_seen.append(h.dtype)
        logits = self.head(h)
        return type("O", (), {"logits": logits})()


# ──────────────────────────────────────────────────────────────────────
# Contract 1 — bridge truncates expert tokens to T (trunk seq_len)
# ──────────────────────────────────────────────────────────────────────


class TestExpertTruncation:
    """The bridge re-tokenises arbitrary trunk text with the expert
    tokenizer, which can produce ``T_expert > T``. The trailing tokens
    are never used (the alignment map only reaches ``T``) yet we still
    pay the full forward cost AND risk CUBLAS instability on large
    matmuls in bf16. Truncate at re-encode time."""

    def test_expert_input_capped_at_trunk_T(self, trunk_tokenizer):
        from neuroslm.experts import LMExpert, VocabBridge

        # Build a minimal LMExpert by-hand to bypass HF download.
        e = object.__new__(LMExpert)
        nn.Module.__init__(e)
        fake_expert_tok = _FakeBPETokenizer()
        # Match counting_lm's vocab to the trunk so the bridge.apply
        # is a same-shape pass-through (identity bridge). This isolates
        # the truncation contract from any vocab-mapping concerns.
        counting_lm = _CountingFakeLM(vocab=trunk_tokenizer.vocab_size)
        e.model_id = "fake"
        e.domain = "general"
        e.lm = counting_lm
        e._trunk_tokenizer = trunk_tokenizer
        e._expert_tokenizer = fake_expert_tok
        e.vocab_bridge = VocabBridge.build(
            trunk_tokenizer=trunk_tokenizer,
            expert_tokenizer=trunk_tokenizer,
        )
        e.is_same_tokenizer = False
        e.vocab_size_trunk = trunk_tokenizer.vocab_size
        e.vocab_size_expert = trunk_tokenizer.vocab_size

        # Trunk seq_len = 512, same as the deployed config.
        T = 512
        ids = torch.randint(0, trunk_tokenizer.vocab_size, (1, T))
        with torch.no_grad():
            _ = e._forward_bridge(ids)

        assert counting_lm.max_seen_T <= T, (
            f"bridge fed the expert {counting_lm.max_seen_T} tokens "
            f"but trunk T = {T}; oversized expert input causes "
            f"CUBLAS_STATUS_EXECUTION_FAILED on bf16 CUDA"
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 2 — expert forward runs with autocast DISABLED
# ──────────────────────────────────────────────────────────────────────


class TestAutocastIsolation:
    """The frozen HF expert lives in fp32 (or whatever dtype it loaded
    in). Bf16 autocast around the harness step must NOT leak into the
    expert's matmuls — frozen weights + autocast-downcast inputs is a
    known CUBLAS instability source on the bridge path."""

    def test_expert_forward_sees_fp32_under_bf16_autocast(
        self, trunk_tokenizer
    ):
        from neuroslm.experts import LMExpert, VocabBridge

        e = object.__new__(LMExpert)
        nn.Module.__init__(e)
        fake_expert_tok = _FakeBPETokenizer()
        counting_lm = _CountingFakeLM(vocab=trunk_tokenizer.vocab_size)
        e.model_id = "fake"
        e.domain = "general"
        e.lm = counting_lm
        e._trunk_tokenizer = trunk_tokenizer
        e._expert_tokenizer = fake_expert_tok
        e.vocab_bridge = VocabBridge.build(
            trunk_tokenizer=trunk_tokenizer,
            expert_tokenizer=trunk_tokenizer,
        )
        e.is_same_tokenizer = False
        e.vocab_size_trunk = trunk_tokenizer.vocab_size
        e.vocab_size_expert = trunk_tokenizer.vocab_size

        ids = torch.randint(0, trunk_tokenizer.vocab_size, (1, 16))

        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            with torch.no_grad():
                _ = e._forward_bridge(ids)

        # Every dtype the fake LM observed must be fp32 — autocast
        # was suppressed for the expert forward.
        assert counting_lm.dtypes_seen, "expert was never invoked"
        non_fp32 = [d for d in counting_lm.dtypes_seen
                    if d != torch.float32]
        assert not non_fp32, (
            f"bf16 autocast leaked into the frozen expert forward: "
            f"saw dtypes {set(counting_lm.dtypes_seen)}. The expert "
            f"must run with autocast disabled to avoid CUBLAS instability."
        )

    def test_same_tok_forward_also_autocast_disabled(
        self, trunk_tokenizer
    ):
        """Same isolation contract for the fast path."""
        from neuroslm.experts import LMExpert, VocabBridge

        e = object.__new__(LMExpert)
        nn.Module.__init__(e)
        counting_lm = _CountingFakeLM(vocab=trunk_tokenizer.vocab_size)
        e.model_id = "fake"
        e.domain = "general"
        e.lm = counting_lm
        e._trunk_tokenizer = trunk_tokenizer
        e._expert_tokenizer = trunk_tokenizer
        e.vocab_bridge = VocabBridge.build(
            trunk_tokenizer=trunk_tokenizer,
            expert_tokenizer=trunk_tokenizer,
        )
        e.is_same_tokenizer = True
        e.vocab_size_trunk = trunk_tokenizer.vocab_size
        e.vocab_size_expert = trunk_tokenizer.vocab_size

        ids = torch.randint(0, trunk_tokenizer.vocab_size, (1, 16))
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            with torch.no_grad():
                _ = e._forward_same_tok(ids)

        non_fp32 = [d for d in counting_lm.dtypes_seen
                    if d != torch.float32]
        assert not non_fp32, (
            f"bf16 autocast leaked into the fast-path expert forward: "
            f"saw dtypes {set(counting_lm.dtypes_seen)}"
        )
