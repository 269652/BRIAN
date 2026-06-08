# -*- coding: utf-8 -*-
"""TDD: ``GPT2SubCortex`` must handle sequences longer than
``config.n_positions`` without crashing.

Reproduces the production failure on ``rcc_bowtie_30m_p4`` at
``seq_len=2048`` against the 1024-position GPT-2 family
(``gpt2`` / ``gpt2-medium`` / ``distilgpt2``)::

    /pytorch/aten/src/ATen/native/cuda/IndexKernelUtils.cu:16:
        vectorized_gather_kernel: ...
        Assertion `ind >= 0 && ind < ind_dim_size
                   && "vectorized gather kernel index out of bounds"`
        failed.
    torch.AcceleratorError: CUDA error: device-side assert triggered

Root cause: GPT-2's positional embedding ``wpe`` has exactly
``config.n_positions`` rows. ``GPT2SubCortex.forward`` previously
forwarded any ``(B, T)`` tensor with ``T > n_positions`` straight to
``self.gpt2(input_ids=ids, ...)``, triggering an out-of-bounds gather
on the position lookup deep inside the GPT-2 forward.

Fix contract (pinned by the tests below):

  1. Short-path back-compat (``T <= n_positions``): output must be
     **bit-identical** to a raw ``self.gpt2(input_ids=ids)`` call.
  2. Long-path no-crash (``T > n_positions``): no exception, correct
     shape, finite output.
  3. Composability: long output must equal a manual non-overlapping
     per-window concatenation (``torch.cat([gpt2(chunk_i) for ...])``)
     bit-for-bit. This pins the *semantics* — non-overlapping chunks,
     each fed positions 0..len-1 (always in-distribution).
  4. Config-driven window size: the chunk size must come from
     ``self.gpt2.config.n_positions``, not be hardcoded to 1024.
     Works for any future variant or shrunken model.

Test architecture: a real ``GPT2Model`` (the production class, not a
mock) instantiated with ``GPT2Config(n_positions=8, ...)`` — small dims
+ random weights, no HF download. The position-embedding bug lives in
the architecture, not the weights, so a tiny real model reproduces it
exactly. This keeps the suite fast (< 1 s) while exercising the same
code paths the production GPT-2 hits.
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2Model    # noqa: E402

from neuroslm.cortex import GPT2SubCortex          # noqa: E402


# ────────────────────────────────────────────────────────────────────
# Fixtures — real GPT2Model, tiny dims, no HF download
# ────────────────────────────────────────────────────────────────────

def _tiny_gpt2(n_positions: int = 8, n_embd: int = 16,
               n_layer: int = 1, n_head: int = 2,
               vocab_size: int = 64) -> GPT2Model:
    """Build a real ``GPT2Model`` with tiny dims (no download).

    This is the *production* class with random weights — every code
    path (token + position embedding, attention, MLP, layernorm) is
    the same one ``from_pretrained("gpt2")`` exercises. Only the
    weights and the dimensions differ.
    """
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        # Eliminate stochastic layers so equality assertions are stable
        attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0, summary_first_dropout=0.0,
    )
    m = GPT2Model(cfg)
    m.eval()    # disable any remaining train-time non-determinism
    return m


@pytest.fixture
def tiny_subcortex():
    """A ``GPT2SubCortex`` wrapping a tiny real ``GPT2Model``
    (n_positions=8). Built via the ``from_module`` classmethod —
    the public injection point that lets callers (production *and*
    tests) supply a pre-built model rather than going through
    ``from_pretrained``."""
    gpt2 = _tiny_gpt2(n_positions=8, n_embd=16)
    return GPT2SubCortex.from_module(
        name="tiny", domain="general", gpt2=gpt2,
        hf_model_id="<in-memory>",
    )


# ────────────────────────────────────────────────────────────────────
# Contract 0 — clean public injection point exists
# ────────────────────────────────────────────────────────────────────

class TestFromModuleClassmethod:
    """The fix must expose a public, non-hacky way to construct
    ``GPT2SubCortex`` from an already-built ``GPT2Model``. Without
    this, callers (and tests) are forced into ``__new__`` gymnastics
    or full ``from_pretrained`` downloads."""

    def test_from_module_returns_gpt2_subcortex(self):
        gpt2 = _tiny_gpt2()
        sc = GPT2SubCortex.from_module(
            name="x", domain="general", gpt2=gpt2,
            hf_model_id="anything",
        )
        assert isinstance(sc, GPT2SubCortex)
        assert sc.gpt2 is gpt2
        assert sc.name == "x"
        assert sc.domain == "general"
        assert sc.d_native == gpt2.config.n_embd
        assert sc.hf_model_id == "anything"

    def test_from_module_freezes_by_default(self):
        gpt2 = _tiny_gpt2()
        sc = GPT2SubCortex.from_module(
            name="x", domain="general", gpt2=gpt2,
            hf_model_id="anything",
        )
        assert all(not p.requires_grad for p in sc.gpt2.parameters()), \
            "from_module should freeze weights by default to match " \
            "the from_pretrained constructor's behaviour"

    def test_from_module_freeze_weights_false_keeps_grads(self):
        gpt2 = _tiny_gpt2()
        sc = GPT2SubCortex.from_module(
            name="x", domain="general", gpt2=gpt2,
            hf_model_id="anything", freeze_weights=False,
        )
        assert all(p.requires_grad for p in sc.gpt2.parameters())


# ────────────────────────────────────────────────────────────────────
# Contract 1 — short-context back-compat (bit-identical to raw GPT-2)
# ────────────────────────────────────────────────────────────────────

class TestShortContextBackCompat:
    """For ``T <= n_positions`` the wrapper must be a no-op layer
    around the raw GPT-2 forward — no rewrapping, no overhead, no
    silent semantic change."""

    def test_T_equals_n_positions_matches_raw_gpt2(self, tiny_subcortex):
        sc = tiny_subcortex
        n_pos = int(sc.gpt2.config.n_positions)    # 8
        ids = torch.randint(0, sc.gpt2.config.vocab_size, (2, n_pos))
        with torch.no_grad():
            wrapped = sc(ids)
            raw = sc.gpt2(input_ids=ids, output_hidden_states=False,
                          return_dict=True).last_hidden_state
        assert wrapped.shape == (2, n_pos, sc.gpt2.config.n_embd)
        assert torch.equal(wrapped, raw), (
            "T == n_positions must be bit-identical to raw GPT-2 "
            "forward — back-compat violated"
        )

    def test_T_less_than_n_positions_matches_raw_gpt2(self, tiny_subcortex):
        sc = tiny_subcortex
        ids = torch.randint(0, sc.gpt2.config.vocab_size, (2, 5))
        with torch.no_grad():
            wrapped = sc(ids)
            raw = sc.gpt2(input_ids=ids, output_hidden_states=False,
                          return_dict=True).last_hidden_state
        assert torch.equal(wrapped, raw)

    def test_T_equals_1_matches_raw_gpt2(self, tiny_subcortex):
        sc = tiny_subcortex
        ids = torch.randint(0, sc.gpt2.config.vocab_size, (1, 1))
        with torch.no_grad():
            wrapped = sc(ids)
            raw = sc.gpt2(input_ids=ids, output_hidden_states=False,
                          return_dict=True).last_hidden_state
        assert torch.equal(wrapped, raw)


# ────────────────────────────────────────────────────────────────────
# Contract 2 — long-context: no crash, correct shape, finite
# ────────────────────────────────────────────────────────────────────

class TestLongContextNoCrash:
    """Reproduces the production failure and asserts the fix is in
    place."""

    def test_T_just_above_n_positions_does_not_crash(self, tiny_subcortex):
        """The minimal failing case: T = n_positions + 1 was enough
        to trigger the CUDA OOB in production."""
        sc = tiny_subcortex
        n_pos = int(sc.gpt2.config.n_positions)
        ids = torch.randint(0, sc.gpt2.config.vocab_size, (2, n_pos + 1))
        with torch.no_grad():
            out = sc(ids)                                  # MUST NOT raise
        assert out.shape == (2, n_pos + 1, sc.gpt2.config.n_embd)
        assert torch.isfinite(out).all()

    def test_T_exactly_2x_n_positions(self, tiny_subcortex):
        sc = tiny_subcortex
        n_pos = int(sc.gpt2.config.n_positions)
        ids = torch.randint(0, sc.gpt2.config.vocab_size, (2, n_pos * 2))
        with torch.no_grad():
            out = sc(ids)
        assert out.shape == (2, n_pos * 2, sc.gpt2.config.n_embd)
        assert torch.isfinite(out).all()

    def test_T_4x_n_positions(self, tiny_subcortex):
        """Mimics the production failure ratio (seq_len=2048,
        n_positions=1024 = 2x); we go 4x here to make sure the
        chunking loop iterates multiple times."""
        sc = tiny_subcortex
        n_pos = int(sc.gpt2.config.n_positions)
        ids = torch.randint(0, sc.gpt2.config.vocab_size, (1, n_pos * 4))
        with torch.no_grad():
            out = sc(ids)
        assert out.shape == (1, n_pos * 4, sc.gpt2.config.n_embd)
        assert torch.isfinite(out).all()

    def test_T_non_multiple_of_n_positions(self, tiny_subcortex):
        """Tail chunk shorter than n_positions: 8 + 8 + 3."""
        sc = tiny_subcortex
        ids = torch.randint(0, sc.gpt2.config.vocab_size, (2, 19))
        with torch.no_grad():
            out = sc(ids)
        assert out.shape == (2, 19, sc.gpt2.config.n_embd)
        assert torch.isfinite(out).all()


# ────────────────────────────────────────────────────────────────────
# Contract 3 — chunked output equals manual per-window concatenation
# ────────────────────────────────────────────────────────────────────

class TestChunkedOutputComposability:
    """The windowed forward must equal a per-window concatenation —
    this pins the *semantics* of the fix (non-overlapping chunks, each
    GPT-2-trained-distribution-valid) and rules out alternatives like
    sliding windows, overlap-and-average, or position interpolation."""

    def test_long_output_equals_manual_per_window_concat(self, tiny_subcortex):
        sc = tiny_subcortex
        n_pos = int(sc.gpt2.config.n_positions)     # 8
        T = n_pos * 2 + 3                            # 8 + 8 + 3 = 19
        ids = torch.randint(0, sc.gpt2.config.vocab_size, (2, T))

        with torch.no_grad():
            # Reference: manual non-overlapping window concat
            manual_parts = []
            for start in range(0, T, n_pos):
                end = min(start + n_pos, T)
                chunk = ids[:, start:end]
                part = sc.gpt2(
                    input_ids=chunk, output_hidden_states=False,
                    return_dict=True,
                ).last_hidden_state
                manual_parts.append(part)
            manual = torch.cat(manual_parts, dim=1)

            actual = sc(ids)

        assert actual.shape == manual.shape == (2, T, sc.gpt2.config.n_embd)
        assert torch.equal(actual, manual), (
            "long-context output must equal manual per-window "
            "concatenation; non-overlapping chunking semantics violated"
        )

    def test_chunked_output_finite_across_long_run(self, tiny_subcortex):
        """Sanity: many iterations don't accumulate NaN/Inf."""
        sc = tiny_subcortex
        ids = torch.randint(
            0, sc.gpt2.config.vocab_size, (1, sc.gpt2.config.n_positions * 8),
        )
        with torch.no_grad():
            out = sc(ids)
        assert torch.isfinite(out).all()


# ────────────────────────────────────────────────────────────────────
# Contract 4 — window size driven by model config, not hardcoded
# ────────────────────────────────────────────────────────────────────

class TestUsesModelConfigForWindowSize:
    """The fix must read the window size from ``self.gpt2.config`` —
    not hardcode 1024 — so it works across gpt2 / gpt2-medium /
    gpt2-large / gpt2-xl and any future variant."""

    def test_smaller_window_respected(self):
        """A model with n_positions=4 must be chunked into 4-token
        windows, not 8-token (the fixture default)."""
        gpt2 = _tiny_gpt2(n_positions=4, n_embd=16)
        sc = GPT2SubCortex.from_module(
            name="tiny4", domain="general", gpt2=gpt2,
            hf_model_id="<in-memory>",
        )
        ids = torch.randint(0, gpt2.config.vocab_size, (1, 10))   # 4+4+2

        with torch.no_grad():
            actual = sc(ids)
            # Manual reference with the right (n_pos=4) chunking
            manual = torch.cat([
                sc.gpt2(input_ids=ids[:, s:s + 4]).last_hidden_state
                for s in range(0, 10, 4)
            ], dim=1)

        assert actual.shape == (1, 10, 16)
        assert torch.equal(actual, manual), (
            "window size must be driven by gpt2.config.n_positions; "
            "using a different chunk size produces different output"
        )

    def test_larger_window_respected(self):
        """A model with n_positions=32 must NOT be chunked when given
        a 20-token input (it fits in one window)."""
        gpt2 = _tiny_gpt2(n_positions=32, n_embd=16)
        sc = GPT2SubCortex.from_module(
            name="tiny32", domain="general", gpt2=gpt2,
            hf_model_id="<in-memory>",
        )
        ids = torch.randint(0, gpt2.config.vocab_size, (1, 20))
        with torch.no_grad():
            actual = sc(ids)
            raw = sc.gpt2(input_ids=ids).last_hidden_state
        assert torch.equal(actual, raw), (
            "T <= n_positions must take the short-path; current "
            "behaviour suggests the wrapper is chunking unnecessarily"
        )


# ────────────────────────────────────────────────────────────────────
# Contract 5 — gradient flow when weights are unfrozen
# ────────────────────────────────────────────────────────────────────

class TestGradientFlowLongContext:
    """Long-context forward must still build a gradient graph when
    weights are trainable — otherwise the windowed path would silently
    detach gradients in a way that broke fine-tuning."""

    def test_gradients_flow_through_long_context_when_unfrozen(self):
        gpt2 = _tiny_gpt2(n_positions=8, n_embd=16)
        sc = GPT2SubCortex.from_module(
            name="tiny", domain="general", gpt2=gpt2,
            hf_model_id="<in-memory>", freeze_weights=False,
        )
        ids = torch.randint(0, gpt2.config.vocab_size, (1, 20))     # 8+8+4
        out = sc(ids)
        loss = out.pow(2).mean()
        loss.backward()
        # At least one parameter must have a non-zero gradient
        grad_norms = [
            p.grad.norm().item() for p in sc.gpt2.parameters()
            if p.grad is not None
        ]
        assert grad_norms, "no GPT-2 parameter received a gradient"
        assert max(grad_norms) > 0.0, "all GPT-2 gradients are zero"
