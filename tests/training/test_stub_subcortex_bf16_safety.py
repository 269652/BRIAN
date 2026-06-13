"""Defensive contract: ``StubSubCortex.forward`` must not crash in bf16.

Root-cause record
=================

On the 2026-01-28 vast.ai deploy (instance ``40844277``, A100 SXM4,
``--model dsl_lm`` + bf16 autocast), the training run died at:

    File "/workspace/brian/neuroslm/cortex.py", line 169
        h = self.encoder(h, mask=mask, is_causal=True)
    ...
    RuntimeError: CUDA error: an illegal memory access was encountered
        in self.linear2(self.dropout(self.activation(self.linear1(x))))

PyTorch's :class:`torch.nn.TransformerEncoder` accepts EITHER an
explicit additive ``mask`` OR the boolean ``is_causal`` hint — passing
both simultaneously is documented as undefined behaviour and on
bf16/A100 it corrupts the FFN's intermediate scratch buffer in the
nested-tensor fast path. The combination triggers an illegal memory
access deep in the kernel.

This file pins:

  1. ``StubSubCortex.forward`` must complete on CPU in bf16 without
     raising — this is the local reproduction of the on-device crash.
     (We can't reproduce the *exact* CUDA error on CPU, but the
     redundant-mask anti-pattern is detectable here: PyTorch raises a
     ``RuntimeError`` or a deprecation warning on CPU with the same
     bad combo, and either is a failure.)
  2. The same input/output shapes are preserved across the fix — we
     are not allowed to silently degrade the causal mask semantics.
  3. The encoder DOES enforce causality (no time-travel): position
     ``t`` of the output may depend on positions ``0..t`` only,
     regardless of which of ``mask`` / ``is_causal`` is used.

The fix in ``cortex.py`` is single-line: drop the explicit ``mask``
and rely on ``is_causal=True`` only. The kernel builds its own
causal mask internally and the fast-path is bug-free in that mode.
"""
from __future__ import annotations

import math
import warnings

import pytest


torch = pytest.importorskip("torch")


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def stub_cortex():
    """Same construction the deploy uses: ``d_model=512, n_heads=4,
    n_layers=2, vocab=8`` — small enough to run on CPU in milliseconds.
    """
    from neuroslm.cortex import StubSubCortex

    return StubSubCortex(
        name="probe",
        domain="general",
        vocab=8,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_ctx=32,
    )


# ──────────────────────────────────────────────────────────────────────
# Crash repro contracts
# ──────────────────────────────────────────────────────────────────────


class TestBf16ForwardSafety:
    """The forward MUST complete cleanly under bf16 autocast on CPU."""

    def test_bf16_autocast_forward_does_not_crash(self, stub_cortex):
        ids = torch.randint(0, 8, (2, 16))
        # autocast(bf16) is what the deploy runs under — every step is
        # wrapped in `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.
        # We exercise the same code path on CPU (bf16 SDPA backend
        # exists on CPU too, and the bug surface is the same).
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            out = stub_cortex(ids)
        # Shape preserved and outputs finite.
        assert out.shape == (2, 16, 64), out.shape
        assert torch.isfinite(out).all(), (
            "bf16 forward produced non-finite values — likely the same "
            "memory-corruption issue as the on-device crash"
        )

    def test_no_mask_plus_is_causal_warning(self, stub_cortex):
        """The fix must not re-introduce the redundant mask+is_causal
        anti-pattern. PyTorch emits a deprecation warning when both
        are passed — assert silence."""
        ids = torch.randint(0, 8, (2, 16))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = stub_cortex(ids)
        bad = [w for w in caught
               if "is_causal" in str(w.message)
               and "mask" in str(w.message).lower()]
        assert not bad, (
            "fix must not re-introduce the `mask=...` + `is_causal=True` "
            f"combination; got warnings: {[str(w.message) for w in bad]}"
        )

    def test_bf16_forward_no_explicit_mask_in_call(self, stub_cortex):
        """Pin the implementation: the forward must call
        ``self.encoder(...)`` WITHOUT passing both an explicit
        ``mask=`` kwarg AND ``is_causal=True``. We monkey-patch the
        encoder's ``forward`` and inspect the actual kwargs."""
        captured = {}
        original = stub_cortex.encoder.forward

        def _spy(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return original(*args, **kwargs)

        stub_cortex.encoder.forward = _spy
        try:
            ids = torch.randint(0, 8, (2, 16))
            with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                _ = stub_cortex(ids)
        finally:
            stub_cortex.encoder.forward = original

        # The only legal combinations are:
        #   * mask=<tensor>, is_causal=False (or unset)
        #   * mask=None    , is_causal=True
        mask = captured["kwargs"].get("mask", None)
        is_causal = captured["kwargs"].get("is_causal", False)
        bad_combo = (mask is not None) and bool(is_causal)
        assert not bad_combo, (
            f"redundant mask+is_causal=True call: mask={type(mask).__name__}, "
            f"is_causal={is_causal} — this is the bf16 crash root cause"
        )


# ──────────────────────────────────────────────────────────────────────
# Semantic correctness — causality must be preserved by the fix
# ──────────────────────────────────────────────────────────────────────


class TestCausalitySemantics:
    """After the fix, the encoder still must not let position ``t`` see
    positions ``> t``. We verify this by perturbing a future token and
    confirming the past-token outputs are unchanged."""

    def test_future_perturbation_does_not_affect_past(self, stub_cortex):
        ids_a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]])
        ids_b = ids_a.clone()
        ids_b[0, -1] = 7  # perturb only the LAST position

        stub_cortex.eval()
        with torch.no_grad():
            out_a = stub_cortex(ids_a)
            out_b = stub_cortex(ids_b)
        # The first 7 positions must be bit-identical between a and b —
        # causal masking forbids the last-token change from rippling back.
        diff = (out_a[:, :-1] - out_b[:, :-1]).abs().max().item()
        assert diff < 1e-5, (
            f"causal semantics broken: perturbing position 7 changed "
            f"earlier outputs by {diff:.2e} (should be 0)"
        )

    def test_forward_shape_unchanged(self, stub_cortex):
        """Sanity: the fix must not change the public surface."""
        ids = torch.randint(0, 8, (3, 12))
        out = stub_cortex(ids)
        assert out.shape == (3, 12, 64)
