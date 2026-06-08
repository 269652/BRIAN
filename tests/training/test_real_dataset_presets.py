# -*- coding: utf-8 -*-
"""TDD: every training preset reachable via ``brian train`` must
consume **real tokens**, not synthetic random batches from ``torch.randint``.

Background
----------
The user reported a training plateau at ``lm_loss ≈ 8.32 = log(4096)`` —
the entropy of uniform-random labels over a vocab of 4096. Diagnosis:
the CLI's ``--preset=tiny`` path routes to
:mod:`colab_train_minimal_cpu`, whose previous implementation built
batches via ``torch.randint(0, vocab_size, ...)``. No matter how long
you trained, the model could never beat ``log(V)`` because **the
targets were uniform noise.**

This suite pins three contracts:

1. **Real data by default** — :func:`colab_train_minimal_cpu.main` must
   build its training batches via the project's real-text pipeline
   (:mod:`neuroslm.data.batch_iterator`), not ``torch.randint``.

2. **Robust fallback chain** — when the upstream HuggingFace stream is
   unreachable (no network, no ``datasets`` install), the script must
   still produce a usable batch source via either a bundled local
   corpus OR a clearly-logged synthetic fallback. It must NEVER
   silently train on random tokens without telling the operator.

3. **Big presets keep their real-data wiring** — the DSL trainer in
   :mod:`neuroslm.train_dsl` already uses ``RealDataSource``; a
   regression guard ensures that contract isn't lost in future refactors.

These tests do not require network: they monkey-patch
``neuroslm.data._stream_iterator`` to a deterministic token generator
(same pattern as :mod:`tests.test_data_adaptive_mixture`).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

class _FakeTokenizer:
    """Tokenizer stand-in: deterministic, no tiktoken / no network."""
    vocab_size = 128
    eos_id = 0

    def encode(self, text: str) -> list[int]:
        # Map char codes into a small range so encode() stays deterministic.
        return [(ord(c) % (self.vocab_size - 1)) + 1 for c in text[:32]]

    def decode(self, ids) -> str:
        return ""


@pytest.fixture
def fake_text_stream(monkeypatch):
    """Replace ``neuroslm.data._stream_iterator`` with a deterministic,
    learnable token pattern. Returns the patched module for assertions.

    The pattern is a fixed short cycle so a small LM can actually learn
    to predict next-token (i.e., loss should drop below ``log(V)``).
    """
    from neuroslm import data as data_mod

    def _fake_stream(_tok, ctx_len, mode, _buf=8192):
        # Repeating cycle of 8 distinct tokens — deterministic, learnable.
        pattern = [10, 20, 30, 40, 50, 60, 70, 80]
        cursor = 0
        while True:
            window = [pattern[(cursor + i) % len(pattern)]
                      for i in range(ctx_len + 1)]
            cursor += 1
            yield window

    monkeypatch.setattr(data_mod, "_stream_iterator", _fake_stream)
    return data_mod


@pytest.fixture
def fake_tokenizer(monkeypatch):
    """Stub the project tokenizer so it doesn't need tiktoken in tests."""
    fake = _FakeTokenizer()

    # Patch on the neuroslm.tokenizer module surface so that any
    # ``from neuroslm.tokenizer import Tokenizer; Tokenizer()`` call
    # returns our fake.
    import neuroslm.tokenizer as tok_mod
    monkeypatch.setattr(tok_mod, "Tokenizer", lambda *a, **k: fake)
    return fake


@pytest.fixture
def cleanup_minimal_cpu(monkeypatch):
    """Ensure the colab_train_minimal_cpu module is reloaded fresh per
    test so monkeypatches inside it take effect."""
    if "colab_train_minimal_cpu" in sys.modules:
        del sys.modules["colab_train_minimal_cpu"]
    yield
    if "colab_train_minimal_cpu" in sys.modules:
        del sys.modules["colab_train_minimal_cpu"]


# ──────────────────────────────────────────────────────────────────────
# Contract 1 — real-data-by-default
# ──────────────────────────────────────────────────────────────────────

class TestMinimalCpuUsesRealDataPath:
    """``colab_train_minimal_cpu.main()`` must pull batches from the
    real-text pipeline (``neuroslm.data.batch_iterator``), not
    ``torch.randint``."""

    def test_main_does_not_call_torch_randint_for_inputs(
        self, monkeypatch, fake_text_stream, fake_tokenizer,
        cleanup_minimal_cpu,
    ):
        """When real data is reachable, ``main()`` must produce no
        ``torch.randint`` calls of vocab-shape during the training loop
        (the original bug)."""
        import torch
        original_randint = torch.randint
        randint_calls = []

        def _spy_randint(*args, **kwargs):
            # Record every randint call so we can detect vocab-shape ones.
            randint_calls.append((args, kwargs))
            return original_randint(*args, **kwargs)

        monkeypatch.setattr(torch, "randint", _spy_randint)

        import colab_train_minimal_cpu as ccpu
        # 2 steps is enough to surface the bug; using `text_source="auto"`
        # exercises the production code path with our fake stream.
        ccpu.main(steps=2, ood_every=999, dna_path="dna/not/here.dna")

        # The previous code did:
        #   input_ids = torch.randint(0, vocab_size, (B, S-1))
        #   target_ids = torch.randint(0, vocab_size, (B, S-1))
        # giving at least 4 vocab-shape randint calls for 2 steps.
        # With real data, those vanish entirely.
        vocab_shape_calls = [
            (a, k) for (a, k) in randint_calls
            if len(a) >= 2 and isinstance(a[1], int) and a[1] >= 64
        ]
        assert not vocab_shape_calls, (
            f"main() still uses torch.randint for input/target ids "
            f"({len(vocab_shape_calls)} vocab-shape calls). "
            f"Real-text batches via batch_iterator are expected. "
            f"Sample: {vocab_shape_calls[:2]}"
        )

    def test_main_invokes_batch_iterator(
        self, monkeypatch, fake_tokenizer, cleanup_minimal_cpu,
    ):
        """The training loop must call ``neuroslm.data.batch_iterator``
        at least once when ``text_source="auto"`` (the default)."""
        from neuroslm import data as data_mod

        original_batch_iter = data_mod.batch_iterator
        calls = {"n": 0}

        def _spy_batch_iter(*args, **kwargs):
            calls["n"] += 1
            return original_batch_iter(*args, **kwargs)

        monkeypatch.setattr(data_mod, "batch_iterator", _spy_batch_iter)

        # Stub the underlying stream so we don't hit HF.
        def _fake_stream(_tok, ctx_len, mode, _buf=8192):
            while True:
                yield [1] * (ctx_len + 1)
        monkeypatch.setattr(data_mod, "_stream_iterator", _fake_stream)

        import colab_train_minimal_cpu as ccpu
        ccpu.main(steps=1, ood_every=999, dna_path="dna/not/here.dna")

        assert calls["n"] >= 1, (
            "main() never called neuroslm.data.batch_iterator — "
            "real-data wiring is missing or bypassed."
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 2 — robust fallback chain (real → local → synthetic)
# ──────────────────────────────────────────────────────────────────────

class TestMinimalCpuFallbackChain:
    """When the real stream fails, the script must degrade gracefully
    AND visibly. The operator must never be left wondering why loss is
    stuck at ``log(V)``."""

    def test_local_corpus_used_when_hf_unavailable(
        self, monkeypatch, fake_tokenizer, capsys, cleanup_minimal_cpu,
    ):
        """When HuggingFace ``datasets`` raises, the script must fall
        back to the bundled local corpus and log that fact clearly."""
        from neuroslm import data as data_mod

        def _broken_open_stream(*_a, **_k):
            raise RuntimeError("simulated: no network / no datasets")

        monkeypatch.setattr(data_mod, "open_stream", _broken_open_stream)

        import colab_train_minimal_cpu as ccpu

        # Should not raise: must fall back to local corpus or synthetic.
        ccpu.main(steps=1, ood_every=999, dna_path="dna/not/here.dna",
                  text_source="auto")

        out = capsys.readouterr().out
        # We require *some* explicit log line — either local corpus
        # success or synthetic fallback warning.
        assert any(marker in out for marker in
                   ("[data] local corpus", "[data] synthetic fallback",
                    "real data unavailable")), (
            "When HF is unreachable, main() must explicitly log the "
            "fallback choice (local corpus or synthetic). Got:\n" + out
        )

    def test_explicit_synthetic_source_is_loud(
        self, monkeypatch, fake_tokenizer, capsys, cleanup_minimal_cpu,
    ):
        """If the operator passes ``text_source="synthetic"``, the script
        must shout WARNING because random tokens produce the
        ``log(V)`` plateau the user already hit."""
        import colab_train_minimal_cpu as ccpu

        ccpu.main(steps=1, ood_every=999, dna_path="dna/not/here.dna",
                  text_source="synthetic")
        out = capsys.readouterr().out

        assert "WARNING" in out.upper() or "synthetic" in out.lower(), (
            "text_source='synthetic' must produce a visible warning so the "
            "operator knows why loss will plateau at log(V). Got:\n" + out
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 3 — learning actually happens (the real test)
# ──────────────────────────────────────────────────────────────────────

class TestMinimalCpuActuallyLearns:
    """The whole point of switching to real data: ``lm_loss`` must drop
    below ``log(vocab_size)`` within a few steps on a *learnable*
    pattern. This is the regression test for the user's original
    plateau bug."""

    def test_loss_drops_below_log_vocab_on_learnable_pattern(
        self, monkeypatch, fake_text_stream, fake_tokenizer,
        cleanup_minimal_cpu,
    ):
        """With our fake repeating-pattern stream + a tiny model,
        ``lm_loss`` after a few steps must beat the uniform-random
        baseline of ``log(vocab_size)``.

        The fake stream emits a cycle of 8 tokens — the smallest LM can
        learn this in <50 steps. If loss stays ≥ log(V), training is
        broken.
        """
        import math
        import colab_train_minimal_cpu as ccpu

        result = ccpu.main(
            steps=30, ood_every=999, dna_path="dna/not/here.dna",
            text_source="auto",
        )
        # `main()` must return a result dict with at least
        # `{"final_lm_loss": float, "vocab_size": int}` so tests can
        # verify learning. (This is a NEW contract; current
        # implementation returns None — the test forces a return value.)
        assert isinstance(result, dict), (
            "main() must return a result dict (e.g. "
            "{'final_lm_loss': float, 'vocab_size': int}) so callers "
            "can verify learning. Got: " + repr(result)
        )
        final_loss = result["final_lm_loss"]
        vocab = result["vocab_size"]
        baseline = math.log(vocab)
        assert final_loss < baseline - 0.5, (
            f"Loss did not learn the deterministic pattern: "
            f"final_lm_loss={final_loss:.3f}, log(vocab={vocab})={baseline:.3f}. "
            f"Must beat the uniform-random baseline by ≥ 0.5 nats. "
            f"If this fails, the trainer is still on random data."
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 4 — full DSL trainer (big presets) regression guard
# ──────────────────────────────────────────────────────────────────────

class TestDslTrainerRealDataContract:
    """The non-tiny presets (rcc_bowtie_30m_p4, 100m, 300m, 1b, 7b) all
    flow through ``neuroslm.train_dsl``. Pin the real-data wiring so
    future refactors can't accidentally regress."""

    def test_train_dsl_exposes_real_data_source(self):
        """``neuroslm.train_dsl`` must expose a ``RealDataSource`` class
        that wraps ``batch_iterator``. This is the contract that all
        non-tiny presets depend on."""
        import neuroslm.train_dsl as td
        assert hasattr(td, "RealDataSource"), (
            "neuroslm.train_dsl.RealDataSource is missing — every "
            "non-tiny preset depends on it. Restore the class."
        )
        # And it must actually call batch_iterator under the hood.
        import inspect
        src = inspect.getsource(td.RealDataSource)
        assert "batch_iterator" in src, (
            "RealDataSource no longer references batch_iterator — "
            "the real-data path is broken for non-tiny presets."
        )

    def test_train_dsl_synthetic_is_last_resort_not_default(self):
        """``SyntheticBatchSource`` may exist as a fallback, but it must
        NOT be the default code path."""
        import neuroslm.train_dsl as td
        import inspect
        # Verify in the build/main path: RealDataSource appears before
        # SyntheticBatchSource and synthetic is only used in a fallback
        # branch.
        src = inspect.getsource(td)
        real_idx = src.find("RealDataSource(")
        synth_idx = src.find("SyntheticBatchSource(")
        if synth_idx >= 0 and real_idx >= 0:
            # Both exist — that's fine, but RealDataSource must come first
            # so it's the primary path.
            assert real_idx < synth_idx or "fallback" in src.lower(), (
                "SyntheticBatchSource appears before RealDataSource in "
                "train_dsl.py — synthetic must be the fallback, not default."
            )


# ──────────────────────────────────────────────────────────────────────
# Contract 5 — CLI tiny entry point passes through new text_source param
# ──────────────────────────────────────────────────────────────────────

class TestTinyCliPassesTextSource:
    """``brian train --preset=tiny`` should default to real data and let
    the operator override via ``--text-source`` (future flag) or just
    use ``auto``. This test asserts that the CLI either passes a
    ``text_source`` kwarg OR omits it (allowing ``main()``'s default
    to be ``"auto"``) — but it must NOT pass ``"synthetic"`` by default.
    """

    def test_cli_does_not_force_synthetic(self, monkeypatch):
        """``brian train --preset=tiny`` must not silently force
        synthetic batches."""
        from neuroslm import cli
        import argparse
        import importlib.util

        captured = {}

        class _Loader:
            def exec_module(self, module):
                def _main(**kwargs):
                    captured.update(kwargs)
                module.main = _main

        class _Spec:
            loader = _Loader()

        monkeypatch.setattr(importlib.util, "spec_from_file_location",
                            lambda *a, **k: _Spec())
        monkeypatch.setattr(importlib.util, "module_from_spec",
                            lambda spec: type("_M", (), {})())

        args = argparse.Namespace(
            preset="tiny", arch=None, dna=None,
            steps=1, ood_every=999,
            batch=None, seq_len=None, d_sem=None,
        )
        rc = cli.cmd_train(args)
        assert rc == 0

        # If text_source is passed, it must not be 'synthetic'.
        if "text_source" in captured:
            assert captured["text_source"] != "synthetic", (
                "CLI is forcing synthetic batches by default — "
                "this triggers the log(V) plateau bug."
            )
