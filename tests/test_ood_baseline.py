# -*- coding: utf-8 -*-
"""GPT-2 baseline through the eval-v2 protocol (`brian ood baseline`).

The two-layer doctrine (architecture.md §14) sets the trunk's success
criterion at "≥ GPT-2-124M under the SAME eval-v2 protocol at matched
params". Published GPT-2 numbers are useless for that comparison — they
use different tokenization/eval conventions. This module pins the
instrument that makes the target honest: run a HuggingFace causal LM
through the IDENTICAL `_eval_all_corpora` / `_eval_ppl_on_texts`
machinery (same tokenizer, same corpora, same sliding truncation, same
NLL aggregation) our trunks are measured with.

Contracts:
  A. `run_baseline_eval` exists, defaults to model_id="gpt2", and
     supports a `model_factory` injection point so these tests never
     touch the HF Hub.
  B. Result dict is protocol-v2 shaped: eval_surface="baseline_hf",
     per-corpus results from the SAME registry, gap_ratio_v2 =
     wikitext/traindist — directly comparable to a trunk's mid-ood JSON.
  C. Correctness: a uniform-logits model must score ppl ≈ vocab_size on
     every corpus (CE = ln V for uniform predictions).
  D. Fail-open per corpus, same as the training-time eval.
  E. CLI: `brian ood baseline [--model gpt2]` is wired.
"""
from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

VOCAB = 50257


class _UniformLM(nn.Module):
    """ids → all-zero logits (uniform distribution over the gpt2 vocab)."""

    max_ctx = 64

    def __init__(self):
        super().__init__()
        # _eval_ppl_on_texts discovers the device via next(parameters()).
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, ids):
        B, T = ids.shape
        return torch.zeros(B, T, VOCAB)


def _tiny_corpora():
    texts = [
        "The quick brown fox jumps over the lazy dog and keeps on "
        "running through the quiet green field until sunset falls.",
        "In the beginning the project had one goal: measure everything "
        "against the same yardstick, or measure nothing at all.",
        "Language models predict the next token; baselines exist so "
        "that the word better always has a number attached to it.",
    ]
    return {
        "wikitext": lambda: iter(texts),
        "pg19": lambda: iter(texts),
        "traindist": lambda: iter(texts),
    }


@pytest.fixture
def patched_registry(monkeypatch):
    import neuroslm.train_dsl as td
    monkeypatch.setattr(td, "_EVAL_CORPORA", _tiny_corpora())
    return td


class TestRunBaselineEvalSurface:
    def test_importable_and_defaults_to_gpt2(self):
        from neuroslm.train_dsl import run_baseline_eval
        sig = inspect.signature(run_baseline_eval)
        assert sig.parameters["model_id"].default == "gpt2"
        assert "model_factory" in sig.parameters

    def test_result_is_protocol_v2_shaped(self, patched_registry):
        td = patched_registry
        out = td.run_baseline_eval(model_factory=_UniformLM)
        assert out["protocol"] == "v2"
        assert out["eval_surface"] == "baseline_hf"
        assert out["model_id"] == "gpt2"
        assert set(out["corpora"]) == {"wikitext", "pg19", "traindist"}
        for res in out["corpora"].values():
            assert res["n_sequences"] > 0
            assert len(res["per_seq_nll"]) == res["n_sequences"]
        assert out["gap_ratio_v2"] == pytest.approx(
            out["corpora"]["wikitext"]["ppl"]
            / out["corpora"]["traindist"]["ppl"])

    def test_uniform_model_scores_vocab_ppl(self, patched_registry):
        td = patched_registry
        out = td.run_baseline_eval(model_factory=_UniformLM)
        for name, res in out["corpora"].items():
            assert res["ppl"] == pytest.approx(VOCAB, rel=0.01), (
                f"uniform logits must score ppl ≈ vocab on {name!r} — "
                "anything else means the baseline path diverges from "
                "the shared evaluator's NLL math")

    def test_fail_open_per_corpus(self, monkeypatch):
        import neuroslm.train_dsl as td
        reg = _tiny_corpora()

        def _boom():
            raise RuntimeError("corpus offline")

        reg["pg19"] = _boom
        monkeypatch.setattr(td, "_EVAL_CORPORA", reg)
        out = td.run_baseline_eval(model_factory=_UniformLM)
        assert "pg19" not in out["corpora"]
        assert {"wikitext", "traindist"} <= set(out["corpora"])
        assert out["pg19_ppl"] is None


class TestCliWiring:
    def test_ood_baseline_parses(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(["ood", "baseline"])
        assert args.model == "gpt2"
        assert callable(args.func)

    def test_model_flag_overrides(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(
            ["ood", "baseline", "--model", "distilgpt2", "--cap", "10"])
        assert args.model == "distilgpt2"
        assert args.cap == 10
