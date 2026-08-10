# -*- coding: utf-8 -*-
"""TDD contracts for eval protocol v2 (H59 falsification apparatus).

Protocol-v1 problems (findings.md H12 sidebar + 2026-07-12 audit):
  * gap_ratio's denominator is the RUNNING TRAIN LOSS — contaminated by
    flooding / label smoothing / EMA phase, so it is not comparable
    across arms with different regularisation.
  * Single OOD corpus (WikiText-103), which is domain-adjacent to the
    FineWeb-Edu training mix.
  * No per-sequence measurements → no statistics; the Welch's-t
    ImprovementGate exists (neuroslm/verification/improvement_gate.py)
    but nothing feeds it.

Protocol v2 contracts:
  A. `_eval_ppl_on_texts(harness, texts, tok, ctx, cap)` — ONE shared
     evaluator for every corpus, returning per-sequence NLLs alongside
     the aggregate ppl.
  B. `_EVAL_CORPORA` registry: "wikitext" (OOD axis 1), "pg19" (OOD
     axis 2, genuinely distant), "traindist" (held-out slice of the
     TRAINING distribution — the clean gap denominator). Loaders are
     lazy callables so tests never touch the network.
  C. `_mid_ood_eval` evaluates every registered corpus, prints a line
     whose prefix stays `[mid-ood] step N: wikitext ppl=<float>` (the
     `brian ps` parser at cli.py:2126 pins this format), computes
     gap_ratio_v2 = wikitext_ppl / traindist_ppl, and persists
     per-corpus per-sequence NLLs in the JSON.
  D. Fail-open: one corpus loader failing must not kill the eval —
     wikitext ppl is still returned and the JSON records the corpora
     that succeeded.
  E. `brian ood compare BEFORE.json AFTER.json` feeds the stored
     per-sequence NLLs through ImprovementGate (Welch one-sided,
     direction="decrease") so "arm A beats arm B" requires statistics,
     not eyeballs.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent

# `brian ps` parses this exact prefix (neuroslm/cli.py:2126) — protocol
# v2 must keep it stable.
_PS_MID_OOD_RE = re.compile(
    r"\[mid-ood\]\s+step\s+(?P<step>\d+):\s+wikitext\s+ppl=(?P<ppl>[\d.]+)")


# ── fixtures ──────────────────────────────────────────────────────────

class _TinyLM(nn.Module):
    """256-vocab byte LM: embedding → linear. Deterministic, no network."""
    max_ctx = 64

    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.emb = nn.Embedding(256, 16)
        self.head = nn.Linear(16, 256)

    def forward(self, ids):
        return self.head(self.emb(ids))


class _FakeHarness(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _TinyLM()

    def forward(self, ids):
        return self.language_model(ids)


class _ByteTok:
    """Stub tokenizer: text → byte ids (<256). No tiktoken, no network."""

    def encode(self, text: str):
        return list(text.encode("utf-8", errors="replace")[:4096])


def _texts(n=6, length=200):
    return [f"sequence {i}: " + ("lorem ipsum dolor sit amet " * 10)[:length]
            for i in range(n)]


# ══════════════════════════════════════════════════════════════════════
# A. Shared evaluator with per-sequence NLLs
# ══════════════════════════════════════════════════════════════════════

class TestEvalPplOnTexts:
    def _run(self, texts, cap=50):
        from neuroslm.train_dsl import _eval_ppl_on_texts
        return _eval_ppl_on_texts(_FakeHarness(), texts, _ByteTok(),
                                  ctx=32, cap=cap)

    def test_returns_aggregate_and_per_seq(self):
        res = self._run(_texts(4))
        assert set(res) >= {"ppl", "n_sequences", "n_tokens", "per_seq_nll"}
        assert res["n_sequences"] == 4
        assert len(res["per_seq_nll"]) == 4
        assert res["n_tokens"] > 0
        assert math.isfinite(res["ppl"]) and res["ppl"] > 0

    def test_cap_respected(self):
        res = self._run(_texts(10), cap=3)
        assert res["n_sequences"] == 3
        assert len(res["per_seq_nll"]) == 3

    def test_short_texts_skipped(self):
        texts = ["tiny", ""] + _texts(2)
        res = self._run(texts)
        assert res["n_sequences"] == 2

    def test_ppl_consistent_with_token_weighted_nll(self):
        """Aggregate ppl must equal exp(total_nll / total_tokens) — the
        per-seq list is per-sequence MEAN nll, the aggregate stays
        token-weighted (identical to protocol v1's arithmetic)."""
        from neuroslm.train_dsl import _eval_ppl_on_texts
        h = _FakeHarness()
        texts = _texts(3)
        res = _eval_ppl_on_texts(h, texts, _ByteTok(), ctx=32, cap=50)
        assert res["ppl"] == pytest.approx(
            math.exp(min(res["total_nll"] / res["n_tokens"], 20.0)), rel=1e-6)

    def test_deterministic(self):
        r1 = self._run(_texts(3))
        r2 = self._run(_texts(3))
        assert r1["per_seq_nll"] == pytest.approx(r2["per_seq_nll"])


# ══════════════════════════════════════════════════════════════════════
# B. Corpora registry
# ══════════════════════════════════════════════════════════════════════

class TestEvalCorporaRegistry:
    def test_registry_has_three_axes(self):
        from neuroslm.train_dsl import _EVAL_CORPORA
        assert set(_EVAL_CORPORA) >= {"wikitext", "pg19", "traindist"}

    def test_loaders_are_lazy_callables(self):
        from neuroslm.train_dsl import _EVAL_CORPORA
        for name, loader in _EVAL_CORPORA.items():
            assert callable(loader), f"{name} loader must be a callable"


class TestTraindistLoaderNeverSkips:
    """2026-08-10 incident: `_load_traindist_texts` used to call
    `.skip(8_000_000)` on a STREAMING HF dataset to reach an "untrained"
    offset into the training split. `.skip()` on an IterableDataset is
    O(n) network+parse work, not an O(1) seek — this hung a live,
    billing A100 indefinitely at the very first mid-training eval, with
    no error, no progress, nothing after "corpus 'pg19' failed
    (skipping)" in the log. Fixed by reading a DIFFERENT, independently-
    sampled FineWeb-Edu subset from its start (no skip needed). These
    contracts pin the fix at the `datasets.load_dataset` call boundary
    so a future edit can't silently reintroduce a `.skip()` on a
    streaming dataset in this loader.
    """

    def test_no_skip_call_on_the_streaming_dataset(self, monkeypatch):
        import neuroslm.train_dsl as td

        calls = {"skip": 0}

        class _FakeStreamingDataset:
            def __iter__(self):
                for i in range(5):
                    yield {"text": f"doc {i} " * 20}

            def skip(self, n):
                calls["skip"] += 1
                raise AssertionError(
                    ".skip() must never be called on the traindist "
                    "streaming dataset — it hung a live A100 for the "
                    "price of the rental (2026-08-10)")

        def _fake_load_dataset(*args, **kwargs):
            return _FakeStreamingDataset()

        monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)
        td._CORPUS_CACHE.pop("traindist", None)

        texts = list(td._load_traindist_texts())
        assert calls["skip"] == 0
        assert len(texts) == 5

    def test_uses_a_different_subset_than_training(self, monkeypatch):
        """The training loader (neuroslm/data.py) reads sample-10BT —
        traindist must read a DIFFERENT sample config, not the same
        split with an in-stream offset."""
        import neuroslm.train_dsl as td

        captured = {}

        class _FakeStreamingDataset:
            def __iter__(self):
                return iter(())

        def _fake_load_dataset(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return _FakeStreamingDataset()

        monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)
        td._CORPUS_CACHE.pop("traindist", None)
        list(td._load_traindist_texts())

        assert captured["kwargs"].get("name") != "sample-10BT", (
            "traindist must not read the exact training split/config — "
            "use a different independently-sampled subset instead of "
            "trying to offset into the same stream")


# ══════════════════════════════════════════════════════════════════════
# C/D. _mid_ood_eval protocol v2
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def fake_corpora(monkeypatch, tmp_path):
    """Patch the registry with offline corpora and sandbox the CWD so
    the JSON lands under tmp_path/logs/..."""
    import neuroslm.train_dsl as td
    wik = _texts(5, length=300)
    pg = [t.replace("lorem", "victorian") for t in _texts(4, length=300)]
    tr = [t.replace("lorem", "fineweb") for t in _texts(4, length=300)]
    monkeypatch.setattr(td, "_EVAL_CORPORA", {
        "wikitext": lambda: iter(wik),
        "pg19": lambda: iter(pg),
        "traindist": lambda: iter(tr),
    })
    monkeypatch.setattr(td, "_EVAL_TOKENIZER_FACTORY", lambda: _ByteTok())
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run_mid_eval(step=500, history=None):
    from neuroslm.train_dsl import _mid_ood_eval
    return _mid_ood_eval(_FakeHarness(), step, None, None,
                         train_ppl_history=history)


class TestMidOodEvalV2:
    def test_returns_wikitext_ppl(self, fake_corpora):
        ppl = _run_mid_eval()
        assert ppl is not None and math.isfinite(ppl) and ppl > 0

    def test_ps_parser_line_format_preserved(self, fake_corpora, capsys):
        _run_mid_eval(step=1500)
        out = capsys.readouterr().out
        m = _PS_MID_OOD_RE.search(out)
        assert m, f"`brian ps` mid-ood regex no longer matches:\n{out}"
        assert m.group("step") == "1500"

    def test_line_reports_traindist_and_gap_v2(self, fake_corpora, capsys):
        _run_mid_eval()
        out = capsys.readouterr().out
        assert "traindist ppl=" in out
        assert "gap_v2=" in out

    def test_json_has_per_corpus_nlls(self, fake_corpora):
        _run_mid_eval(step=500)
        out_files = list(
            (fake_corpora / "logs/vast/benchmarks/ood").glob("*.json"))
        assert len(out_files) == 1
        data = json.loads(out_files[0].read_text(encoding="utf-8"))
        assert data["protocol"] == "v2"
        assert set(data["corpora"]) == {"wikitext", "pg19", "traindist"}
        for name, c in data["corpora"].items():
            assert c["n_sequences"] == len(c["per_seq_nll"]) > 0, name
            assert c["ppl"] > 0
        assert data["gap_ratio_v2"] == pytest.approx(
            data["corpora"]["wikitext"]["ppl"]
            / data["corpora"]["traindist"]["ppl"], rel=1e-6)
        # v1 keys kept for analyze-log / cli_metrics backcompat
        assert data["ood_ppl"] == pytest.approx(
            data["corpora"]["wikitext"]["ppl"], rel=1e-6)
        assert data["kind"] == "mid-training"

    def test_fail_open_on_broken_corpus(self, fake_corpora, monkeypatch):
        import neuroslm.train_dsl as td

        def _boom():
            raise RuntimeError("no network")

        reg = dict(td._EVAL_CORPORA)
        reg["pg19"] = _boom
        monkeypatch.setattr(td, "_EVAL_CORPORA", reg)
        ppl = _run_mid_eval()
        assert ppl is not None, "one broken corpus must not kill the eval"
        out_files = list(
            (fake_corpora / "logs/vast/benchmarks/ood").glob("*.json"))
        data = json.loads(out_files[0].read_text(encoding="utf-8"))
        assert "pg19" not in data["corpora"]
        assert "wikitext" in data["corpora"]


# ══════════════════════════════════════════════════════════════════════
# E. brian ood compare — the statistics gate
# ══════════════════════════════════════════════════════════════════════

def _result_json(path: Path, nlls, corpus="wikitext"):
    path.write_text(json.dumps({
        "corpora": {corpus: {"ppl": math.exp(sum(nlls) / len(nlls)),
                             "n_sequences": len(nlls),
                             "n_tokens": 1000,
                             "per_seq_nll": nlls}},
        "protocol": "v2",
    }), encoding="utf-8")
    return path


class TestOodCompare:
    def _compare(self, tmp_path, before_nll, after_nll, **kw):
        import argparse
        from neuroslm.cli import cmd_ood_compare
        a = _result_json(tmp_path / "before.json", before_nll)
        b = _result_json(tmp_path / "after.json", after_nll)
        ns = argparse.Namespace(
            before=str(a), after=str(b),
            corpus=kw.get("corpus", "wikitext"),
            alpha=kw.get("alpha", 0.05),
            min_effect=kw.get("min_effect", 0.01))
        return cmd_ood_compare(ns)

    def test_clear_improvement_admitted(self, tmp_path, capsys):
        rc = self._compare(tmp_path,
                           before_nll=[5.0, 5.1, 4.9, 5.05, 4.95, 5.02],
                           after_nll=[3.0, 3.1, 2.9, 3.05, 2.95, 3.02])
        assert rc == 0
        out = capsys.readouterr().out
        assert "admitted=True" in out
        assert "p=" in out

    def test_noise_rejected(self, tmp_path, capsys):
        rc = self._compare(tmp_path,
                           before_nll=[5.0, 5.1, 4.9, 5.05, 4.95, 5.02],
                           after_nll=[5.01, 5.09, 4.91, 5.04, 4.96, 5.01])
        assert rc == 0
        out = capsys.readouterr().out
        assert "admitted=False" in out

    def test_missing_corpus_errors(self, tmp_path, capsys):
        import argparse
        from neuroslm.cli import cmd_ood_compare
        a = _result_json(tmp_path / "b.json", [5.0, 5.1], corpus="wikitext")
        b = _result_json(tmp_path / "a.json", [4.0, 4.1], corpus="pg19")
        ns = argparse.Namespace(before=str(a), after=str(b),
                                corpus="wikitext", alpha=0.05,
                                min_effect=0.01)
        rc = cmd_ood_compare(ns)
        assert rc != 0

    def test_cli_parses_ood_compare(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(
            ["ood", "compare", "before.json", "after.json",
             "--corpus", "pg19"])
        assert args.before == "before.json"
        assert args.after == "after.json"
        assert args.corpus == "pg19"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
