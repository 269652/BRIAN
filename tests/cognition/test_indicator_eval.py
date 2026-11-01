# -*- coding: utf-8 -*-
"""Consciousness-indicator battery (architecture.md §14.11).

The cognition layer is judged by measurable function the bare trunk
lacks (§14.1). This battery turns indicator properties from the
theory-derived checklist (Butlin, Long et al. 2023, "Consciousness in
Artificial Intelligence") into behavioural probes, each run under three
conditions: the full runtime, a single-mechanism ablation, and a bare
LM with the SAME context budget (sliding transcript window of
``max_prompt_chars``).

The fakes here model "an LM that reads its context truthfully": a
probe's outcome depends only on what the cognitive loop puts in front
of the generator — which is exactly the thing under test.
"""
from __future__ import annotations

import json
import math
import re

import pytest

from neuroslm.cognition.runtime import MindConfig, MindSeams, ThoughtScore

_STOP = {"what", "is", "my", "the", "a", "an", "i", "in", "of", "on",
         "which", "did", "do", "to", "was", "where", "when", "it", "at"}


def _bow(text: str):
    """Bag-of-words 64-dim embedding, stopwords dropped — cosine is
    high exactly when two texts share content words."""
    v = [0.0] * 64
    for w in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if w not in _STOP:
            v[sum(map(ord, w)) % 64] += 1.0
    if not any(v):
        v[0] = 1e-3
    return v


class _Reader:
    """Fake LM that reads its context: when the CURRENT input (last
    'Just now:' / 'User:' line) is a question, it answers with the
    bank answer that appears EARLIEST in the prompt; otherwise it just
    acknowledges. Never volunteers facts on non-questions, so a fact
    cannot leak forward through its own replies."""

    def __init__(self, bank):
        self.bank = [w.lower() for w in bank]

    def __call__(self, prompt: str, max_new_tokens: int) -> str:
        lines = [l for l in prompt.splitlines() if l.strip()]
        cur = next((l for l in reversed(lines)
                    if l.startswith(("Just now:", "User:"))), "")
        if not cur.rstrip().endswith("?"):
            return "Noted."
        low = prompt.lower()
        hits = []
        for w in self.bank:
            m = re.search(rf"\b{re.escape(w)}\b", low)
            if m:
                hits.append((m.start(), w))
        return f"I think {min(hits)[1]}." if hits else "I am not sure."


def _flat_score(text):
    return ThoughtScore(mean_nll=3.0, entropy_norm=0.5)


def _seams(gen, embed=_bow, score=_flat_score):
    return MindSeams(generate_fn=gen, score_fn=score, embed_fn=embed)


# ── Statistics helpers ───────────────────────────────────────────────

class TestStats:
    def test_wilson_interval_matches_reference(self):
        from neuroslm.cognition.indicators import wilson_interval
        lo, hi = wilson_interval(5, 10)
        assert lo == pytest.approx(0.2366, abs=1e-3)
        assert hi == pytest.approx(0.7634, abs=1e-3)

    def test_wilson_interval_edges(self):
        from neuroslm.cognition.indicators import wilson_interval
        assert wilson_interval(0, 0) == (0.0, 1.0)
        lo, hi = wilson_interval(10, 10)
        assert hi == pytest.approx(1.0) and 0.65 < lo < 0.75

    def test_auroc_perfect_inverted_tied_and_undefined(self):
        from neuroslm.cognition.indicators import auroc
        assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
        assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
        assert auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5
        assert auroc([0.3, 0.7], [1, 1]) is None, (
            "type-2 AUROC is undefined with only one outcome class")

    def test_mentions_is_word_bounded(self):
        from neuroslm.cognition.indicators import mentions
        assert mentions("It is in the red box.", ("red",))
        assert not mentions("It was 18 minutes.", ("8",))
        assert not mentions("bored", ("red",))
        assert mentions("The answer: 8.", ("eight", "8"))

    def test_sequence_novelty_and_streaks(self):
        from neuroslm.cognition.indicators import (
            longest_streak_below, sequence_novelty)
        a, b = [1.0, 0.0], [0.0, 1.0]
        nov = sequence_novelty([a, a, b, b, a])
        assert nov[0] == 1.0
        assert nov[1] == pytest.approx(0.0)
        assert nov[2] == pytest.approx(1.0)
        assert nov[3] == pytest.approx(0.0)
        assert nov[4] == pytest.approx(0.0)
        assert longest_streak_below([0.9, 0.1, 0.1, 0.5, 0.1], 0.25) == 2
        assert longest_streak_below([], 0.25) == 0


# ── Bare baseline: a bounded-context LM with the SAME budget ─────────

class TestSlidingWindowLM:
    def test_keeps_only_the_most_recent_chars(self):
        from neuroslm.cognition.indicators import SlidingWindowLM
        seen = []

        def gen(prompt, n):
            seen.append(prompt)
            return "ok"
        lm = SlidingWindowLM(gen, max_chars=60, n_tok=8)
        lm.hear("first message that is fairly long")
        lm.hear("second message that is also long")
        lm.hear("third")
        assert len(seen[-1]) <= 60
        assert "first message" not in seen[-1]
        assert seen[-1].rstrip().endswith("My reply:")


# ── Probe: episodic recall beyond the context window (GWT-3) ─────────

def _recall_items():
    from neuroslm.cognition.indicators import RecallItem
    return [
        RecallItem("My dog is called Pemberton.", "What is my dog called?",
                   ("pemberton",)),
        RecallItem("My bike lock code is 3852.",
                   "What is my bike lock code?", ("3852",)),
    ]


class TestEpisodicRecallProbe:
    def _run(self, cfg=None, n_distractors=4):
        from neuroslm.cognition.indicators import probe_episodic_recall
        items = _recall_items()
        gen = _Reader([a for it in items for a in it.answers])
        return probe_episodic_recall(
            _seams(gen), items=items, n_distractors=n_distractors,
            cfg=cfg or MindConfig(n_candidates=2), seed=0)

    def test_full_runtime_recalls_the_fact(self):
        res = self._run()
        assert res["indicator"] == "GWT-3"
        assert res["conditions"]["full"]["accuracy"] == 1.0

    def test_ablating_recall_destroys_it(self):
        res = self._run()
        assert res["conditions"]["no_recall"]["accuracy"] == 0.0
        assert res["contrasts"]["full_minus_no_recall"] == 1.0

    def test_bare_lm_succeeds_only_while_the_fact_fits_its_window(self):
        roomy = self._run(n_distractors=2)
        assert roomy["conditions"]["bare"]["accuracy"] == 1.0
        tight = self._run(cfg=MindConfig(n_candidates=2,
                                         max_prompt_chars=120),
                          n_distractors=6)
        assert tight["conditions"]["bare"]["accuracy"] == 0.0

    def test_reports_wilson_ci_and_raw_replies(self):
        cond = self._run()["conditions"]["full"]
        assert cond["n"] == 2 and cond["correct"] == 2
        lo, hi = cond["ci95"]
        assert 0.0 <= lo <= cond["accuracy"] <= hi <= 1.0
        assert all("pemberton" in r.lower() or "3852" in r
                   for r in cond["replies"])


# ── Probe: source / reality monitoring (HOT-2) ───────────────────────

def _source_embed(text):
    """Controlled geometry: the self-generated lures sit CLOSER to the
    question (cos .95) than the observed fact (cos .85), so only the
    provenance penalty (0.15) can rank reality first."""
    t = (text or "").lower()
    if "which box" in t:
        return [1.0, 0.0, 0.0]
    if "blue box" in t:
        return [0.95, math.sqrt(1 - 0.95 ** 2), 0.0]
    if "red box" in t:
        return [0.85, math.sqrt(1 - 0.85 ** 2), 0.0]
    return [0.0, 0.0, 1.0]


class TestSourceMonitoringProbe:
    def _run(self):
        from neuroslm.cognition.indicators import (
            SourceItem, probe_source_monitoring)
        item = SourceItem(
            observed="The key is in the red box.",
            lures=("I imagine the key is in the blue box.",
                   "Maybe the key ended up in the blue box.",
                   "I picture the key lying in the blue box."),
            question="Which box is the key in?",
            answers=("red",), lure_answers=("blue",))
        return probe_source_monitoring(
            _seams(_Reader(["red", "blue"]), embed=_source_embed),
            items=[item], cfg=MindConfig(n_candidates=2), seed=0)

    def test_provenance_penalty_lets_reality_win(self):
        res = self._run()
        assert res["indicator"] == "HOT-2"
        assert res["conditions"]["full"]["accuracy"] == 1.0

    def test_without_provenance_the_mind_believes_its_own_musings(self):
        res = self._run()
        assert res["conditions"]["no_provenance"]["accuracy"] == 0.0
        assert res["contrasts"]["full_minus_no_provenance"] == 1.0


# ── Probe: loop escape / state-dependent attention (GWT-4) ───────────

class TestLoopEscapeProbe:
    def _run(self):
        from neuroslm.cognition.indicators import probe_loop_escape
        return probe_loop_escape(
            _seams(lambda p, n: "The same thought again."),
            seed_texts=["I watched the river flow past the old mill."],
            n_idle_ticks=10, cfg=MindConfig(n_candidates=2), seed=0)

    def test_full_runtime_detects_the_loop_and_reorients(self):
        res = self._run()
        assert res["indicator"] == "GWT-4"
        assert res["conditions"]["full"]["reorients"] > 0

    def test_ablation_removes_every_control_act(self):
        nc = self._run()["conditions"]["no_control"]
        assert nc["reorients"] == 0 and nc["replays"] == 0

    def test_novelty_is_measured_externally_on_the_thought_stream(self):
        res = self._run()
        for name in ("full", "no_control", "bare"):
            c = res["conditions"][name]
            assert c["mean_novelty"] == pytest.approx(0.0), name
            assert c["n_thoughts"] >= 8, name
            assert c["longest_low_streak"] >= c["n_thoughts"] - 2, name
            assert c["distinct_ratio"] == pytest.approx(
                1 / c["n_thoughts"]), name


# ── Probe: metacognitive sensitivity (HOT-2, type-2 AUROC) ───────────

class TestMetacognitionProbe:
    def _run(self):
        from neuroslm.cognition.indicators import QAItem, probe_metacognition
        items = [QAItem("What is the capital of France?", ("paris",)),
                 QAItem("What is the capital of Mongolia?",
                        ("ulaanbaatar",))]

        def gen(prompt, n):
            return "Paris." if "France" in prompt else "Probably Kyoto."

        def score(text):
            ok = "paris" in text.lower()
            return ThoughtScore(mean_nll=1.0 if ok else 4.0,
                                entropy_norm=0.1 if ok else 0.9)
        return probe_metacognition(_seams(gen, score=score), items=items,
                                   cfg=MindConfig(n_candidates=2), seed=0)

    def test_confidence_that_tracks_correctness_scores_auroc_one(self):
        res = self._run()
        assert res["indicator"] == "HOT-2"
        for name in ("full", "bare"):
            c = res["conditions"][name]
            assert c["accuracy"] == 0.5
            assert c["auroc_entropy"] == 1.0
            assert c["auroc_nll"] == 1.0


# ── IIT-family probe: perturbational complexity (PCI analogue) ──────

class TestLempelZiv:
    def test_kaspar_schuster_reference_sequence(self):
        from neuroslm.cognition.indicators import lz76_complexity
        # Kaspar & Schuster (1987): 0·001·10·100·1000·101 → c = 6
        bits = [int(c) for c in "0001101001000101"]
        assert lz76_complexity(bits) == 6

    def test_constant_sequence_is_minimally_complex(self):
        from neuroslm.cognition.indicators import lz76_complexity
        assert lz76_complexity([0] * 32) == 2
        assert lz76_complexity([]) == 0

    def test_pci_normalisation(self):
        import random as _r
        from neuroslm.cognition.indicators import pci_normalized
        assert pci_normalized([[0] * 8 for _ in range(4)]) == 0.0
        rng = _r.Random(0)
        noisy = [[rng.randint(0, 1) for _ in range(64)] for _ in range(16)]
        periodic = [[(t // 2) % 2 for t in range(64)] for _ in range(16)]
        assert pci_normalized(noisy) > 0.8, "random binary ≈ 1 by design"
        assert pci_normalized(periodic) < 0.3


def _fairground_gen(prompt, n):
    """Thought content depends on the prompt — a pulse can only change
    behaviour by changing what reaches THINK."""
    return ("I keep thinking about the fairground."
            if "fairground" in prompt.lower()
            else "I keep thinking about the river.")


class TestPerturbationalComplexityProbe:
    def _run(self, pulses=("null", "da", "memory")):
        from neuroslm.cognition.indicators import (
            probe_perturbational_complexity)
        return probe_perturbational_complexity(
            _seams(_fairground_gen),
            seed_texts=["I watched the river flow past the old mill."],
            n_warmup=2, n_response_ticks=6, pulses=pulses,
            cfg=MindConfig(n_candidates=2), seed=0)

    def test_null_pulse_twins_never_diverge(self):
        res = self._run()
        assert res["indicator"] == "IIT/PCI"
        for cond in res["conditions"].values():
            assert cond["null"]["pci"] == 0.0
            assert cond["null"]["fraction_diverged"] == 0.0

    def test_da_pulse_registers_on_the_nt_channel(self):
        full = self._run()["conditions"]["full"]
        assert full["da"]["pci"] > 0.0
        assert "DA" in full["da"]["runs"][0]["channels"]

    def test_memory_pulse_propagates_into_thought_only_through_recall(self):
        res = self._run()
        full_ch = res["conditions"]["full"]["memory"]["runs"][0]["channels"]
        cut_ch = res["conditions"]["no_recall"]["memory"]["runs"][0]["channels"]
        assert "recall" in full_ch and "thought" in full_ch
        assert "recall" not in cut_ch and "thought" not in cut_ch
        assert (res["conditions"]["full"]["memory"]["channels_reached"]
                > res["conditions"]["no_recall"]["memory"]["channels_reached"])

    def test_rejects_unknown_pulse(self):
        with pytest.raises(ValueError):
            self._run(pulses=("tms_to_the_moon",))


# ── Battery driver ───────────────────────────────────────────────────

class TestBattery:
    def test_runs_a_probe_subset_and_is_json_serialisable(self):
        from neuroslm.cognition.indicators import (
            INDICATORS, run_indicator_battery)
        items = _recall_items()
        report = run_indicator_battery(
            _seams(_Reader([a for it in items for a in it.answers])),
            probes=["episodic_recall"], cfg=MindConfig(n_candidates=2),
            n_items=1, n_distractors=2, seed=3)
        json.dumps(report)
        assert set(report["probes"]) == {"episodic_recall"}
        assert report["meta"]["seed"] == 3
        assert report["meta"]["indicators"] == INDICATORS
        assert report["probes"]["episodic_recall"]["conditions"]["full"]["n"] == 1

    def test_format_report_summarises_every_condition(self):
        from neuroslm.cognition.indicators import (
            format_report, run_indicator_battery)
        items = _recall_items()
        report = run_indicator_battery(
            _seams(_Reader([a for it in items for a in it.answers])),
            probes=["episodic_recall"], cfg=MindConfig(n_candidates=2),
            n_items=1, n_distractors=2)
        text = format_report(report)
        assert "episodic_recall" in text and "GWT-3" in text
        for cond in ("full", "no_recall", "bare"):
            assert cond in text

    def test_rejects_unknown_probe(self):
        from neuroslm.cognition.indicators import run_indicator_battery
        with pytest.raises(ValueError):
            run_indicator_battery(_seams(_Reader([])), probes=["telepathy"])

    def test_reseed_hook_called_per_item(self):
        from neuroslm.cognition.indicators import run_indicator_battery
        calls = []
        seams = _seams(_Reader(["pemberton"]))
        seams.reseed = calls.append
        run_indicator_battery(seams, probes=["metacognition"], n_items=3,
                              cfg=MindConfig(n_candidates=2), seed=1)
        assert len(calls) >= 3 and len(set(calls)) >= 3


# ── CLI + deploy surface ─────────────────────────────────────────────

class TestCli:
    def test_eval_mind_parses_with_defaults(self):
        from neuroslm.cli import _build_parser
        a = _build_parser().parse_args(["eval", "mind"])
        assert a.eval_kind == "mind"
        # Instruct, not base: a base model has no turn-end token and
        # degenerates in the respond path (experts.py registry note).
        assert a.expert == "smollm2_360m_instruct"
        assert a.probes is None and a.out is None

    def test_eval_mind_flags(self):
        from neuroslm.cli import _build_parser
        a = _build_parser().parse_args(
            ["eval", "mind", "--probes", "episodic_recall,metacognition",
             "--n-items", "4", "--seed", "7", "--device", "cuda"])
        assert a.probes == "episodic_recall,metacognition"
        assert a.n_items == 4 and a.seed == 7 and a.device == "cuda"

    def test_deploy_mind_eval_parses(self):
        from neuroslm.cli import _build_parser
        a = _build_parser().parse_args(["deploy-mind-eval"])
        assert a.expert == "smollm2_360m_instruct"
        assert a.no_auto_destroy is False

    def test_deploy_mind_eval_requires_human_confirmation_first(
            self, monkeypatch):
        import neuroslm.cli as cli
        from neuroslm.connectors import vast_mind_eval
        launched = []
        monkeypatch.setattr(vast_mind_eval.VastMindEvalConnector, "launch",
                            lambda self, cfg: launched.append(cfg) or 0)

        def _deny(*a, **k):
            raise SystemExit(1)
        monkeypatch.setattr(cli, "_require_human_confirmation", _deny)
        args = cli._build_parser().parse_args(["deploy-mind-eval"])
        with pytest.raises(SystemExit):
            args.func(args)
        assert launched == [], "a paid launch must never precede the gate"


class TestMindEvalOnstart:
    def _onstart(self, **kw):
        from neuroslm.connectors.vast_mind_eval import build_mind_eval_onstart
        env = {"BRANCH": "master", "EXPERT": "smollm2_360m",
               "EVAL_ARGS": "--n-items 4", "LABEL": "neuroslm-mind-eval",
               "AUTO_DESTROY": "1"}
        env.update(kw)
        return build_mind_eval_onstart(env)

    def test_runs_the_battery_on_cuda_and_pushes_results(self):
        s = self._onstart()
        assert "neuroslm.cli eval mind" in s
        assert "--device cuda" in s and "--n-items 4" in s
        assert "results/mind_eval" in s
        assert "push_artifacts" in s
        assert "__" not in s.replace("__init__", ""), "unexpanded placeholder"

    def test_self_destroys_by_default_and_not_with_opt_out(self):
        assert "vastai destroy instance" in self._onstart()
        assert "vastai destroy instance" not in self._onstart(AUTO_DESTROY="0")


class TestMultiExpertEval:
    """One deploy compares several trunks: `--expert a,b` runs the
    battery once per expert, one report each."""

    def _patch(self, monkeypatch, calls):
        import neuroslm.cognition.indicators as ind
        import neuroslm.cognition.runtime as rtm
        import neuroslm.train_dsl as td
        monkeypatch.setattr(rtm, "build_hf_seams",
                            lambda expert, device="cpu": calls.append(expert)
                            or _seams(_Reader([])))
        monkeypatch.setattr(ind, "run_indicator_battery",
                            lambda seams, **kw: {"meta": {}, "probes": {}})
        monkeypatch.setattr(td, "_git_commit_info", lambda: ("sha", "br"))

    def test_runs_the_battery_per_expert_into_out_dir(self, monkeypatch,
                                                      tmp_path):
        from neuroslm.cli import _build_parser
        calls = []
        self._patch(monkeypatch, calls)
        a = _build_parser().parse_args(
            ["eval", "mind", "--expert",
             "smollm2_360m_instruct,deepseek_r1_qwen_1_5b",
             "--out-dir", str(tmp_path)])
        assert a.func(a) == 0
        assert calls == ["smollm2_360m_instruct", "deepseek_r1_qwen_1_5b"]
        names = sorted(p.name for p in tmp_path.glob("*.json"))
        assert len(names) == 2
        assert any(n.endswith("_deepseek-r1-qwen-1-5b.json") for n in names)
        meta = json.loads((tmp_path / names[0]).read_text())["meta"]
        assert meta["git_commit"] == "sha" and "expert" in meta

    def test_single_out_file_with_several_experts_is_rejected(
            self, monkeypatch, tmp_path):
        from neuroslm.cli import _build_parser
        self._patch(monkeypatch, [])
        a = _build_parser().parse_args(
            ["eval", "mind", "--expert", "a1,b2",
             "--out", str(tmp_path / "x.json")])
        assert a.func(a) == 2

    def test_onstart_writes_into_the_results_dir(self):
        from neuroslm.connectors.vast_mind_eval import build_mind_eval_onstart
        s = build_mind_eval_onstart(
            {"EXPERT": "smollm2_360m_instruct,deepseek_r1_qwen_1_5b",
             "AUTO_DESTROY": "1"})
        assert "--expert 'smollm2_360m_instruct,deepseek_r1_qwen_1_5b'" in s
        assert "--out-dir results/mind_eval" in s
        assert "'results/mind_eval'" in s, "push the whole results dir"
