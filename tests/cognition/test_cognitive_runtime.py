# -*- coding: utf-8 -*-
"""CognitiveRuntime — the always-on cognition layer (architecture.md §14).

The two-layer doctrine says the neuroanatomical machinery THINKS in
natural language on top of a trained trunk instead of shaping its
pretraining gradient. This is that runtime, and this file is the first
slice of the Layer-A evaluation battery the doctrine requires: the
cognition layer is judged by measurable cognitive function, never by
next-token ppl.

One tick = one cognitive cycle:

    SENSE   drain the sensory queue (user text, environment events)
    RECALL  similarity retrieval from EpisodicMemory (hippocampus)
    THINK   trunk generates K candidate thoughts from persona+recall
    GATE    basal-ganglia selection: DA sets exploration temperature,
            GABA above threshold inhibits the whole act (silence)
    STORE   surprise-gated episodic write (novel thoughts remembered,
            repetitive ones not)
    DRIVE   the DrivenNTSystem integrates the tick's signals so NT
            state carries across ticks

Layer-A contracts pinned here:
  A. Episodic recall — a fact observed at tick t is retrieved and
     enters the thinking prompt when related input arrives later.
  B. NT-manipulation behavior change — high DA measurably shifts
     selection away from greedy; low DA is greedy.
  C. Inhibition — high GABA produces silence (no thought, no write,
     no generation compute).
  D. Surprise-gated memory — repetitive thoughts stop being stored;
     novel ones are.
Plus mechanism contracts: cosine retrieval math, NT integration via
the real DrivenNTSystem, ChatDaemon hosting.
"""
from __future__ import annotations

import math
import random

import pytest

from neuroslm.memory.episodic import EpisodicMemory


# ── Deterministic fakes (the runtime is dependency-injected) ─────────

def _vec_for(text: str):
    """Cheap deterministic 8-dim 'embedding': trigger words dominate
    an axis each so cosine similarity is controllable in tests."""
    axes = ("code", "weather", "music", "launch", "coffee", "river",
            "chess", "moon")
    v = [0.05] * len(axes)
    low = text.lower()
    for i, w in enumerate(axes):
        if w in low:
            v[i] += 1.0
    return v


class _ScriptedGen:
    """GenerateFn returning scripted candidates; records every prompt."""

    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.prompts = []
        self.calls = 0

    def __call__(self, prompt: str, max_new_tokens: int) -> str:
        self.prompts.append(prompt)
        out = self.candidates[self.calls % len(self.candidates)]
        self.calls += 1
        return out


class _FakeNT:
    """Duck-typed NT system with directly settable levels."""

    def __init__(self, **levels):
        base = {"DA": 0.15, "NE": 0.20, "5HT": 0.50, "ACh": 0.30,
                "eCB": 0.10, "Glu": 0.45, "GABA": 0.15}
        base.update(levels)
        self._levels = base
        self.step_calls = []

    def levels(self):
        return dict(self._levels)

    def baselines(self):
        return {"DA": 0.15, "NE": 0.20, "5HT": 0.50, "ACh": 0.30,
                "eCB": 0.10, "Glu": 0.45, "GABA": 0.15}

    def step_full(self, **kw):
        self.step_calls.append(kw)


def _score_map(mapping, default_nll=5.0):
    from neuroslm.cognition.runtime import ThoughtScore

    def _score(text):
        return ThoughtScore(mean_nll=mapping.get(text, default_nll),
                            entropy_norm=0.5)
    return _score


def _mk_runtime(gen, scores=None, nt=None, cfg=None, seed=0):
    from neuroslm.cognition.runtime import CognitiveRuntime, MindConfig
    return CognitiveRuntime(
        generate_fn=gen,
        score_fn=scores or _score_map({}),
        embed_fn=_vec_for,
        nt=nt if nt is not None else _FakeNT(),
        memory=EpisodicMemory(maxlen=64),
        cfg=cfg or MindConfig(n_candidates=2),
        rng=random.Random(seed),
    )


# ── Mechanism: cosine retrieval on EpisodicMemory ────────────────────

class TestEpisodicRetrieve:
    def test_retrieves_nearest_by_cosine(self):
        m = EpisodicMemory(maxlen=16)
        m.add("about the launch", content_vec=_vec_for("launch"))
        m.add("about coffee", content_vec=_vec_for("coffee"))
        m.add("about music", content_vec=_vec_for("music"))
        got = m.retrieve(_vec_for("the launch happened"), k=1)
        assert len(got) == 1
        assert got[0]["content"] == "about the launch"

    def test_k_bounds_and_ordering(self):
        m = EpisodicMemory(maxlen=16)
        m.add("launch one", content_vec=_vec_for("launch"))
        m.add("coffee talk", content_vec=_vec_for("coffee"))
        got = m.retrieve(_vec_for("launch"), k=5)
        assert [e["content"] for e in got][0] == "launch one"
        assert len(got) == 2

    def test_skips_episodes_without_vectors(self):
        m = EpisodicMemory(maxlen=16)
        m.add("no vector here")
        m.add("launch fact", content_vec=_vec_for("launch"))
        got = m.retrieve(_vec_for("launch"), k=3)
        assert [e["content"] for e in got] == ["launch fact"]


# ── Layer-A contract A: episodic recall shapes future thinking ───────

class TestEpisodicRecall:
    def test_observed_fact_enters_later_thinking_prompt(self):
        gen = _ScriptedGen(["a thought"])
        rt = _mk_runtime(gen)
        rt.observe("The launch code is BLUE-7.")
        rt.observe("I had coffee this morning.")
        gen.prompts.clear()
        rt.observe("When is the launch?")
        rt.tick()
        assert gen.prompts, "tick must generate"
        assert "BLUE-7" in gen.prompts[0], (
            "the related episode must be retrieved into the thinking "
            "prompt — this is the hippocampal read path")

    def test_percepts_are_always_stored(self):
        rt = _mk_runtime(_ScriptedGen(["x"]))
        rt.observe("the river is high today")
        contents = [e["content"] for e in rt.memory.all()]
        assert any("river" in c for c in contents)


# ── Layer-A contract B: DA gates exploration in selection ────────────

class TestNTGatedSelection:
    CANDS = ["good thought", "worse thought"]

    def _scores(self):
        return _score_map({"good thought": 2.0, "worse thought": 4.0})

    def test_low_da_is_greedy(self):
        from neuroslm.cognition.runtime import MindConfig
        gen = _ScriptedGen(self.CANDS)
        cfg = MindConfig(n_candidates=2, selection_temp_base=0.01)
        rt = _mk_runtime(gen, scores=self._scores(),
                         nt=_FakeNT(DA=0.15), cfg=cfg)
        picks = {rt.tick().thought for _ in range(12)}
        assert picks == {"good thought"}, (
            "at baseline DA and near-zero temperature, selection is "
            "greedy on trunk NLL")

    def test_high_da_explores(self):
        from neuroslm.cognition.runtime import MindConfig
        gen = _ScriptedGen(self.CANDS)
        cfg = MindConfig(n_candidates=2, selection_temp_base=0.01,
                         da_temp_gain=400.0)
        rt = _mk_runtime(gen, scores=self._scores(),
                         nt=_FakeNT(DA=0.95), cfg=cfg, seed=1)
        picks = [rt.tick().thought for _ in range(24)]
        assert "worse thought" in picks, (
            "high DA must raise selection temperature enough that "
            "non-greedy thoughts occur — dopaminergic exploration")

    def test_selection_temperature_monotone_in_da(self):
        from neuroslm.cognition.runtime import selection_temperature, MindConfig
        cfg = MindConfig()
        base = {"DA": 0.15}
        t_lo = selection_temperature(0.15, 0.15, cfg)
        t_mid = selection_temperature(0.5, 0.15, cfg)
        t_hi = selection_temperature(0.95, 0.15, cfg)
        assert t_lo <= t_mid <= t_hi
        assert t_lo == pytest.approx(cfg.selection_temp_base)


# ── Layer-A contract C: GABA inhibition = silence ────────────────────

class TestInhibition:
    def test_high_gaba_silences_the_tick(self):
        gen = _ScriptedGen(["a thought"])
        rt = _mk_runtime(gen, nt=_FakeNT(GABA=0.9))
        res = rt.tick()
        assert res.inhibited is True
        assert res.thought is None
        assert res.stored is False
        assert gen.calls == 0, (
            "inhibition suppresses the act itself — no generation "
            "compute is spent on an inhibited tick")

    def test_low_gaba_thinks(self):
        gen = _ScriptedGen(["a thought"])
        rt = _mk_runtime(gen, nt=_FakeNT(GABA=0.1))
        res = rt.tick()
        assert res.inhibited is False
        assert res.thought == "a thought"


# ── Layer-A contract D: surprise-gated episodic writes ───────────────

class TestSurpriseGatedWrites:
    def test_first_thought_is_stored_repetition_is_not(self):
        from neuroslm.cognition.runtime import MindConfig
        gen = _ScriptedGen(["same thought"])
        scores = _score_map({"same thought": 3.0})
        cfg = MindConfig(n_candidates=1, surprise_write_z=0.05)
        rt = _mk_runtime(gen, scores=scores, cfg=cfg)
        first = rt.tick()
        assert first.stored is True, "bootstrap: first thought is novel"
        repeats = [rt.tick().stored for _ in range(6)]
        assert repeats[-1] is False, (
            "an unchanging NLL converges to its EMA — repetitive "
            "thoughts must stop being written to episodic memory")

    def test_novel_high_nll_thought_is_stored_again(self):
        from neuroslm.cognition.runtime import MindConfig, ThoughtScore
        outs = ["same thought"] * 8 + ["astonishing new idea"]
        gen = _ScriptedGen(outs)

        def scores(text):
            return ThoughtScore(
                mean_nll=9.0 if "astonishing" in text else 3.0,
                entropy_norm=0.5)

        cfg = MindConfig(n_candidates=1, surprise_write_z=0.05)
        rt = _mk_runtime(gen, scores=scores, cfg=cfg)
        results = [rt.tick() for _ in range(9)]
        assert results[-1].thought == "astonishing new idea"
        assert results[-1].stored is True, (
            "a thought far from the NLL EMA is surprising — it must "
            "be written")


# ── NT integration: the real DrivenNTSystem carries state ────────────

class TestRealNTIntegration:
    def test_default_construction_uses_driven_nt(self):
        from neuroslm.cognition.runtime import CognitiveRuntime
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        rt = CognitiveRuntime(
            generate_fn=_ScriptedGen(["t"]),
            score_fn=_score_map({}),
            embed_fn=_vec_for,
        )
        assert isinstance(rt.nt, DrivenNTSystem)

    def test_tick_drives_the_nt_system(self):
        nt = _FakeNT()
        rt = _mk_runtime(_ScriptedGen(["t"]), nt=nt)
        rt.tick()
        assert nt.step_calls, "every tick must advance NT dynamics"
        assert "loss" in nt.step_calls[-1]

    def test_tick_result_reports_nt_levels_and_phi(self):
        rt = _mk_runtime(_ScriptedGen(["t"]))
        res = rt.tick()
        assert set(res.nt_levels) >= {"DA", "NE", "GABA"}
        assert 0.0 <= res.phi_proxy <= 1.0


# ── Introspection: seeing the inner processes (§14.5) ─────────────────
# "I want to see the inner processes when talking to it — the action
# the basal ganglia produces; a line about how many memories recalled
# by the hippocampus" — one formatted line per tick summarising the
# whole cognitive cycle's decision, not just its text output.

class TestMindWandering:
    """DMN condition: no sensory input this tick → THINK is biased
    toward associative, reflective continuation instead of a flat
    prompt completion. External input present → task-positive, no
    wandering framing."""

    def test_idle_tick_uses_a_wander_prompt(self):
        gen = _ScriptedGen(["a wandering thought"])
        rt = _mk_runtime(gen)
        rt.tick()  # nothing observed — pure idle/DMN tick
        assert gen.prompts, "must generate"
        prompt = gen.prompts[0]
        assert any(w in prompt for w in rt.cfg.wander_prompts), (
            "an idle tick with no sensory input IS the DMN condition — "
            "it must be framed as mind-wandering, not a flat "
            "continuation of nothing")

    def test_tick_with_sensory_input_skips_wander_framing(self):
        gen = _ScriptedGen(["a response"])
        rt = _mk_runtime(gen)
        rt.observe("what time is it")
        gen.prompts.clear()
        rt.tick()
        prompt = gen.prompts[0]
        assert not any(w in prompt for w in rt.cfg.wander_prompts), (
            "responding to real input is task-positive, not DMN "
            "wandering — the wander framing must not leak in")
        assert "what time is it" in prompt

    def test_wander_prompts_rotate(self):
        gen = _ScriptedGen(["t"])
        rt = _mk_runtime(gen)
        rt.tick()
        first = gen.prompts[-1]
        rt.tick()
        second = gen.prompts[-1]
        assert first != second, (
            "repeating the exact same wander line every idle tick "
            "would make the mind-wandering feel scripted, not organic")


class TestTickLineageAndWandering:
    def test_tick_n_increments(self):
        rt = _mk_runtime(_ScriptedGen(["t1", "t2"]))
        r1 = rt.tick()
        r2 = rt.tick()
        assert r1.tick_n >= 1
        assert r2.tick_n == r1.tick_n + 1

    def test_prior_thought_captures_lineage(self):
        rt = _mk_runtime(_ScriptedGen(["first", "second"]))
        r1 = rt.tick()
        assert r1.prior_thought is None, "nothing precedes the first tick"
        r2 = rt.tick()
        assert r2.prior_thought == r1.thought, (
            "each tick's thought must trace back to what came before "
            "it, so the debug log can show how thoughts evolve")

    def test_wandering_flag_reflects_sensory_presence(self):
        rt = _mk_runtime(_ScriptedGen(["t"]))
        assert rt.tick().wandering is True
        rt.observe("hello")
        assert rt.tick().wandering is False


class TestFormatDebugTrace:
    """'debug log the actions generated by basal ganglia' — the full
    deliberation (every candidate + its score, which one won), not
    just the compact one-line summary format_introspection gives."""

    def _normal_result(self):
        from neuroslm.cognition.runtime import TickResult, ThoughtScore
        return TickResult(
            thought="good one", tick_n=3, prior_thought="previous idea",
            wandering=True,
            candidates=["good one", "meh idea"],
            scores=[ThoughtScore(mean_nll=2.1, entropy_norm=0.4),
                    ThoughtScore(mean_nll=3.5, entropy_norm=0.6)],
            recalled=[{"content": "episode A"}, {"content": "episode B"}],
            stored=True, inhibited=False,
            nt_levels={"DA": 0.15, "GABA": 0.15}, phi_proxy=0.4)

    def test_every_candidate_shown_with_score_and_selection_marked(self):
        from neuroslm.cognition.runtime import format_debug_trace
        s = format_debug_trace(self._normal_result())
        assert "good one" in s and "meh idea" in s
        assert "2.10" in s
        sel_line = [l for l in s.splitlines() if "good one" in l][0]
        other_line = [l for l in s.splitlines() if "meh idea" in l][0]
        assert "SELECTED" in sel_line and "SELECTED" not in other_line

    def test_recalled_episodes_shown(self):
        from neuroslm.cognition.runtime import format_debug_trace
        s = format_debug_trace(self._normal_result())
        assert "episode A" in s and "episode B" in s

    def test_lineage_from_prior_thought_shown(self):
        from neuroslm.cognition.runtime import format_debug_trace
        s = format_debug_trace(self._normal_result())
        assert "previous idea" in s

    def test_tick_number_shown(self):
        from neuroslm.cognition.runtime import format_debug_trace
        s = format_debug_trace(self._normal_result())
        assert "tick 3" in s

    def test_inhibited_tick_notes_no_deliberation(self):
        from neuroslm.cognition.runtime import format_debug_trace, TickResult
        r = TickResult(thought=None, inhibited=True, tick_n=5,
                       nt_levels={"GABA": 0.9})
        s = format_debug_trace(r)
        assert "inhibit" in s.lower()
        assert "SELECTED" not in s


class TestFormatIntrospection:
    def _result(self, **kw):
        from neuroslm.cognition.runtime import TickResult
        base = dict(
            thought="a thought", candidates=["a thought", "other"],
            scores=[], recalled=[{"content": "x"}, {"content": "y"}],
            stored=True, inhibited=False,
            nt_levels={"DA": 0.15, "NE": 0.20, "5HT": 0.50, "ACh": 0.30,
                      "eCB": 0.10, "Glu": 0.45, "GABA": 0.15},
            phi_proxy=0.42)
        base.update(kw)
        return TickResult(**base)

    def test_normal_tick_reports_bg_hc_nt_phi(self):
        from neuroslm.cognition.runtime import format_introspection
        s = format_introspection(self._result())
        assert "Φ=0.42" in s
        assert "DA=0.15" in s and "GABA=0.15" in s
        assert "recall=2" in s, "hippocampal recall count must be visible"
        assert "n=2" in s, "basal-ganglia candidate count must be visible"
        assert "write=yes" in s

    def test_selected_candidate_index_shown(self):
        from neuroslm.cognition.runtime import format_introspection
        s = format_introspection(self._result(
            thought="other", candidates=["a thought", "other"]))
        assert "pick=1" in s

    def test_unstored_thought_shows_write_no(self):
        from neuroslm.cognition.runtime import format_introspection
        s = format_introspection(self._result(stored=False))
        assert "write=no" in s

    def test_inhibited_tick_reports_silence_not_bg_hc(self):
        from neuroslm.cognition.runtime import format_introspection
        s = format_introspection(self._result(
            inhibited=True, thought=None, candidates=[], recalled=[],
            stored=False))
        assert "inhibit" in s.lower()
        assert "GABA=0.15" in s, "NT snapshot still shown on silence"
        assert "recall=" not in s and "n=" not in s


# ── Hosting: ChatDaemon runs the mind ────────────────────────────────

class TestChatDaemonHostsTheMind:
    def _daemon(self, rt):
        from neuroslm.chat_daemon import ChatDaemon, ChatDaemonConfig
        return ChatDaemon(_ScriptedGen(["reply"]),
                          ChatDaemonConfig(), use_color=False, mind=rt)

    def test_think_once_delegates_to_mind(self):
        gen = _ScriptedGen(["mind thought"])
        rt = _mk_runtime(gen)
        d = self._daemon(rt)
        out = d.think_once()
        assert out == "mind thought"
        assert gen.calls >= 1
        kinds = [e.kind for e in d.memory.recent(8)]
        assert "thought" in kinds

    def test_post_user_feeds_the_sensory_queue(self):
        gen = _ScriptedGen(["mind thought"])
        rt = _mk_runtime(gen)
        d = self._daemon(rt)
        d.post_user("the launch code is BLUE-7")
        contents = [e["content"] for e in rt.memory.all()]
        assert any("BLUE-7" in c for c in contents), (
            "user turns are sensory input — they must reach the "
            "mind's episodic memory")

    def test_think_once_records_last_tick_and_introspect_entry(self):
        gen = _ScriptedGen(["mind thought"])
        rt = _mk_runtime(gen)
        d = self._daemon(rt)
        d.think_once()
        assert d.last_tick is not None and d.last_tick.thought == "mind thought"
        kinds = [e.kind for e in d.memory.recent(8)]
        assert "introspect" in kinds, (
            "the inner-state summary (BG/HC/NT/Φ) must land in memory "
            "alongside the thought, not just the thought text")

    def test_inhibited_tick_still_records_introspection(self):
        gen = _ScriptedGen(["mind thought"])
        rt = _mk_runtime(gen, nt=_FakeNT(GABA=0.9))
        d = self._daemon(rt)
        out = d.think_once()
        assert out is None
        assert d.last_tick is not None and d.last_tick.inhibited is True
        entries = d.memory.recent(8)
        assert any(e.kind == "introspect" for e in entries), (
            "silence is a decision — it must be visible too, not just "
            "spoken thoughts")
        assert not any(e.kind == "thought" for e in entries)

    def test_log_stream_receives_introspection_and_thought(self):
        import io
        gen = _ScriptedGen(["mind thought"])
        rt = _mk_runtime(gen)
        from neuroslm.chat_daemon import ChatDaemon, ChatDaemonConfig
        buf = io.StringIO()
        d = ChatDaemon(_ScriptedGen(["reply"]), ChatDaemonConfig(),
                       use_color=False, mind=rt, log_stream=buf)
        d.think_once()
        out = buf.getvalue()
        assert "mind thought" in out
        assert "GABA=" in out, (
            "server-side (no client attached) DMN visibility: the box "
            "log must show inner state, not just the spoken thought")

    def test_log_stream_receives_full_bg_deliberation_trace(self):
        import io
        gen = _ScriptedGen(["candidate one", "candidate two"])
        rt = _mk_runtime(gen)
        from neuroslm.chat_daemon import ChatDaemon, ChatDaemonConfig
        buf = io.StringIO()
        d = ChatDaemon(_ScriptedGen(["reply"]), ChatDaemonConfig(),
                       use_color=False, mind=rt, log_stream=buf)
        d.think_once()
        out = buf.getvalue()
        assert "SELECTED" in out, (
            "'debug log the actions generated by basal ganglia' — the "
            "full candidate deliberation must reach the log, not just "
            "the winning thought's text")
        assert "tick 1" in out

    def test_no_log_stream_is_a_silent_no_op(self):
        gen = _ScriptedGen(["mind thought"])
        rt = _mk_runtime(gen)
        d = self._daemon(rt)  # log_stream defaults to None
        d.think_once()  # must not raise

    def test_daemon_without_mind_keeps_legacy_path(self):
        from neuroslm.chat_daemon import ChatDaemon, ChatDaemonConfig
        d = ChatDaemon(_ScriptedGen(["legacy"]),
                       ChatDaemonConfig(), use_color=False)
        assert d.think_once() == "legacy"


class TestCliWiring:
    def test_chat_mind_flag_parses(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(["chat", "--mind"])
        assert args.mind is True

    def test_mind_defaults_off(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(["chat"])
        assert args.mind is False


# ── Expert backend: think with a frozen pretrained LM, no trunk ──────

class _FakeHFModel:
    """Duck-typed HF causal LM: __call__(input_ids=..) → .logits, plus
    an embedding surface. Vocab 32, d 4; logits peak on token 1 so the
    scorer sees a deterministic distribution."""

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    def __init__(self):
        import torch
        self.config = type("C", (), {"n_positions": 64})()
        self._emb = torch.arange(32 * 4, dtype=torch.float32).reshape(32, 4)
        self.to_calls = []

    def eval(self):
        return self

    def to(self, device):
        self.to_calls.append(device)
        return self

    def get_input_embeddings(self):
        emb = self._emb

        class _E:
            weight = emb
        return _E()

    def __call__(self, input_ids=None):
        import torch
        B, T = input_ids.shape
        logits = torch.zeros(B, T, 32)
        logits[..., 1] = 4.0
        return self._Out(logits)


class _FakeHFTokenizer:
    def encode(self, text):
        return [(ord(c) % 30) + 1 for c in (text or " ")[:16]]

    def decode(self, ids, **kw):
        return "decoded-" + "".join(chr(97 + (i % 26)) for i in ids)


class TestExpertBackend:
    def _rt(self, cfg=None):
        from neuroslm.cognition.runtime import build_runtime_from_hf_lm
        return build_runtime_from_hf_lm(
            "fake/expert",
            model_factory=_FakeHFModel,
            tokenizer_factory=_FakeHFTokenizer,
            cfg=cfg,
        )

    def test_builder_exists_with_injection_points(self):
        import inspect
        from neuroslm.cognition.runtime import build_runtime_from_hf_lm
        sig = inspect.signature(build_runtime_from_hf_lm)
        assert "model_factory" in sig.parameters
        assert "tokenizer_factory" in sig.parameters

    def test_expert_mind_thinks_without_any_trunk(self):
        rt = self._rt()
        rt.observe("hello there")
        res = rt.tick()
        assert res.thought, (
            "the mind must run on a frozen pretrained expert alone — "
            "usable before any trunk training")
        assert res.scores and res.scores[0].mean_nll > 0.0
        assert 0.0 <= res.phi_proxy <= 1.0

    def test_expert_embeddings_power_recall(self):
        rt = self._rt()
        rt.observe("alpha beta")
        eps = rt.memory.all()
        assert eps and eps[-1]["content_vec"] is not None
        assert len(eps[-1]["content_vec"]) == 4, (
            "episodic vectors must come from the expert's own "
            "embedding rows (d=4 in the fake)")

    def test_chat_expert_flag_parses(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(["chat", "--expert"])
        assert args.expert == "smollm2_360m"
        args = _build_parser().parse_args(
            ["chat", "--expert", "Qwen/Qwen2.5-0.5B"])
        assert args.expert == "Qwen/Qwen2.5-0.5B"
        args = _build_parser().parse_args(["chat"])
        assert args.expert is None


class _FakeHFModelWithGenerate(_FakeHFModel):
    """HF-shaped model exposing .generate — the KV-cache fast path.

    2026-08-11 incident: the expert backend reused the daemon's naive
    per-token loop, which re-runs a FULL forward over the growing
    sequence for every generated token (no KV cache). Tolerable for
    the small DSL trunk it was written for; on a 360M HF model on CPU
    a single 96-token reply takes minutes — the user-visible symptom
    is `brian chat --expert` "hanging" on the first message.
    """

    def __init__(self):
        super().__init__()
        self.generate_calls = []

    def generate(self, input_ids=None, **kw):
        import torch
        self.generate_calls.append(kw)
        B, T = input_ids.shape
        n_new = int(kw.get("max_new_tokens", 4))
        new = torch.full((B, n_new), 7, dtype=torch.long)
        return torch.cat([input_ids, new], dim=-1)


class TestExpertModelMovedToDevice:
    """2026-08-12 live incident: a mind box deployed on an A100 ran
    each DMN tick in ~55-60s instead of a couple seconds, and manual
    /think calls kept returning None. Root cause: AutoModelForCausalLM
    .from_pretrained() loads on CPU by default — build_runtime_from_hf_lm
    never called .to(device), so a "GPU" deploy silently ran the whole
    1.5B model on CPU. Each tick held the daemon's non-blocking
    inference lock for the full CPU-bound duration, so any client
    /think landing mid-tick bounced off the lock and returned None."""

    def test_model_moved_to_requested_device(self):
        from neuroslm.cognition.runtime import build_runtime_from_hf_lm
        model = _FakeHFModel()
        build_runtime_from_hf_lm(
            "fake/expert", device="cuda",
            model_factory=lambda: model,
            tokenizer_factory=_FakeHFTokenizer)
        assert model.to_calls == ["cuda"], (
            ".to(device) must be called unconditionally — a silently "
            "CPU-bound 'GPU' deploy is exactly what happened live")

    def test_cpu_device_also_moves_explicitly(self):
        from neuroslm.cognition.runtime import build_runtime_from_hf_lm
        model = _FakeHFModel()
        build_runtime_from_hf_lm(
            "fake/expert", device="cpu",
            model_factory=lambda: model,
            tokenizer_factory=_FakeHFTokenizer)
        assert model.to_calls == ["cpu"], (
            "the .to() call must be unconditional, not an "
            "if device != 'cpu' special case that's easy to drift "
            "out of sync with a future default change")


class TestExpertGenerateUsesKVCache:
    def _rt(self):
        from neuroslm.cognition.runtime import build_runtime_from_hf_lm
        model = _FakeHFModelWithGenerate()
        rt = build_runtime_from_hf_lm(
            "fake/expert",
            model_factory=lambda: model,
            tokenizer_factory=_FakeHFTokenizer,
        )
        return rt, model

    def test_gen_fn_routes_through_model_generate(self):
        rt, model = self._rt()
        out = rt._gen("hello world", 4)
        assert model.generate_calls, (
            "an HF model exposing .generate must be sampled through it "
            "(KV cache) — the naive per-token full-forward loop is "
            "O(T²) full forwards and reads as a hang on CPU")
        assert model.generate_calls[0].get("use_cache") is True
        assert model.generate_calls[0].get("max_new_tokens") == 4

    def test_gen_fn_decodes_only_new_tokens(self):
        rt, model = self._rt()
        out = rt._gen("hello", 3)
        # fake generate appends token id 7 three times → decode of [7,7,7]
        assert out == _FakeHFTokenizer().decode([7, 7, 7])

    def test_model_without_generate_falls_back_to_naive_loop(self):
        rt = TestExpertBackend()._rt()
        out = rt._gen("hello", 2)
        assert isinstance(out, str) and out, (
            "models without .generate (e.g. bare DSL wrappers) must "
            "keep working through the daemon's naive seam")


class _FakeChatTokenizer(_FakeHFTokenizer):
    """Tokenizer exposing a chat template (instruct-model shape)."""
    chat_template = "{{ messages }}"  # any truthy value — content unused

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=True):
        text = messages[-1]["content"]
        return self.encode(f"<user>{text}<assistant>")


class TestGenerationQuality:
    """2026-08-11 live incident: `brian chat connect` produced a
    degenerate loop —

        USER: I have the answer.
        BRIAN: Well, I do.
        USER: I have the answer.
        BRIAN: Well, I do.  (repeat)

    Root cause: the base (non-instruct) expert has no learned turn-end
    token, so unconstrained generation hallucinates the next USER turn
    and then falls into n-gram repetition. Fix: repetition controls
    always on, a stop-string safety net for base models specifically,
    and proper chat-template formatting when the tokenizer has one
    (instruct models know their own turn-end token and don't need the
    stop-string hack).
    """

    def _rt(self, tok_factory):
        from neuroslm.cognition.runtime import build_runtime_from_hf_lm
        model = _FakeHFModelWithGenerate()
        rt = build_runtime_from_hf_lm(
            "fake/expert", model_factory=lambda: model,
            tokenizer_factory=tok_factory)
        return rt, model

    def test_repetition_controls_always_applied(self):
        rt, model = self._rt(_FakeHFTokenizer)
        rt._gen("hello", 8)
        kw = model.generate_calls[-1]
        assert kw.get("repetition_penalty", 1.0) > 1.0
        assert kw.get("no_repeat_ngram_size", 0) >= 2

    def test_stop_strings_for_base_models(self):
        rt, model = self._rt(_FakeHFTokenizer)
        rt._gen("hello", 8)
        kw = model.generate_calls[-1]
        assert "stop_strings" in kw
        assert any("USER" in s for s in kw["stop_strings"]), (
            "a base model has no learned turn-end token — without a "
            "stop string it hallucinates the next USER turn, exactly "
            "the live degenerate loop reported")
        assert kw.get("tokenizer") is not None, (
            "HF stop_strings requires the tokenizer= kwarg on generate()")

    def test_chat_template_used_when_available(self):
        rt, model = self._rt(_FakeChatTokenizer)
        rt._gen("hello", 8)
        kw = model.generate_calls[-1]
        assert "stop_strings" not in kw, (
            "instruct models know their own turn-end token via the "
            "chat template — the crude stop-string safety net is only "
            "needed for base models")

    def test_chat_template_path_still_decodes_new_tokens_only(self):
        rt, model = self._rt(_FakeChatTokenizer)
        out = rt._gen("hello", 3)
        assert out == _FakeChatTokenizer().decode([7, 7, 7])
