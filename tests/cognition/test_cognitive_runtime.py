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


def _mk_runtime(gen, scores=None, nt=None, cfg=None, seed=0, osc=False):
    """Mechanism tests isolate their mechanism: the §14.13 workspace
    (which gates storage on ignition) is OFF unless a test opts in
    with ``osc=True``. Ignition gating has its own contracts."""
    import dataclasses
    from neuroslm.cognition.runtime import CognitiveRuntime, MindConfig
    cfg = cfg or MindConfig(n_candidates=2)
    if not osc:
        cfg = dataclasses.replace(cfg, workspace_oscillator=False)
    return CognitiveRuntime(
        generate_fn=gen,
        score_fn=scores or _score_map({}),
        embed_fn=_vec_for,
        nt=nt if nt is not None else _FakeNT(),
        memory=EpisodicMemory(maxlen=64),
        cfg=cfg,
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


# ── Layer-A contract D: novelty-gated episodic writes ────────────────
# (2026-08-12: the gate switched from NLL-EMA distance to SEMANTIC
# novelty — see TestSemanticNoveltyGate for the full new contract and
# the live evidence that motivated the change. These two keep the
# original Layer-A behavioral claims pinned under the new mechanism.)

class TestNoveltyGatedWrites:
    def test_first_thought_is_stored_repetition_is_not(self):
        from neuroslm.cognition.runtime import MindConfig
        gen = _ScriptedGen(["same thought"])
        cfg = MindConfig(n_candidates=1)
        rt = _mk_runtime(gen, cfg=cfg)
        first = rt.tick()
        assert first.stored is True, "bootstrap: first thought is novel"
        repeats = [rt.tick().stored for _ in range(6)]
        assert repeats[-1] is False, (
            "identical content has zero semantic novelty — repetitive "
            "thoughts must stop being written to episodic memory")

    def test_semantically_novel_thought_is_stored_again(self):
        from neuroslm.cognition.runtime import MindConfig
        outs = ["same thought"] * 8 + ["coffee by the river at dawn"]
        gen = _ScriptedGen(outs)
        # storage-dependent contract, independent of §14.13 ignition
        # (pinned in TestOscillatoryWorkspace) — workspace off.
        cfg = MindConfig(n_candidates=1, workspace_oscillator=False)
        rt = _mk_runtime(gen, cfg=cfg)
        results = [rt.tick() for _ in range(9)]
        assert results[-1].thought == "coffee by the river at dawn"
        assert results[-1].stored is True, (
            "semantically distant content is novel — it must be written")


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

class TestThoughtTokenBudget:
    def test_default_budget_is_generous_enough_to_avoid_mid_sentence_cutoff(self):
        """Live report: thoughts consistently ran out of budget
        mid-sentence ('...complement human'). 32 tokens is too tight
        for a verbose instruct model; raised to 96."""
        from neuroslm.cognition.runtime import MindConfig
        assert MindConfig().thought_n_tok >= 96


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


class TestActionTaxonomy:
    """The basal ganglia produces exactly one of three actions per
    tick: 'respond' (external input present — always surfaces),
    'speak' (idle tick, but the thought was novel enough to pass the
    hippocampal surprise gate — voiced spontaneously), or 'think'
    (idle tick, thought converged/repetitive — stays internal). No
    new NT machinery: reuses the existing wandering + novelty-gate
    (stored) signals honestly rather than inventing a parallel gate."""

    def test_sensory_present_is_always_respond(self):
        rt = _mk_runtime(_ScriptedGen(["a reply"]))
        rt.observe("hello there")
        r = rt.tick()
        assert r.action == "respond"

    def test_idle_novel_thought_is_speak(self):
        from neuroslm.cognition.runtime import MindConfig
        # empty memory -> the first thought is maximally novel -> stored.
        cfg = MindConfig(n_candidates=1)
        rt = _mk_runtime(_ScriptedGen(["a fresh idea"]), cfg=cfg)
        r = rt.tick()
        assert r.wandering is True
        assert r.stored is True
        assert r.action == "speak"

    def test_idle_repetitive_thought_is_think(self):
        from neuroslm.cognition.runtime import MindConfig
        cfg = MindConfig(n_candidates=1)
        gen = _ScriptedGen(["same idea"])
        rt = _mk_runtime(gen, scores=_score_map({"same idea": 3.0}), cfg=cfg)
        rt.tick()  # bootstrap: first is always novel/stored
        r2 = rt.tick()  # identical text -> zero semantic novelty
        assert r2.stored is False
        assert r2.action == "think"

    def test_inhibited_tick_reports_pending_respond_if_sensory_queued(self):
        rt = _mk_runtime(_ScriptedGen(["t"]), nt=_FakeNT(GABA=0.9))
        rt.observe("urgent thing")
        r = rt.tick()
        assert r.inhibited is True
        assert r.action == "respond", (
            "inhibition suppressed the act, but the debug trace should "
            "still show WHAT was suppressed — a pending response, not "
            "generic silence")

    def test_dmn_resumes_wandering_after_handling_external_input(self):
        """'if the DMN loop is interrupted by an action... it should
        spontaneously enter mind wandering again' — sensory drains
        after one tick consumes it, so the very next tick reverts to
        wandering with no special-case code required."""
        rt = _mk_runtime(_ScriptedGen(["reply", "wander again"]))
        rt.observe("a question")
        r1 = rt.tick()
        assert r1.wandering is False and r1.action == "respond"
        r2 = rt.tick()
        assert r2.wandering is True and r2.action in ("speak", "think")


class TestIITFlavoredMetrics:
    """'add more IIT metrics' — honest proxies computed from what the
    cognition layer actually has (candidate NLLs + the selection
    softmax), explicitly NOT a rigorous Φ: selection_entropy (how
    decisive the basal-ganglia pick was — low = confident exclusion
    of alternatives) and differentiation (how distinguishable the
    candidate repertoire was — the spread of trunk NLL across
    options)."""

    def test_selection_entropy_in_unit_range(self):
        rt = _mk_runtime(_ScriptedGen(["a", "b", "c"]))
        r = rt.tick()
        assert 0.0 <= r.selection_entropy <= 1.0

    def test_single_candidate_has_zero_selection_entropy(self):
        from neuroslm.cognition.runtime import MindConfig
        cfg = MindConfig(n_candidates=1)
        rt = _mk_runtime(_ScriptedGen(["only option"]), cfg=cfg)
        r = rt.tick()
        assert r.selection_entropy == pytest.approx(0.0)

    def test_differentiation_reflects_nll_spread(self):
        from neuroslm.cognition.runtime import MindConfig
        cfg = MindConfig(n_candidates=2)
        gen = _ScriptedGen(["close one", "close two"])
        rt_close = _mk_runtime(
            gen, scores=_score_map({"close one": 3.0, "close two": 3.05}),
            cfg=cfg)
        r_close = rt_close.tick()
        gen2 = _ScriptedGen(["spread one", "spread two"])
        rt_spread = _mk_runtime(
            gen2, scores=_score_map({"spread one": 1.0, "spread two": 8.0}),
            cfg=cfg)
        r_spread = rt_spread.tick()
        assert r_spread.differentiation > r_close.differentiation


class TestFormatDebugTrace:
    """'debug log the actions generated by basal ganglia' — the full
    deliberation (every candidate + its score, which one won), not
    just the compact one-line summary format_introspection gives."""

    def _normal_result(self):
        from neuroslm.cognition.runtime import TickResult, ThoughtScore
        return TickResult(
            thought="good one", tick_n=3, prior_thought="previous idea",
            wandering=True, action="speak",
            candidates=["good one", "meh idea"],
            scores=[ThoughtScore(mean_nll=2.1, entropy_norm=0.4),
                    ThoughtScore(mean_nll=3.5, entropy_norm=0.6)],
            recalled=[{"content": "episode A"}, {"content": "episode B"}],
            stored=True, inhibited=False,
            nt_levels={"DA": 0.15, "NE": 0.20, "5HT": 0.50, "ACh": 0.30,
                      "eCB": 0.10, "Glu": 0.45, "GABA": 0.15},
            phi_proxy=0.4, selection_entropy=0.62, differentiation=0.99)

    def test_action_shown(self):
        from neuroslm.cognition.runtime import format_debug_trace
        s = format_debug_trace(self._normal_result())
        assert "action=speak" in s

    def test_nt_levels_shown(self):
        from neuroslm.cognition.runtime import format_debug_trace
        s = format_debug_trace(self._normal_result())
        assert "GABA=0.15" in s

    def test_iit_metrics_shown(self):
        from neuroslm.cognition.runtime import format_debug_trace
        s = format_debug_trace(self._normal_result())
        assert "H=0.62" in s or "0.62" in s
        assert "0.99" in s

    def test_selected_thought_is_never_truncated(self):
        from neuroslm.cognition.runtime import format_debug_trace, TickResult, ThoughtScore
        long_thought = "x" * 250
        r = TickResult(
            thought=long_thought, tick_n=1, wandering=True, action="speak",
            candidates=[long_thought],
            scores=[ThoughtScore(mean_nll=2.0, entropy_norm=0.3)],
            nt_levels={"GABA": 0.1})
        s = format_debug_trace(r)
        assert long_thought in s, (
            "the WINNING thought must render in full — only the "
            "rejected alternatives in the menu get truncated")

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


class TestEpisodicRepresentation:
    """'Don't make HC remember what BRIAN said. Make HC remember what
    happened to BRIAN, what state BRIAN was in, what BRIAN noticed,
    what BRIAN thought, and what resulted from it.'

    Phases 1+2 of the proposed roadmap (episodic representation +
    context/state metadata) — reusing EpisodicMemory's EXISTING
    tags/context slots (present in the schema, never populated by the
    cognition runtime) rather than inventing a parallel Episode type.
    Phases 3-5 (graph relationships, situation/trigger-based
    retrieval operations, consolidation into semantic memory) are
    NOT built here — this is the data model those would need, not
    those mechanisms themselves.
    """

    def test_observed_percept_is_tagged_and_contextualised(self):
        rt = _mk_runtime(_ScriptedGen(["t"]))
        rt.observe("the launch code is BLUE-7", source="user")
        ep = rt.memory.all()[-1]
        assert "kind=observed" in ep["tags"]
        assert ep["context"]["kind"] == "observed"
        assert ep["context"]["source"] == "user"

    def test_inferred_thought_carries_full_context_layer(self):
        from neuroslm.cognition.runtime import MindConfig
        cfg = MindConfig(n_candidates=1)
        rt = _mk_runtime(_ScriptedGen(["a fresh idea"]), cfg=cfg)
        rt.tick()
        ep = [e for e in rt.memory.all() if e["content"] == "a fresh idea"][0]
        ctx = ep["context"]
        assert ctx["kind"] == "inferred"
        assert ctx["action"] in ("speak", "think", "respond")
        assert ctx["wandering"] is True
        assert "trigger" in ctx and ctx["trigger"]
        assert "confidence" in ctx and 0.0 <= ctx["confidence"] <= 1.0
        assert "phi_proxy" in ctx
        assert "selection_entropy" in ctx
        assert "differentiation" in ctx

    def test_associations_record_which_episodes_shaped_this_thought(self):
        from neuroslm.cognition.runtime import MindConfig
        cfg = MindConfig(n_candidates=1, recall_k=2)
        rt = _mk_runtime(_ScriptedGen(["first", "second"]), cfg=cfg)
        rt.tick()
        rt.tick()
        ep = rt.memory.all()[-1]
        assert isinstance(ep["context"]["associations"], list)

    def test_respond_trigger_is_the_actual_user_text(self):
        # Distinct embedding axes (launch vs coffee) so the reply
        # passes the semantic novelty gate and actually gets stored.
        rt = _mk_runtime(_ScriptedGen(["coffee first, then work"]))
        rt.observe("what is the launch plan?")
        rt.tick()
        ep = [e for e in rt.memory.all()
             if e["content"] == "coffee first, then work"][0]
        assert ep["context"]["trigger"] == "what is the launch plan?"

    def test_wandering_trigger_is_the_wander_prompt_used(self):
        from neuroslm.cognition.runtime import MindConfig
        # storage-dependent contract, independent of §14.13 ignition
        # (pinned in TestOscillatoryWorkspace) — workspace off.
        rt = _mk_runtime(_ScriptedGen(["a wander"]),
                         cfg=MindConfig(n_candidates=2,
                                        workspace_oscillator=False))
        r = rt.tick()
        ep = [e for e in rt.memory.all() if e["content"] == "a wander"][0]
        assert ep["context"]["trigger"] in rt.cfg.wander_prompts

    def test_recalled_episodes_labelled_observed_vs_inferred_in_debug_trace(self):
        from neuroslm.cognition.runtime import format_debug_trace, TickResult, ThoughtScore
        r = TickResult(
            thought="x", tick_n=1, wandering=True, action="speak",
            candidates=["x"], scores=[ThoughtScore(mean_nll=2.0, entropy_norm=0.3)],
            recalled=[
                {"content": "a fact I was told", "context": {"kind": "observed"}},
                {"content": "a thought I had", "context": {"kind": "inferred"}},
            ],
            nt_levels={"GABA": 0.1})
        s = format_debug_trace(r)
        assert "[observed]" in s and "[inferred]" in s


class TestSemanticNoveltyGate:
    """A of the curiosity loop: the hippocampal write gate measures
    SEMANTIC novelty (1 − max cosine vs stored episodes, using the
    embedding machinery that already exists) instead of NLL-distance
    from an EMA. Live evidence for the change: ticks 2-9 on box
    47509954 showed NLL flat in a 3.0-3.9 band while the CONTENT
    orbited one topic — the NLL gate saw 'novelty' where there was
    none, wrote nothing (`write=no` nine ticks straight), and recall
    served the same single episode forever."""

    def test_first_thought_into_empty_memory_is_stored(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["the launch is tomorrow"]),
                         cfg=MindConfig(n_candidates=1))
        r = rt.tick()
        assert r.stored is True
        assert r.novelty == pytest.approx(1.0), (
            "nothing stored yet — maximal novelty by definition")

    def test_identical_repetition_is_not_stored(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["the launch is tomorrow"]),
                         cfg=MindConfig(n_candidates=1))
        rt.tick()
        r2 = rt.tick()
        assert r2.novelty == pytest.approx(0.0, abs=1e-6), (
            "identical text embeds identically — semantic novelty 0")
        assert r2.stored is False
        assert r2.action == "think"

    def test_semantically_distinct_thought_is_stored_again(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(
            _ScriptedGen(["the launch is tomorrow",
                         "I want coffee and music"]),
            cfg=MindConfig(n_candidates=1))
        rt.tick()
        r2 = rt.tick()
        assert r2.novelty > rt.cfg.novelty_write_threshold
        assert r2.stored is True
        assert r2.action == "speak"

    def test_novelty_reported_in_unit_range(self):
        rt = _mk_runtime(_ScriptedGen(["a thought"]))
        r = rt.tick()
        assert 0.0 <= r.novelty <= 1.0


class TestBoredomCuriosityLoop:
    """B of the curiosity loop: falling novelty accumulates into
    boredom, and boredom raises the basal ganglia's exploration
    temperature THROUGH the existing DA→T path's multiplier — the
    exploration knob that existed all along but was never driven
    (NT pinned at baseline for nine straight live ticks)."""

    def test_boredom_rises_over_repetitive_ticks(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["same thought forever"]),
                         cfg=MindConfig(n_candidates=1))
        first = rt.tick()
        for _ in range(4):
            last = rt.tick()
        assert last.boredom > first.boredom

    def test_curious_temperature_monotone_in_boredom(self):
        from neuroslm.cognition.runtime import (
            MindConfig, curious_selection_temperature,
        )
        cfg = MindConfig()
        t0 = curious_selection_temperature(0.15, 0.15, 0.0, cfg)
        t_mid = curious_selection_temperature(0.15, 0.15, 0.5, cfg)
        t_hi = curious_selection_temperature(0.15, 0.15, 1.0, cfg)
        assert t0 < t_mid < t_hi
        assert t0 == pytest.approx(cfg.selection_temp_base)

    def test_novelty_drives_the_nt_activation_channel(self):
        from neuroslm.cognition.runtime import MindConfig
        nt = _FakeNT()
        rt = _mk_runtime(_ScriptedGen(["a thought"]), nt=nt,
                         cfg=MindConfig(n_candidates=1))
        r = rt.tick()
        assert nt.step_calls[-1].get("activation") == pytest.approx(r.novelty), (
            "novel content is arousing — semantic novelty is the "
            "honest activation driver, replacing the entropy proxy")


class TestInhibitionOfReturn:
    """C1: recently-recalled episodes are transiently suppressed so
    the same memory can't dominate every tick (live evidence: the
    SAME [inferred] episode recalled nine ticks straight)."""

    @staticmethod
    def _seed(rt, *contents):
        # Seed memory WITHOUT queueing sensory input (observe() would
        # turn the next tick into a respond and defeat the wandering-
        # path assertions these tests make).
        for c in contents:
            rt.memory.add(c, content_vec=_vec_for(c),
                          context={"kind": "observed"})

    def test_same_episode_not_recalled_twice_in_a_row(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["thinking about the launch"]),
                         cfg=MindConfig(n_candidates=1, recall_k=1))
        self._seed(rt, "the launch code is ready",
                  "coffee tastes great today")
        rt._last_thought = "thinking about launch stuff"  # launch anchor
        r1 = rt.tick()
        assert any("launch code" in e["content"] for e in r1.recalled)
        r2 = rt.tick()   # A now inhibited → B gets its turn
        assert not any("launch code" in e["content"] for e in r2.recalled), (
            "inhibition of return: the episode recalled last tick "
            "must be suppressed this tick")

    def test_ior_yields_rather_than_blinds(self):
        """When ONLY suppressed episodes exist, IOR must yield (return
        the best match anyway) — an empty recall would be worse than a
        repeated one."""
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["thinking about the launch"]),
                         cfg=MindConfig(n_candidates=1, recall_k=1))
        self._seed(rt, "the launch code is ready")  # the ONLY episode
        r1 = rt.tick()
        r2 = rt.tick()
        assert r2.recalled, "sole episode must still be recallable"

    def test_respond_ticks_are_exempt(self):
        """IOR is a DMN anti-fixation device. On a respond tick the
        most relevant memory must be available even if it was recalled
        recently — found by the §14.11 episodic_recall probe: a fact
        recalled on its own percept tick was still suppressed five
        ticks later when the user asked about it."""
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["ok"]),
                         cfg=MindConfig(n_candidates=1, recall_k=1))
        rt.observe("the launch code is ready")
        rt.tick()      # respond tick recalls the fresh launch percept
        rt.observe("coffee tastes great today")
        rt.tick()
        rt.observe("what about the launch code?")
        r = rt.tick()
        assert any("launch code is ready" in e["content"]
                   for e in r.recalled), (
            "respond ticks must not inhibit the relevant memory")

    def test_respond_tick_never_recalls_its_own_percept(self):
        """observe() stores the percept BEFORE the tick, so it is its
        own nearest neighbour (cos 1). It is already in the prompt as
        'Just now:' — recalling it too wastes a recall slot on a
        duplicate and displaces the memory the input is about."""
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["ok"]),
                         cfg=MindConfig(n_candidates=1, recall_k=1))
        rt.memory.add("the launch code is ready",
                      content_vec=_vec_for("the launch code is ready"),
                      context={"kind": "observed"})
        # launch+moon axes: cos(percept, itself)=1 > cos(percept, fact)
        rt.observe("is the launch code ready for the moon?")
        r = rt.tick()
        contents = [e["content"] for e in r.recalled]
        assert "is the launch code ready for the moon?" not in contents
        assert contents == ["the launch code is ready"]


class TestReplayAnchoredWandering:
    """C2: when bored, the wander tick anchors RECALL on a randomly
    sampled stored episode instead of the last thought — hippocampal
    replay, the mechanism behind associative jumps. Without it the
    anchor chain last_thought→similarity→same basin never breaks."""

    def test_replay_fires_when_bored(self):
        from neuroslm.cognition.runtime import MindConfig
        cfg = MindConfig(n_candidates=1)
        rt = _mk_runtime(_ScriptedGen(["same thought forever"]), cfg=cfg)
        TestInhibitionOfReturn._seed(rt, "the launch code is ready",
                                     "coffee tastes great today")
        rt._boredom = 1.0   # force the bored state directly
        r = rt.tick()
        assert r.wandering is True
        assert r.replay is True

    def test_no_replay_when_engaged(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["a fresh thought"]),
                         cfg=MindConfig(n_candidates=1))
        TestInhibitionOfReturn._seed(rt, "the launch code is ready")
        r = rt.tick()   # boredom starts at 0 — engaged wandering
        assert r.wandering is True
        assert r.replay is False

    def test_no_replay_during_respond(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["a reply"]),
                         cfg=MindConfig(n_candidates=1))
        rt.observe("what is the plan?")
        rt._boredom = 1.0
        r = rt.tick()
        assert r.action == "respond" and r.replay is False, (
            "replay is a DMN mechanism — answering real input must "
            "stay anchored on the input")


class TestProvenanceAwareRetrieval:
    """Control fix 1 (2026-08-12 log analysis): retrieval was
    provenance-blind — an [inferred] musing and an [observed] fact
    competed as equals on cosine alone, so the mind's own prose
    out-retrieved reality ('autobiographical contamination'). The
    stored kind now carries a retrieval penalty for self-generated
    episodes."""

    def _rt(self, cfg=None):
        from neuroslm.cognition.runtime import CognitiveRuntime, MindConfig
        return CognitiveRuntime(
            generate_fn=_ScriptedGen(["a thought"]),
            score_fn=_score_map({}),
            embed_fn=lambda t: [1.0, 0.0],   # constant anchor
            nt=_FakeNT(), memory=EpisodicMemory(maxlen=64),
            cfg=cfg or MindConfig(n_candidates=1, recall_k=1),
            rng=random.Random(0))

    def test_observed_beats_inferred_at_equal_cosine(self):
        rt = self._rt()
        rt.memory.add("my own musing", content_vec=[1.0, 0.0],
                      context={"kind": "inferred"})
        rt.memory.add("a real fact", content_vec=[1.0, 0.0],
                      context={"kind": "observed"})
        r = rt.tick()
        assert [e["content"] for e in r.recalled] == ["a real fact"]

    def test_penalty_outweighs_a_small_cosine_edge(self):
        rt = self._rt()
        # inferred is a PERFECT cosine match; observed is slightly off
        # — the self-generated penalty (default 0.15) must flip it.
        rt.memory.add("my own musing", content_vec=[1.0, 0.0],
                      context={"kind": "inferred"})
        rt.memory.add("a real fact", content_vec=[0.95, 0.312],
                      context={"kind": "observed"})
        r = rt.tick()
        assert [e["content"] for e in r.recalled] == ["a real fact"]

    def test_a_large_cosine_edge_still_wins(self):
        rt = self._rt()
        # provenance is a thumb on the scale, not a veto: a strongly
        # more relevant inferred episode must still be retrievable.
        rt.memory.add("my own musing", content_vec=[1.0, 0.0],
                      context={"kind": "inferred"})
        rt.memory.add("a real fact", content_vec=[0.0, 1.0],
                      context={"kind": "observed"})
        r = rt.tick()
        assert [e["content"] for e in r.recalled] == ["my own musing"]


class TestReplayPulse:
    """Control fix 2: live trace showed persistent '(replay)' from
    tick 10 onward — boredom crossed the threshold and STAYED there,
    so replay fired every tick and became a drift amplifier. Replay
    is a pulse, not a state: the jump consumes the boredom, and a
    refractory window spaces jumps out."""

    def _bored_rt(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["same thought forever"]),
                         cfg=MindConfig(n_candidates=1))
        TestInhibitionOfReturn._seed(rt, "the launch code is ready",
                                     "coffee tastes great today")
        rt._boredom = 1.0
        return rt

    def test_replay_consumes_boredom(self):
        rt = self._bored_rt()
        r = rt.tick()
        assert r.replay is True
        assert r.boredom < rt.cfg.replay_boredom_threshold, (
            "the associative jump must RELIEVE boredom — otherwise "
            "replay locks on and fires every tick")

    def test_refractory_blocks_back_to_back_replay(self):
        rt = self._bored_rt()
        r1 = rt.tick()
        assert r1.replay is True
        rt._boredom = 1.0   # even if boredom is forced straight back up
        r2 = rt.tick()
        assert r2.replay is False, (
            "a second jump within the refractory window must not fire")

    def test_replay_returns_after_refractory_expires(self):
        rt = self._bored_rt()
        rt.tick()
        rt._boredom = 1.0
        rt._last_replay_tick = -100   # simulate the window expiring
        r = rt.tick()
        assert r.replay is True


class TestReorientPolicy:
    """Control fix 3: boredom sensed the loop but the policy only ever
    answered with MORE generation. When novelty stays low for
    `reorient_after` consecutive ticks (= the LOOPING condition), the
    next wandering tick is a forced state transition: anchor on the
    last OBSERVED input if one exists, else a clean-slate tick with
    recall suppressed — not another continuation of the loop."""

    def _looping_rt(self, observe_first=None):
        from neuroslm.cognition.runtime import MindConfig
        cfg = MindConfig(n_candidates=1, reorient_after=2)
        rt = _mk_runtime(_ScriptedGen(["same thought forever"]), cfg=cfg)
        if observe_first:
            rt.observe(observe_first)
            # The respond tick generates+stores "same thought forever"
            # too (streak update is unconditional on action), so it
            # already plays the "novelty 1.0 or high — streak 0" role
            # below — one fewer generic tick is needed to reach the
            # reorient_after boundary.
            rt.tick()      # consume the sensory queue (respond tick)
            rt.tick()      # identical -> novelty 0 -> streak 1
        else:
            rt.tick()          # novelty 1.0 or high — streak 0
            rt.tick()          # identical -> novelty 0 -> streak 1
        rt.tick()          # streak 2 == reorient_after
        return rt

    def test_loop_triggers_reorient(self):
        rt = self._looping_rt()
        r = rt.tick()
        assert r.reorient is True
        assert r.action == "reorient"

    def test_reorient_anchors_on_last_observed_input(self):
        rt = self._looping_rt(observe_first="the launch plan matters")
        r = rt.tick()
        assert r.reorient is True
        assert any("launch plan" in e["content"] for e in r.recalled), (
            "RETURN_TO_LAST_USER_INPUT: the reorient tick must anchor "
            "recall on the last observed episode, not the loop's own "
            "last thought")

    def test_reorient_without_observed_input_is_clean_slate(self):
        rt = self._looping_rt()
        r = rt.tick()
        assert r.reorient is True
        assert r.recalled == [], (
            "no observed episode to return to — reorient suppresses "
            "recall entirely rather than re-feeding the loop's own "
            "stored thoughts")

    def test_streak_resets_after_reorient(self):
        rt = self._looping_rt()
        rt.tick()           # the reorient tick
        r_next = rt.tick()  # streak was reset — no immediate re-trigger
        assert r_next.reorient is False

    def test_respond_never_reorients(self):
        rt = self._looping_rt()
        rt.observe("a direct question arrives")
        r = rt.tick()
        assert r.action == "respond"
        assert r.reorient is False


class TestPerceptUtilityGate:
    """Control fix 4: observe() stored EVERY percept ('external events
    are salient by default') — so 'Hellohello' became a permanent
    autobiographical anchor that replay later resurfaced. Trivial
    percepts are still PROCESSED (queued as sensory input, replied
    to) but not committed to episodic memory."""

    def _stored_contents(self, rt):
        return [e["content"] for e in rt.memory.all()]

    def test_one_word_statement_is_not_stored(self):
        rt = _mk_runtime(_ScriptedGen(["t"]))
        rt.observe("Hellohello")
        assert "Hellohello" not in self._stored_contents(rt)

    def test_short_greeting_is_not_stored(self):
        rt = _mk_runtime(_ScriptedGen(["t"]))
        rt.observe("Hey, good morning")
        assert "Hey, good morning" not in self._stored_contents(rt)

    def test_contentful_greeting_is_stored(self):
        rt = _mk_runtime(_ScriptedGen(["t"]))
        text = "Hello, my name is Moritz and I am building a mind"
        rt.observe(text)
        assert text in self._stored_contents(rt)

    def test_short_but_salient_insult_is_stored(self):
        rt = _mk_runtime(_ScriptedGen(["t"]))
        rt.observe("You suck")
        assert "You suck" in self._stored_contents(rt), (
            "length is not salience — a two-word insult is an "
            "emotionally salient event and belongs in memory")

    def test_ungated_percept_still_reaches_the_sensory_queue(self):
        gen = _ScriptedGen(["a reply"])
        rt = _mk_runtime(gen)
        rt.observe("Hellohello")
        r = rt.tick()
        assert r.action == "respond", (
            "gating STORAGE must not gate PROCESSING — a trivial "
            "greeting still gets responded to")
        assert "Hellohello" in gen.prompts[0]


class TestInnerSpeechRegister:
    """D1: DMN ticks generate through a separate wander seam (raw
    completion — no chat template, no imaginary addressee); respond
    ticks keep the chat seam. Live evidence: every wandering thought
    opened 'Brian, ...' or 'Thank you! Let's continue...' — the chat
    template made the instruct model roleplay a conversation instead
    of thinking."""

    def test_wander_tick_uses_the_wander_seam(self):
        from neuroslm.cognition.runtime import CognitiveRuntime, MindConfig
        chat_gen = _ScriptedGen(["chat output"])
        wander_gen = _ScriptedGen(["inner monologue output"])
        rt = CognitiveRuntime(
            generate_fn=chat_gen, score_fn=_score_map({}),
            embed_fn=_vec_for, nt=_FakeNT(),
            memory=EpisodicMemory(maxlen=64), rng=random.Random(0),
            cfg=MindConfig(n_candidates=1),
            generate_wander_fn=wander_gen)
        r = rt.tick()
        assert r.thought == "inner monologue output"
        assert wander_gen.calls == 1 and chat_gen.calls == 0

    def test_respond_tick_uses_the_chat_seam(self):
        from neuroslm.cognition.runtime import CognitiveRuntime, MindConfig
        chat_gen = _ScriptedGen(["chat output"])
        wander_gen = _ScriptedGen(["inner monologue output"])
        rt = CognitiveRuntime(
            generate_fn=chat_gen, score_fn=_score_map({}),
            embed_fn=_vec_for, nt=_FakeNT(),
            memory=EpisodicMemory(maxlen=64), rng=random.Random(0),
            cfg=MindConfig(n_candidates=1),
            generate_wander_fn=wander_gen)
        rt.observe("hello there")
        r = rt.tick()
        assert r.thought == "chat output"
        assert chat_gen.calls == 1 and wander_gen.calls == 0

    def test_wander_seam_defaults_to_the_main_seam(self):
        rt = _mk_runtime(_ScriptedGen(["t"]))
        assert rt._gen_wander is rt._gen


class TestSecondPersonPrior:
    """D2: a BG prior penalizes second-person address in WANDERING
    candidates (inner speech has no addressee) — a value-shaping term
    on the existing −NLL/T utility, exactly how action priors enter
    basal-ganglia models. Never applied to respond ticks, where an
    addressee genuinely exists."""

    def test_wandering_prefers_the_first_person_candidate(self):
        from neuroslm.cognition.runtime import MindConfig
        cands = ["You should think about this, Brian",
                "I keep coming back to the launch"]
        cfg = MindConfig(n_candidates=2, selection_temp_base=0.01)
        rt = _mk_runtime(
            _ScriptedGen(cands),
            scores=_score_map({c: 3.0 for c in cands}), cfg=cfg)
        r = rt.tick()
        assert r.thought == "I keep coming back to the launch", (
            "equal NLL — the second-person candidate must lose on the "
            "inner-speech prior alone")

    def test_respond_does_not_penalize_second_person(self):
        from neuroslm.cognition.runtime import MindConfig
        cands = ["You are right about that", "I am not sure"]
        cfg = MindConfig(n_candidates=2, selection_temp_base=0.01)
        rt = _mk_runtime(
            _ScriptedGen(cands),
            scores=_score_map({"You are right about that": 2.0,
                              "I am not sure": 3.0}), cfg=cfg)
        rt.observe("was I right?")
        r = rt.tick()
        assert r.thought == "You are right about that", (
            "a reply legitimately addresses someone — the prior must "
            "not apply on respond ticks")


class TestClassifyFnInjection:
    """The mind classifies with its OWN trunk/expert when one is
    wired in (classify_fn=...); the regex lexicon is the fallback
    used only when nothing is injected — never the other way round."""

    def test_default_runtime_uses_lexicon_classifier(self):
        rt = _mk_runtime(_ScriptedGen(["t"]))
        rt.observe("You suck")
        assert rt.memory.all()[-1]["context"]["action_class"] == "insult"

    def test_injected_classify_fn_is_used_instead(self):
        from neuroslm.cognition.runtime import CognitiveRuntime, ActionClassification
        calls = []

        def custom_classify(text):
            calls.append(text)
            return ActionClassification(primary="request", candidates=["request"])

        rt = CognitiveRuntime(
            generate_fn=_ScriptedGen(["t"]), score_fn=_score_map({}),
            embed_fn=_vec_for, nt=_FakeNT(),
            memory=EpisodicMemory(maxlen=64), rng=random.Random(0),
            classify_fn=custom_classify)
        rt.observe("You suck")
        assert calls == ["You suck"]
        assert rt.memory.all()[-1]["context"]["action_class"] == "request", (
            "an injected classifier must override the lexicon default")


class TestActionClassStoredOnEpisodes:
    """Every stored episode carries its generalized action class
    alongside the literal text — "You suck" stores BOTH the literal
    utterance and action_class="insult"."""

    def test_observed_percept_carries_action_class(self):
        rt = _mk_runtime(_ScriptedGen(["t"]))
        rt.observe("You suck")
        ep = rt.memory.all()[-1]
        assert ep["context"]["action_class"] == "insult"

    def test_inferred_thought_carries_action_class(self):
        from neuroslm.cognition.runtime import MindConfig
        cfg = MindConfig(n_candidates=1)
        rt = _mk_runtime(_ScriptedGen(["Thanks for that"]), cfg=cfg)
        rt.tick()
        ep = [e for e in rt.memory.all()
             if e["content"] == "Thanks for that"][0]
        assert ep["context"]["action_class"] == "gratitude"


class TestDetectPatterns:
    """'It should learn causal and temporal relations... when it
    observes enough times that insulting causes a negative response
    it should learn that.' Real implementation: Apriori-derived
    association mining over the OBSERVED+INFERRED action-class
    sequence already stored — see neuroslm/cognition/patterns.py for
    why this is honestly 'association', not 'causation'."""

    def test_detect_patterns_mines_the_stored_episode_sequence(self):
        rt = _mk_runtime(_ScriptedGen(["t"]))
        for _ in range(4):
            rt.observe("You suck")
            rt.observe("No, that's wrong")
        rules = rt.detect_patterns(min_confidence=0.0)
        assert any(r.antecedent == "insult" and r.consequent == "disagreement"
                  for r in rules)

    def test_detect_patterns_returns_association_rule_objects(self):
        from neuroslm.cognition.patterns import AssociationRule
        rt = _mk_runtime(_ScriptedGen(["t"]))
        rt.observe("Hello")
        rt.observe("Hi there")
        rules = rt.detect_patterns(min_confidence=0.0)
        assert all(isinstance(r, AssociationRule) for r in rules)

    def test_empty_memory_returns_no_rules(self):
        rt = _mk_runtime(_ScriptedGen(["t"]))
        assert rt.detect_patterns() == []


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
        from neuroslm.cognition.runtime import MindConfig
        gen = _ScriptedGen(["mind thought"])
        # storage-dependent contract, independent of §14.13 ignition
        # (pinned in TestOscillatoryWorkspace) — workspace off.
        rt = _mk_runtime(gen, cfg=MindConfig(n_candidates=2,
                                             workspace_oscillator=False))
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

    def test_think_action_tick_is_not_posted_to_the_thoughts_pane(self):
        """A 'think' action (idle, repetitive/converged — the novelty
        gate rejected it) stays internal: it must NOT appear as a
        first-class 'thought' in the dashboard/memory, even though it
        still fully happened and is fully logged."""
        from neuroslm.cognition.runtime import MindConfig
        cfg = MindConfig(n_candidates=1)
        gen = _ScriptedGen(["repetitive idea"])
        rt = _mk_runtime(
            gen, scores=_score_map({"repetitive idea": 3.0}), cfg=cfg)
        d = self._daemon(rt)
        d.think_once()   # bootstrap tick: always novel -> speak
        before = len(d.memory.recent(20, kinds=("thought",)))
        d.think_once()  # identical text -> zero semantic novelty -> think
        assert d.last_tick.action == "think"
        after = len(d.memory.recent(20, kinds=("thought",)))
        assert after == before, (
            "an internal 'think' action must not be posted as a "
            "user-visible thought")
        # but it's still in the debug channel:
        introspects = d.memory.recent(20, kinds=("introspect",))
        assert introspects, "internal thinking must still be logged"

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

    def test_respond_routes_through_the_mind_when_attached(self):
        """'respond action in case of external input' — a real basal-
        ganglia act, not telemetry synthesised around a bypassed
        generation path. respond() with a mind attached must actually
        run mind.tick(), not the legacy direct-_gen seam."""
        gen = _ScriptedGen(["a considered reply"])
        rt = _mk_runtime(gen)
        d = self._daemon(rt)
        reply = d.respond("what is the plan?")
        assert reply == "a considered reply"
        assert d.last_tick is not None
        assert d.last_tick.action == "respond"
        assert d.last_tick.wandering is False
        kinds = [e.kind for e in d.memory.recent(8)]
        assert "reply" in kinds and "introspect" in kinds

    def test_respond_without_mind_keeps_legacy_path(self):
        from neuroslm.chat_daemon import ChatDaemon, ChatDaemonConfig
        d = ChatDaemon(_ScriptedGen(["legacy reply"]),
                       ChatDaemonConfig(), use_color=False)
        assert d.respond("hi") == "legacy reply"

    def test_dmn_resumes_wandering_after_a_respond_call(self):
        """The spontaneous-re-entry claim, exercised end to end
        through the daemon: after respond() consumes the sensory
        queue, the NEXT background think_once() tick must be idle
        wandering again — no special-case 'resume' code needed, it's
        emergent from the queue draining."""
        gen = _ScriptedGen(["a reply", "a wandering thought"])
        rt = _mk_runtime(gen)
        d = self._daemon(rt)
        d.respond("a question")
        assert d.last_tick.wandering is False
        d.think_once()
        assert d.last_tick.wandering is True


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

    def test_wires_a_generation_based_classifier_not_the_bare_lexicon(self):
        """The mind classifies with its OWN trunk/expert — production
        wiring must NOT fall back to the bare regex lexicon by
        default; that's the test-only default, not what a real
        deployed mind uses."""
        from neuroslm.cognition.patterns import classify_action
        rt = self._rt()
        assert rt._classify is not classify_action
        # still returns a sane, well-typed result even against the
        # gibberish the fake model generates (safety-net fallback).
        from neuroslm.cognition.patterns import ACTION_TAXONOMY
        result = rt._classify("You suck")
        assert result.primary in set(ACTION_TAXONOMY) | {"statement"}

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
        rt.observe("alpha beta gamma delta epsilon")
        eps = rt.memory.all()
        assert eps and eps[-1]["content_vec"] is not None
        assert len(eps[-1]["content_vec"]) == 4, (
            "episodic vectors must come from the expert's own "
            "embedding rows (d=4 in the fake)")

    def test_chat_expert_flag_parses(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(["chat", "--expert"])
        from neuroslm.cognition.runtime import DEFAULT_MIND_EXPERT
        assert args.expert == DEFAULT_MIND_EXPERT
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


class _FakeChatTokenizerV5(_FakeChatTokenizer):
    """transformers >= 5: apply_chat_template(tokenize=True) returns a
    dict-like BatchEncoding, not a list of ids."""

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=True, **kw):
        ids = super().apply_chat_template(messages, add_generation_prompt,
                                          tokenize)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


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

    def test_chat_template_accepts_transformers_v5_batch_encoding(self):
        """Fresh installs resolve transformers 5.x, where the old
        list(apply_chat_template(...)) yielded the dict's KEYS and
        crashed every respond tick (found by the first real
        `brian eval mind` run)."""
        rt, model = self._rt(_FakeChatTokenizerV5)
        out = rt._gen("hello", 3)
        assert out == _FakeChatTokenizer().decode([7, 7, 7])


# ── Reasoning experts (DeepSeek-R1 family): think trace as introspection

class _CharTokenizer:
    """Char-level tokenizer: text round-trips exactly, so a scripted
    model can emit precise <think> sequences."""
    eos_token_id = 0
    chat_template = "{% generation %}<think>\n"   # R1-style marker

    def encode(self, text):
        return [ord(c) % 256 for c in (text or " ")]

    def decode(self, ids, **kw):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=True, **kw):
        return {"input_ids": self.encode(
            f"<u>{messages[-1]['content']}<a><think>\n")}


class _ScriptedReasoningModel(_FakeHFModel):
    """generate() emits scripted text per call, cut at max_new_tokens."""

    def __init__(self, scripts):
        import torch
        super().__init__()
        self.config = type("C", (), {"n_positions": 8192})()
        self._emb = torch.randn(256, 4)
        self.scripts = list(scripts)
        self.calls = []

    def __call__(self, input_ids=None):
        import torch
        B, T = input_ids.shape
        return self._Out(torch.zeros(B, T, 256))

    def generate(self, input_ids=None, **kw):
        import torch
        self.calls.append((input_ids[0].tolist(), kw))
        text = self.scripts[(len(self.calls) - 1) % len(self.scripts)]
        new = [ord(c) for c in text][: int(kw["max_new_tokens"])]
        return torch.cat([input_ids, torch.tensor([new])], dim=-1)


def _reasoning_seams(scripts, budget=32):
    from neuroslm.cognition.runtime import build_hf_seams
    model = _ScriptedReasoningModel(scripts)
    seams = build_hf_seams("fake/r1", model_factory=lambda: model,
                           tokenizer_factory=_CharTokenizer,
                           reasoning_budget=budget)
    return seams, model


class TestSplitReasoning:
    def test_closed_block_with_opening_tag(self):
        from neuroslm.cognition.runtime import split_reasoning
        assert split_reasoning("<think>hmm, rivers</think>\n\nRivers bend.") \
            == ("Rivers bend.", "hmm, rivers")

    def test_template_opened_block(self):
        """R1 chat templates put '<think>\n' in the PROMPT, so the
        output starts inside the block."""
        from neuroslm.cognition.runtime import split_reasoning
        assert split_reasoning("hmm</think>Answer.") == ("Answer.", "hmm")

    def test_unterminated_block_has_no_answer(self):
        from neuroslm.cognition.runtime import split_reasoning
        assert split_reasoning("<think>still going") == ("", "still going")

    def test_plain_text_is_all_answer(self):
        from neuroslm.cognition.runtime import split_reasoning
        assert split_reasoning("Just a reply.") == ("Just a reply.", "")


class TestReasoningExpert:
    def test_closed_think_returns_answer_and_records_trace(self):
        seams, model = _reasoning_seams(["I wonder.</think>\n\nHello there."],
                                        budget=64)
        assert seams.generate_fn("hi", 16) == "Hello there."
        assert seams.reasoning_fn() == "I wonder."
        assert len(model.calls) == 1
        assert model.calls[0][1]["max_new_tokens"] == 64 + 16

    def test_budget_forcing_closes_an_overlong_think(self):
        """s1-style budget forcing: an unterminated think block is
        closed by appending '</think>' and the answer is generated as
        a continuation."""
        seams, model = _reasoning_seams(
            ["thinking forever and ever and ever", "Final."], budget=10)
        assert seams.generate_fn("hi", 16) == "Final."
        assert seams.reasoning_fn() == "thinking f"
        assert len(model.calls) == 2
        second_prompt = "".join(chr(i) for i in model.calls[1][0])
        assert second_prompt.endswith("thinking f\n</think>\n\n")

    def test_non_reasoning_tokenizer_is_untouched(self):
        from neuroslm.cognition.runtime import build_hf_seams
        seams = build_hf_seams("fake/expert",
                               model_factory=_FakeHFModelWithGenerate,
                               tokenizer_factory=_FakeChatTokenizer)
        assert seams.reasoning_fn() is None

    def test_tick_surfaces_the_selected_thoughts_reasoning(self):
        from neuroslm.cognition.runtime import MindConfig, format_debug_trace
        seams, _ = _reasoning_seams(
            ["Why did they say that?</think>\n\nI hear you."], budget=64)
        rt = seams.runtime(cfg=MindConfig(n_candidates=2))
        rt.observe("tell me about rivers please")
        res = rt.tick()
        assert res.thought == "I hear you."
        assert res.reasoning == "Why did they say that?"
        assert "Why did they say that?" in format_debug_trace(res)


# ── §14.13: dead NT channels wired + oscillatory workspace ───────────

class TestNTDriversWired:
    """Live run 58670004: NE, ACh and GABA sat at baseline with sd=0
    for every tick — the loop only ever passed loss + activation."""

    def test_every_tick_passes_all_five_drivers(self):
        from neuroslm.cognition.runtime import MindConfig
        nt = _FakeNT()
        rt = _mk_runtime(_ScriptedGen(["thinking about the river"]), nt=nt,
                         cfg=MindConfig(n_candidates=2), osc=True)
        rt.tick()
        kw = nt.step_calls[-1]
        for k in ("loss", "activation", "grad_norm", "ignition_rate",
                  "attn_entropy_norm"):
            assert kw.get(k) is not None, k

    def test_drivers_carry_the_ticks_own_signals(self):
        from neuroslm.cognition.runtime import MindConfig
        nt = _FakeNT()
        rt = _mk_runtime(_ScriptedGen(["thinking about the river"]), nt=nt,
                         cfg=MindConfig(n_candidates=2), osc=True)
        r = rt.tick()
        kw = nt.step_calls[-1]
        assert kw["attn_entropy_norm"] == pytest.approx(r.selection_entropy)
        assert kw["ignition_rate"] == (1.0 if r.ignited else 0.0)
        # NE: novelty surprise vs a novelty expectation SEEDED with the
        # first observation — no cold-start transient (run 58676023:
        # NE drifted 0.20→0.16 as boredom warmed up from 0)
        assert kw["grad_norm"] == pytest.approx(0.0)

    def test_ignition_falls_back_to_storage_without_oscillator(self):
        from neuroslm.cognition.runtime import MindConfig
        nt = _FakeNT()
        rt = _mk_runtime(_ScriptedGen(["thinking about the river"]), nt=nt,
                         cfg=MindConfig(n_candidates=2,
                                        workspace_oscillator=False), osc=True)
        r = rt.tick()
        assert r.ignited is None
        assert nt.step_calls[-1]["ignition_rate"] == (1.0 if r.stored else 0.0)

    def test_real_nt_channels_now_move(self):
        from neuroslm.cognition.runtime import MindConfig
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        topics = ["code", "weather", "music", "launch", "coffee", "river",
                  "chess", "moon"]
        gen = _ScriptedGen([f"thinking about the {t}" for t in topics])
        nt = DrivenNTSystem()
        # driver wiring, not ignition: with the deterministic fakes the
        # workspace never ignites, nothing is stored and novelty (hence
        # the NE driver) would be constant by construction.
        rt = _mk_runtime(gen, nt=nt, cfg=MindConfig(
            n_candidates=2, workspace_oscillator=False), osc=True)
        seen = {k: set() for k in ("NE", "ACh", "GABA")}
        for _ in range(30):
            rt.tick()
            for k in seen:
                seen[k].add(round(nt.levels()[k], 4))
        for k, vals in seen.items():
            assert len(vals) > 3, f"{k} still flat: {vals}"


class TestOscillatoryWorkspace:
    def test_on_by_default(self):
        from neuroslm.cognition.runtime import MindConfig
        assert MindConfig().workspace_oscillator is True

    def test_tick_reports_workspace_state(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["thinking about the river"]),
                         cfg=MindConfig(n_candidates=2), osc=True)
        r = rt.tick()
        assert isinstance(r.ignited, bool)
        assert 0.0 <= r.sync_r <= 1.0
        assert r.gamma_peak_hz > 0.0
        assert 0.0 <= r.gamma_power <= 1.0

    def test_off_reports_nothing(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["thinking about the river"]),
                         cfg=MindConfig(n_candidates=2,
                                        workspace_oscillator=False), osc=True)
        r = rt.tick()
        assert r.ignited is None and r.sync_r is None

    def test_no_ignition_means_no_broadcast(self):
        """A novel thought that fails to ignite stays local: not
        written to memory, not spoken."""
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["thinking about the river"]),
                         cfg=MindConfig(n_candidates=2,
                                        osc_ignition_threshold=1.01), osc=True)
        r = rt.tick()
        assert r.novelty >= rt.cfg.novelty_write_threshold
        assert r.ignited is False
        assert r.stored is False and r.action == "think"
        assert not rt.memory.all()

    def test_ignition_plus_novelty_broadcasts(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["thinking about the river"]),
                         cfg=MindConfig(n_candidates=2,
                                        osc_ignition_mode="fixed",
                                        osc_ignition_threshold=0.0), osc=True)
        r = rt.tick()
        assert r.ignited is True and r.stored is True
        assert r.action == "speak"

    def test_debug_trace_shows_the_workspace(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["thinking about the river"]),
                         cfg=MindConfig(n_candidates=2), osc=True)
        text = format_debug_trace_of(rt.tick())
        assert "WS[" in text and "Hz" in text


def format_debug_trace_of(r):
    from neuroslm.cognition.runtime import format_debug_trace
    return format_debug_trace(r)


class TestDeepSeekDefaultTrunk:
    """2026-11-01: the mind's default trunk is a DeepSeek-R1 distill
    (thought quality on SmolLM2-360M was the floor in run 58670004).
    One constant, every mind entry point — no drift between them."""

    def test_default_is_a_deepseek_r1_distill(self):
        from neuroslm.cognition.runtime import DEFAULT_MIND_EXPERT
        from neuroslm.experts import resolve_expert_alias
        assert resolve_expert_alias(DEFAULT_MIND_EXPERT).startswith(
            "deepseek-ai/DeepSeek-R1-Distill")

    def test_every_mind_entry_point_uses_it(self):
        import inspect
        from neuroslm.cli import _build_parser
        from neuroslm.cognition.runtime import (
            build_hf_seams, build_runtime_from_hf_lm)
        from neuroslm.connectors.vast_mind import MindDeployConfig
        from neuroslm.connectors.vast_mind_eval import MindEvalDeployConfig
        from neuroslm.cognition.runtime import DEFAULT_MIND_EXPERT as D
        p = _build_parser()
        assert p.parse_args(["chat", "--expert"]).expert == D
        assert p.parse_args(["deploy-mind"]).expert == D
        assert p.parse_args(["eval", "mind"]).expert == D
        assert p.parse_args(["deploy-mind-eval"]).expert == D
        assert MindDeployConfig().expert == D
        assert MindEvalDeployConfig().expert == D
        for fn in (build_hf_seams, build_runtime_from_hf_lm):
            assert inspect.signature(fn).parameters["model_id"].default == D

    def test_gpu_loads_in_the_checkpoint_dtype_cpu_in_fp32(self):
        """transformers 4.x defaults to fp32 — a 7B model would double
        its VRAM and halve throughput on the A100."""
        from neuroslm.cognition.runtime import _hf_load_kwargs
        # `dtype`, not `torch_dtype`: run 58673415 (transformers 4.57)
        # logged "`torch_dtype` is deprecated! Use `dtype` instead!"
        assert _hf_load_kwargs("cuda") == {"dtype": "auto"}
        assert _hf_load_kwargs("cpu") == {}



# ── Fixes from live run 58673415 (R1-7B + oscillator) ───────────────

class TestEmbeddingsAreMeanCentred:
    """Run 58673415: 16/16 ticks 'think' — tick 2 ignited yet was not
    stored, so the novelty gate rejected a math thought against river
    memories. Mean-pooled INPUT embeddings share a large common
    component (anisotropy), so raw cosines between unrelated texts sit
    high and the absolute 0.3 novelty threshold stops transferring
    across trunks. Fix: subtract the embedding-table mean (the
    standard anisotropy correction). Unverified on R1 until the next
    run — this pins the math only."""

    def test_embed_is_mean_pooled_rows_minus_table_mean(self):
        import torch
        from neuroslm.cognition.runtime import build_hf_seams
        model = _FakeHFModelWithGenerate()
        seams = build_hf_seams("fake/expert", model_factory=lambda: model,
                               tokenizer_factory=_FakeHFTokenizer)
        tok = _FakeHFTokenizer()
        ids = tok.encode("alpha beta")
        table = model._emb
        expected = table[torch.tensor(ids)].mean(0) - table.mean(0)
        got = torch.tensor(seams.embed_fn("alpha beta"))
        assert torch.allclose(got, expected, atol=1e-5)

    def test_the_common_component_no_longer_dominates_cosine(self):
        """A table whose rows share a big common offset: raw mean-pooled
        vectors of disjoint token sets have cos≈1; centred ones don't."""
        import torch
        from neuroslm.cognition.runtime import build_hf_seams
        model = _FakeHFModelWithGenerate()
        g = torch.Generator().manual_seed(0)
        model._emb = torch.randn(32, 4, generator=g) + 50.0

        class _Tok(_FakeHFTokenizer):
            def encode(self, text):
                return [1, 2, 3] if text == "a" else [20, 21, 22]
        seams = build_hf_seams("fake/expert", model_factory=lambda: model,
                               tokenizer_factory=_Tok)
        cos = torch.nn.functional.cosine_similarity
        raw = cos(model._emb[[1, 2, 3]].mean(0),
                  model._emb[[20, 21, 22]].mean(0), dim=0)
        cen = cos(torch.tensor(seams.embed_fn("a")),
                  torch.tensor(seams.embed_fn("b")), dim=0)
        assert raw > 0.99
        assert cen < 0.9


class TestNTStandardisesOnTheRuntimeTimescale:
    """Run 58673415: DA 0.15→0.05, GABA 0.15→0.10 — monotone within
    each run. DrivenNTSystem's driver z-scores use driver_alpha=0.005
    (~200-step memory, tuned for training steps); over a dozen ticks
    the running mean stays pinned at the FIRST sample, so one early
    ignition reads as 'normal' forever and every later tick is a
    persistent negative z."""

    def test_runtime_builds_its_nt_with_the_config_alpha(self):
        from neuroslm.cognition.runtime import CognitiveRuntime, MindConfig
        rt = CognitiveRuntime(generate_fn=_ScriptedGen(["x"]),
                              score_fn=_score_map({}), embed_fn=_vec_for,
                              cfg=MindConfig(nt_driver_alpha=0.1))
        assert all(st.alpha == 0.1 for st in rt.nt._stats.values())
        assert MindConfig().nt_driver_alpha >= 0.05

    def test_a_single_early_ignition_does_not_shift_gaba_for_good(self):
        from neuroslm.emergent.driven_nt import DrivenNTSystem

        def run(alpha):
            nt = DrivenNTSystem(driver_alpha=alpha)
            nt.step_full(ignition_rate=1.0)
            for _ in range(40):
                nt.step_full(ignition_rate=0.0)
            return abs(nt.levels()["GABA"] - 0.15)
        assert run(0.005) > 0.02, "slow stats: the first sample sticks"
        assert run(0.1) < run(0.005) / 3


class TestTraceExplainsTheGates:
    """Run 58673415's trace could not say WHY a thought stayed 'think'
    — novelty and the workspace inputs were never logged."""

    def test_trace_shows_novelty_boredom_and_workspace_inputs(self):
        from neuroslm.cognition.runtime import MindConfig, format_debug_trace
        rt = _mk_runtime(_ScriptedGen(["thinking about the river"]),
                         cfg=MindConfig(n_candidates=2), osc=True)
        r = rt.tick()
        assert set(r.workspace_inputs) == {"sensory", "memory", "language",
                                           "selection", "novelty"}
        text = format_debug_trace(r)
        assert "CUR[nov=" in text and "bore=" in text
        assert "in[s=" in text, "WS line carries the module activations"


class TestGenerateHasAnAttentionMask:
    """Run 58673415: 'The attention mask is not set ... pad token is
    same as eos token' (R1: pad == eos)."""

    def test_generate_receives_an_all_ones_mask(self):
        from neuroslm.cognition.runtime import build_hf_seams
        model = _FakeHFModelWithGenerate()
        seams = build_hf_seams("fake/expert", model_factory=lambda: model,
                               tokenizer_factory=_FakeChatTokenizer)
        seams.generate_fn("hello", 3)
        mask = model.generate_calls[-1].get("attention_mask")
        assert mask is not None and bool(mask.all())



class TestNoveltyExpectationForNE:
    def test_constant_novelty_gives_a_flat_ne_driver(self):
        from neuroslm.cognition.runtime import MindConfig
        nt = _FakeNT()
        rt = _mk_runtime(_ScriptedGen(["same thought"]), nt=nt,
                         cfg=MindConfig(n_candidates=1,
                                        workspace_oscillator=False,
                                        novelty_write_threshold=2.0,
                                        wm_size=0))  # constant novelty
        for _ in range(8):
            rt.tick()
        drivers = [c["grad_norm"] for c in nt.step_calls]
        assert max(drivers) == pytest.approx(0.0, abs=1e-9)

    def test_a_novelty_jump_drives_ne(self):
        from neuroslm.cognition.runtime import MindConfig
        nt = _FakeNT()
        gen = _ScriptedGen(["thinking about code"] * 6
                           + ["thinking about the moon"])
        rt = _mk_runtime(gen, nt=nt, cfg=MindConfig(
            n_candidates=1, workspace_oscillator=False))
        rs = [rt.tick() for _ in range(7)]
        assert nt.step_calls[-1]["grad_norm"] > 0.3
        assert rs[-1].novelty > rs[-2].novelty


class TestAdaptiveWorkspaceDefaults:
    def test_runtime_default_is_adaptive(self):
        from neuroslm.cognition.runtime import MindConfig
        assert MindConfig().osc_ignition_mode == "adaptive"

    def test_trace_shows_the_threshold(self):
        from neuroslm.cognition.runtime import MindConfig, format_debug_trace
        rt = _mk_runtime(_ScriptedGen(["thinking about the river"]),
                         cfg=MindConfig(n_candidates=2), osc=True)
        assert "θ=" in format_debug_trace(rt.tick())


class TestThoughtsAreNotCutOff:
    """Runs 58673415/58676023: R1's thoughts ended mid-sentence ("…so
    we can write:") at the 96-token budget, and the 1200-char prompt,
    truncated from the LEFT, could drop the persona and recalled
    memories once thoughts got long."""

    def test_budgets_fit_a_verbose_trunk(self):
        from neuroslm.cognition.runtime import MindConfig
        c = MindConfig()
        assert c.thought_n_tok >= 192
        assert c.max_prompt_chars >= 4000

    def test_trim_to_last_complete_sentence(self):
        from neuroslm.cognition.runtime import trim_to_sentence
        assert trim_to_sentence("One idea. Then another that trails"
                                ) == "One idea."
        assert trim_to_sentence("Is it? Maybe so! And then the") \
            == "Is it? Maybe so!"
        assert trim_to_sentence("Complete thought.") == "Complete thought."
        assert trim_to_sentence('She said "yes." And') == 'She said "yes."'
        assert trim_to_sentence("no boundary at all") == "no boundary at all"
        assert trim_to_sentence("Pi is 3.14 and") == "Pi is 3.14 and", (
            "a decimal point is not a sentence end")

    def test_candidates_are_trimmed_before_selection(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["A full thought. And a cut-off tr"]),
                         cfg=MindConfig(n_candidates=1,
                                        workspace_oscillator=False))
        assert rt.tick().thought == "A full thought."



class TestTraceShowsFullText:
    """2026-11-01: 'a lot of thoughts are cut off' — 39/40 selected R1
    thoughts were complete; the '…' came from the trace truncating
    rejected candidates, recalled episodes and lineage at 90 chars."""

    def _r(self):
        from neuroslm.cognition.runtime import ThoughtScore, TickResult
        long_a, long_b = "a" * 300, "b" * 300
        return TickResult(
            thought=long_a, tick_n=1, wandering=True, action="speak",
            candidates=[long_a, long_b],
            scores=[ThoughtScore(2.0, 0.3), ThoughtScore(3.0, 0.3)],
            recalled=[{"content": "c" * 300, "context": {"kind": "observed"}}],
            prior_thought="d" * 300, nt_levels={"GABA": 0.1})

    def test_rejected_candidates_recall_and_lineage_render_in_full(self):
        from neuroslm.cognition.runtime import format_debug_trace
        text = format_debug_trace(self._r())
        for ch in "abcd":
            assert ch * 300 in text, ch
        assert "…" not in text


class TestWanderingDoesNotRecallItsOwnLastThought:
    """Run 58676023: 4/5 ignitions had memory input m=1.00 — a wandering
    tick anchors RECALL on the last thought, and once that thought was
    stored it came back as its own nearest neighbour (cos 1), although
    it is already in the prompt as 'My last thought:'. That inflated
    the workspace's memory activation right after every broadcast —
    an artificial positive ignition autocorrelation (store → m=1 →
    ignite → store), against H65(c)'s negative lag-1."""

    def test_last_thought_is_excluded_from_recall(self):
        from neuroslm.cognition.runtime import MindConfig
        # moon axis: novel against the launch memory, so it is stored
        gen = _ScriptedGen(["thinking about the moon",
                            "more on the launch"])
        rt = _mk_runtime(gen, cfg=MindConfig(n_candidates=1, recall_k=3))
        rt.memory.add("the launch code is ready",
                      content_vec=_vec_for("the launch code is ready"),
                      context={"kind": "observed"})
        first = rt.tick()
        assert first.stored, "setup: the first thought must be stored"
        second = rt.tick()
        contents = [e["content"] for e in second.recalled]
        assert "thinking about the moon" not in contents
        assert "the launch code is ready" in contents

    def test_memory_activation_reflects_other_episodes(self):
        from neuroslm.cognition.runtime import MindConfig
        gen = _ScriptedGen(["thinking about the launch",
                            "more on the launch"])
        rt = _mk_runtime(gen, cfg=MindConfig(n_candidates=1), osc=True)
        rt.memory.add("coffee tastes great today",
                      content_vec=_vec_for("coffee tastes great today"),
                      context={"kind": "observed"})
        rt.tick()
        rt.memory.add("thinking about the launch",
                      content_vec=_vec_for("thinking about the launch"),
                      context={"kind": "inferred"})
        r = rt.tick()
        assert r.workspace_inputs["memory"] < 0.99



class TestConfirmatoryFreezeFixes:
    """Fixes from run 58678026, shipped before the confirmatory run."""

    def test_runtime_nt_uses_warmup_variance_init(self):
        from neuroslm.cognition.runtime import CognitiveRuntime, MindConfig
        assert MindConfig().nt_driver_var_init == "warmup"
        rt = CognitiveRuntime(generate_fn=_ScriptedGen(["x"]),
                              score_fn=_score_map({}), embed_fn=_vec_for)
        assert all(st.var_init == "warmup" for st in rt.nt._stats.values())

    def test_respond_prompt_instructs_a_first_person_reply(self):
        from neuroslm.cognition.runtime import MindConfig
        gen = _ScriptedGen(["I hear you."])
        rt = _mk_runtime(gen, cfg=MindConfig(n_candidates=1))
        rt.observe("the river was beautiful today")
        rt.tick()
        prompt = gen.prompts[-1]
        assert rt.cfg.respond_instruction in prompt
        assert prompt.index(rt.cfg.respond_instruction) < prompt.index(
            "My reply:")

    def test_wandering_prompt_has_no_reply_instruction(self):
        from neuroslm.cognition.runtime import MindConfig
        gen = _ScriptedGen(["a wander"])
        rt = _mk_runtime(gen, cfg=MindConfig(n_candidates=1))
        rt.tick()
        assert rt.cfg.respond_instruction not in gen.prompts[-1]

    def test_third_person_self_narration_loses_selection(self):
        """Run 58678026: 'The passage reflects Brian's deep
        contemplation…' — R1 analysed the persona instead of being it."""
        from neuroslm.cognition.runtime import MindConfig
        narr = "The passage reflects Brian's deep contemplation."
        reply = "I loved watching it too."
        gen = _ScriptedGen([narr, reply])
        rt = _mk_runtime(gen, scores=_score_map({narr: 3.0, reply: 3.8}),
                         cfg=MindConfig(n_candidates=2,
                                        selection_temp_base=0.05))
        rt.observe("the river was beautiful today")
        assert rt.tick().thought == reply

    def test_narration_detector(self):
        from neuroslm.cognition.runtime import _is_self_narration
        assert _is_self_narration("The passage reflects Brian's mood.")
        assert _is_self_narration("So Brian continues this line of thinking")
        assert not _is_self_narration("I think the river is calm.")
        assert not _is_self_narration("Brian, look at this")  # address



# ── H67: control layer that works on any trunk (confirmatory run 58680199) ──

class TestWorkingMemoryNovelty:
    """Run 58680199 (R1-7B): a ~30-tick fluid-dynamics→probability basin
    never registered as repetitive — novelty is measured against STORED
    episodes, and with ignition at ~9% almost nothing was stored. The
    last N selected thoughts now count too."""

    def test_repetition_registers_without_any_storage(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["the same thought"]),
                         cfg=MindConfig(n_candidates=1,
                                        novelty_write_threshold=2.0))
        rs = [rt.tick() for _ in range(3)]
        assert not any(r.stored for r in rs)
        assert rs[0].novelty == pytest.approx(1.0)
        assert rs[1].novelty == pytest.approx(0.0, abs=1e-6)

    def test_wm_size_zero_restores_storage_only_novelty(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["the same thought"]),
                         cfg=MindConfig(n_candidates=1, wm_size=0,
                                        novelty_write_threshold=2.0))
        rs = [rt.tick() for _ in range(3)]
        assert rs[1].novelty == pytest.approx(1.0)

    def test_wm_is_bounded(self):
        from neuroslm.cognition.runtime import MindConfig
        topics = ["code", "weather", "music", "launch", "coffee"]
        rt = _mk_runtime(_ScriptedGen([f"about {t}" for t in topics]),
                         cfg=MindConfig(n_candidates=1, wm_size=3,
                                        novelty_write_threshold=2.0))
        for _ in range(5):
            rt.tick()
        assert len(rt._wm) == 3

    def test_unstored_repetition_now_triggers_reorient(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["the same thought"]),
                         cfg=MindConfig(n_candidates=1,
                                        novelty_write_threshold=2.0))
        rs = [rt.tick() for _ in range(7)]
        assert any(r.reorient for r in rs)


class TestRelativeControlThresholds:
    """Run 58680199: reorient (novelty < 0.25) and replay (boredom ≥ 0.6)
    never fired — R1's centred novelty sat at 0.34-0.77 and boredom
    peaked at 0.59. Absolute thresholds tuned on SmolLM2 don't transfer;
    relative ones compare against the run's own slow novelty baseline."""

    def test_default_is_relative(self):
        from neuroslm.cognition.runtime import MindConfig
        assert MindConfig().control_thresholds == "relative"

    def test_loop_threshold_scales_with_the_baseline(self):
        from neuroslm.cognition.runtime import loop_threshold, MindConfig
        c = MindConfig()
        assert loop_threshold(0.6, c) == pytest.approx(0.75 * 0.6)
        assert loop_threshold(0.3, c) == pytest.approx(0.75 * 0.3)
        a = MindConfig(control_thresholds="absolute")
        assert loop_threshold(0.6, a) == a.loop_novelty_threshold

    def test_replay_threshold_scales_with_the_baseline(self):
        from neuroslm.cognition.runtime import replay_threshold, MindConfig
        c = MindConfig()
        assert replay_threshold(0.6, c) == pytest.approx(0.4 + c.replay_margin)
        a = MindConfig(control_thresholds="absolute")
        assert replay_threshold(0.6, a) == a.replay_boredom_threshold

    def test_a_sustained_drop_below_baseline_counts_as_looping(self):
        """High-novelty trunk: novelty ~0.65 for a while, then a basin at
        ~0.40 — above the old absolute 0.25, below 0.75 × baseline."""
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["x"]), cfg=MindConfig(
            n_candidates=1, wm_size=0))
        for v in [0.65, 0.66, 0.64, 0.65, 0.40, 0.41, 0.39]:
            rt._update_control(v)
        assert rt._low_nov_streak >= 3

    def test_baseline_is_seeded_with_the_first_novelty(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["x"]), cfg=MindConfig(n_candidates=1))
        rt._update_control(0.62)
        assert rt._nov_base == pytest.approx(0.62)


class TestNarrationAndScaffold:
    def test_generic_third_person_narration_is_penalised(self):
        from neuroslm.cognition.runtime import _is_self_narration
        assert _is_self_narration(
            "The user started by thinking about modeling experiments.")
        assert _is_self_narration("The passage describes a calm river.")
        # mentioning the user is fine; narrating them is not
        assert not _is_self_narration("I could ask the user about it later?")
        assert not _is_self_narration("Users of maps often get lost.")

    def test_scaffold_echo_is_cut(self):
        from neuroslm.cognition.runtime import strip_scaffold
        assert strip_scaffold(
            'What does "flow" mean here?\nPrevious thoughts: rivers flowing'
        ) == 'What does "flow" mean here?'
        assert strip_scaffold("An idea.\nJust now: something") == "An idea."
        assert strip_scaffold("Plain thought.") == "Plain thought."
        assert strip_scaffold("I remember: nothing") == ""

    def test_runtime_strips_scaffold_before_selection(self):
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(
            ["A real thought.\nMy last thought: echo"]),
            cfg=MindConfig(n_candidates=1))
        assert rt.tick().thought == "A real thought."
