# -*- coding: utf-8 -*-
"""Memory persistence for CognitiveRuntime (architecture.md §14.14) —
save_mind_memory/load_mind_memory in neuroslm/memory/store.py, reusing
the Brain-side .mem checkpoint format's pattern (pickle + JSON sidecar,
shared _stream_state/_restore_stream helpers) rather than inventing a
second serialization scheme.

Before this, CognitiveRuntime had NO persistence at all — episodic
memory (and, since §14.12, narrative state) was lost on every process
restart. Mirrors tests/test_memory.py's existing Brain-side
save/load-roundtrip pattern.
"""
from __future__ import annotations

import pickle
import random

import pytest

from neuroslm.cognition.runtime import CognitiveRuntime, MindConfig, ThoughtScore
from neuroslm.memory.episodic import EpisodicMemory


def _vec_for(text: str):
    axes = ("code", "weather", "music", "launch")
    v = [0.05] * len(axes)
    low = text.lower()
    for i, w in enumerate(axes):
        if w in low:
            v[i] += 1.0
    return v


class _ScriptedGen:
    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.calls = 0

    def __call__(self, prompt, max_new_tokens):
        out = self.candidates[self.calls % len(self.candidates)]
        self.calls += 1
        return out


def _score(text):
    return ThoughtScore(mean_nll=2.0, entropy_norm=0.5)


def _mk_runtime(maxlen=64, enable_narrative=True, reflection_interval=0, seed=0):
    cfg = MindConfig(n_candidates=2, enable_narrative=enable_narrative,
                     reflection_interval=reflection_interval,
                     novelty_write_threshold=0.0)
    return CognitiveRuntime(
        generate_fn=_ScriptedGen(["first thought", "second thought"]),
        score_fn=_score, embed_fn=_vec_for,
        memory=EpisodicMemory(maxlen=maxlen),
        cfg=cfg, rng=random.Random(seed))


class TestSaveMindMemory:
    def test_save_writes_mem_and_json_sidecar(self, tmp_path):
        from neuroslm.memory.store import save_mind_memory
        rt = _mk_runtime()
        rt.observe("I noticed the weather change")
        path = tmp_path / "mind.mem"
        stats = save_mind_memory(path, rt)
        assert path.exists()
        assert (tmp_path / "mind.mem.json").exists()
        assert stats["n_episodes"] == 1


class TestLoadMindMemory:
    def test_load_restores_episodic_buffer_content_and_order(self, tmp_path):
        from neuroslm.memory.store import load_mind_memory, save_mind_memory
        rt = _mk_runtime()
        rt.observe("first observed event about the weather")
        rt.observe("second observed event about the launch")
        path = tmp_path / "mind.mem"
        save_mind_memory(path, rt)

        fresh = _mk_runtime()
        load_mind_memory(path, fresh)
        assert [e["content"] for e in fresh.memory.all()] == \
            [e["content"] for e in rt.memory.all()]

    def test_load_restores_narrative_streams(self, tmp_path):
        from neuroslm.memory.store import load_mind_memory, save_mind_memory
        rt = _mk_runtime(enable_narrative=True)
        rt.observe("first observed event about the weather")
        rt.tick()  # STORE -> record_autobiographical
        path = tmp_path / "mind.mem"
        save_mind_memory(path, rt)
        assert rt.narrative is not None

        fresh = _mk_runtime(enable_narrative=True)
        load_mind_memory(path, fresh)
        assert fresh.narrative is not None
        assert (len(fresh.narrative.world.events)
               == len(rt.narrative.world.events) == 1)
        assert (len(fresh.narrative.autobiographical.events)
               == len(rt.narrative.autobiographical.events) == 1)
        assert (fresh.narrative.autobiographical.events[0].content
               == rt.narrative.autobiographical.events[0].content)

    def test_load_with_narrative_disabled_skips_narrative_silently(self, tmp_path):
        """A file saved with narrative enabled, loaded into a runtime
        with cfg.enable_narrative=False, must not crash — narrative
        content is simply not restored."""
        from neuroslm.memory.store import load_mind_memory, save_mind_memory
        rt = _mk_runtime(enable_narrative=True)
        rt.observe("first observed event about the weather")
        path = tmp_path / "mind.mem"
        save_mind_memory(path, rt)

        fresh = _mk_runtime(enable_narrative=False)
        stats = load_mind_memory(path, fresh)  # must not raise
        assert fresh.narrative is None
        assert stats is not None

    def test_roundtrip_preserves_tick_counters_and_boredom(self, tmp_path):
        from neuroslm.memory.store import load_mind_memory, save_mind_memory
        rt = _mk_runtime()
        for _ in range(3):
            rt.tick()
        path = tmp_path / "mind.mem"
        save_mind_memory(path, rt)

        fresh = _mk_runtime()
        load_mind_memory(path, fresh)
        assert fresh._tick_n == rt._tick_n
        assert fresh._boredom == pytest.approx(rt._boredom)

    def test_roundtrip_preserves_mined_rules(self, tmp_path):
        from neuroslm.memory.store import load_mind_memory, save_mind_memory
        rt = _mk_runtime(reflection_interval=1)
        for _ in range(4):
            rt.observe("You suck")
            rt.observe("No, that's wrong")
        rt.tick()
        assert rt._mined_rules  # reflection_interval=1 mines on tick 1
        path = tmp_path / "mind.mem"
        save_mind_memory(path, rt)

        fresh = _mk_runtime()
        load_mind_memory(path, fresh)
        assert [r.antecedent for r in fresh._mined_rules] == \
            [r.antecedent for r in rt._mined_rules]

    def test_load_rejects_wrong_format_file(self, tmp_path):
        from neuroslm.memory.store import load_mind_memory
        path = tmp_path / "bad.mem"
        with open(path, "wb") as f:
            pickle.dump({"format": "neuroslm.memory.v1"}, f)  # Brain's tag
        fresh = _mk_runtime()
        with pytest.raises(ValueError):
            load_mind_memory(path, fresh)

    def test_episodic_buffer_respects_maxlen_on_load(self, tmp_path):
        from neuroslm.memory.store import load_mind_memory, save_mind_memory
        rt = _mk_runtime(maxlen=1000)
        for i in range(600):
            rt.observe(f"event number {i} about the weather")
        path = tmp_path / "mind.mem"
        save_mind_memory(path, rt)

        fresh = _mk_runtime(maxlen=512)
        load_mind_memory(path, fresh)
        loaded = fresh.memory.all()
        original = rt.memory.all()
        assert len(loaded) == 512
        assert loaded[-1]["content"] == original[-1]["content"]
        assert loaded[0]["content"] == original[-512]["content"]
