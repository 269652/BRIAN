"""Hierarchical memory (neuroslm/memory/hierarchy.py): one index, two
hierarchies over the same episodes.

Semantic: episodes consolidate into gists (an episode joins its nearest
gist unless it is unusually far from all of them — adaptive, like
ignition — and then founds a new one). Temporal: an unusually large
workspace state change is an event boundary (event segmentation); an
event that does not fit its chapter opens a new chapter."""
from __future__ import annotations

import math

import pytest

from neuroslm.memory.episodic import EpisodicMemory


def _v(*axes, n=6):
    v = [0.05] * n
    for a in axes:
        v[a] += 1.0
    return v


def _mem(vecs):
    m = EpisodicMemory(maxlen=64)
    for i, v in enumerate(vecs):
        m.add(f"episode {i}", content_vec=v, context={"kind": "observed"})
    return m


class TestGists:
    def test_similar_episodes_share_a_gist_an_outlier_founds_one(self):
        from neuroslm.memory.hierarchy import MemoryHierarchy
        h = MemoryHierarchy()
        river = [_v(0), _v(0, 1), _v(0), _v(0, 1), _v(0)]
        m = _mem(river + [_v(4)])
        h.sync(m.all())
        g = [e["context"]["gist"] for e in m.all()]
        assert len(set(g[:5])) == 1
        assert g[5] != g[0]

    def test_a_gist_centroid_is_the_mean_of_its_members(self):
        from neuroslm.memory.hierarchy import MemoryHierarchy
        h = MemoryHierarchy()
        m = _mem([_v(0), _v(0, 1)])
        h.sync(m.all())
        (gist,) = h.gists
        assert gist.centroid == pytest.approx(
            [(a + b) / 2 for a, b in zip(_v(0), _v(0, 1))])
        assert gist.size == 2

    def test_sync_is_incremental(self):
        from neuroslm.memory.hierarchy import MemoryHierarchy
        h = MemoryHierarchy()
        m = _mem([_v(0), _v(0)])
        h.sync(m.all())
        h.sync(m.all())
        assert sum(g.size for g in h.gists) == 2

    def test_episodes_without_a_vector_are_skipped(self):
        from neuroslm.memory.hierarchy import MemoryHierarchy
        m = EpisodicMemory(maxlen=8)
        m.add("no vector")
        h = MemoryHierarchy()
        h.sync(m.all())
        assert h.gists == [] and "gist" not in m.all()[0]["context"]


class TestEventsAndChapters:
    def test_a_large_state_change_is_an_event_boundary(self):
        from neuroslm.memory.hierarchy import MemoryHierarchy
        h = MemoryHierarchy()
        steady = [h.boundary(0.3 + 0.01 * (i % 3), _v(0)) for i in range(20)]
        assert sum(steady[5:]) <= 3
        assert h.boundary(0.95, _v(0)), "an unusual jump"
        assert h.n_events >= 2

    def test_episodes_record_their_event_and_chapter(self):
        from neuroslm.memory.hierarchy import MemoryHierarchy
        h = MemoryHierarchy()
        m = _mem([_v(0)])
        h.sync(m.all())
        ctx = m.all()[0]["context"]
        assert ctx["event"] == 0 and ctx["chapter"] == 0

    def test_an_event_unlike_its_chapter_opens_a_new_chapter(self):
        from neuroslm.memory.hierarchy import MemoryHierarchy
        h = MemoryHierarchy()
        for i in range(12):                      # several river events
            for _ in range(3):
                h.boundary(0.3, _v(0))
            h.boundary(0.99, _v(0, 1) if i % 2 else _v(0))
        ch = h.current_chapter
        for _ in range(3):
            h.boundary(0.3, _v(4))               # a chess event closes
        h.boundary(0.99, _v(4))
        assert h.current_chapter == ch + 1

    def test_similar_events_stay_in_one_chapter(self):
        from neuroslm.memory.hierarchy import MemoryHierarchy
        h = MemoryHierarchy()
        for i in range(12):
            for _ in range(3):
                h.boundary(0.3, _v(0))
            h.boundary(0.99, _v(0))
        assert h.n_events > 5 and h.n_chapters <= 2


class TestTopDownRecall:
    def test_recall_spans_gists_before_near_duplicates(self):
        from neuroslm.memory.hierarchy import best_per_gist
        eps = [{"content": f"r{i}", "context": {"gist": 0}} for i in range(3)]
        eps.append({"content": "chess", "context": {"gist": 1}})
        scored = [(0.9, eps[0]), (0.89, eps[1]), (0.88, eps[2]),
                  (0.5, eps[3])]
        out = best_per_gist(scored)
        assert [e["content"] for _, e in out] == ["r0", "chess", "r1", "r2"]

    def test_ungisted_episodes_keep_their_place(self):
        from neuroslm.memory.hierarchy import best_per_gist
        a = {"content": "a", "context": {}}
        b = {"content": "b", "context": {"gist": 0}}
        assert [e["content"] for _, e in best_per_gist(
            [(0.9, a), (0.8, b)])] == ["a", "b"]
