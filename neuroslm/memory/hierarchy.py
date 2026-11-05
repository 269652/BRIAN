"""Hierarchical memory: two hierarchies indexed over the same episodes.

Semantic (consolidation into gists). An episode joins its nearest gist
unless its similarity to it is unusually low — below the typical match
by more than k·σ, with running statistics as for ignition — in which
case it founds a new gist. A gist's centroid is the mean of its
members.

Temporal (event segmentation, Zacks et al. 2007). An unusually large
workspace state change is an event boundary. When an event closes, it
either joins the current chapter or — if its centroid is unusually far
from the chapter's — opens a new chapter.

``sync`` annotates each new episode's ``context`` with ``gist``,
``event`` and ``chapter``; ``best_per_gist`` reorders a recall ranking
top-down: the best episode of each gist first, then the rest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_SD0 = 0.05
"""Seed and floor of the running deviations: a just-noticeable
difference, so a near-constant signal does not make every flicker an
outlier."""


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


class _Running:
    """EWMA mean/variance, updated after each decision."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.mu: Optional[float] = None
        self.var = _SD0 ** 2

    @property
    def sd(self) -> float:
        return max(math.sqrt(self.var), _SD0)

    def update(self, x: float) -> None:
        if self.mu is None:
            self.mu = x
            return
        d = x - self.mu
        self.mu += self.alpha * d
        self.var = (1 - self.alpha) * (self.var + self.alpha * d * d)


@dataclass
class Gist:
    id: int
    centroid: List[float]
    size: int = 1
    members: List[str] = field(default_factory=list)


class MemoryHierarchy:
    def __init__(self, k: float = 1.0, alpha: float = 0.2) -> None:
        self.k = float(k)
        self.gists: List[Gist] = []
        self._match = _Running(alpha)
        self._change = _Running(alpha)
        self._fit = _Running(alpha)
        self.current_event = 0
        self.current_chapter = 0
        self._event_vecs: List[List[float]] = []
        self._chapter_centroid: Optional[List[float]] = None
        self._chapter_events = 0

    @property
    def n_events(self) -> int:
        return self.current_event + 1

    @property
    def n_chapters(self) -> int:
        return self.current_chapter + 1

    # ── semantic ──────────────────────────────────────────────────────

    def sync(self, episodes: Sequence[dict]) -> None:
        for e in episodes:
            ctx = e.get("context")
            vec = e.get("content_vec")
            if vec is None or ctx is None or "gist" in ctx:
                continue
            ctx["gist"] = self._assign(list(vec), str(e.get("content", "")))
            ctx["event"] = self.current_event
            ctx["chapter"] = self.current_chapter

    def _assign(self, vec: List[float], content: str) -> int:
        if not self.gists:
            return self._found(vec, content)
        sims = [_cos(vec, g.centroid) for g in self.gists]
        i = max(range(len(sims)), key=sims.__getitem__)
        s = sims[i]
        outlier = (self._match.mu is not None
                   and s < self._match.mu - self.k * self._match.sd)
        self._match.update(s)
        if outlier:
            return self._found(vec, content)
        g = self.gists[i]
        g.size += 1
        g.centroid = [c + (v - c) / g.size for c, v in zip(g.centroid, vec)]
        g.members.append(content)
        return g.id

    def _found(self, vec: List[float], content: str) -> int:
        g = Gist(len(self.gists), list(vec), 1, [content])
        self.gists.append(g)
        return g.id

    # ── temporal ──────────────────────────────────────────────────────

    def boundary(self, state_change: Optional[float],
                 state_vec: Optional[Sequence[float]]) -> bool:
        """Feed one tick's workspace state change and state vector;
        returns True when the tick opens a new event."""
        is_boundary = False
        if state_change is not None:
            x = float(state_change)
            is_boundary = (self._change.mu is not None
                           and x > self._change.mu + self.k * self._change.sd)
            self._change.update(x)
        if is_boundary:
            self._close_event()
            self.current_event += 1
        if state_vec is not None:
            self._event_vecs.append(list(state_vec))
        return is_boundary

    def _close_event(self) -> None:
        if not self._event_vecs:
            return
        n = len(self._event_vecs)
        centroid = [sum(c) / n for c in zip(*self._event_vecs)]
        self._event_vecs = []
        if self._chapter_centroid is None:
            self._chapter_centroid, self._chapter_events = centroid, 1
            return
        c = _cos(centroid, self._chapter_centroid)
        new = (self._fit.mu is not None
               and c < self._fit.mu - self.k * self._fit.sd)
        self._fit.update(c)
        if new:
            self.current_chapter += 1
            self._chapter_centroid, self._chapter_events = centroid, 1
            return
        self._chapter_events += 1
        w = 1.0 / self._chapter_events
        self._chapter_centroid = [a + w * (b - a) for a, b in
                                  zip(self._chapter_centroid, centroid)]

    def summary(self) -> Dict:
        sizes = [g.size for g in self.gists]
        return {"gists": len(self.gists),
                "mean_gist_size": (sum(sizes) / len(sizes)) if sizes else 0.0,
                "events": self.n_events, "chapters": self.n_chapters}


def _split(scored):
    first, rest, seen = [], [], set()
    for item in scored:
        g = (item[1].get("context") or {}).get("gist")
        if g is None or g not in seen:
            first.append(item)
            if g is not None:
                seen.add(g)
        else:
            rest.append(item)
    return first, rest


def one_per_gist(scored: Sequence[Tuple[float, dict]]
                 ) -> List[Tuple[float, dict]]:
    """The best-ranked episode of each gist (and every ungisted one)."""
    return _split(scored)[0]


def best_per_gist(scored: Sequence[Tuple[float, dict]]
                  ) -> List[Tuple[float, dict]]:
    """Top-down recall order: ``one_per_gist`` in rank order, then the
    remaining episodes."""
    first, rest = _split(scored)
    return first + rest
