# -*- coding: utf-8 -*-
"""Global workspace content (architecture.md §14.13; Baars 1988,
Dehaene & Changeux 2011).

A capacity-limited store (Cowan's ~4 chunks) where percepts, recalled
memories and selected thoughts compete by salience. Every item fades
each tick; the weakest is displaced when the store is full. What
survives is broadcast as the context of the next thought.
:class:`~neuroslm.cognition.oscillator.GammaWorkspace` decides
ignition; this module holds what is in mind."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple


@dataclass
class WorkspaceItem:
    text: str
    source: str
    vec: List[float]
    salience: float
    seq: int = 0


_FLOAT_ZERO = 1e-9


def divisive_normalise(x: float, baseline: Optional[float]) -> float:
    """x / (x + baseline) (Carandini & Heeger 2012): 0 at x = 0, 0.5 at
    the baseline, → 1 far above it — independent of the scale x lives
    on. No baseline yet counts as typical (0.5)."""
    x = max(0.0, float(x))
    if x < _FLOAT_ZERO:     # 1 − cos(v, v) is ~1e-16, not 0
        return 0.0
    b = x if baseline is None else max(0.0, float(baseline))
    return x / (x + b)


def thought_salience(novelty: float, baseline: Optional[float],
                     ignited: bool) -> float:
    """Ignition sustains a thought at full salience; otherwise its raw
    novelty divisively normalised against the run's raw baseline."""
    return 1.0 if ignited else divisive_normalise(novelty, baseline)


class GlobalWorkspace:
    def __init__(self, capacity: int = 4, decay: float = 0.7):
        self.capacity = max(1, int(capacity))
        self.decay_rate = float(decay)
        self._items: List[WorkspaceItem] = []
        self._seq = 0

    def decay(self) -> None:
        for it in self._items:
            it.salience *= self.decay_rate

    def offer(self, text: str, source: str, vec: Sequence[float],
              salience: float) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._seq += 1
        for it in self._items:
            if it.text == text:
                if salience > it.salience:
                    it.salience = float(salience)
                    it.seq = self._seq
                return
        self._items.append(WorkspaceItem(text, source, list(vec),
                                         float(salience), self._seq))
        while len(self._items) > self.capacity:
            weakest = min(self._items, key=lambda it: (it.salience, it.seq))
            self._items.remove(weakest)

    def state_vector(self) -> Optional[List[float]]:
        """The workspace as ONE latent: the salience-weighted mean of
        its items' vectors (items without a vector are skipped). None
        when nothing salient with a vector is held."""
        held = [it for it in self._items if it.vec and it.salience > 0.0]
        total = sum(it.salience for it in held)
        if not held or total <= 0.0:
            return None
        dim = len(held[0].vec)
        out = [0.0] * dim
        for it in held:
            w = it.salience / total
            for i, x in enumerate(it.vec[:dim]):
                out[i] += w * x
        return out

    def drop(self, source: str) -> None:
        self._items = [it for it in self._items if it.source != source]

    def items(self) -> List[WorkspaceItem]:
        """Most salient LAST — closest to where generation starts."""
        return sorted(self._items, key=lambda it: (it.salience, it.seq))

    def items_view(self) -> List[Tuple[str, float, str]]:
        return [(it.source, it.salience, it.text) for it in self.items()]

    def texts(self) -> Set[str]:
        return {it.text for it in self._items}
