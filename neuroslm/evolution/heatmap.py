# -*- coding: utf-8 -*-
"""TrainingHeatmap — incremental per-element heat over the Hypergraph IR.

During training, each HypergraphIR element (node = module/gene, edge =
synapse/modulation path) accumulates a *heat* value: an exponential
moving average of a per-element signal (gradient magnitude). Hot
elements are where learning concentrates; cold elements are inert.

The heatmap is updated incrementally (cheaply, every N steps), persisted
to JSON as a live artifact, and queried for hot/cold paths that drive the
epigenetic mutation pipeline (propose -> gate -> Lean proof).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class HeatEntry:
    """Accumulated heat for one IR element."""
    id: str
    kind: str = "node"          # "node" | "edge"
    heat: float = 0.0           # EMA of the per-element signal
    last_signal: float = 0.0    # most recent raw signal
    updates: int = 0            # number of times this element was updated

    def to_dict(self) -> Dict:
        return {"id": self.id, "kind": self.kind, "heat": self.heat,
                "last_signal": self.last_signal, "updates": self.updates}

    @classmethod
    def from_dict(cls, d: Dict) -> "HeatEntry":
        return cls(id=d["id"], kind=d.get("kind", "node"),
                   heat=float(d.get("heat", 0.0)),
                   last_signal=float(d.get("last_signal", 0.0)),
                   updates=int(d.get("updates", 0)))


class TrainingHeatmap:
    """Incremental EMA heatmap keyed by HypergraphIR element id."""

    def __init__(self, beta: float = 0.1) -> None:
        if not (0.0 < beta <= 1.0):
            raise ValueError(f"beta must be in (0, 1], got {beta!r}")
        self.beta = beta
        self.step = 0
        self.entries: Dict[str, HeatEntry] = {}

    # ── incremental update ───────────────────────────────────────────────

    def update(self, signals: Dict[str, float],
               kinds: Optional[Dict[str, str]] = None,
               step: Optional[int] = None) -> None:
        """Fold a batch of per-element signals into the EMA heat.

        Args:
            signals: element id -> signal value (e.g. gradient L2 norm)
            kinds:   element id -> "node"|"edge" (only needed first time
                     an id is seen; defaults to "node")
            step:    explicit training step; if None, advances by 1.
        """
        kinds = kinds or {}
        for eid, sig in signals.items():
            entry = self.entries.get(eid)
            if entry is None:
                entry = HeatEntry(id=eid, kind=kinds.get(eid, "node"))
                self.entries[eid] = entry
            elif eid in kinds:
                entry.kind = kinds[eid]
            if entry.updates == 0:
                entry.heat = float(sig)
            else:
                entry.heat = (1.0 - self.beta) * entry.heat + self.beta * float(sig)
            entry.last_signal = float(sig)
            entry.updates += 1
        self.step = step if step is not None else self.step + 1

    # ── queries ──────────────────────────────────────────────────────────

    def heat(self, eid: str) -> float:
        entry = self.entries.get(eid)
        return entry.heat if entry else 0.0

    def normalized(self) -> Dict[str, float]:
        """Heat scaled to [0, 1] by the current maximum."""
        if not self.entries:
            return {}
        max_heat = max(e.heat for e in self.entries.values())
        if max_heat <= 0.0:
            return {eid: 0.0 for eid in self.entries}
        return {eid: e.heat / max_heat for eid, e in self.entries.items()}

    def rank(self) -> List[Tuple[str, float]]:
        """Elements sorted by raw heat, descending."""
        return sorted(((eid, e.heat) for eid, e in self.entries.items()),
                      key=lambda kv: kv[1], reverse=True)

    def hot_paths(self, threshold: float = 0.7,
                  kind: Optional[str] = None) -> List[str]:
        """Ids whose normalized heat exceeds ``threshold`` (hottest first)."""
        norm = self.normalized()
        out = [eid for eid, v in norm.items()
               if v > threshold and (kind is None or self.entries[eid].kind == kind)]
        out.sort(key=lambda eid: norm[eid], reverse=True)
        return out

    def cold_paths(self, threshold: float = 0.1,
                   kind: Optional[str] = None) -> List[str]:
        """Ids whose normalized heat is below ``threshold`` (coldest first)."""
        norm = self.normalized()
        out = [eid for eid, v in norm.items()
               if v < threshold and (kind is None or self.entries[eid].kind == kind)]
        out.sort(key=lambda eid: norm[eid])
        return out

    # ── persistence ──────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "version": "heatmap/1.0",
            "beta": self.beta,
            "step": self.step,
            "entries": {eid: e.to_dict() for eid, e in self.entries.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TrainingHeatmap":
        hm = cls(beta=float(d.get("beta", 0.1)))
        hm.step = int(d.get("step", 0))
        hm.entries = {eid: HeatEntry.from_dict(ed)
                      for eid, ed in d.get("entries", {}).items()}
        return hm

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingHeatmap":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
