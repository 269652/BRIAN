# -*- coding: utf-8 -*-
"""Persistent search ledger — a growing record of what the explorer already tried.

Every searched program is keyed by a **semantic signature**: a hash of its
quantized semantic vector (op histogram + structure), so it is
hyperparameter-invariant — SGD@0.01 and SGD@0.9 share a signature. The ledger
persists to JSON, so a new run loads the accumulated history and skips **duds**
(patterns already tried that were rejected or didn't improve) instead of
rediscovering them every time.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from neuroslm.genetic.language import Program


@dataclass
class SearchRecord:
    signature: str
    source: str
    outcome: str            # "kept" | "rejected" | "searched"
    delta: float = 0.0      # metric change (negative = improvement for loss/ppl)
    metric_before: float = float("nan")
    metric_after: float = float("nan")
    step: int = 0
    run_id: str = ""
    kind: str = "modulation"
    count: int = 1          # how many times this signature was encountered


class SearchLedger:
    """A persistent, signature-keyed record of searched patterns."""

    def __init__(self, path, quantize: int = 2):
        self.path = None if str(path) == ":memory:" else Path(path)
        self.quantize = quantize
        self._by_sig: Dict[str, SearchRecord] = {}
        if self.path is not None and self.path.exists():
            self._load()

    # -- signature ----------------------------------------------------------
    def signature(self, program: Program) -> str:
        v = np.round(program.semantic_vector(), self.quantize)
        return hashlib.sha1(v.tobytes()).hexdigest()[:16]

    # -- queries ------------------------------------------------------------
    def has_searched(self, program: Program) -> bool:
        return self.signature(program) in self._by_sig

    def outcome_of(self, program: Program) -> Optional[str]:
        rec = self._by_sig.get(self.signature(program))
        return rec.outcome if rec else None

    def is_dud(self, program: Program) -> bool:
        """Searched before, and it did not help (rejected or delta ≥ 0)."""
        rec = self._by_sig.get(self.signature(program))
        if rec is None:
            return False
        return rec.outcome == "rejected" or (rec.outcome != "kept" and rec.delta >= 0.0)

    def get(self, program: Program) -> Optional[SearchRecord]:
        return self._by_sig.get(self.signature(program))

    def all(self) -> List[SearchRecord]:
        return list(self._by_sig.values())

    def stats(self) -> dict:
        out = {"total": len(self._by_sig), "kept": 0, "rejected": 0, "searched": 0}
        for r in self._by_sig.values():
            out[r.outcome] = out.get(r.outcome, 0) + 1
        return out

    # -- mutation -----------------------------------------------------------
    def record(self, program: Program, outcome: str, delta: float = 0.0,
               metric_before: float = float("nan"), metric_after: float = float("nan"),
               step: int = 0, run_id: str = "", kind: str = "modulation") -> SearchRecord:
        sig = self.signature(program)
        existing = self._by_sig.get(sig)
        rec = SearchRecord(
            signature=sig, source=program.to_source(), outcome=outcome, delta=delta,
            metric_before=metric_before, metric_after=metric_after, step=step,
            run_id=run_id, kind=kind, count=(existing.count + 1 if existing else 1),
        )
        self._by_sig[sig] = rec  # latest outcome wins; count accumulates
        return rec

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(r) for r in self._by_sig.values()], indent=1),
            encoding="utf-8")

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        for d in data:
            rec = SearchRecord(**{k: d.get(k) for k in SearchRecord.__dataclass_fields__})
            self._by_sig[rec.signature] = rec
