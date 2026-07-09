# -*- coding: utf-8 -*-
"""Install banked discovery winners into a training run — evidence-gated.

The multi-site probe (layer_probe.py) banks site-tagged winners to
``modulations/`` with their measured Δ. This module closes the loop: at the
start of the next run, recurring winners are re-validated live and installed
into ``DSLLanguageCortex._layer_modulations`` so they take effect in training.

Two gates, both mandatory:

1. **Evidence grouping** (``select_installable``) — winners are grouped by
   (site, program semantics); recurrence pools their Δ evidence. A group's
   ``count`` decides how strict the live gate is.
2. **Live validation** (``install_modulations``) — each selection is measured
   on a real, FRESH batch of the CURRENT model before it sticks. Recurring
   winners (count>=2) must not get worse; single-shot winners must STRICTLY
   improve (probe batch + install batch on the same weights = 2-fold
   cross-batch validation). A stale winner from an older checkpoint cannot
   hurt the new run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F

from neuroslm.genetic.language import Program

_SITE_RE = re.compile(r"_L(\d+)_step\d+$")


def parse_site(name: str):
    """Layer index from a probe-tagged record name, or None."""
    m = _SITE_RE.search(name)
    return int(m.group(1)) if m else None


@dataclass
class Selection:
    layer: int
    program: Program
    mean_delta: float
    count: int
    name: str            # representative (best-Δ) record


def select_installable(records: Sequence, *, min_count: int = 1,
                       min_delta: float = 0.0) -> List[Selection]:
    """Recurrence gate: group by (site, program semantics), keep groups seen
    >= min_count times with mean Δ > min_delta; best group per layer."""
    from neuroslm.genetic.normalize import semantic_signature

    groups: dict = {}
    for r in records:
        k = parse_site(r.name)
        if k is None:
            continue
        try:
            sig = semantic_signature(r.program)
        except Exception:
            sig = r.program.to_source()
        groups.setdefault((k, sig), []).append(r)

    per_layer: dict = {}
    for (k, _sig), rs in groups.items():
        if len(rs) < min_count:
            continue
        deltas = [float(r.metrics.get("delta_ce", 0.0)) for r in rs]
        mean_d = sum(deltas) / len(deltas)
        if mean_d <= min_delta:
            continue
        best = max(rs, key=lambda r: float(r.metrics.get("delta_ce", 0.0)))
        sel = Selection(layer=k, program=best.program, mean_delta=mean_d,
                        count=len(rs), name=best.name)
        cur = per_layer.get(k)
        if cur is None or sel.mean_delta > cur.mean_delta:
            per_layer[k] = sel
    return sorted(per_layer.values(), key=lambda s: -s.mean_delta)


def install_modulations(lm, selections: Sequence[Selection], *, val_fn=None,
                        tol: float = 1e-4, strict_improve: float = 1e-3) -> dict:
    """Live-validation gate, count-aware. Installs accumulate, so each
    candidate is judged on top of the ones already accepted.

    - ``count >= 2`` (recurred across probes): lenient — CE must not get worse.
    - ``count == 1`` (banked once): strict — CE must IMPROVE by at least
      ``strict_improve`` on this fresh batch. Probe batch + install batch on
      the same weights = 2-fold cross-batch validation, so a single-shot
      winner that generalizes earns its install; batch-specific noise doesn't.
    """
    from neuroslm.genetic.neuro_evolve import _make_modulator

    installed, rejected = [], []
    for sel in selections:
        entry = {"name": sel.name, "layer": sel.layer,
                 "mean_delta": sel.mean_delta, "count": sel.count}
        mod = _make_modulator(sel.program)
        prev = lm._layer_modulations.get(sel.layer)
        before = val_fn() if val_fn is not None else None
        lm._layer_modulations[sel.layer] = mod
        if val_fn is not None:
            after = val_fn()
            entry["ce_before"], entry["ce_after"] = before, after
            required = (before + tol) if sel.count >= 2 else (before - strict_improve)
            if not (after <= required):
                if prev is None:
                    lm._layer_modulations.pop(sel.layer, None)
                else:
                    lm._layer_modulations[sel.layer] = prev
                rejected.append(entry)
                continue
        installed.append(entry)
    return {"installed": installed, "rejected": rejected}


def install_from_store(lm, store_dir, ids: torch.Tensor, targets: torch.Tensor,
                       *, min_count: int = 1, min_delta: float = 0.0) -> dict:
    """Load ``modulations/``, apply both gates against the live model, install."""
    from neuroslm.genetic.modulation_store import ModulationStore

    store_dir = Path(store_dir)
    if not store_dir.exists():
        return {"installed": [], "rejected": []}
    records = ModulationStore(store_dir).list_all()
    selections = select_installable(records, min_count=min_count,
                                    min_delta=min_delta)
    if not selections:
        return {"installed": [], "rejected": []}

    was_training = lm.training
    lm.eval()
    try:
        def val_fn() -> float:
            with torch.no_grad():
                logits = lm(ids)
                return float(F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)))

        return install_modulations(lm, selections, val_fn=val_fn)
    finally:
        if was_training:
            lm.train()
