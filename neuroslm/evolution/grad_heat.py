# -*- coding: utf-8 -*-
"""Gradient heat collector — map model grads onto the Hypergraph IR.

The training signal for the heatmap is gradient magnitude. This module
turns ``model.named_parameters()`` gradients into per-IR-element signals:

  - node signal = L2 aggregate of the grad-norms of every parameter whose
    top-level name token matches the node's name (e.g. params ``gws.*``
    -> node ``population:gws``). An ``alias`` map handles model/IR name
    mismatches (e.g. ``hippo`` -> ``hippocampus``).
  - edge signal = mean of its endpoint nodes' signals, so synapses /
    modulations between hot modules read hot.

``update_heatmap`` folds the signals into a TrainingHeatmap and (when a
publisher is given) triggers the cadence-based commit/push.
"""
from __future__ import annotations
import math
from typing import Dict, Iterable, Optional, Tuple


def parameter_grad_norms(named_parameters: Iterable) -> Dict[str, float]:
    """name -> L2 norm of its gradient (params without a grad are skipped)."""
    out: Dict[str, float] = {}
    for name, param in named_parameters:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        try:
            out[name] = float(grad.detach().norm(2).item())
        except AttributeError:
            # Allow plain floats/sequences in tests.
            try:
                out[name] = float(grad)
            except (TypeError, ValueError):
                continue
    return out


def signals_from_grad_norms(
    grad_norms: Dict[str, float],
    ir,
    alias: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Map parameter grad-norms to node+edge signals over the IR.

    Returns (signals, kinds) ready for ``TrainingHeatmap.update``.
    """
    alias = alias or {}
    name_to_id = {n.name: n.id for n in ir.nodes}

    # ── node signals: L2 aggregate per matched node ──────────────────────
    sumsq: Dict[str, float] = {}
    for pname, gn in grad_norms.items():
        token = pname.split(".", 1)[0]
        node_name = alias.get(token, token)
        nid = name_to_id.get(node_name)
        if nid is None:
            continue
        sumsq[nid] = sumsq.get(nid, 0.0) + float(gn) ** 2

    signals: Dict[str, float] = {nid: math.sqrt(s) for nid, s in sumsq.items()}
    kinds: Dict[str, str] = {nid: "node" for nid in signals}

    # ── edge signals: mean of endpoint node signals ──────────────────────
    for edge in ir.hyperedges:
        vals = []
        for member in edge.members:
            mid = name_to_id.get(member)
            if mid is not None:
                vals.append(signals.get(mid, 0.0))
        edge_signal = (sum(vals) / len(vals)) if vals else 0.0
        signals[edge.id] = edge_signal
        kinds[edge.id] = "edge"

    return signals, kinds


def update_heatmap(
    heatmap,
    grad_norms: Dict[str, float],
    ir,
    step: Optional[int] = None,
    publisher=None,
    alias: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Fold model grad-norms into the heatmap; trigger publisher cadence.

    Returns the per-element signals that were folded in.
    """
    signals, kinds = signals_from_grad_norms(grad_norms, ir, alias=alias)
    heatmap.update(signals, kinds=kinds, step=step)
    if publisher is not None and step is not None:
        publisher.maybe_publish(heatmap, step)
    return signals
