# -*- coding: utf-8 -*-
"""Multi-site discovery probe — search only the *optimizable* regions of a trunk.

H46's structural null taught the lesson this module encodes: the terminal
hidden state is the single most end-to-end-optimized point in the network, so
a post-hoc modulation there has nothing to exploit (measured Δ=0). Intermediate
layers are different — their outputs are shaped by *indirect* pressure only, so
real slack can survive there. The Lion-style recipe is: search cheap candidates
where the objective is measured on the REAL loss, and only spend budget where a
cheap headroom scan proves the loss surface still moves.

Three stages:

1. ``headroom_scan``   — per layer, apply a small deterministic battery of
   structured perturbations to that layer's output, re-run the TRUE tail
   (``forward_from_layer``), and record (a) sensitivity — mean |ΔCE|, i.e.
   does this site have any leverage on the loss at all — and (b) improvement —
   best CE drop, i.e. does a trivial perturbation ALREADY beat the trained
   forward. improvement > 0 is measured slack: the site is under-optimized.
2. ``select_sites``    — spend the GA budget on measured slack first; skip
   insensitive sites always; fall back to the most promising sensitive site
   (least-negative improvement) when nothing shows trivial slack.
3. ``probe_optimizable_regions`` — run the NGL modulation search at the chosen
   sites, scoring every candidate by the true next-token CE through the real
   tail. Strictly read-only (no weight/buffer/stash writes, mode restored);
   winners persist to the modulation store with their measured Δ and site.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
TailFn = Callable[[int, Tensor], Tensor]     # (layer_index, hidden) -> logits


@dataclass
class SiteReport:
    layer: int
    sensitivity: float          # mean |ΔCE| over the battery — loss leverage
    improvement: float          # best (baseline_ce - perturbed_ce); >0 = slack
    best_perturbation: str      # which battery member achieved `improvement`

    def line(self) -> str:
        mark = "←slack" if self.improvement > 0 else (
            "flat" if self.sensitivity < 1e-6 else "tight")
        return (f"L{self.layer}: sens={self.sensitivity:.4g} "
                f"improve={self.improvement:+.4g} ({self.best_perturbation}) {mark}")


def _ce(logits: Tensor, targets: Tensor) -> float:
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 targets.reshape(-1)))


def _battery(h: Tensor, seed: int):
    """Small, deterministic, structured perturbations — cheap slack detectors."""
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(h.shape, generator=g).to(device=h.device, dtype=h.dtype)
    scale = float(h.detach().float().std()) * 0.05 + 1e-8
    return [
        ("scale_up", h * 1.05),
        ("scale_dn", h * 0.95),
        ("damp",     h * 0.8),
        ("noise",    h + noise * scale),
    ]


def headroom_scan(layers: Sequence[Tensor], tail_fn: TailFn, targets: Tensor,
                  *, seed: int = 0, baseline_ce: Optional[float] = None
                  ) -> List[SiteReport]:
    """Measure per-site loss leverage + trivial slack under the true tail."""
    with torch.no_grad():
        if baseline_ce is None:
            k_last = len(layers) - 1
            baseline_ce = _ce(tail_fn(k_last, layers[k_last]), targets)
        reports: List[SiteReport] = []
        for k, h in enumerate(layers):
            deltas, best_gain, best_name = [], float("-inf"), "-"
            for name, hp in _battery(h, seed + k):
                ce = _ce(tail_fn(k, hp), targets)
                deltas.append(abs(ce - baseline_ce))
                gain = baseline_ce - ce
                if gain > best_gain:
                    best_gain, best_name = gain, name
            reports.append(SiteReport(
                layer=k,
                sensitivity=float(sum(deltas) / max(1, len(deltas))),
                improvement=float(best_gain),
                best_perturbation=best_name))
    return reports


def select_sites(reports: Sequence[SiteReport], *, top_k: int = 1,
                 slack_tol: float = 1e-4, min_sensitivity: float = 1e-6
                 ) -> List[int]:
    """Budget policy: measured slack first; never insensitive sites."""
    sensitive = [r for r in reports if r.sensitivity >= min_sensitivity]
    slack = sorted((r for r in sensitive if r.improvement > slack_tol),
                   key=lambda r: -r.improvement)
    if slack:
        return [r.layer for r in slack[:top_k]]
    # No trivial slack anywhere — search the most promising sensitive site
    # (least-negative improvement) speculatively; the GA can find nonlinear
    # wins the battery cannot.
    speculative = sorted(sensitive, key=lambda r: -r.improvement)
    return [r.layer for r in speculative[:top_k]]


def probe_optimizable_regions(lm, ids: Tensor, targets: Tensor, *,
                              store=None, config=None, step: int = 0,
                              run_id: str = "trunk", top_k: int = 1,
                              seed: int = 0, progress=None) -> dict:
    """Headroom-gated multi-site NGL modulation search on a real trunk.

    ``lm`` must expose ``forward_from_layer`` and stash ``_last_layer_outputs``
    on forward (``DSLLanguageCortex``). Read-only by construction: everything
    runs under ``torch.no_grad()`` in eval mode, and the training/eval mode is
    restored on exit.
    """
    from neuroslm.genetic.ledger import SearchLedger
    from neuroslm.genetic.training_explorer import probe_hidden_modulation

    def _say(msg: str) -> None:
        if progress is not None:
            progress(msg)
        else:
            print(msg, flush=True)

    was_training = lm.training
    lm.eval()
    try:
        with torch.no_grad():
            logits = lm(ids)
            stashed = getattr(lm, "_last_layer_outputs", None)
            if not stashed:
                raise RuntimeError("model did not stash _last_layer_outputs")
            layers = [o.detach() for o in stashed]
            baseline_ce = _ce(logits, targets)

        tail_fn: TailFn = lambda k, h: lm.forward_from_layer(k, h)
        reports = headroom_scan(layers, tail_fn, targets, seed=seed,
                                baseline_ce=baseline_ce)
        _say(f"[probe @ step {step}] baseline_ce={baseline_ce:.4f} | "
             + " | ".join(r.line() for r in reports))
        searched = select_sites(reports, top_k=top_k)

        best_ce, evaluated, saved_names, results = baseline_ce, 0, [], {}
        for k in searched:
            res = probe_hidden_modulation(
                layers[k],
                lambda h, _k=k: lm.forward_from_layer(_k, h),
                targets,
                ledger=SearchLedger(":memory:"),   # trunk moves every probe —
                store=store,                        # cross-checkpoint dedup is wrong
                config=config, step=step, run_id=f"{run_id}-L{k}")
            results[k] = res
            evaluated += res.get("evaluated", 0)
            if res["best_ce"] < best_ce:
                best_ce = res["best_ce"]
            if res.get("saved"):
                saved_names.append(res["saved"])
    finally:
        if was_training:
            lm.train()

    delta = baseline_ce - best_ce
    return {
        "baseline_ce": baseline_ce,
        "best_ce": best_ce,
        "delta_ce": delta,
        "improved": delta > 1e-6,
        "saved": ",".join(saved_names) if saved_names else None,
        "evaluated": evaluated,
        "reports": reports,
        "searched": searched,
        "results": results,
    }
