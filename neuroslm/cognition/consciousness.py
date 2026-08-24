# -*- coding: utf-8 -*-
"""Real IIT-flavored Φ + a GWT-flavored broadcast-strength proxy for
CognitiveRuntime's tick loop (architecture.md §14.11).

Four separate, non-integrated "Φ" implementations existed in this repo
before this module: the cheap softmax-entropy proxy CognitiveRuntime
already reports as ``phi_proxy`` (honest, but not integration-theoretic),
a covariance-norm proxy behind the orphaned TripleGuard, the fully
orphaned ``neuroslm.thsd.engine.PhiDynamicsComputer`` (a correlation
stub matching formal_framework.md §6.2's own admission that it is "a
tractable proxy" pending "a real algorithm" — CLAUDE.md §12 documents
its real job as Lean-proof vocabulary, not a runtime metric), and the
one genuinely IIT-flavored estimator that actually exists:
``NeuralOrchestrator.gaussian_mi_mip_phi`` — Gaussian mutual information
over an exhaustive minimum-information-partition search, wired into
Brain's own forward pass and loss.

Rather than inventing a fifth proxy, this module reuses that estimator
verbatim for CognitiveRuntime's tick loop — the "mind" that is actually
deployed and running, but has no NFG module graph like Brain's to
partition. ``bucket_reduce`` is the only new piece: it makes
CognitiveRuntime's real per-tick subsystem signals (a 7-dim NT vector
next to hundreds-dim embeddings) comparable on equal footing before
handing them to the shared estimator.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def bucket_reduce(vec: Sequence[float], k: int = 8) -> List[float]:
    """Deterministic mean-pool of ``vec`` into ``k`` contiguous buckets.

    ``gaussian_mi_mip_phi``'s caller in ``NeuralOrchestrator`` truncates
    every module row to the length of the SHORTEST vector
    (``_stack_module_outputs``) — fine when every module comes off the
    same NFG trunk at comparable width, but CognitiveRuntime's natural
    per-tick signals are wildly heterogeneous (the 7-dim NT vector vs.
    an ``embed_dim()``-sized thought/recall/sensory embedding). Naively
    stacking-then-truncating would collapse every embedding down to its
    first few components. Bucket-reducing to one shared fixed width
    first keeps every module's full extent represented, however long or
    short the source vector.
    """
    v = list(vec)
    n = len(v)
    k = max(1, int(k))
    if n == 0:
        return [0.0] * k
    out: List[float] = []
    for i in range(k):
        lo = (i * n) // k
        hi = ((i + 1) * n) // k
        if hi <= lo:
            # Fewer source elements than buckets — this bucket sees none.
            out.append(0.0)
            continue
        seg = v[lo:hi]
        out.append(sum(seg) / len(seg))
    return out


def compute_phi_iit(modules: Dict[str, Sequence[float]], k: int = 8
                    ) -> Tuple[float, int]:
    """Real IIT-flavored Φ over this tick's named subsystem vectors.

    Bucket-reduces every module vector to width ``k``, stacks into an
    ``(n, k)`` matrix, and hands it to
    :meth:`NeuralOrchestrator.gaussian_mi_mip_phi` — reused verbatim,
    not reimplemented. Returns ``(0.0, n)`` when fewer than 2 modules
    have content this tick (nothing to partition) or on any numerical
    failure (mirrors ``NeuralOrchestrator.compute_phi_proxy``'s own
    NaN/Inf guard — telemetry must never surface a broken number).
    """
    names = [name for name, v in modules.items() if v]
    n = len(names)
    if n < 2:
        return 0.0, n

    import torch

    from neuroslm.intelligence.orchestrator import NeuralOrchestrator

    rows = [bucket_reduce(modules[name], k=k) for name in names]
    M = torch.tensor(rows, dtype=torch.float32)
    try:
        phi = NeuralOrchestrator.gaussian_mi_mip_phi(M)
        val = float(phi.item())
    except Exception:
        return 0.0, n
    if val != val or val in (float("inf"), float("-inf")):
        return 0.0, n
    return max(0.0, val), n


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Pure-python cosine, same style as EpisodicMemory.retrieve_scored's
    own ``_cos`` — zero-norm inputs read as "no signal" (0.0), not an
    error, since a degenerate tick's telemetry must stay a plain float."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def compute_broadcast_strength(modules: Dict[str, Sequence[float]],
                               winner_key: str, k: int = 8) -> float:
    """GWT-flavored proxy: cosine(winner vector, mean of every OTHER
    module vector this tick) — how much the winning content correlates
    with the rest of the tick's subsystem state, a zero-training stand-
    in for Dehaene's "ignition" (a winning coalition becomes globally
    accessible).

    Chosen over porting ``neuroslm.modules.workspace.GlobalWorkspace``:
    that module needs LEARNED parameters (slot queries, thresholds)
    trained as part of the NFG trunk — there is no checkpoint for them
    on CognitiveRuntime's frozen-HF-expert escape hatch, and
    instantiating it untrained would be exactly the decorative stub
    CLAUDE.md §14 forbids. This is an honest proxy, like this
    codebase's existing ``selection_entropy``/``differentiation`` — not
    a claim of rigor.

    Returns 0.0 if ``winner_key`` is missing/empty or fewer than 2
    modules total have content this tick.
    """
    winner = modules.get(winner_key)
    others = [v for name, v in modules.items() if name != winner_key and v]
    if not winner or not others:
        return 0.0
    w = bucket_reduce(winner, k=k)
    reduced_others = [bucket_reduce(o, k=k) for o in others]
    mean_other = [sum(col) / len(reduced_others) for col in zip(*reduced_others)]
    return _cosine(w, mean_other)
