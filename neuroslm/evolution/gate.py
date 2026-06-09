# -*- coding: utf-8 -*-
"""L4 — gate proposals.

Wraps :class:`neuroslm.verification.improvement_gate.ImprovementGate`
(+ optional :class:`neuroslm.verification.triple_guard.TripleGuard`)
to admit/reject :class:`~neuroslm.compiler.ribosome.DNAPatch`
proposals coming out of L3 (:func:`neuroslm.evolution.mutator.
propose_mutations`).

Design contract
---------------
``gate_proposals(proposals, evidence_by_target, *, ...) ->
(admitted, rejected)`` — the two return lists are disjoint and
partition the input. Every patch carries the verdict (or rejection
reasons) in its ``metadata`` so the audit trail is preserved end-to-
end without a sidecar log.

Direction policy
----------------
Each DNAPatch ``kind`` has a default direction interpretation. The
caller can override per-evidence (``ImprovementEvidence.direction``)
or per-call (``default_direction``):

============  ==========  =======================================
kind          direction   typical metric
============  ==========  =======================================
node_mutation  increase    Φ / accuracy / intelligence-density
edge_strengthen  decrease    ppl / loss / OOD-gap
edge_prune        decrease    no-regression on ppl
============  ==========  =======================================

This matches §10.2 of ``docs/formal_framework.md``: a hot node
admitting a non-negative coupling cannot decrease Φ (H001), so
``direction="increase"`` is the right Welch hypothesis; a
strengthened edge / pruned edge should not regress generation
perplexity (H002), so ``direction="decrease"``.

Integration with L5 (Lean)
--------------------------
This module is the *empirical* gate. The Lean gate (L5) plugs in
behind the same surface: a future ``lean_backend`` kwarg lets the
caller short-circuit empirical evaluation when the mutation form
matches a hypothesis ``Brian.PhiMonotone``-style theorem in
``hypothesis/proofs/``. See plan ``docs/heatmap_evolution_plan.md``
§L5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable,
)

from neuroslm.compiler.ribosome import DNAPatch
from neuroslm.verification.improvement_gate import (
    ImprovementGate, ImprovementVerdict,
)


# ── public surface ────────────────────────────────────────────────


__all__ = [
    "ImprovementEvidence",
    "gate_proposals",
    "DEFAULT_DIRECTION_BY_KIND",
]


# Per-kind direction defaults. Used unless the caller pins direction
# on the evidence or passes ``default_direction``.
DEFAULT_DIRECTION_BY_KIND: Dict[str, str] = {
    "node_mutation":   "increase",   # Φ / accuracy
    "edge_strengthen": "decrease",   # ppl / loss
    "edge_prune":      "decrease",   # no-regression on ppl
}


@dataclass
class ImprovementEvidence:
    """Empirical evidence for one proposal's target.

    ``direction`` is optional; if ``None`` the gate falls back to the
    per-kind default (see :data:`DEFAULT_DIRECTION_BY_KIND`) and then
    to the call-level ``default_direction`` argument.
    """
    before: Sequence[float]
    after:  Sequence[float]
    direction: Optional[str] = None


# ── duck-typed TripleGuard adapter ────────────────────────────────


@runtime_checkable
class _TripleGuardLike(Protocol):
    def admit(self, before: Any, after: Any, mutation: Any = None) -> Any: ...


# ── helpers ───────────────────────────────────────────────────────


def _resolve_direction(
    *,
    evidence: ImprovementEvidence,
    patch: DNAPatch,
    default_direction: Optional[str],
) -> str:
    if evidence.direction is not None:
        return evidence.direction
    if patch.kind in DEFAULT_DIRECTION_BY_KIND:
        return DEFAULT_DIRECTION_BY_KIND[patch.kind]
    if default_direction is not None:
        return default_direction
    raise ValueError(
        f"no direction for patch kind {patch.kind!r} on target "
        f"{patch.target!r}: not in DEFAULT_DIRECTION_BY_KIND, and "
        f"neither evidence.direction nor default_direction was set"
    )


def _annotate_admitted(
    patch: DNAPatch,
    *,
    improvement_verdict: Optional[ImprovementVerdict] = None,
    triple_guard_verdict: Optional[Any] = None,
    lean_verdict: Optional[Any] = None,
) -> DNAPatch:
    md = dict(patch.metadata)
    if improvement_verdict is not None:
        md["gate_verdict"] = improvement_verdict.to_dict()
    if triple_guard_verdict is not None:
        md["triple_guard_verdict"] = _verdict_to_dict(triple_guard_verdict)
    if lean_verdict is not None:
        md["lean_verdict"] = _lean_verdict_to_dict(lean_verdict)
    return DNAPatch(
        version=patch.version, step=patch.step, kind=patch.kind,
        target=patch.target, delta=list(patch.delta), metadata=md,
    )


def _annotate_rejected(
    patch: DNAPatch,
    reasons: List[str],
    *,
    lean_verdict: Optional[Any] = None,
) -> DNAPatch:
    md = dict(patch.metadata)
    md["rejection_reasons"] = list(reasons)
    if lean_verdict is not None:
        md["lean_verdict"] = _lean_verdict_to_dict(lean_verdict)
    return DNAPatch(
        version=patch.version, step=patch.step, kind=patch.kind,
        target=patch.target, delta=list(patch.delta), metadata=md,
    )


def _verdict_to_dict(verdict: Any) -> Dict[str, Any]:
    """Coerce a TripleGuard-style verdict to a dict for metadata
    embedding. Falls back to ``{"admitted": bool(verdict.admitted),
    "reasons": [...]}`` if ``to_dict`` is absent."""
    if hasattr(verdict, "to_dict"):
        return verdict.to_dict()
    return {
        "admitted": bool(getattr(verdict, "admitted", False)),
        "reasons":  list(getattr(verdict, "reasons", [])),
    }


def _lean_verdict_to_dict(verdict: Any) -> Dict[str, Any]:
    """Coerce a LeanVerdict to a dict for metadata embedding."""
    return {
        "status":   str(getattr(verdict, "status", "unknown")),
        "file":     str(getattr(verdict, "file", "")),
        "errors":   list(getattr(verdict, "errors", [])),
        "warnings": list(getattr(verdict, "warnings", [])),
        "n_sorry":  int(getattr(verdict, "n_sorry", 0)),
    }


# ── public API ────────────────────────────────────────────────────


def gate_proposals(
    proposals: Sequence[DNAPatch],
    evidence_by_target: Dict[str, ImprovementEvidence],
    *,
    improvement_gate: Optional[ImprovementGate] = None,
    triple_guard: Optional[_TripleGuardLike] = None,
    structural_by_target: Optional[Dict[str, Tuple[Any, Any]]] = None,
    default_direction: Optional[str] = None,
    lean_backend: Optional[Any] = None,
) -> Tuple[List[DNAPatch], List[DNAPatch]]:
    """Admit/reject proposals via Improvement (+ optional TripleGuard) gates.

    Parameters
    ----------
    proposals
        L3 :class:`DNAPatch` proposals (typically the output of
        :func:`neuroslm.evolution.mutator.propose_mutations`).
    evidence_by_target
        ``{patch.target: ImprovementEvidence}`` map. A proposal whose
        target is missing from this dict is rejected with the
        ``no_evidence`` reason (never silently admitted) — *unless* a
        ``lean_backend`` admits it first.
    improvement_gate
        Optional pre-configured gate; defaults to
        ``ImprovementGate()``.
    triple_guard
        Optional structural gate (anything with an ``admit(before,
        after, mutation=None) -> Verdict`` method). When supplied, a
        proposal is admitted iff *both* the improvement gate AND the
        triple guard admit. Proposals with no structural evidence
        in ``structural_by_target`` skip the TripleGuard check (no
        evidence → no opinion → no rejection).
    structural_by_target
        ``{patch.target: (before_chk, after_chk)}`` map fed to the
        TripleGuard.
    default_direction
        Fallback direction if neither the per-evidence override nor
        the per-kind default applies (rare; only used for custom kinds).
    lean_backend
        Optional Lean proof backend (typically
        :class:`neuroslm.evolution.lean_gate.LeanProofBackend`).
        Must expose ``admit_proposal(patch) -> Optional[LeanVerdict]``.
        When provided and the verdict is ``status='verified'``, the
        proposal is admitted **without** consulting the empirical or
        structural gates — a formal proof strictly dominates a
        statistical signal. Any other Lean status falls through to
        the empirical gate. This kwarg is purely additive: it can
        only admit proposals the empirical gate would have rejected;
        it can never reject one the empirical gate would have admitted.

    Returns
    -------
    (admitted, rejected) — disjoint partition of ``proposals``.
    Every patch carries either a ``gate_verdict`` (admitted via
    empirical gate), a ``lean_verdict`` (admitted via Lean, or
    consulted-but-fell-through), or a ``rejection_reasons`` list.
    """
    gate = improvement_gate or ImprovementGate()
    structural_by_target = structural_by_target or {}

    admitted: List[DNAPatch] = []
    rejected: List[DNAPatch] = []

    for patch in proposals:
        # ── Lean short-circuit (only if backend supplied) ──────────
        lean_verdict = None
        if lean_backend is not None:
            try:
                lean_verdict = lean_backend.admit_proposal(patch)
            except Exception as exc:
                lean_verdict = None
                # We swallow backend exceptions; the Lean kwarg is
                # purely additive (see docstring). Surface via metadata
                # so the auditor can see we tried.
                # (No reject; we just fall through.)
                lean_verdict_meta_extra = {"lean_backend_error": str(exc)}
            else:
                lean_verdict_meta_extra = {}

            if (lean_verdict is not None
                    and getattr(lean_verdict, "status", "") == "verified"):
                admitted.append(_annotate_admitted(
                    patch, lean_verdict=lean_verdict,
                ))
                continue
        else:
            lean_verdict_meta_extra = {}

        # ── empirical (Improvement) gate ───────────────────────────
        evidence = evidence_by_target.get(patch.target)
        if evidence is None:
            rejected.append(_annotate_rejected(
                patch,
                [f"no_evidence for target {patch.target!r} — proposal "
                 f"cannot be admitted without before/after samples"],
                lean_verdict=lean_verdict,
            ))
            continue

        try:
            direction = _resolve_direction(
                evidence=evidence, patch=patch,
                default_direction=default_direction,
            )
            improvement_verdict = gate.admit(
                evidence.before, evidence.after, direction=direction,
            )
        except ValueError as exc:
            rejected.append(_annotate_rejected(
                patch, [str(exc)], lean_verdict=lean_verdict,
            ))
            continue

        if not improvement_verdict.admitted:
            rejected.append(_annotate_rejected(
                patch, improvement_verdict.reasons,
                lean_verdict=lean_verdict,
            ))
            continue

        # ── optional structural (TripleGuard) gate ─────────────────
        triple_verdict = None
        if triple_guard is not None:
            struct_pair = structural_by_target.get(patch.target)
            if struct_pair is not None:
                before_chk, after_chk = struct_pair
                triple_verdict = triple_guard.admit(
                    before_chk, after_chk, mutation={
                        "kind": patch.kind,
                        "target": patch.target,
                        "delta": list(patch.delta),
                        "metadata": dict(patch.metadata),
                    },
                )
                if not getattr(triple_verdict, "admitted", False):
                    reasons = list(getattr(triple_verdict, "reasons", []))
                    rejected.append(_annotate_rejected(
                        patch,
                        # Surface the improvement verdict too so the
                        # auditor sees "empirical OK, structural NO".
                        [f"[triple_guard] {r}" for r in reasons]
                        + [f"[improvement] (admitted) effect="
                           f"{improvement_verdict.effect:+.6g}, "
                           f"p={improvement_verdict.p_value:.4g}"],
                        lean_verdict=lean_verdict,
                    ))
                    continue

        admitted.append(_annotate_admitted(
            patch,
            improvement_verdict=improvement_verdict,
            triple_guard_verdict=triple_verdict,
            lean_verdict=lean_verdict,
        ))

    return admitted, rejected
