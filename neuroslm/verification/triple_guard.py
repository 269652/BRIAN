# -*- coding: utf-8 -*-
"""Triple Guard — the live admission gate for evolutionary mutations.

Implements the **normative gate** specified in
``docs/formal_framework.md`` §7:

.. math::

    \\Phi(K) > 0 \\;\\wedge\\;
    \\|H^1(K;F)\\| \\to 0 \\;\\wedge\\;
    \\lambda_1(L_F) > \\lambda_{\\min}

A trained checkpoint — or, more interestingly, a *candidate mutation* —
is **admitted** only if it satisfies all three guards simultaneously
compared to the architecture's *before* state.  Rejected mutations are
never persisted into the RAID-5 DNA stream; instead they are written
to a sibling ``step_<N>.rejected.json`` audit file so the human (or
the next-generation evolutionary loop) can see what the architecture
refused to become.

Mathematical primitives are delegated to the already-shipped
``CohomologyValidator`` in :mod:`neuroslm.verification.verifier`.  This
module is therefore a *composer*, not a new mathematical model.

Surfaces exposed
----------------
* :class:`Verdict`            — structured admit/reject record
* :class:`TripleGuard`        — the gate itself, with
    - ``admit(before, after, mutation) -> Verdict``
    - ``from_arch_neuro(path)`` factory that reads the
      ``formal_spec { triple_guard { ... } }`` block

Design notes
------------
The decision is *strict*: a mutation is admitted iff **all three**
guards pass.  Each guard returns a structured reason on rejection so
the verdict.reasons list tells the auditor exactly which invariant
was violated.

The thresholds are intentionally configurable per architecture (via
the DSL block) so the rcc_bowtie can choose, e.g., a permissive
``phi_min = 0.0`` during early infancy training and tighten it after
awakening.  See ``docs/formal_framework.md`` §6.3 and §7.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from neuroslm.dsl.thg_ir import THGCheckpoint
from neuroslm.verification.verifier import (
    CohomologyValidator,
    InvariantChecker,
)


# ──────────────────────────────────────────────────────────────────────
# Verdict
# ──────────────────────────────────────────────────────────────────────


@dataclass
class Verdict:
    """The structured result of a single :py:meth:`TripleGuard.admit`
    call.

    Attributes
    ----------
    admitted : bool
        ``True`` iff the candidate mutation passed all three guards.
    phi_before, phi_after : float
        Integrated information (Φ) of the architecture before and after
        applying the candidate mutation.
    h1_before, h1_after : float
        Norm-proxy for ``‖H¹(K;F)‖`` before and after the mutation.
    lambda_before, lambda_after : float
        Sheaf-Laplacian spectral-gap proxy (λ₁) before and after.
    reasons : list[str]
        One human-readable reason per failed guard; empty when
        ``admitted is True``.  Each reason mentions the guard symbol
        (``"Phi"``, ``"H1"``, ``"lambda"``) so callers can ``grep`` on
        a single canonical token.
    """

    admitted: bool
    phi_before: float
    phi_after: float
    h1_before: float
    h1_after: float
    lambda_before: float
    lambda_after: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form for embedding in patch metadata."""
        return {
            "admitted": bool(self.admitted),
            "phi_before": float(self.phi_before),
            "phi_after": float(self.phi_after),
            "h1_before": float(self.h1_before),
            "h1_after": float(self.h1_after),
            "lambda_before": float(self.lambda_before),
            "lambda_after": float(self.lambda_after),
            "reasons": list(self.reasons),
        }


# ──────────────────────────────────────────────────────────────────────
# TripleGuard
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TripleGuard:
    """The composite (Φ ∧ H¹ ∧ λ₁) admission gate.

    Parameters
    ----------
    phi_min : float
        Minimum admissible Φ *after* the mutation.  Default ``0.0`` —
        purely safety net (rejects only when Φ becomes negative, which
        can happen with numerically unstable proxies); set to a strict
        positive value via the DSL block to enforce the §6.3 Φ Guard.
    h1_max : float
        Maximum admissible ``‖H¹‖`` proxy *after* the mutation.
        Default ``5.0`` matches ``CohomologyValidator``'s built-in
        hallucination threshold.
    lambda_min : float
        Minimum admissible spectral gap λ₁ *after* the mutation.
        Default ``0.0`` — disabled until §4.2 Tonnetz filter lands as a
        real Laplacian computation; for now we use a connectedness
        proxy (≥ 0 iff the graph is non-empty).
    """

    phi_min: float = 0.0
    h1_max: float = 5.0
    lambda_min: float = 0.0

    # Composed validators — exposed so callers can substitute custom
    # implementations (e.g. once the Tonnetz λ₁ from formal_framework
    # §4.2 lands, swap in the real Laplacian computer).
    cohomology: CohomologyValidator = field(default_factory=CohomologyValidator)
    invariants: InvariantChecker = field(default_factory=InvariantChecker)

    # ── Public API ──────────────────────────────────────────────────

    def admit(
        self,
        before: THGCheckpoint,
        after: THGCheckpoint,
        mutation: Optional[Dict[str, Any]] = None,
    ) -> Verdict:
        """Decide whether a candidate mutation may be persisted.

        Parameters
        ----------
        before, after : THGCheckpoint
            The architecture state immediately before and after the
            mutation is applied.  ``after`` is a *candidate*: this
            method never mutates it (see
            ``TestTripleGuardDecisions.test_admit_does_not_mutate_inputs``).
        mutation : dict, optional
            The mutation dictionary as used by ``EvolutionaryTrainingContext``
            (kind / target / delta / metadata).  Currently unused by
            the decision but accepted so the signature is stable for
            future per-mutation heuristics.

        Returns
        -------
        Verdict
            A structured record of the decision and the six scores.
        """
        del mutation  # not needed for the current decision

        phi_before = self._safe_phi(before)
        phi_after = self._safe_phi(after)
        h1_before = self._h1_norm(before)
        h1_after = self._h1_norm(after)
        lambda_before = self._lambda_proxy(before)
        lambda_after = self._lambda_proxy(after)

        reasons: List[str] = []

        # Φ guard — see formal_framework.md §6.3.
        if not (phi_after >= self.phi_min and math.isfinite(phi_after)):
            reasons.append(
                f"Phi guard violated: phi_after={phi_after:.6g} "
                f"< phi_min={self.phi_min:.6g}"
            )

        # H¹ guard — see formal_framework.md §2.3.
        if not (h1_after <= self.h1_max and math.isfinite(h1_after)):
            reasons.append(
                f"H1 guard violated: h1_after={h1_after:.6g} "
                f"> h1_max={self.h1_max:.6g}"
            )

        # λ₁ guard — see formal_framework.md §4.2 (Tonnetz).
        if not (lambda_after >= self.lambda_min and math.isfinite(lambda_after)):
            reasons.append(
                f"lambda guard violated: lambda_after={lambda_after:.6g} "
                f"< lambda_min={self.lambda_min:.6g}"
            )

        return Verdict(
            admitted=(len(reasons) == 0),
            phi_before=phi_before,
            phi_after=phi_after,
            h1_before=h1_before,
            h1_after=h1_after,
            lambda_before=lambda_before,
            lambda_after=lambda_after,
            reasons=reasons,
        )

    # ── Factory ─────────────────────────────────────────────────────

    @classmethod
    def from_arch_neuro(cls, path: str) -> "TripleGuard":
        """Build a TripleGuard from an ``arch.neuro`` file.

        Reads any ``formal_spec NAME { rule: "triple_guard", phi_min: ...,
        h1_max: ..., lambda_min: ... }`` block in the file and uses its
        thresholds.  When no such block exists, returns a default
        guard (still wired into the evolution loop so the safety net
        is always present).

        We parse with a small regex rather than going through the full
        ``NeuroMLCompiler`` because ``formal_spec`` blocks live at the
        top level of arch.neuro and the compiler currently flattens
        their fields into a generic ``properties`` dict — we want the
        raw numeric thresholds with no IR round-trip.
        """
        text = Path(path).read_text(encoding="utf-8")
        block = _extract_triple_guard_block(text)
        if block is None:
            return cls()  # default thresholds — opt-in safety net
        return cls(
            phi_min=float(block.get("phi_min", cls.phi_min)),
            h1_max=float(block.get("h1_max", cls.h1_max)),
            lambda_min=float(block.get("lambda_min", cls.lambda_min)),
        )

    # ── Internal score computations ────────────────────────────────

    def _safe_phi(self, thg: THGCheckpoint) -> float:
        """``CohomologyValidator.compute_phi`` with a stable fallback
        when the THG is too small (fewer than 2 nodes ⇒ Φ undefined,
        treated as ``+inf`` for guard purposes so we never reject a
        trivially-small architecture for low Φ alone)."""
        if len(thg.nodes) < 2:
            return float("inf")
        try:
            return float(self.cohomology.compute_phi(thg))
        except Exception:
            return float("nan")  # surfaces as guard violation

    def _h1_norm(self, thg: THGCheckpoint) -> float:
        """Maximum per-node operator-embedding L2 norm — the same proxy
        ``CohomologyValidator.check_h1_consistency`` uses internally.
        Reported as a float so the verdict carries the numeric value
        rather than only the boolean."""
        if not thg.nodes:
            return 0.0
        max_norm = 0.0
        for node in thg.nodes.values():
            n = math.sqrt(sum(float(e) ** 2 for e in node.operator_embedding))
            if n > max_norm:
                max_norm = n
        return max_norm

    def _lambda_proxy(self, thg: THGCheckpoint) -> float:
        """Spectral-gap proxy until the true Tonnetz λ₁ (§4.2) lands.

        Current proxy: ``min(edge_count / max_possible_edges, 1.0)``,
        which is non-negative whenever the architecture has at least
        one edge and rises toward 1 as the graph approaches a
        complete sheaf.  This is monotone in connectedness, which is
        the property the formal_framework's λ₁ guard ultimately wants
        to protect (geometric dispersion ⇒ disconnected sheaf ⇒
        λ₁ → 0).
        """
        n = len(thg.nodes)
        if n < 2:
            # Single-vertex K has λ₁ undefined; treat as +inf so the
            # guard never rejects on this score alone.
            return float("inf")
        max_edges = n * (n - 1)  # directed; cellular sheaf is oriented
        if max_edges == 0:
            return float("inf")
        return min(1.0, len(thg.edges) / max_edges)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


# Regex matches every ``formal_spec NAME { ... }`` top-level block.
# We don't try to handle nested braces here because the triple_guard
# block is intentionally a flat key-value list per the framework spec.
_FORMAL_SPEC_RE = re.compile(
    r"\bformal_spec\s+\w+\s*\{([^{}]*)\}",
    re.DOTALL,
)
# Matches numeric assignments such as ``phi_min: 0.25`` or
# ``rule: "triple_guard"`` inside a formal_spec body.
_KV_RE = re.compile(
    r"(\w+)\s*:\s*(\"[^\"]*\"|'[^']*'|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
)


def _extract_triple_guard_block(text: str) -> Optional[Dict[str, Any]]:
    """Return the contents of the first ``formal_spec`` block whose
    ``rule`` field equals ``"triple_guard"``, as a flat ``{key: value}``
    dict where numeric values are parsed to floats.  Returns ``None``
    when no such block is present.
    """
    for match in _FORMAL_SPEC_RE.finditer(text):
        body = match.group(1)
        kvs: Dict[str, Any] = {}
        for k, v in _KV_RE.findall(body):
            v = v.strip()
            if v.startswith(("\"", "'")):
                kvs[k] = v[1:-1]
            else:
                try:
                    kvs[k] = float(v)
                except ValueError:
                    kvs[k] = v
        if str(kvs.get("rule", "")).lower() == "triple_guard":
            return kvs
    return None
