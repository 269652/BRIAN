# -*- coding: utf-8 -*-
"""L5 — Lean proof gate.

Wires the L4 :func:`gate_proposals` to the Lean backend shipped in
commit ``41df700`` (``neuroslm.discoveries.lean``). When a proposal's
``kind`` maps to a hypothesis whose ``.lean`` file is admitted by
the Lean kernel, the proposal is admitted *without* needing
empirical evidence — a formal proof strictly dominates statistical
significance.

Default mapping (proposal.kind -> hypothesis.id):

    node_mutation   -> H001  (Phi monotone under coupling addition)
    edge_strengthen -> H002  (OOD gap decrease under CDGA)
    edge_prune      -> H002  (no-regression on ppl)

Fallback semantics
------------------
The L4 empirical gate **always** runs when Lean returns anything other
than ``verified`` — ``compiles`` (the proof body still has ``sorry``),
``error`` (Lean rejected), ``skipped`` (no Lean binary). This means the
pipeline keeps working without a Lean install: the Lean kwarg is purely
*additive* — it can only **admit** proposals the L4 gate would have
rejected; it can never reject a proposal the L4 gate would have admitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from neuroslm.compiler.ribosome import DNAPatch
from neuroslm.discoveries.lean import LeanVerdict, verify_lean_proof


__all__ = [
    "DEFAULT_KIND_TO_HYPOTHESIS",
    "kind_to_hypothesis_id",
    "LeanProofBackend",
]


# Map proposal kinds (DNAPatch.kind values, see L3 mutator.py) to the
# canonical hypothesis id whose Lean theorem subsumes the obligation.
DEFAULT_KIND_TO_HYPOTHESIS: Dict[str, str] = {
    # H001 - Phi(theta') >= Phi(theta) when the mutation adds a
    #        non-negative coupling. A hot-node mutation that perturbs
    #        a population's parameters falls under this monotonicity.
    "node_mutation":   "H001",
    # H002 - Adding lambda * CDGA to the loss cannot increase the OOD
    #        gap. Both strengthening a hot edge and pruning a cold
    #        edge are local-structural changes whose OOD effect is
    #        bounded by H002's no-regression theorem (when proved).
    "edge_strengthen": "H002",
    "edge_prune":      "H002",
}


def kind_to_hypothesis_id(kind: str) -> Optional[str]:
    """Return the canonical hypothesis id for a DNAPatch kind, or
    ``None`` if no formal obligation has been registered yet."""
    return DEFAULT_KIND_TO_HYPOTHESIS.get(kind)


@dataclass
class LeanProofBackend:
    """Adapter from L4's gate_proposals to discoveries.lean.

    Holds the hypothesis root (where the ``proofs/`` subdir lives) and,
    for each proposal, looks up the canonical hypothesis id, resolves
    the ``.lean`` file, and shells out to ``lean --json`` via
    :func:`neuroslm.discoveries.lean.verify_lean_proof`.

    The adapter is a simple data carrier; the kernel work happens in
    the shipped :func:`verify_lean_proof`. The split exists so the L4
    gate can take a *protocol-shaped* object (anything with
    ``admit_proposal(patch) -> Optional[LeanVerdict]``) and test
    doubles in unit tests stay tiny.
    """
    hypothesis_root: Path = field(
        default_factory=lambda: _default_hypothesis_root())
    kind_to_id: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_KIND_TO_HYPOTHESIS))
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.hypothesis_root = Path(self.hypothesis_root)

    # ── lookup ─────────────────────────────────────────────────────

    def resolve_proof_path(self, hypothesis_id: str) -> Optional[Path]:
        """Return the path to ``<root>/proofs/<id>_*.lean`` if it
        exists, else ``None``. We glob by id-prefix because the
        slug part of the filename is the title and may evolve over time."""
        proofs_dir = self.hypothesis_root / "proofs"
        if not proofs_dir.is_dir():
            return None
        matches = sorted(proofs_dir.glob(f"{hypothesis_id}_*.lean"))
        return matches[0] if matches else None

    # ── public surface used by gate_proposals ──────────────────────

    def admit_proposal(self, patch: DNAPatch) -> Optional[LeanVerdict]:
        """Try to admit ``patch`` by Lean.

        Returns:
          - ``None`` if the kind has no mapping or the proof file is
            absent — caller should fall through to the empirical gate.
          - A :class:`LeanVerdict` otherwise. ``status='verified'``
            means Lean admitted; anything else means caller should
            still consult the empirical gate.
        """
        hid = self.kind_to_id.get(patch.kind)
        if hid is None:
            return None
        proof = self.resolve_proof_path(hid)
        if proof is None:
            return None
        return verify_lean_proof(str(proof), timeout=self.timeout)


# ── helpers ────────────────────────────────────────────────────────


def _default_hypothesis_root() -> Path:
    """Walk up from this file to the repo root, then point at
    ``hypothesis/``. Mirrors the convention used elsewhere in the
    codebase (e.g. ``neuroslm/cli.py::_hypothesis_root``)."""
    here = Path(__file__).resolve()
    # neuroslm/evolution/lean_gate.py -> repo root is parent.parent.parent
    repo_root = here.parent.parent.parent
    return repo_root / "hypothesis"
