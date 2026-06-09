# -*- coding: utf-8 -*-
"""TDD: L5 Lean proof gate.

Wires the L4 :func:`gate_proposals` to the Lean backend shipped in
commit ``41df700`` (``neuroslm.discoveries.lean``). When a proposal's
kind matches a hypothesis with a verified Lean proof, the proposal
is admitted *without* needing empirical evidence (a formal proof
strictly dominates statistical significance).

Mapping (proposal.kind -> hypothesis.id):

  node_mutation   -> H001  (Phi monotone under coupling addition)
  edge_strengthen -> H002  (OOD gap decrease under CDGA)
  edge_prune      -> H002  (no-regression on ppl)

When the Lean binary is absent or the proof verdict is not
``verified``, we **fall back** to the empirical ImprovementGate
behaviour from L4 (so the pipeline keeps working without Lean).
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from neuroslm.compiler.ribosome import DNAPatch
from neuroslm.evolution.gate import (
    gate_proposals, ImprovementEvidence,
)
from neuroslm.evolution.lean_gate import (
    LeanProofBackend, kind_to_hypothesis_id, DEFAULT_KIND_TO_HYPOTHESIS,
)
from neuroslm.discoveries.lean import LeanVerdict


REPO_ROOT = Path(__file__).resolve().parent.parent
HYPOTHESIS_ROOT = REPO_ROOT / "hypothesis"
PROOFS_ROOT = HYPOTHESIS_ROOT / "proofs"


# ── helpers ────────────────────────────────────────────────────────


def _mk_patch(kind: str, target: str, step: int = 1000) -> DNAPatch:
    return DNAPatch(
        version="1.0", step=step, kind=kind, target=target,
        delta=[0.05] * 4,
        metadata={"reason": "hot_path", "heat": 0.9, "element_id": target},
    )


def _improving(direction: str, n: int = 32, seed: int = 0):
    rng = random.Random(seed)
    if direction == "decrease":
        before = [1.0 + rng.gauss(0, 0.02) for _ in range(n)]
        after  = [0.7 + rng.gauss(0, 0.02) for _ in range(n)]
    else:
        before = [0.5 + rng.gauss(0, 0.02) for _ in range(n)]
        after  = [0.8 + rng.gauss(0, 0.02) for _ in range(n)]
    return before, after


# ── kind -> hypothesis id mapping ────────────────────────────────


class TestKindToHypothesisMapping:
    def test_node_mutation_maps_to_h001(self):
        assert kind_to_hypothesis_id("node_mutation") == "H001"

    def test_edge_strengthen_maps_to_h002(self):
        assert kind_to_hypothesis_id("edge_strengthen") == "H002"

    def test_edge_prune_maps_to_h002(self):
        assert kind_to_hypothesis_id("edge_prune") == "H002"

    def test_unknown_kind_returns_none(self):
        assert kind_to_hypothesis_id("topology_change") is None
        assert kind_to_hypothesis_id("totally_made_up") is None

    def test_default_map_is_complete_for_l3_kinds(self):
        """Every kind L3's propose_mutations emits must have a mapping."""
        for k in ("node_mutation", "edge_strengthen", "edge_prune"):
            assert k in DEFAULT_KIND_TO_HYPOTHESIS


# ── LeanProofBackend resolves & invokes ──────────────────────────


class TestLeanProofBackendResolve:
    def test_resolves_proof_file_for_h001(self):
        backend = LeanProofBackend(
            hypothesis_root=HYPOTHESIS_ROOT,
        )
        proof = backend.resolve_proof_path("H001")
        assert proof is not None
        assert proof.exists()
        assert proof.suffix == ".lean"
        assert "H001" in proof.name

    def test_resolves_proof_file_for_h002(self):
        backend = LeanProofBackend(hypothesis_root=HYPOTHESIS_ROOT)
        assert backend.resolve_proof_path("H002").exists()

    def test_unknown_id_returns_none(self):
        backend = LeanProofBackend(hypothesis_root=HYPOTHESIS_ROOT)
        assert backend.resolve_proof_path("H999") is None

    def test_admits_proposal_returns_falsy_when_unknown_kind(self):
        """A proposal whose kind has no mapping cannot be Lean-admitted."""
        backend = LeanProofBackend(hypothesis_root=HYPOTHESIS_ROOT)
        p = _mk_patch("topology_change", "x")
        verdict = backend.admit_proposal(p)
        assert verdict is None      # no opinion; caller must use empirical gate


# ── short-circuit semantics in gate_proposals ────────────────────


class _StubLeanBackend:
    """Pure-Python stand-in for LeanProofBackend so the test doesn't
    need a Lean install."""

    def __init__(self, verdict_by_kind):
        self.verdict_by_kind = verdict_by_kind
        self.calls = []

    def admit_proposal(self, patch):
        self.calls.append((patch.kind, patch.target))
        return self.verdict_by_kind.get(patch.kind)


def _verified_verdict():
    return LeanVerdict(status="verified", file="dummy.lean", n_sorry=0)


def _compiles_verdict():
    return LeanVerdict(status="compiles", file="dummy.lean", n_sorry=1)


def _error_verdict():
    return LeanVerdict(status="error", file="dummy.lean",
                       errors=["proof failed"])


def _skipped_verdict():
    return LeanVerdict(status="skipped", file="dummy.lean",
                       warnings=["lean binary not on PATH"])


class TestLeanShortCircuit:
    def test_verified_lean_admits_without_empirical_evidence(self):
        """If Lean proves the obligation, the proposal is admitted even
        when the evidence dict is empty."""
        p = _mk_patch("node_mutation", "x")
        backend = _StubLeanBackend({"node_mutation": _verified_verdict()})
        admitted, rejected = gate_proposals(
            [p], {},                        # no empirical evidence
            lean_backend=backend,
        )
        assert len(admitted) == 1 and len(rejected) == 0
        v = admitted[0].metadata["lean_verdict"]
        assert v["status"] == "verified"
        # backend was consulted
        assert ("node_mutation", "x") in backend.calls

    def test_verified_lean_admits_even_when_empirical_would_reject(self):
        """Formal proof strictly dominates a noisy / contradicting
        empirical signal."""
        p = _mk_patch("node_mutation", "x")
        backend = _StubLeanBackend({"node_mutation": _verified_verdict()})
        # Empirical evidence is INTENTIONALLY contradictory.
        bad_b, bad_a = _improving("decrease")        # wrong direction
        admitted, rejected = gate_proposals(
            [p], {"x": ImprovementEvidence(bad_b, bad_a)},
            lean_backend=backend,
        )
        assert len(admitted) == 1
        # Empirical gate NOT consulted (no gate_verdict).
        assert "gate_verdict" not in admitted[0].metadata
        assert admitted[0].metadata["lean_verdict"]["status"] == "verified"

    def test_unverified_lean_falls_through_to_empirical(self):
        """compiles/error/skipped Lean verdicts -> use the L4 empirical gate."""
        p = _mk_patch("node_mutation", "x")
        backend = _StubLeanBackend({"node_mutation": _compiles_verdict()})
        b, a = _improving("increase")
        admitted, rejected = gate_proposals(
            [p], {"x": ImprovementEvidence(b, a)},
            lean_backend=backend,
        )
        assert len(admitted) == 1
        # Empirical gate WAS consulted (gate_verdict present).
        assert admitted[0].metadata["gate_verdict"]["admitted"] is True
        assert admitted[0].metadata["lean_verdict"]["status"] == "compiles"

    def test_lean_error_falls_back_to_empirical(self):
        p = _mk_patch("node_mutation", "x")
        backend = _StubLeanBackend({"node_mutation": _error_verdict()})
        b, a = _improving("increase")
        admitted, _ = gate_proposals(
            [p], {"x": ImprovementEvidence(b, a)},
            lean_backend=backend,
        )
        assert len(admitted) == 1
        assert "gate_verdict" in admitted[0].metadata
        assert admitted[0].metadata["lean_verdict"]["status"] == "error"

    def test_lean_skipped_falls_back_to_empirical(self):
        """No Lean binary -> verdict.status='skipped' -> empirical gate runs."""
        p = _mk_patch("node_mutation", "x")
        backend = _StubLeanBackend({"node_mutation": _skipped_verdict()})
        b, a = _improving("increase")
        admitted, _ = gate_proposals(
            [p], {"x": ImprovementEvidence(b, a)},
            lean_backend=backend,
        )
        assert len(admitted) == 1
        assert admitted[0].metadata["lean_verdict"]["status"] == "skipped"
        assert "gate_verdict" in admitted[0].metadata

    def test_unmappable_kind_falls_back_to_empirical(self):
        """A kind with no hypothesis mapping -> Lean backend returns
        None -> empirical gate runs."""
        p = _mk_patch("node_mutation", "x")
        backend = _StubLeanBackend({})              # no mapping for node_mutation
        b, a = _improving("increase")
        admitted, _ = gate_proposals(
            [p], {"x": ImprovementEvidence(b, a)},
            lean_backend=backend,
        )
        assert len(admitted) == 1
        assert "gate_verdict" in admitted[0].metadata
        assert admitted[0].metadata.get("lean_verdict") is None


# ── real Lean kernel — skipped without binary ────────────────────


class TestRealLeanKernel:
    @pytest.mark.skipif(
        shutil.which("lean") is None,
        reason="lean binary not installed; install via elan to run",
    )
    def test_h001_against_real_kernel(self):
        """Smoke-test: run the actual Lean kernel against H001's stub.
        Stubs use `sorry`, so the verdict should be `compiles` (no
        errors) rather than `verified`. Once the Brian Lean library
        lands and the `sorry` is replaced, this becomes `verified`.
        """
        backend = LeanProofBackend(hypothesis_root=HYPOTHESIS_ROOT)
        from neuroslm.discoveries.lean import verify_lean_proof
        proof = backend.resolve_proof_path("H001")
        verdict = verify_lean_proof(proof)
        assert verdict.status in ("verified", "compiles"), \
            f"Lean errors on H001: {verdict.errors}"
