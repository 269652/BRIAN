# -*- coding: utf-8 -*-
"""TDD: L4 proposal gate.

Wraps :class:`neuroslm.verification.improvement_gate.ImprovementGate`
(+ optional :class:`neuroslm.verification.triple_guard.TripleGuard`)
to admit/reject the :class:`DNAPatch` proposals coming out of L3
(:func:`neuroslm.evolution.mutator.propose_mutations`).

Contract under test:

    gate_proposals(
        proposals,                  # list[DNAPatch]  (from L3)
        evidence_by_target,         # dict[str, ImprovementEvidence]
        *,
        improvement_gate=None,      # defaults to ImprovementGate()
        triple_guard=None,          # optional structural gate
        structural_by_target=None,  # dict[str, (before, after)] for TripleGuard
        default_direction=None,     # override per-kind defaults
    ) -> (admitted: list[DNAPatch], rejected: list[DNAPatch])

The two return lists are disjoint, partition the input, and every
patch carries the verdict (or rejection reasons) in its ``metadata``
so the audit trail is preserved.

Per-kind direction defaults:
  - node_mutation   -> "increase"  (Φ / accuracy / intelligence-density)
  - edge_strengthen -> "decrease"  (ppl / loss / OOD-gap)
  - edge_prune      -> "decrease"  (no-regression: must not make ppl worse)
"""
from __future__ import annotations

import random

import pytest

from neuroslm.compiler.ribosome import DNAPatch
from neuroslm.evolution.gate import (
    gate_proposals, ImprovementEvidence,
)
from neuroslm.verification.improvement_gate import ImprovementGate


# ── fixtures ───────────────────────────────────────────────────────


def _mk_patch(kind: str, target: str, *, step: int = 1000) -> DNAPatch:
    return DNAPatch(
        version="1.0",
        step=step,
        kind=kind,
        target=target,
        delta=[0.05] * 4,
        metadata={"reason": "hot_path", "heat": 0.9, "element_id": target},
    )


def _improving(direction: str, *, n: int = 32, seed: int = 0):
    """Return (before, after) samples that clearly improve in ``direction``."""
    rng = random.Random(seed)
    if direction == "decrease":
        before = [1.0 + rng.gauss(0, 0.02) for _ in range(n)]
        after  = [0.7 + rng.gauss(0, 0.02) for _ in range(n)]
    else:
        before = [0.5 + rng.gauss(0, 0.02) for _ in range(n)]
        after  = [0.8 + rng.gauss(0, 0.02) for _ in range(n)]
    return before, after


def _flat(*, n: int = 32, seed: int = 0):
    """Indistinguishable before/after — should always be rejected."""
    rng = random.Random(seed)
    before = [0.5 + rng.gauss(0, 0.02) for _ in range(n)]
    after  = [0.5 + rng.gauss(0, 0.02) for _ in range(n)]
    return before, after


# ── core admission cases ──────────────────────────────────────────


class TestImprovementEvidenceGate:
    def test_real_improvement_admitted(self):
        """A node_mutation with clear improvement evidence is admitted."""
        p = _mk_patch("node_mutation", "cortex")
        b, a = _improving("increase")          # node_mutation -> increase
        admitted, rejected = gate_proposals(
            [p], {"cortex": ImprovementEvidence(b, a)},
        )
        assert len(admitted) == 1 and len(rejected) == 0
        assert admitted[0].target == "cortex"
        v = admitted[0].metadata["gate_verdict"]
        assert v["admitted"] is True
        assert v["direction"] == "increase"

    def test_wrong_direction_rejected(self):
        """An edge_strengthen patch with a metric that went *up* is rejected
        (kind default is "decrease")."""
        p = _mk_patch("edge_strengthen", "synapse:cortex->striatum")
        # Generate an "increase" trend but pair it with a "decrease" gate -> wrong dir.
        b, a = _improving("increase")
        admitted, rejected = gate_proposals(
            [p], {p.target: ImprovementEvidence(b, a)},
        )
        assert len(admitted) == 0 and len(rejected) == 1
        reasons = rejected[0].metadata["rejection_reasons"]
        assert any("wrong direction" in r.lower() for r in reasons)

    def test_noise_rejected(self):
        """Flat / indistinguishable evidence fails Welch significance."""
        p = _mk_patch("node_mutation", "cortex")
        b, a = _flat()
        admitted, rejected = gate_proposals(
            [p], {"cortex": ImprovementEvidence(b, a)},
        )
        assert len(rejected) == 1
        rs = rejected[0].metadata["rejection_reasons"]
        assert any("statistically significant" in r or "wrong direction" in r
                   for r in rs)

    def test_missing_evidence_rejected(self):
        """Proposal with no evidence in the dict is rejected with a clean
        ``no_evidence`` reason — never silently admitted."""
        p = _mk_patch("node_mutation", "missing_target")
        admitted, rejected = gate_proposals([p], {})
        assert len(rejected) == 1
        rs = rejected[0].metadata["rejection_reasons"]
        assert any("no_evidence" in r or "no evidence" in r.lower()
                   for r in rs)


# ── per-kind direction defaults ───────────────────────────────────


class TestPerKindDirectionDefaults:
    def test_node_mutation_defaults_to_increase(self):
        p = _mk_patch("node_mutation", "x")
        b, a = _improving("increase")
        admitted, _ = gate_proposals([p], {"x": ImprovementEvidence(b, a)})
        assert admitted[0].metadata["gate_verdict"]["direction"] == "increase"

    def test_edge_strengthen_defaults_to_decrease(self):
        p = _mk_patch("edge_strengthen", "syn")
        b, a = _improving("decrease")
        admitted, _ = gate_proposals([p], {"syn": ImprovementEvidence(b, a)})
        assert admitted[0].metadata["gate_verdict"]["direction"] == "decrease"

    def test_edge_prune_defaults_to_decrease(self):
        """edge_prune is also direction='decrease' (no-regression)."""
        p = _mk_patch("edge_prune", "syn")
        b, a = _improving("decrease")
        admitted, _ = gate_proposals([p], {"syn": ImprovementEvidence(b, a)})
        assert admitted[0].metadata["gate_verdict"]["direction"] == "decrease"

    def test_evidence_direction_overrides_default(self):
        """Caller can pin direction per-evidence; that wins over the
        kind default."""
        p = _mk_patch("node_mutation", "x")           # default "increase"
        b, a = _improving("decrease")
        admitted, _ = gate_proposals(
            [p], {"x": ImprovementEvidence(b, a, direction="decrease")},
        )
        assert admitted[0].metadata["gate_verdict"]["direction"] == "decrease"


# ── batch / partition properties ──────────────────────────────────


class TestBatchPartition:
    def test_partitions_inputs(self):
        """admitted ∪ rejected == input, admitted ∩ rejected == ∅."""
        good = _mk_patch("node_mutation", "good")
        bad  = _mk_patch("node_mutation", "bad")
        bg, ag = _improving("increase", seed=1)
        bb, ab = _flat(seed=2)
        admitted, rejected = gate_proposals(
            [good, bad],
            {"good": ImprovementEvidence(bg, ag),
             "bad":  ImprovementEvidence(bb, ab)},
        )
        assert len(admitted) + len(rejected) == 2
        admitted_targets = {p.target for p in admitted}
        rejected_targets = {p.target for p in rejected}
        assert admitted_targets.isdisjoint(rejected_targets)
        assert "good" in admitted_targets
        assert "bad" in rejected_targets

    def test_empty_proposal_list(self):
        assert gate_proposals([], {}) == ([], [])

    def test_returns_dnapatch_instances(self):
        p = _mk_patch("node_mutation", "x")
        b, a = _improving("increase")
        admitted, _ = gate_proposals([p], {"x": ImprovementEvidence(b, a)})
        assert all(isinstance(q, DNAPatch) for q in admitted)


# ── custom gate / configurability ─────────────────────────────────


class TestCustomGate:
    def test_custom_alpha_is_honored(self):
        """A weak but real signal that the *default* gate admits should be
        rejected once alpha is set strict enough — this proves the kwarg
        actually flows through to the underlying gate."""
        p = _mk_patch("node_mutation", "x")
        rng = random.Random(7)
        # Tiny effect (0.50 -> 0.51) + sizable noise -> default p ≈ 0.01-0.1
        # so default alpha=0.05 may admit; alpha=1e-9 definitely won't.
        before = [0.50 + rng.gauss(0, 0.05) for _ in range(8)]
        after  = [0.51 + rng.gauss(0, 0.05) for _ in range(8)]
        strict = ImprovementGate(alpha=1e-9, min_effect=0.0)
        admitted, rejected = gate_proposals(
            [p], {"x": ImprovementEvidence(before, after)},
            improvement_gate=strict,
        )
        assert len(rejected) == 1
        assert any("statistically significant" in r
                   for r in rejected[0].metadata["rejection_reasons"])

    def test_custom_min_effect_is_honored(self):
        p = _mk_patch("node_mutation", "x")
        b, a = _improving("increase")
        # 99 % min_effect -> nothing realistic clears it
        strict = ImprovementGate(alpha=0.5, min_effect=0.99)
        admitted, rejected = gate_proposals(
            [p], {"x": ImprovementEvidence(b, a)},
            improvement_gate=strict,
        )
        assert len(rejected) == 1


# ── optional TripleGuard chain ────────────────────────────────────


class _FakeTripleGuard:
    """Stand-in that returns a verdict-like object — keeps the test
    independent of the heavy Φ/H¹/λ machinery."""

    def __init__(self, *, admit: bool, reasons=None):
        self._admit = admit
        self._reasons = reasons or []

    def admit(self, before, after, mutation=None):
        from neuroslm.verification.triple_guard import Verdict
        return Verdict(
            admitted=self._admit,
            phi_before=0.5, phi_after=0.6,
            h1_before=0.1, h1_after=0.1,
            lambda_before=0.0, lambda_after=0.0,
            reasons=list(self._reasons),
        )


class TestTripleGuardChain:
    def test_both_gates_must_admit(self):
        """If TripleGuard rejects, the patch is rejected even if the
        improvement gate would have admitted."""
        p = _mk_patch("node_mutation", "x")
        b, a = _improving("increase")
        admitted, rejected = gate_proposals(
            [p], {"x": ImprovementEvidence(b, a)},
            triple_guard=_FakeTripleGuard(admit=False, reasons=["Phi guard violated"]),
            structural_by_target={"x": ("before-chk", "after-chk")},
        )
        assert len(admitted) == 0 and len(rejected) == 1
        assert any("Phi guard violated" in r for r in rejected[0].metadata["rejection_reasons"])

    def test_both_gates_admit_admitted(self):
        p = _mk_patch("node_mutation", "x")
        b, a = _improving("increase")
        admitted, rejected = gate_proposals(
            [p], {"x": ImprovementEvidence(b, a)},
            triple_guard=_FakeTripleGuard(admit=True),
            structural_by_target={"x": ("before-chk", "after-chk")},
        )
        assert len(admitted) == 1 and len(rejected) == 0
        # The structural verdict is also surfaced in the metadata.
        assert "triple_guard_verdict" in admitted[0].metadata

    def test_no_structural_evidence_skips_triple_guard(self):
        """TripleGuard supplied but no structural evidence for the target
        -> we only consult the improvement gate (and admit if it admits)."""
        p = _mk_patch("node_mutation", "x")
        b, a = _improving("increase")
        admitted, _ = gate_proposals(
            [p], {"x": ImprovementEvidence(b, a)},
            triple_guard=_FakeTripleGuard(admit=False, reasons=["Phi guard violated"]),
            structural_by_target={},          # nothing for "x"
        )
        # Improvement admits -> proposal admitted, TripleGuard not consulted.
        assert len(admitted) == 1
