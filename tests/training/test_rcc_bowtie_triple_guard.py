# -*- coding: utf-8 -*-
"""TDD acceptance suite — Triple Guard wired into rcc_bowtie auto-evolution.

This suite pins the contract that the math primitives documented in
``docs/formal_framework.md`` (Φ guard, H¹ guard, λ₁ guard — together
the *Triple Guard* of §7) become **live constraints** during
``rcc_bowtie`` auto-evolution, not merely descriptive text.

Concretely:

1. ``neuroslm.verification.triple_guard.TripleGuard`` exists and exposes
   a single ``admit(thg_before, thg_after, mutation) -> Verdict``
   decision composing Φ + H¹ + λ₁.
2. The verdict carries before/after scores for all three guards and an
   ``admitted`` boolean, so every accept/reject is auditable.
3. ``EvolutionaryTrainingContext.save_checkpoint`` is augmented to
   *gate* mutations through a configured ``TripleGuard``: rejected
   mutations never reach disk; accepted ones land with the verdict
   embedded in ``patch.metadata.triple_guard``.
4. Rejected mutations are persisted to a separate audit file
   (``step_<N>.rejected.json``) so the human/automatic reviewer can
   see *what* the architecture refused to become.
5. ``architectures/master/arch.neuro`` (the canonical bowtie arch,
   renamed 2026-06-14 from ``rcc_bowtie``) can declare a
   ``formal_spec { triple_guard { ... } }`` block which compiles into
   a ``TripleGuard`` instance — closing the loop from DSL → live gate.

Reference: ``docs/formal_framework.md`` §7 (Triple Guard).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from neuroslm.compiler.ribosome import DNAPatch, LatentDNA
from neuroslm.dsl.thg_ir import THGCheckpoint, THGEdge, THGNode


# ──────────────────────────────────────────────────────────────────────
# Fixtures — small, deterministic THGs we can predict scores for
# ──────────────────────────────────────────────────────────────────────

def _make_thg(node_specs: List[tuple]) -> THGCheckpoint:
    """Build a THGCheckpoint with the given (id, embedding) pairs.

    Each embedding is a Python list[float] of identical length so that
    Φ/H¹/λ₁ are computable.  Edges are constructed pairwise so the
    sheaf is connected (otherwise λ₁ ≡ 0 trivially).
    """
    nodes = {
        nid: THGNode(id=nid, kind="population",
                     operator_embedding=emb, metadata={})
        for nid, emb in node_specs
    }
    edges = {}
    ids = [nid for nid, _ in node_specs]
    for i in range(len(ids) - 1):
        eid = f"{ids[i]}__{ids[i + 1]}"
        edges[eid] = THGEdge(id=eid, src=ids[i], dst=ids[i + 1],
                             kind="synapse", weight=1.0)
    return THGCheckpoint(
        version="2.0",
        nodes=nodes,
        edges=edges,
        gene_state={"learning_rate": 1e-3, "baseline_nt": 0.1},
        step=0,
        metadata={},
    )


@pytest.fixture()
def healthy_thg():
    """A THG with non-trivial correlation between node embeddings →
    non-zero Φ and small ‖H¹‖."""
    return _make_thg([
        ("n1", [0.1, 0.2, 0.3, 0.1]),
        ("n2", [0.2, 0.3, 0.4, 0.2]),
        ("n3", [0.15, 0.25, 0.35, 0.15]),
    ])


@pytest.fixture()
def healthy_thg_smallmut(healthy_thg):
    """``healthy_thg`` after a *tiny* additive mutation on n2 — both
    Φ and H¹ shift but stay inside admissible bands."""
    mutated = _make_thg([
        ("n1", list(healthy_thg.nodes["n1"].operator_embedding)),
        ("n2", [e + 0.01 for e in healthy_thg.nodes["n2"].operator_embedding]),
        ("n3", list(healthy_thg.nodes["n3"].operator_embedding)),
    ])
    return mutated


@pytest.fixture()
def healthy_thg_hallucinated(healthy_thg):
    """``healthy_thg`` after a mutation that explodes n2's embedding
    norm — must trigger the H¹ (hallucination) guard."""
    mutated = _make_thg([
        ("n1", list(healthy_thg.nodes["n1"].operator_embedding)),
        # 100x bigger than the default hallucination_threshold (=5.0).
        ("n2", [500.0, 500.0, 500.0, 500.0]),
        ("n3", list(healthy_thg.nodes["n3"].operator_embedding)),
    ])
    return mutated


@pytest.fixture()
def healthy_thg_phi_collapsed(healthy_thg):
    """``healthy_thg`` after a mutation that decorrelates n3 — Φ drops
    toward zero so the Φ guard rejects it (when phi_min > 0)."""
    mutated = _make_thg([
        ("n1", list(healthy_thg.nodes["n1"].operator_embedding)),
        ("n2", list(healthy_thg.nodes["n2"].operator_embedding)),
        # Anti-correlated direction → drives off-diagonal covariance to 0.
        ("n3", [-0.15, -0.25, -0.35, -0.15]),
    ])
    return mutated


def _patch(step: int, target: str, delta: List[float]) -> dict:
    """A mutation dict in the shape EvolutionaryTrainingContext expects."""
    return {
        "kind": "node_mutation",
        "target": target,
        "delta": delta,
        "metadata": {"reason": "test"},
    }


# ──────────────────────────────────────────────────────────────────────
# 1.  Importability & construction
# ──────────────────────────────────────────────────────────────────────

class TestTripleGuardImport:
    """The TripleGuard primitive must be importable and constructible
    from explicit thresholds *and* from a parsed DSL ``formal_spec``
    block."""

    def test_triple_guard_importable(self):
        from neuroslm.verification.triple_guard import TripleGuard  # noqa: F401

    def test_construction_with_defaults(self):
        from neuroslm.verification.triple_guard import TripleGuard
        guard = TripleGuard()
        # Defaults are normative — they live in formal_framework.md §7.
        assert guard.phi_min >= 0.0
        assert guard.h1_max > 0.0
        assert guard.lambda_min >= 0.0

    def test_construction_with_explicit_thresholds(self):
        from neuroslm.verification.triple_guard import TripleGuard
        guard = TripleGuard(phi_min=0.1, h1_max=1.0, lambda_min=0.05)
        assert guard.phi_min == 0.1
        assert guard.h1_max == 1.0
        assert guard.lambda_min == 0.05

    def test_from_arch_neuro_with_triple_guard_block(self, tmp_path):
        """If arch.neuro contains a ``formal_spec { triple_guard { ... } }``
        block, ``TripleGuard.from_arch_neuro(path)`` must return a
        guard whose thresholds came from that block."""
        from neuroslm.verification.triple_guard import TripleGuard
        dsl = """
        architecture demo { d_sem: 64, dt: 0.01 }
        population P { count: 32, dynamics: "rate_code" }
        formal_spec demo_spec {
            rule: "triple_guard",
            phi_min: 0.25,
            h1_max: 0.5,
            lambda_min: 0.15
        }
        """
        path = tmp_path / "arch.neuro"
        path.write_text(dsl, encoding="utf-8")
        guard = TripleGuard.from_arch_neuro(str(path))
        assert guard.phi_min == pytest.approx(0.25)
        assert guard.h1_max == pytest.approx(0.5)
        assert guard.lambda_min == pytest.approx(0.15)

    def test_from_arch_neuro_without_block_returns_default_guard(self, tmp_path):
        """If no ``triple_guard`` block is present, the factory returns
        a default-configured guard (not None) — opt-in safety net."""
        from neuroslm.verification.triple_guard import TripleGuard
        dsl = """
        architecture demo { d_sem: 64, dt: 0.01 }
        population P { count: 32, dynamics: "rate_code" }
        """
        path = tmp_path / "arch.neuro"
        path.write_text(dsl, encoding="utf-8")
        guard = TripleGuard.from_arch_neuro(str(path))
        assert isinstance(guard, TripleGuard)


# ──────────────────────────────────────────────────────────────────────
# 2.  Decision contract — admit / reject + verdict structure
# ──────────────────────────────────────────────────────────────────────

class TestTripleGuardDecisions:
    """The single decision function ``admit(before, after, mutation)``
    must return a structured verdict, never raise, and use the three
    guards together (not the worst of the three by accident)."""

    def test_admit_returns_verdict_object(self, healthy_thg, healthy_thg_smallmut):
        from neuroslm.verification.triple_guard import TripleGuard, Verdict
        guard = TripleGuard()
        verdict = guard.admit(
            healthy_thg, healthy_thg_smallmut,
            _patch(step=1, target="n2", delta=[0.01, 0.01, 0.01, 0.01]),
        )
        assert isinstance(verdict, Verdict)
        # Required fields — these are what land in patch.metadata.
        assert hasattr(verdict, "admitted")
        assert hasattr(verdict, "phi_before") and hasattr(verdict, "phi_after")
        assert hasattr(verdict, "h1_before") and hasattr(verdict, "h1_after")
        assert hasattr(verdict, "lambda_before") and hasattr(verdict, "lambda_after")
        assert hasattr(verdict, "reasons")
        assert isinstance(verdict.reasons, list)

    def test_admits_a_benign_mutation(self, healthy_thg, healthy_thg_smallmut):
        from neuroslm.verification.triple_guard import TripleGuard
        guard = TripleGuard()  # default thresholds → permissive
        verdict = guard.admit(
            healthy_thg, healthy_thg_smallmut,
            _patch(step=1, target="n2", delta=[0.01] * 4),
        )
        assert verdict.admitted is True
        assert verdict.reasons == []  # nothing to complain about

    def test_rejects_a_hallucinating_mutation(
        self, healthy_thg, healthy_thg_hallucinated
    ):
        from neuroslm.verification.triple_guard import TripleGuard
        guard = TripleGuard()  # default h1_max < 100
        verdict = guard.admit(
            healthy_thg, healthy_thg_hallucinated,
            _patch(step=1, target="n2", delta=[500.0] * 4),
        )
        assert verdict.admitted is False
        assert any("H1" in r or "h1" in r for r in verdict.reasons), (
            f"expected H1 reason; got {verdict.reasons}"
        )

    def test_rejects_phi_collapse_when_phi_min_strict(
        self, healthy_thg, healthy_thg_phi_collapsed
    ):
        from neuroslm.verification.triple_guard import TripleGuard
        # Set a strict phi_min that the collapsed THG cannot meet.
        guard = TripleGuard(phi_min=1e6)
        verdict = guard.admit(
            healthy_thg, healthy_thg_phi_collapsed,
            _patch(step=1, target="n3", delta=[-0.3] * 4),
        )
        assert verdict.admitted is False
        assert any("Phi" in r or "phi" in r or "Φ" in r for r in verdict.reasons)

    def test_verdict_is_serialisable(self, healthy_thg, healthy_thg_smallmut):
        """``verdict.to_dict()`` returns a JSON-serialisable dict so the
        patch metadata round-trips through disk cleanly."""
        from neuroslm.verification.triple_guard import TripleGuard
        guard = TripleGuard()
        verdict = guard.admit(
            healthy_thg, healthy_thg_smallmut,
            _patch(step=1, target="n2", delta=[0.01] * 4),
        )
        d = verdict.to_dict()
        # Round-trip through JSON to prove serialisability.
        encoded = json.dumps(d)
        decoded = json.loads(encoded)
        assert decoded["admitted"] == verdict.admitted
        assert "phi_before" in decoded
        assert "h1_after" in decoded

    def test_admit_does_not_mutate_inputs(
        self, healthy_thg, healthy_thg_smallmut
    ):
        """Calling ``admit`` must be side-effect free on its arguments."""
        from neuroslm.verification.triple_guard import TripleGuard
        before_snapshot = json.dumps(
            {nid: n.operator_embedding for nid, n in healthy_thg.nodes.items()},
            sort_keys=True,
        )
        after_snapshot = json.dumps(
            {nid: n.operator_embedding
             for nid, n in healthy_thg_smallmut.nodes.items()},
            sort_keys=True,
        )
        guard = TripleGuard()
        _ = guard.admit(
            healthy_thg, healthy_thg_smallmut,
            _patch(step=1, target="n2", delta=[0.01] * 4),
        )
        after_before_snapshot = json.dumps(
            {nid: n.operator_embedding for nid, n in healthy_thg.nodes.items()},
            sort_keys=True,
        )
        after_after_snapshot = json.dumps(
            {nid: n.operator_embedding
             for nid, n in healthy_thg_smallmut.nodes.items()},
            sort_keys=True,
        )
        assert before_snapshot == after_before_snapshot
        assert after_snapshot == after_after_snapshot


# ──────────────────────────────────────────────────────────────────────
# 3.  EvolutionaryTrainingContext gating
# ──────────────────────────────────────────────────────────────────────

class TestEvolutionContextGating:
    """The evolution context's ``save_checkpoint`` must, when a guard
    is configured, filter out rejected mutations *before* persisting
    them to disk."""

    def _make_minimal_ctx(self, tmp_path, guard=None):
        """Build an EvolutionaryTrainingContext that has just enough
        state for ``save_checkpoint`` to run.  We don't go through
        ``__enter__`` (which needs a real DNA file) — we just inject
        the fields manually so the test stays focused on the gating
        behaviour."""
        from neuroslm.utils.colab import EvolutionaryTrainingContext
        # Create a fake DNA file so the constructor's mkdir path works.
        dna = tmp_path / "base.dna"
        dna.write_text("{}", encoding="utf-8")
        ctx = EvolutionaryTrainingContext(
            dna_path=str(dna),
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        if guard is not None:
            ctx.set_triple_guard(guard)
        return ctx

    def test_set_triple_guard_attaches_guard(self, tmp_path):
        from neuroslm.verification.triple_guard import TripleGuard
        ctx = self._make_minimal_ctx(tmp_path)
        guard = TripleGuard()
        ctx.set_triple_guard(guard)
        # Public attribute so the training loop and tests can inspect.
        assert ctx.triple_guard is guard

    def test_save_checkpoint_without_guard_persists_all_mutations(
        self, tmp_path
    ):
        """Backward compatibility — when no guard is set, behaviour is
        identical to today (every mutation is written)."""
        ctx = self._make_minimal_ctx(tmp_path)
        mutations = [
            _patch(step=10, target="n1", delta=[0.1] * 4),
            _patch(step=10, target="n2", delta=[0.2] * 4),
        ]
        ctx.save_checkpoint(step=10, mutations=mutations)
        files = list((tmp_path / "ckpt").glob("step_*.patch.dna"))
        assert len(files) >= 1  # at least one patch persisted

    def test_save_checkpoint_with_guard_filters_rejected(
        self, tmp_path, healthy_thg, healthy_thg_hallucinated
    ):
        """When a guard is configured AND thg_before/thg_after are
        supplied, rejected mutations must not produce patch files."""
        from neuroslm.verification.triple_guard import TripleGuard
        guard = TripleGuard()
        ctx = self._make_minimal_ctx(tmp_path, guard=guard)

        bad_mut = _patch(step=20, target="n2", delta=[500.0] * 4)
        good_mut = _patch(step=20, target="n1", delta=[0.01] * 4)

        # Supply per-mutation before/after via the new optional
        # `thg_pairs` keyword on save_checkpoint.
        ctx.save_checkpoint(
            step=20,
            mutations=[good_mut, bad_mut],
            thg_pairs=[
                (healthy_thg, healthy_thg),                    # good: no-op
                (healthy_thg, healthy_thg_hallucinated),        # bad
            ],
        )
        # Only the good mutation should land on disk.
        files = list((tmp_path / "ckpt").glob("step_*.patch.dna"))
        targets = {DNAPatch.load(str(f)).target for f in files}
        assert "n1" in targets
        assert "n2" not in targets, (
            "rejected hallucinating mutation must NOT be persisted"
        )

    def test_rejected_mutations_are_audit_logged(
        self, tmp_path, healthy_thg, healthy_thg_hallucinated
    ):
        """Rejected mutations must be persisted to ``step_<N>.rejected.json``
        with the verdict, so the auditor can see what was thrown away."""
        from neuroslm.verification.triple_guard import TripleGuard
        guard = TripleGuard()
        ctx = self._make_minimal_ctx(tmp_path, guard=guard)

        bad_mut = _patch(step=30, target="n2", delta=[500.0] * 4)
        ctx.save_checkpoint(
            step=30,
            mutations=[bad_mut],
            thg_pairs=[(healthy_thg, healthy_thg_hallucinated)],
        )
        reject_files = list((tmp_path / "ckpt").glob("step_*.rejected.json"))
        assert len(reject_files) == 1
        data = json.loads(reject_files[0].read_text(encoding="utf-8"))
        # Audit record must include the verdict and the mutation that
        # was thrown away.
        assert "rejected" in data
        assert len(data["rejected"]) == 1
        rec = data["rejected"][0]
        assert rec["target"] == "n2"
        assert "verdict" in rec
        assert rec["verdict"]["admitted"] is False

    def test_accepted_patch_carries_verdict_metadata(
        self, tmp_path, healthy_thg, healthy_thg_smallmut
    ):
        """Accepted patches must embed the Triple-Guard verdict in
        their metadata so downstream tooling (incl. the next training
        run loading the patch) can introspect why the mutation
        survived."""
        from neuroslm.verification.triple_guard import TripleGuard
        guard = TripleGuard()
        ctx = self._make_minimal_ctx(tmp_path, guard=guard)

        good_mut = _patch(step=40, target="n2", delta=[0.01] * 4)
        ctx.save_checkpoint(
            step=40,
            mutations=[good_mut],
            thg_pairs=[(healthy_thg, healthy_thg_smallmut)],
        )
        files = list((tmp_path / "ckpt").glob("step_*.patch.dna"))
        assert len(files) == 1
        patch = DNAPatch.load(str(files[0]))
        tg_meta = patch.metadata.get("triple_guard")
        assert tg_meta is not None, (
            f"patch.metadata missing triple_guard: {patch.metadata}"
        )
        assert tg_meta["admitted"] is True
        assert "phi_after" in tg_meta and "h1_after" in tg_meta


# ──────────────────────────────────────────────────────────────────────
# 4.  Integration — rcc_bowtie/arch.neuro opts into the guard
# ──────────────────────────────────────────────────────────────────────

class TestRccBowtieIntegration:
    """End-to-end: ``architectures/master/arch.neuro`` (canonical bowtie
    arch, renamed 2026-06-14 from ``rcc_bowtie``) must be able to
    declare a ``formal_spec { triple_guard { ... } }`` block, and
    compiling the architecture must surface that block as a usable
    ``TripleGuard`` on the resulting context."""

    def test_rcc_bowtie_arch_neuro_compiles(self):
        """Sanity: the current bowtie arch compiles at all (so any
        ``formal_spec`` addition we make won't break the build)."""
        from neuroslm.dsl.multifile import compile_folder
        arch_root = Path(__file__).resolve().parents[2] / "architectures" / "master"
        assert arch_root.exists()
        ir = compile_folder(str(arch_root))
        assert ir is not None
        # The bowtie has populations; this lets us confirm we got the
        # real arch, not a stub.
        assert len(ir.populations) > 0

    def test_rcc_bowtie_triple_guard_from_arch_neuro(self):
        """``TripleGuard.from_arch_neuro`` must work on the production
        bowtie arch.neuro — either reading a triple_guard block if
        present, or falling back to defaults if not.  Either way: a
        TripleGuard instance comes back."""
        from neuroslm.verification.triple_guard import TripleGuard
        arch_neuro = (
            Path(__file__).resolve().parents[2]
            / "architectures" / "master" / "arch.neuro"
        )
        guard = TripleGuard.from_arch_neuro(str(arch_neuro))
        assert isinstance(guard, TripleGuard)
        # Whether opted-in or default, the three thresholds must be set.
        assert guard.phi_min >= 0.0
        assert guard.h1_max > 0.0
        assert guard.lambda_min >= 0.0
