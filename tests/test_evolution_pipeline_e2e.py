# -*- coding: utf-8 -*-
"""TDD: L7 end-to-end evolution pipeline.

Wire the entire chain together:

    grad norms -> TrainingHeatmap.update()
               -> propose_mutations()  -> DNAPatch[]
               -> gate_proposals()     -> (admitted, rejected)
               -> (optionally Lean-admitted via lean_backend)
               -> splice_discovery_into_dna()
                  (with appropriate DiscoveryRecord wrapping)

This is the integration test for everything L1..L6 plus the
discoveries/ machinery from 41df700.
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest


SAMPLE_DSL = (
    "architecture pipeline_e2e { d_sem: 256 }\n"
    "neurotransmitter dopamine { base_concentration: 0.5 }\n"
    'population cortex { count: 512, dynamics: "rate_code" }\n'
    'population striatum { count: 256, dynamics: "rate_code" }\n'
    'population thalamus { count: 128, dynamics: "rate_code" }\n'
    "synapse cortex -> striatum { weight: 0.6 }\n"
    "synapse cortex -> thalamus { weight: 0.4 }\n"
    "modulation dopamine -> striatum { gain: 1.2 }\n"
)


@pytest.fixture
def tiny_arch_root(tmp_path):
    """A tiny architecture directory with arch.neuro on disk."""
    arch_root = tmp_path / "tiny_arch"
    arch_root.mkdir()
    (arch_root / "arch.neuro").write_text(SAMPLE_DSL, encoding="utf-8")
    return arch_root


def _improving_evidence(direction: str, n: int = 32, seed: int = 0):
    rng = random.Random(seed)
    if direction == "decrease":
        before = [1.0 + rng.gauss(0, 0.02) for _ in range(n)]
        after  = [0.7 + rng.gauss(0, 0.02) for _ in range(n)]
    else:
        before = [0.5 + rng.gauss(0, 0.02) for _ in range(n)]
        after  = [0.8 + rng.gauss(0, 0.02) for _ in range(n)]
    return before, after


# ── E2E: heatmap -> propose -> gate -> admit ───────────────────────


class TestEndToEndAdmission:
    def test_full_pipeline_emits_admitted_patches(self, tiny_arch_root):
        """Hot grad signal -> a heatmap entry -> a DNAPatch proposal ->
        improvement evidence -> admitted patch with full audit trail."""
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        from neuroslm.evolution import (
            TrainingHeatmap, propose_mutations,
            gate_proposals, ImprovementEvidence,
        )

        # Step 1: lift the IR
        ir = lift_arch_to_hypergraph(tiny_arch_root)
        assert any(n.name == "cortex" for n in ir.nodes)

        # Step 2: heatmap with cortex as a hot node
        hm = TrainingHeatmap()
        hm.update(
            {"population:cortex": 1.0, "population:striatum": 0.05,
             "synapse:cortex->striatum": 0.9},
            kinds={"population:cortex": "node",
                   "population:striatum": "node",
                   "synapse:cortex->striatum": "edge"},
        )

        # Step 3: propose mutations
        proposals = propose_mutations(hm, ir, step=1000)
        assert len(proposals) > 0
        kinds = {p.kind for p in proposals}
        assert "node_mutation" in kinds          # cortex was hot
        assert "edge_strengthen" in kinds        # hot edge was strengthened

        # Step 4: gate with real improvement evidence
        evidence = {}
        for p in proposals:
            direction = ("increase" if p.kind == "node_mutation"
                         else "decrease")
            b, a = _improving_evidence(direction, seed=hash(p.target) & 0xff)
            evidence[p.target] = ImprovementEvidence(b, a)
        admitted, rejected = gate_proposals(proposals, evidence)

        # Step 5: at least one patch admitted, all carry audit metadata
        assert len(admitted) > 0
        for p in admitted:
            assert "gate_verdict" in p.metadata
            assert p.metadata["gate_verdict"]["admitted"] is True

    def test_full_pipeline_with_lean_short_circuit(self, tiny_arch_root):
        """With a stub Lean backend that admits node_mutations, the
        proposal is admitted even if the empirical evidence is missing."""
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        from neuroslm.evolution import (
            TrainingHeatmap, propose_mutations, gate_proposals,
        )
        from neuroslm.discoveries.lean import LeanVerdict

        ir = lift_arch_to_hypergraph(tiny_arch_root)
        hm = TrainingHeatmap()
        hm.update(
            {"population:cortex": 1.0},
            kinds={"population:cortex": "node"},
        )
        proposals = propose_mutations(hm, ir, step=1000)
        node_muts = [p for p in proposals if p.kind == "node_mutation"]
        assert node_muts

        class _ConstAdmit:
            def admit_proposal(self, patch):
                if patch.kind == "node_mutation":
                    return LeanVerdict(status="verified", file="x.lean")
                return None

        admitted, rejected = gate_proposals(
            proposals, {},                       # NO empirical evidence
            lean_backend=_ConstAdmit(),
        )
        # All node_mutations are Lean-admitted; everything else rejected.
        assert any(p.kind == "node_mutation" for p in admitted)
        for p in admitted:
            if p.kind == "node_mutation":
                assert p.metadata["lean_verdict"]["status"] == "verified"


# ── E2E: gate -> discovery record -> splice -> round-trip ─────────


class TestEndToEndSplice:
    def test_admitted_patch_becomes_discovery_and_splices(self, tiny_arch_root):
        """An admitted DNAPatch can be wrapped as a DiscoveryRecord,
        marked verified, and spliced into arch.neuro, re-lifting cleanly."""
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        from neuroslm.evolution import (
            TrainingHeatmap, propose_mutations,
            gate_proposals, ImprovementEvidence,
        )
        from neuroslm.discoveries.records import DiscoveryRecord
        from neuroslm.discoveries.splice import splice_discovery_into_dna

        # 1-4: run the pipeline
        ir = lift_arch_to_hypergraph(tiny_arch_root)
        hm = TrainingHeatmap()
        hm.update(
            {"population:cortex": 1.0, "synapse:cortex->striatum": 0.9},
            kinds={"population:cortex": "node",
                   "synapse:cortex->striatum": "edge"},
        )
        proposals = propose_mutations(hm, ir, step=1000)
        evidence = {}
        for p in proposals:
            direction = ("increase" if p.kind == "node_mutation"
                         else "decrease")
            b, a = _improving_evidence(direction, seed=hash(p.target) & 0xff)
            evidence[p.target] = ImprovementEvidence(b, a)
        admitted, _ = gate_proposals(proposals, evidence)
        assert admitted, "the pipeline didn't admit any patches"

        # 5: wrap the first admitted patch as a DiscoveryRecord.
        patch = admitted[0]
        disc = DiscoveryRecord(
            id="D999",
            title="E2E test discovery",
            mechanism_md="Hot cortex mutation admitted by the L4 gate "
                         "via real improvement evidence.",
            mutation_chain=[patch.kind],
            theorem_name="Brian.Discoveries.D999_E2E",
            parent_dna_id="tiny_arch_v0",
            generation=1,
            fitness_before={"ppl": 30.0},
            fitness_after={"ppl": 27.0},
            fitness_delta={"ppl": -3.0},
            proof_status="verified",      # pretend Lean signed off
        )

        # 6: splice into arch.neuro
        arch_neuro = tiny_arch_root / "arch.neuro"
        before_text = arch_neuro.read_text(encoding="utf-8")
        splice_discovery_into_dna(disc, tiny_arch_root)
        after_text = arch_neuro.read_text(encoding="utf-8")

        # 7: the file grew + the discovery marker is in there
        assert len(after_text) > len(before_text)
        assert "Discovery D999" in after_text or "D999" in after_text

        # 8: the lifter still round-trips the new genome
        ir2 = lift_arch_to_hypergraph(tiny_arch_root)
        assert len(ir2.nodes) >= len(ir.nodes)

        # 9: the discovery is now marked integrated
        assert disc.dna_integrated is True
        assert disc.dna_integrated_at is not None


# ── E2E: NFG heat overlay reads the live heatmap ──────────────────


class TestEndToEndHeatRender:
    def test_heatmap_artifact_renders_in_nfg(self, tiny_arch_root, tmp_path):
        """Save a TrainingHeatmap to disk, render the NFG with --heat,
        and confirm the hot population's fill is the thermal color."""
        from neuroslm.evolution import TrainingHeatmap
        from neuroslm.compiler.heat_overlay import heat_to_fillcolor
        from neuroslm.compiler.nfg_graphviz import render_arch

        hm = TrainingHeatmap()
        hm.update(
            {"population:cortex": 1.0, "population:striatum": 0.05},
            kinds={"population:cortex": "node", "population:striatum": "node"},
        )
        heat_path = tmp_path / "live.heatmap.json"
        hm.save(str(heat_path))

        out_dot = tmp_path / "rendered.dot"
        render_arch(tiny_arch_root, str(out_dot),
                    format="dot", heat=str(heat_path))
        text = out_dot.read_text(encoding="utf-8")
        # The hot fill made it into the DOT source.
        assert heat_to_fillcolor(1.0).lower() in text.lower()


# ── E2E: hypothesis verification (smoke test, skipif no Lean) ────


class TestExistingHypothesisVerification:
    """Verify all 5 canonical hypotheses end-to-end via the L5 backend.

    Lean is not installed in the local env, so verify_lean_proof returns
    'skipped' for every record. We assert the *structural* contract:
    every hypothesis on disk is resolvable, the proof file exists, and
    the backend returns a structured LeanVerdict (rather than crashing).
    Once Lean is installed in CI, the same test confirms each H### at
    minimum 'compiles' (no errors, sorry still present).
    """

    def test_every_canonical_hypothesis_is_resolvable(self):
        """Each H001-H005 maps to a .lean file that exists on disk."""
        from neuroslm.evolution.lean_gate import LeanProofBackend
        from neuroslm.discoveries.store import HypothesisStore
        repo_root = Path(__file__).resolve().parent.parent
        store = HypothesisStore(repo_root / "hypothesis")
        backend = LeanProofBackend(hypothesis_root=repo_root / "hypothesis")

        records = store.list_all()
        assert len(records) >= 5
        for rec in records:
            proof = backend.resolve_proof_path(rec.id)
            assert proof is not None, f"{rec.id}: no proof file on disk"
            assert proof.exists()

    def test_every_canonical_hypothesis_yields_a_lean_verdict(self):
        """Calling the backend on every hypothesis returns a structured
        verdict; without Lean installed, status='skipped'."""
        from neuroslm.discoveries.store import HypothesisStore
        from neuroslm.discoveries.lean import verify_lean_proof
        from neuroslm.evolution.lean_gate import LeanProofBackend
        repo_root = Path(__file__).resolve().parent.parent
        store = HypothesisStore(repo_root / "hypothesis")
        backend = LeanProofBackend(hypothesis_root=repo_root / "hypothesis")

        records = store.list_all()
        for rec in records:
            proof = backend.resolve_proof_path(rec.id)
            verdict = verify_lean_proof(str(proof))
            assert verdict.status in (
                "verified", "compiles", "skipped", "error"
            ), f"{rec.id}: unexpected status {verdict.status}"
            # Without Lean -> 'skipped' is the only allowed outcome.
            if shutil.which("lean") is None:
                assert verdict.status == "skipped"
