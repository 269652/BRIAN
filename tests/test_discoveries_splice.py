# -*- coding: utf-8 -*-
"""TDD: ``splice_discovery_into_dna`` — promotes a verified discovery
back into the genome.

A discovery describes a mutation as a hypergraph delta plus a mutation
chain (e.g. ``["add_modulation"]`` with ``add_modulation`` arguments).
The splice function applies that mutation to a target architecture by
appending the corresponding DSL declarations to ``arch.neuro`` (or the
appropriate module) and rewriting the file.

The function refuses to splice unless the discovery's ``proof_status``
is ``"verified"`` — this is the bright line between *"the engine
proposed this"* and *"the engine proposed this AND a formal proof
showed it doesn't regress the invariants"*.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest


# ───────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────

def _seed_minimal_arch(root: Path) -> Path:
    """Create a tiny synthetic architecture so we don't mutate real
    rcc_bowtie during tests."""
    arch_dir = root / "tiny_arch"
    arch_dir.mkdir()
    (arch_dir / "arch.neuro").write_text(
        "architecture tiny {\n"
        "    d_sem: 16,\n"
        "    dt: 0.01\n"
        "}\n"
        "\n"
        "neurotransmitter dopamine {\n"
        "    base_concentration: 0.10,\n"
        "    release_rate: 0.20,\n"
        "    reuptake_rate: 0.80,\n"
        "    diffusion_rate: 0.02\n"
        "}\n"
        "\n"
        "population pfc {\n"
        "    count: 32, timescale: 0.02, dynamics: \"rate_code\"\n"
        "}\n",
        encoding="utf-8",
    )
    return arch_dir


def _disc(did="D001",
          mutation_chain=("add_modulation",),
          mutation_args=None,
          proof_status="verified"):
    from neuroslm.discoveries.records import DiscoveryRecord
    if mutation_args is None:
        mutation_args = [{
            "op": "add_modulation",
            "nt": "dopamine",
            "target": "pfc",
            "effect": "multiplicative",
            "gain": 0.3,
        }]
    return DiscoveryRecord(
        id=did, title="add dopamine -> pfc modulation",
        mechanism_md="A +0.3 dopaminergic modulation onto pfc",
        mutation_chain=list(mutation_chain),
        parent_dna_id="tiny_arch",
        fitness_before={"ood_ppl": 250.0},
        fitness_after={"ood_ppl": 238.5},
        generation=4,
        theorem_name=f"Brian.Discoveries.{did}_X",
        proof_status=proof_status,
        mutation_args_json=__import__("json").dumps(mutation_args),
    )


# ───────────────────────────────────────────────────────────────────
# 1. Splice contract
# ───────────────────────────────────────────────────────────────────

class TestSpliceDiscoveryIntoDna:

    def test_appends_modulation_to_arch_neuro(self, tmp_path: Path):
        from neuroslm.discoveries.splice import splice_discovery_into_dna
        arch = _seed_minimal_arch(tmp_path)
        d = _disc()
        result = splice_discovery_into_dna(d, arch)
        text = (arch / "arch.neuro").read_text(encoding="utf-8")
        # The DSL should now contain a modulation declaration we asked for.
        assert "modulation dopamine -> pfc" in text, \
            "splice did not append the modulation"
        assert result.success is True
        assert result.touched_files == [str((arch / "arch.neuro").resolve())]

    def test_records_d_id_in_appended_block(self, tmp_path: Path):
        """The appended block must be tagged with the discovery id so
        a human (or another tool) can backtrack from genome to ledger."""
        from neuroslm.discoveries.splice import splice_discovery_into_dna
        arch = _seed_minimal_arch(tmp_path)
        d = _disc("D042")
        splice_discovery_into_dna(d, arch)
        text = (arch / "arch.neuro").read_text(encoding="utf-8")
        assert "D042" in text, "appended block must reference discovery id"

    def test_refuses_when_proof_not_verified(self, tmp_path: Path):
        from neuroslm.discoveries.splice import splice_discovery_into_dna
        arch = _seed_minimal_arch(tmp_path)
        d = _disc("D050", proof_status="stub")
        with pytest.raises(RuntimeError, match="verified"):
            splice_discovery_into_dna(d, arch)

    def test_marks_discovery_integrated(self, tmp_path: Path):
        from neuroslm.discoveries.splice import splice_discovery_into_dna
        arch = _seed_minimal_arch(tmp_path)
        d = _disc("D060")
        assert d.dna_integrated is False
        splice_discovery_into_dna(d, arch)
        assert d.dna_integrated is True
        assert d.dna_integrated_at is not None

    def test_idempotent_no_double_splice(self, tmp_path: Path):
        """Splicing the same discovery twice must be a no-op the second
        time — once the bit is set in the record we refuse to write
        the block again."""
        from neuroslm.discoveries.splice import splice_discovery_into_dna
        arch = _seed_minimal_arch(tmp_path)
        d = _disc("D070")
        splice_discovery_into_dna(d, arch)
        text_after_first = (arch / "arch.neuro").read_text(encoding="utf-8")
        # Second call should be a no-op or explicit error — not a duplicate
        result2 = splice_discovery_into_dna(d, arch)
        text_after_second = (arch / "arch.neuro").read_text(encoding="utf-8")
        assert text_after_first == text_after_second
        assert result2.success is False
        assert "already" in (result2.reason or "").lower()

    def test_round_trips_through_lifter(self, tmp_path: Path):
        """After splicing, ``lift_arch_to_hypergraph`` must see the new
        modulation in the IR — closing the loop from discovery →
        DSL → IR."""
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        from neuroslm.discoveries.splice import splice_discovery_into_dna
        arch = _seed_minimal_arch(tmp_path)
        d = _disc("D080")
        splice_discovery_into_dna(d, arch)
        ir = lift_arch_to_hypergraph(arch)
        mod_edges = [e for e in ir.hyperedges if e.kind == "modulation"]
        ids = {e.id for e in mod_edges}
        assert "modulation:dopamine->pfc" in ids, \
            f"lifter did not see spliced modulation; ids={ids}"
