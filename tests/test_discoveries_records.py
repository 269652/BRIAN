# -*- coding: utf-8 -*-
"""TDD: ``HypothesisRecord`` + ``DiscoveryRecord`` schema contract.

Both records are the unit of provenance for formally-tracked insights:

* :class:`HypothesisRecord` — a *human-authored* mathematical claim
  about the architecture (e.g. "$\\Phi$ is monotone under non-negative
  coupling additions"). Lives under ``hypothesis/`` with a sibling Lean
  file under ``hypothesis/proofs/``.
* :class:`DiscoveryRecord` — an *engine-authored* mutation that the
  evolutionary loop found and the admission gates accepted. Lives under
  ``discoveries/`` with a sibling Lean file under ``discoveries/proofs/``.

The tests below pin the storage contract: every field must round-trip
through ``to_dict``/``from_dict`` and through Markdown front-matter so
both human-authored ``.md`` files and machine-generated records share a
single, auditable serialisation format.
"""
from __future__ import annotations


# ───────────────────────────────────────────────────────────────────
# 1. HypothesisRecord schema
# ───────────────────────────────────────────────────────────────────

class TestHypothesisRecordSchema:
    """The hypothesis record must capture the mathematical statement,
    the Lean theorem name, code references, and the proof status."""

    def test_can_construct_minimal_record(self):
        from neuroslm.discoveries.records import HypothesisRecord
        h = HypothesisRecord(
            id="H001",
            title="Φ monotone under coupling addition",
            statement_md=r"$\Phi(\theta') \ge \Phi(\theta)$",
            theorem_name="Brian.PhiMonotone",
        )
        assert h.id == "H001"
        assert h.theorem_name == "Brian.PhiMonotone"
        # Sensible defaults — status starts at "draft", proof unfilled.
        assert h.status == "draft"
        assert h.proof_status == "missing"
        assert h.proof_path is None

    def test_to_dict_round_trips(self):
        from neuroslm.discoveries.records import HypothesisRecord
        h = HypothesisRecord(
            id="H002", title="OOD gap decrease",
            statement_md=r"$\Delta_{\mathrm{OOD}}(\theta+\lambda\cdot\mathrm{CDGA}) \le \Delta_{\mathrm{OOD}}(\theta)$",
            theorem_name="Brian.OodGapDecrease",
            status="stated",
            references=["formal_framework.md §10.2"],
            code_refs=["neuroslm/verification/improvement_gate.py"],
            test_refs=["tests/verification/test_improvement_gate.py"],
            proof_path="hypothesis/proofs/H002_ood_gap_decrease.lean",
            proof_status="stub",
            tags=["ood", "monotonicity"],
        )
        d = h.to_dict()
        h2 = HypothesisRecord.from_dict(d)
        assert h2 == h, "round-trip must preserve every field"

    def test_to_front_matter_round_trips(self):
        """The ``.md`` on disk is YAML front-matter + a Markdown body —
        every field except ``statement_md`` lives in the front-matter,
        ``statement_md`` is the document body."""
        from neuroslm.discoveries.records import HypothesisRecord
        h = HypothesisRecord(
            id="H003", title="Symbolic sparsity collapse",
            statement_md=r"As $\tau \to 0$, $|U(x)|_0 \to 1$",
            theorem_name="Brian.SymbolicSparsity",
        )
        text = h.to_markdown()
        # YAML front-matter must be present and parseable
        assert text.startswith("---\n")
        assert "id: H003" in text
        assert "theorem_name: Brian.SymbolicSparsity" in text
        # Body must include the statement
        assert r"As $\tau \to 0$" in text
        # Round-trip
        h2 = HypothesisRecord.from_markdown(text)
        assert h2.id == h.id
        assert h2.theorem_name == h.theorem_name
        assert h2.statement_md.strip() == h.statement_md.strip()

    def test_unknown_status_rejected(self):
        from neuroslm.discoveries.records import HypothesisRecord
        import pytest
        with pytest.raises(ValueError, match="status"):
            HypothesisRecord(
                id="H999", title="bogus", statement_md="x",
                theorem_name="Brian.Bogus",
                status="totally_invalid",
            )

    def test_unknown_proof_status_rejected(self):
        from neuroslm.discoveries.records import HypothesisRecord
        import pytest
        with pytest.raises(ValueError, match="proof_status"):
            HypothesisRecord(
                id="H999", title="x", statement_md="x",
                theorem_name="Brian.Bogus",
                proof_status="not_a_real_state",
            )

    def test_id_must_be_well_formed(self):
        """``id`` is a stable sort key; enforce a strict format."""
        from neuroslm.discoveries.records import HypothesisRecord
        import pytest
        # Good
        HypothesisRecord(id="H001", title="x", statement_md="x",
                         theorem_name="Brian.X")
        HypothesisRecord(id="H042", title="x", statement_md="x",
                         theorem_name="Brian.X")
        # Bad — must match H\d{3,}
        for bad in ["H1", "h001", "001", "Hyp-001", "H001a", ""]:
            with pytest.raises(ValueError, match="id"):
                HypothesisRecord(id=bad, title="x", statement_md="x",
                                 theorem_name="Brian.X")


# ───────────────────────────────────────────────────────────────────
# 2. DiscoveryRecord schema
# ───────────────────────────────────────────────────────────────────

class TestDiscoveryRecordSchema:
    """An autodiscovered mutation: the engine populates every field
    from the evolutionary run, then the Lean backend either verifies
    or stubs out the proof. ``dna_integrated`` is the bit that
    promotes a verified discovery back into the genome."""

    def test_can_construct_from_evolution_event(self):
        from neuroslm.discoveries.records import DiscoveryRecord
        d = DiscoveryRecord(
            id="D001",
            title="Add dopamine -> pfc modulation",
            mechanism_md="Adds a +0.3-gain dopaminergic modulation onto `pfc`",
            mutation_chain=["add_modulation"],
            parent_dna_id="dsl_arch_step1000",
            fitness_before={"ood_ppl": 250.0, "phi": 0.40},
            fitness_after={"ood_ppl": 238.5, "phi": 0.43},
            generation=4,
            theorem_name="Brian.Discoveries.D001_NoRegression",
        )
        # fitness_delta is autocomputed
        assert d.fitness_delta["ood_ppl"] == -11.5
        assert abs(d.fitness_delta["phi"] - 0.03) < 1e-9
        assert d.dna_integrated is False
        assert d.proof_status == "missing"

    def test_to_dict_round_trips(self):
        from neuroslm.discoveries.records import DiscoveryRecord
        d = DiscoveryRecord(
            id="D002", title="Slot-C inhibition saturation discovery",
            mechanism_md="Trunk inhibition saturated at α=0.97",
            mutation_chain=["mutate_numeric", "add_modulation"],
            parent_dna_id="rcc_bowtie@step5000",
            fitness_before={"loss": 4.2}, fitness_after={"loss": 3.9},
            generation=7,
            theorem_name="Brian.Discoveries.D002_SlotCSaturation",
            proof_status="stub",
            proof_path="discoveries/proofs/D002_slot_c.lean",
            dna_integrated=False,
            tags=["multi_cortex", "alpha_gating"],
            hypergraph_delta_json='{"added_nodes": [], "added_edges": []}',
        )
        d2 = DiscoveryRecord.from_dict(d.to_dict())
        assert d2 == d

    def test_promote_marks_integrated(self):
        """``promote_to_dna(at=ts)`` flips ``dna_integrated`` and
        stamps ``dna_integrated_at`` — once a discovery is in the
        lineage it cannot be silently re-promoted."""
        from neuroslm.discoveries.records import DiscoveryRecord
        d = DiscoveryRecord(
            id="D003", title="x", mechanism_md="x",
            mutation_chain=["mutate_numeric"],
            parent_dna_id="x",
            fitness_before={"x": 1.0}, fitness_after={"x": 0.9},
            generation=1,
            theorem_name="Brian.Discoveries.D003_X",
            proof_status="verified",
        )
        assert d.dna_integrated is False
        d.promote_to_dna(at="2026-06-09T10:00:00Z")
        assert d.dna_integrated is True
        assert d.dna_integrated_at == "2026-06-09T10:00:00Z"

    def test_promote_blocked_without_verified_proof(self):
        """Only ``proof_status == "verified"`` discoveries may be
        promoted — the integrity guarantee the whole pipeline exists
        to provide."""
        from neuroslm.discoveries.records import DiscoveryRecord
        import pytest
        d = DiscoveryRecord(
            id="D004", title="x", mechanism_md="x",
            mutation_chain=["x"], parent_dna_id="x",
            fitness_before={"x": 1.0}, fitness_after={"x": 0.9},
            generation=1,
            theorem_name="Brian.Discoveries.D004_X",
            proof_status="stub",  # not verified yet
        )
        with pytest.raises(RuntimeError, match="verified"):
            d.promote_to_dna(at="2026-06-09T10:00:00Z")

    def test_id_format_strict(self):
        from neuroslm.discoveries.records import DiscoveryRecord
        import pytest
        for bad in ["D1", "d001", "001", "Disc-001", "D001a"]:
            with pytest.raises(ValueError, match="id"):
                DiscoveryRecord(
                    id=bad, title="x", mechanism_md="x",
                    mutation_chain=["x"], parent_dna_id="x",
                    fitness_before={"x": 1.0}, fitness_after={"x": 0.9},
                    generation=1, theorem_name="Brian.X",
                )
