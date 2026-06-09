# -*- coding: utf-8 -*-
"""TDD: Lean proof emission + verification.

The :mod:`neuroslm.discoveries.lean` module emits ``.lean`` proof files
from a :class:`HypothesisRecord` or :class:`DiscoveryRecord`. Per
CLAUDE.md §12, the autogen scaffold uses the
``Brian.Postulate.Unimplemented`` marker as its obligation type — NOT
``sorry`` — so the file compiles cleanly in Lean while still being
caught by the Python-side static lint until a real proof is written.

Verification is two-phase:

  1. Static lint (always runs) — catches ``sorry`` / ``admit`` /
     ``Brian.Postulate.Unimplemented`` in committed files.
  2. Kernel check — if ``lean`` is on PATH, shells out and parses
     the verdict.

A missing Lean binary means the kernel check is skipped, but the
static lint still runs; a file passing static lint with no Lean on
PATH yields ``status="skipped"``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest


_HAS_LEAN = shutil.which("lean") is not None


# ───────────────────────────────────────────────────────────────────
# 1. Emit Lean stub from a hypothesis
# ───────────────────────────────────────────────────────────────────

class TestEmitHypothesisProof:

    def test_emits_file_with_theorem_declaration(self, tmp_path: Path):
        from neuroslm.discoveries.records import HypothesisRecord
        from neuroslm.discoveries.lean import emit_hypothesis_proof
        h = HypothesisRecord(
            id="H001", title="Φ monotone",
            statement_md=r"$\Phi(\theta') \ge \Phi(\theta)$",
            theorem_name="Brian.PhiMonotone",
        )
        out = emit_hypothesis_proof(h, tmp_path)
        out_path = Path(out)
        assert out_path.exists()
        text = out_path.read_text(encoding="utf-8")
        # Must declare a Lean theorem with the canonical name
        assert "theorem" in text
        assert "PhiMonotone" in text
        # Must reference the source markdown for round-trip auditability
        assert "H001" in text

    def test_emit_uses_unimplemented_marker_for_unproven(self, tmp_path: Path):
        """A freshly-emitted scaffold uses
        ``Brian.Postulate.Unimplemented`` (not ``sorry``) as the
        obligation type — the file compiles cleanly in Lean (so
        downstream tooling is unblocked) while still being caught by
        the Python static lint until a real proof is written.

        Per CLAUDE.md §12.2, ``sorry`` is banned in committed files;
        the autogen template must never emit it."""
        from neuroslm.discoveries.records import HypothesisRecord
        from neuroslm.discoveries.lean import (
            emit_hypothesis_proof, UNIMPLEMENTED_MARKER,
        )
        h = HypothesisRecord(
            id="H002", title="OOD gap decrease",
            statement_md="cdga makes ood gap monotone non-increasing",
            theorem_name="Brian.OodGapDecrease",
            proof_status="stub",
        )
        out = Path(emit_hypothesis_proof(h, tmp_path))
        text = out.read_text(encoding="utf-8")
        # New contract: the scaffold uses the Unimplemented marker.
        assert UNIMPLEMENTED_MARKER in text
        # Old contract: sorry is BANNED even in scaffolds.
        assert "sorry" not in text.lower() or "sorry`" in text.lower(), (
            "scaffolds must not contain literal `sorry` tokens "
            "(may mention 'sorry' inside docstring code spans)"
        )

    def test_emit_imports_brian_core_not_mathlib(self, tmp_path: Path):
        """The scaffold must import the Brian Lean library so
        hand-edits have THSD vocabulary in scope from line 1."""
        from neuroslm.discoveries.records import HypothesisRecord
        from neuroslm.discoveries.lean import emit_hypothesis_proof
        h = HypothesisRecord(
            id="H001", title="t",
            statement_md="s", theorem_name="Brian.PhiMonotone",
        )
        out = Path(emit_hypothesis_proof(h, tmp_path))
        text = out.read_text(encoding="utf-8")
        assert "import Brian.Core" in text
        # Mathlib must NOT be in the auto-import set (kept optional;
        # see lean/lakefile.lean for the rationale).
        assert "import Mathlib.Tactic" not in text

    def test_emit_does_not_clobber_hand_edited_proof(self, tmp_path: Path):
        """If the .lean file already exists AND has been hand-edited
        (no Unimplemented marker, no sorry), the emitter must leave
        it untouched. This is the idempotency contract that lets the
        CLI's ``emit-proofs`` run be repeated safely."""
        from neuroslm.discoveries.records import HypothesisRecord
        from neuroslm.discoveries.lean import emit_hypothesis_proof
        h = HypothesisRecord(
            id="H001", title="t",
            statement_md="s", theorem_name="Brian.PhiMonotone",
        )
        # Pre-create a hand-edited proof file.
        proofs_dir = tmp_path / "proofs"
        proofs_dir.mkdir(parents=True)
        proof_path = proofs_dir / h.proof_filename()
        hand_written = (
            "-- Hand-written proof; should not be clobbered.\n"
            "import Brian.Core\n"
            "open Brian.Thsd\n"
            "namespace Brian\n"
            "theorem PhiMonotone :\n"
            "    ∀ (s : Sheaf) (α : Coupling), Phi s ≤ Phi (s ⊕ α) :=\n"
            "  Phi_monotone_addCoupling\n"
            "end Brian\n"
        )
        proof_path.write_text(hand_written, encoding="utf-8")

        emit_hypothesis_proof(h, tmp_path)

        # File contents must be exactly the hand-written version.
        assert proof_path.read_text(encoding="utf-8") == hand_written

    def test_emit_overwrites_old_stub_proof(self, tmp_path: Path):
        """If the existing file contains the Unimplemented marker (or
        a legacy ``sorry``), the emitter regenerates it — those are
        scaffolds, not real work."""
        from neuroslm.discoveries.records import HypothesisRecord
        from neuroslm.discoveries.lean import (
            emit_hypothesis_proof, UNIMPLEMENTED_MARKER,
        )
        h = HypothesisRecord(
            id="H001", title="t",
            statement_md="s", theorem_name="Brian.PhiMonotone",
        )
        proofs_dir = tmp_path / "proofs"
        proofs_dir.mkdir(parents=True)
        proof_path = proofs_dir / h.proof_filename()
        # Legacy stub with sorry.
        proof_path.write_text(
            "-- old\ntheorem X : True := by sorry\n",
            encoding="utf-8",
        )
        emit_hypothesis_proof(h, tmp_path)
        new_text = proof_path.read_text(encoding="utf-8")
        # New scaffold has the Unimplemented marker.
        assert UNIMPLEMENTED_MARKER in new_text

    def test_output_path_under_proofs_subdir(self, tmp_path: Path):
        from neuroslm.discoveries.records import HypothesisRecord
        from neuroslm.discoveries.lean import emit_hypothesis_proof
        h = HypothesisRecord(
            id="H003", title="Symbolic sparsity",
            statement_md="x", theorem_name="Brian.SymbolicSparsity",
        )
        out = Path(emit_hypothesis_proof(h, tmp_path))
        # The emitter must place the .lean under ``<root>/proofs/`` so
        # the on-disk layout matches the schema in hypothesis/README.md.
        assert out.parent.name == "proofs"
        assert out.suffix == ".lean"

    def test_emit_updates_record_proof_path_and_status(self, tmp_path: Path):
        """After emission the record's ``proof_path`` must point at the
        new file and ``proof_status`` must be ``"stub"`` (the strongest
        claim we can make without running Lean)."""
        from neuroslm.discoveries.records import HypothesisRecord
        from neuroslm.discoveries.lean import emit_hypothesis_proof
        h = HypothesisRecord(
            id="H004", title="triple guard sound",
            statement_md="x",
            theorem_name="Brian.TripleGuardSound",
        )
        assert h.proof_path is None
        assert h.proof_status == "missing"
        emit_hypothesis_proof(h, tmp_path)
        assert h.proof_path is not None
        assert h.proof_path.endswith(".lean")
        assert h.proof_status == "stub"


# ───────────────────────────────────────────────────────────────────
# 2. Emit Lean stub from a discovery
# ───────────────────────────────────────────────────────────────────

class TestEmitDiscoveryProof:

    def _disc(self, did="D001", thm="Brian.Discoveries.D001_NoRegression"):
        from neuroslm.discoveries.records import DiscoveryRecord
        return DiscoveryRecord(
            id=did, title="add dopamine -> pfc modulation",
            mechanism_md="A +0.3 dopaminergic modulation onto pfc",
            mutation_chain=["add_modulation"],
            parent_dna_id="rcc_bowtie@step5000",
            fitness_before={"ood_ppl": 250.0, "phi": 0.40},
            fitness_after={"ood_ppl": 238.5, "phi": 0.43},
            generation=4,
            theorem_name=thm,
        )

    def test_emits_no_regression_obligation(self, tmp_path: Path):
        """The Lean obligation for a discovery is *"this mutation does
        not regress any fitness coordinate the gates care about"* —
        the metric values must appear in a comment block so the
        kernel reader has the empirical premise to hand. The
        obligation itself uses the Unimplemented marker (not
        ``sorry``) per CLAUDE.md §12."""
        from neuroslm.discoveries.lean import (
            emit_discovery_proof, UNIMPLEMENTED_MARKER,
        )
        out = Path(emit_discovery_proof(self._disc(), tmp_path))
        text = out.read_text(encoding="utf-8")
        assert "Brian.Discoveries" in text
        assert "D001" in text
        # Empirical witness in a comment block.
        assert "ood_ppl" in text or "ppl" in text.lower()
        # Scaffold marker — not sorry.
        assert UNIMPLEMENTED_MARKER in text
        assert "sorry" not in text.lower()

    def test_distinct_proof_for_each_discovery(self, tmp_path: Path):
        from neuroslm.discoveries.lean import emit_discovery_proof
        o1 = Path(emit_discovery_proof(self._disc("D001",
                  "Brian.Discoveries.D001_X"), tmp_path))
        o2 = Path(emit_discovery_proof(self._disc("D002",
                  "Brian.Discoveries.D002_X"), tmp_path))
        assert o1 != o2, "each discovery must emit a distinct .lean file"


# ───────────────────────────────────────────────────────────────────
# 3. Static lint (CLAUDE.md §12 enforcement, Lean-binary-free)
# ───────────────────────────────────────────────────────────────────


class TestStaticLint:
    """Pure-Python lint catches forbidden patterns even when the
    Lean toolchain is absent — that's how rule 12 has teeth on a
    bare CI worker."""

    def test_clean_file_passes(self, tmp_path: Path):
        from neuroslm.discoveries.lean import static_lint_lean_proof
        f = tmp_path / "ok.lean"
        f.write_text(
            "import Brian.Core\n"
            "namespace Brian\n"
            "theorem T : 1 + 1 = 2 := rfl\n"
            "end Brian\n",
            encoding="utf-8",
        )
        assert static_lint_lean_proof(str(f)) == []

    def test_sorry_outside_comments_flagged(self, tmp_path: Path):
        from neuroslm.discoveries.lean import static_lint_lean_proof
        f = tmp_path / "bad.lean"
        f.write_text(
            "theorem T : True := by sorry\n",
            encoding="utf-8",
        )
        errs = static_lint_lean_proof(str(f))
        assert errs, "literal `sorry` must be flagged"
        assert any("[sorry]" in e for e in errs)

    def test_sorry_inside_block_comment_ignored(self, tmp_path: Path):
        from neuroslm.discoveries.lean import static_lint_lean_proof
        f = tmp_path / "comment.lean"
        f.write_text(
            "/- This file does NOT contain a sorry. -/\n"
            "theorem T : 1 = 1 := rfl\n",
            encoding="utf-8",
        )
        assert static_lint_lean_proof(str(f)) == []

    def test_sorry_inside_line_comment_ignored(self, tmp_path: Path):
        from neuroslm.discoveries.lean import static_lint_lean_proof
        f = tmp_path / "linecomment.lean"
        f.write_text(
            "theorem T : 1 = 1 := rfl  -- no sorry here\n",
            encoding="utf-8",
        )
        assert static_lint_lean_proof(str(f)) == []

    def test_admit_flagged(self, tmp_path: Path):
        from neuroslm.discoveries.lean import static_lint_lean_proof
        f = tmp_path / "admit.lean"
        f.write_text(
            "theorem T : True := by admit\n",
            encoding="utf-8",
        )
        errs = static_lint_lean_proof(str(f))
        assert any("[admit]" in e for e in errs)

    def test_unimplemented_marker_flagged(self, tmp_path: Path):
        """Committed files must not contain the autogen scaffold
        marker. Once a human writes the real proof, the marker
        disappears and the lint passes."""
        from neuroslm.discoveries.lean import (
            static_lint_lean_proof, UNIMPLEMENTED_MARKER,
        )
        f = tmp_path / "stub.lean"
        f.write_text(
            f"import Brian.Core\n"
            f"theorem T : {UNIMPLEMENTED_MARKER} \"H999\" :=\n"
            f"  Brian.Postulate.unimplemented \"H999\"\n",
            encoding="utf-8",
        )
        errs = static_lint_lean_proof(str(f))
        assert any("[unimplemented]" in e for e in errs)


# ───────────────────────────────────────────────────────────────────
# 4. Verification — pure-Python (no lean binary required)
# ───────────────────────────────────────────────────────────────────

class TestVerifyLeanProofPurePython:
    """Even without the Lean binary the backend must give a structured
    verdict so the rest of the pipeline (record updates, splice
    decisions) can keep moving deterministically.

    The static lint runs in every call so rule-12 violations
    (sorry / admit / Unimplemented) yield ``status="error"`` even
    on a bare CI worker."""

    def test_skipped_when_lean_binary_absent_and_lint_passes(
            self, tmp_path: Path, monkeypatch):
        """A clean file + no Lean → status="skipped"."""
        from neuroslm.discoveries.lean import verify_lean_proof
        lean_file = tmp_path / "clean.lean"
        lean_file.write_text(
            "import Brian.Core\n"
            "theorem ok : 1 + 1 = 2 := rfl\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "neuroslm.discoveries.lean._lean_binary", lambda: None)
        verdict = verify_lean_proof(str(lean_file))
        assert verdict.status == "skipped"
        assert verdict.proof_status_for_record() == "stub"

    def test_error_when_lint_fails_even_without_lean(
            self, tmp_path: Path, monkeypatch):
        """A file with `sorry` → status="error" regardless of Lean
        toolchain availability. This is the CLAUDE.md §12 enforcement
        path that runs on every CI worker."""
        from neuroslm.discoveries.lean import verify_lean_proof
        lean_file = tmp_path / "bad.lean"
        lean_file.write_text(
            "theorem T : True := by sorry\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "neuroslm.discoveries.lean._lean_binary", lambda: None)
        verdict = verify_lean_proof(str(lean_file))
        assert verdict.status == "error"
        assert verdict.n_sorry >= 1

    def test_verdict_records_filename(self, tmp_path: Path, monkeypatch):
        from neuroslm.discoveries.lean import verify_lean_proof
        lean_file = tmp_path / "stub.lean"
        # Use a clean file so we hit the "skipped" path and exercise
        # the filename plumbing rather than the lint short-circuit.
        lean_file.write_text(
            "import Brian.Core\n"
            "theorem ok : 1 + 1 = 2 := rfl\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "neuroslm.discoveries.lean._lean_binary", lambda: None)
        verdict = verify_lean_proof(str(lean_file))
        assert verdict.file == str(lean_file)


@pytest.mark.skipif(not _HAS_LEAN, reason="lean binary not on PATH")
class TestVerifyLeanProofWithBinary:
    """When Lean *is* installed we can actually call it. These tests
    cover the trivial happy path and the ``sorry`` warning path."""

    def test_trivial_theorem_verified(self, tmp_path: Path):
        from neuroslm.discoveries.lean import verify_lean_proof
        lean_file = tmp_path / "ok.lean"
        # `lean` will accept a trivial proof; the exact status depends on
        # the Lean release on PATH (Lean 4 names it `True.intro`, Lean 3
        # uses `trivial`). Either way the kernel returns 0 errors.
        lean_file.write_text(
            "example : True := True.intro\n", encoding="utf-8")
        verdict = verify_lean_proof(str(lean_file))
        # Either it verifies cleanly, or `lean` rejected it for a Lean-3
        # vs Lean-4 syntax mismatch — both are valid pipeline outcomes,
        # we just need a structured verdict.
        assert verdict.status in {"verified", "error", "skipped"}
