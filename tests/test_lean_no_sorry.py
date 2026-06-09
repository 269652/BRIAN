# -*- coding: utf-8 -*-
"""CLAUDE.md §12 enforcement — every committed `.lean` file passes
static lint.

Walks `lean/` and `hypothesis/proofs/` (the two trees holding
committed Lean source) and asserts each file:

  * does NOT contain `sorry` outside comments,
  * does NOT contain `admit` outside comments,
  * does NOT contain the `Brian.Postulate.Unimplemented` autogen
    scaffold marker outside comments,
  * imports `Brian.Core` (or a `Brian.*` submodule) if it lives under
    `hypothesis/proofs/` — those are obligations whose theorem
    statements must use THSD vocabulary.

This test runs without the Lean toolchain — it's the rule-12
enforcement that CI workers without `lean` still execute on every
commit.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _all_lean_files() -> list[Path]:
    """Every committed `.lean` file, sorted for deterministic output."""
    roots = [_REPO_ROOT / "lean", _REPO_ROOT / "hypothesis" / "proofs"]
    files: list[Path] = []
    for r in roots:
        if r.is_dir():
            files.extend(sorted(r.rglob("*.lean")))
    return files


@pytest.fixture(scope="module")
def lean_files() -> list[Path]:
    files = _all_lean_files()
    assert files, "expected at least one .lean file under lean/ or hypothesis/proofs/"
    return files


def test_no_sorry_in_any_committed_lean_file(lean_files: list[Path]) -> None:
    """No committed .lean file contains `sorry` outside comments."""
    from neuroslm.discoveries.lean import static_lint_lean_proof
    offending: list[tuple[Path, list[str]]] = []
    for f in lean_files:
        errs = [e for e in static_lint_lean_proof(str(f)) if "[sorry]" in e]
        if errs:
            offending.append((f, errs))
    if offending:
        msg = "\n".join(
            f"  {f.relative_to(_REPO_ROOT)}:\n    " + "\n    ".join(es)
            for f, es in offending
        )
        pytest.fail("`sorry` found in committed .lean files:\n" + msg)


def test_no_admit_in_any_committed_lean_file(lean_files: list[Path]) -> None:
    """No committed .lean file contains `admit` outside comments."""
    from neuroslm.discoveries.lean import static_lint_lean_proof
    offending: list[tuple[Path, list[str]]] = []
    for f in lean_files:
        errs = [e for e in static_lint_lean_proof(str(f)) if "[admit]" in e]
        if errs:
            offending.append((f, errs))
    if offending:
        msg = "\n".join(
            f"  {f.relative_to(_REPO_ROOT)}:\n    " + "\n    ".join(es)
            for f, es in offending
        )
        pytest.fail("`admit` found in committed .lean files:\n" + msg)


def test_no_unimplemented_marker_in_any_committed_lean_file(
        lean_files: list[Path]) -> None:
    """No committed .lean file contains the autogen Unimplemented marker.

    The marker is the autogen scaffold's placeholder; once a hand
    edit lands the marker is gone. A committed file with the marker
    means an unfinished scaffold was committed by mistake."""
    from neuroslm.discoveries.lean import static_lint_lean_proof
    offending: list[tuple[Path, list[str]]] = []
    for f in lean_files:
        errs = [
            e for e in static_lint_lean_proof(str(f))
            if "[unimplemented]" in e
        ]
        if errs:
            offending.append((f, errs))
    if offending:
        msg = "\n".join(
            f"  {f.relative_to(_REPO_ROOT)}:\n    " + "\n    ".join(es)
            for f, es in offending
        )
        pytest.fail(
            "`Brian.Postulate.Unimplemented` (autogen scaffold) found in "
            "committed .lean files:\n" + msg
        )


def test_hypothesis_proofs_import_brian_core(lean_files: list[Path]) -> None:
    """Every `hypothesis/proofs/H###_*.lean` must `import Brian.Core` (or
    a narrower `Brian.*` submodule) so the obligation uses THSD
    vocabulary (CLAUDE.md §12.1)."""
    hypothesis_proofs = [
        f for f in lean_files
        if "hypothesis" in f.parts and "proofs" in f.parts
    ]
    assert hypothesis_proofs, "expected hypothesis/proofs/H###_*.lean files"
    offending: list[Path] = []
    for f in hypothesis_proofs:
        text = f.read_text(encoding="utf-8")
        if "import Brian" not in text:
            offending.append(f)
    if offending:
        msg = "\n".join(
            f"  {f.relative_to(_REPO_ROOT)}" for f in offending
        )
        pytest.fail(
            "hypothesis proofs missing `import Brian.Core` (or a narrower "
            "Brian.* submodule):\n" + msg
        )


def test_hypothesis_proofs_dont_use_trivial_true_obligation(
        lean_files: list[Path]) -> None:
    """Per CLAUDE.md §12.2, theorems in hypothesis proofs must not have
    the trivial `: True := by trivial` obligation. The obligation
    type must be a meaningful proposition in THSD vocabulary."""
    import re
    pattern = re.compile(r"theorem\s+\w+[^:=]*:\s*True\s*:=")
    hypothesis_proofs = [
        f for f in lean_files
        if "hypothesis" in f.parts and "proofs" in f.parts
    ]
    offending: list[Path] = []
    for f in hypothesis_proofs:
        text = f.read_text(encoding="utf-8")
        if pattern.search(text):
            offending.append(f)
    if offending:
        msg = "\n".join(
            f"  {f.relative_to(_REPO_ROOT)}" for f in offending
        )
        pytest.fail(
            "hypothesis proof theorem with trivial `: True` obligation "
            "(CLAUDE.md §12.2):\n" + msg
        )


def test_all_committed_proofs_yield_clean_verdict(lean_files: list[Path]) -> None:
    """Smoke: every committed file passes :func:`verify_lean_proof`'s
    static-lint phase. The Lean toolchain may be absent (verdict =
    `"skipped"`); the rule is that no committed file should yield
    `"error"` from lint alone."""
    from neuroslm.discoveries.lean import verify_lean_proof
    bad: list[tuple[Path, list[str]]] = []
    for f in lean_files:
        v = verify_lean_proof(str(f))
        if v.status == "error":
            bad.append((f, v.errors))
    if bad:
        msg = "\n".join(
            f"  {f.relative_to(_REPO_ROOT)}:\n    "
            + "\n    ".join(es) for f, es in bad
        )
        pytest.fail("verify_lean_proof returned `error` for:\n" + msg)
