"""Lint checks on the H015..H018 Lean proofs.

These mirror the discipline enforced by ``neuroslm.discoveries.lean``:

* no bare ``sorry`` / ``admit`` in the proof body,
* no ``Brian.Postulate.Unimplemented`` token (the autogen scaffold),
* every proof imports ``Brian.Core``,
* every proof file is referenced by its hypothesis markdown record
  via the ``proof_path`` front-matter key.

Lean is not required to be on PATH; the Python-side static lint already
runs without it (the kernel check is a separate ``verify`` step). When
Lean is installed, the proofs additionally compile with ``lake build``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_PROOFS = _ROOT / "hypothesis" / "proofs"
_INDEX = _ROOT / "hypothesis" / "index.json"

NFO_PROOFS = [
    "H015_kuramoto_coherence_phi_lower_bound.lean",
    "H016_coherence_gate_information_preserving.lean",
    "H017_swift_hohenberg_contractive.lean",
    "H018_nfo_readout_zero_init_identity.lean",
]


def _read(name: str) -> str:
    p = _PROOFS / name
    assert p.exists(), f"missing proof file: {p}"
    return p.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", NFO_PROOFS)
def test_proof_imports_brian_core(name: str):
    src = _read(name)
    assert "import Brian.Core" in src, (
        f"{name}: every hypothesis proof must `import Brian.Core` per "
        f"CLAUDE.md §12.1"
    )


@pytest.mark.parametrize("name", NFO_PROOFS)
def test_proof_has_no_sorry(name: str):
    src = _read(name)
    # `sorry` may appear inside doc-comments. Strip block + line comments.
    src_no_block = re.sub(r"/-(.|\n)*?-/", "", src)
    src_no_line = re.sub(r"--.*", "", src_no_block)
    for banned in ("sorry", "admit", "Brian.Postulate.Unimplemented"):
        assert banned not in src_no_line, (
            f"{name}: forbidden token `{banned}` outside doc-comments"
        )


@pytest.mark.parametrize("name", NFO_PROOFS)
def test_proof_has_namespace_brian(name: str):
    src = _read(name)
    assert "namespace Brian" in src, (
        f"{name}: every hypothesis proof must declare `namespace Brian`"
    )
    assert "end Brian" in src


def test_index_lists_all_four_nfo_hypotheses():
    data = json.loads(_INDEX.read_text(encoding="utf-8"))
    ids = {row["id"] for row in data["records"]}
    for hid in ("H015", "H016", "H017", "H018"):
        assert hid in ids, f"{hid} missing from hypothesis index"


def test_nfo_lean_library_compiles_referenced_in_core():
    """Brian.Nfo must be wired into Brian.Core so the hypothesis proofs
    can `import Brian.Core` and pull the vocabulary."""
    core = (_ROOT / "lean" / "Brian" / "Core.lean").read_text(encoding="utf-8")
    assert "import Brian.Nfo" in core, (
        "Brian.Core must import Brian.Nfo (the H015/H016/H018 vocabulary)"
    )
    assert "import Brian.Postulate.Nfo" in core, (
        "Brian.Core must import Brian.Postulate.Nfo (the H017 admission)"
    )


def test_postulate_nfo_file_exists_and_is_namespaced_correctly():
    p = _ROOT / "lean" / "Brian" / "Postulate" / "Nfo.lean"
    assert p.exists(), "Brian.Postulate.Nfo file missing"
    src = p.read_text(encoding="utf-8")
    assert "namespace Brian.Postulate.Nfo" in src
    # CLAUDE.md §12.2 audit: every postulate file declares the namespace
    # AND has a doc-comment citing the empirical evidence.
    assert "tests/modules/test_nfo.py" in src, (
        "Brian.Postulate.Nfo must cite the empirical evidence test"
    )
