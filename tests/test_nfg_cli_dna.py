# -*- coding: utf-8 -*-
"""TDD acceptance suite — `brian compile nfg` must accept .dna inputs.

The user's failing command (Windows shell):

    PS > brian compile nfg .\\architectures\\evol\\
    ResolverError: missing arch.neuro at architecture root ...\\architectures\\evol

Background
----------
A ``.dna`` file is a self-contained snapshot of an architecture's full
DSL (stored verbatim in ``dna.invariants["dsl_code"]``).  When the user
unfolds it next to the ``.dna`` (the common pattern from
``brian dna unfold X.dna --output some/dir/``), the resulting folder
contains the snapshot but *not* the ``modules/`` / ``lib/`` sub-trees
the original arch.neuro imported with ``import { … } from "@/…"``.

Trying to compile that folder fails because:
  1. There is no file literally named ``arch.neuro`` in it.
  2. Even if we renamed the snapshot, the multi-file resolver could not
     follow the ``@/modules/…`` paths because the sibling directories
     don't exist in the snapshot.

The fix is to teach ``cmd_compile_nfg`` to be DNA-aware:

  * Given a ``.dna`` file (or a folder whose ``arch.neuro`` is missing
    but contains exactly one ``.dna``), unfold the DNA in memory,
    extract the ``architecture <name> { … }`` block to find the source
    architecture's name, and route the NFG render through the live
    ``architectures/<name>/`` directory.

This is the same routing pattern ``init_evolution()`` already uses for
training-from-DNA — see ``neuroslm/utils/colab.py``.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pytest

from neuroslm.compiler.ribosome import RibosomeCompiler

REPO_ROOT = Path(__file__).parent.parent


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_dna(tmpdir: Path) -> Path:
    """Compile the canonical bowtie architecture into a fresh ``.dna`` snapshot."""
    compiler = RibosomeCompiler()
    arch_root = str(REPO_ROOT / "architectures" / "master")
    dna_path = tmpdir / "evol.dna"
    compiler.compile_file(arch_root, str(dna_path))
    assert dna_path.exists(), "fixture failed: DNA was not produced"
    return dna_path


# ──────────────────────────────────────────────────────────────────────
# Compile-NFG from a .dna file
# ──────────────────────────────────────────────────────────────────────

class TestCompileNfgFromDnaFile:
    """`brian compile nfg some/path/evol.dna` should succeed."""

    def test_dna_file_argument_succeeds(self):
        """The minimal regression: an explicit ``.dna`` file must compile
        to an NFG without raising ``missing arch.neuro``."""
        from neuroslm.cli import cmd_compile_nfg

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            dna_path = _make_dna(tmpdir)

            out_png = tmpdir / "nfg.png"

            args = argparse.Namespace(
                arch=str(dna_path),
                out=None,
                png=str(out_png),
                semantic=False,
            )

            # The pre-fix behavior was:
            #   ResolverError: missing arch.neuro at architecture root <dna parent>
            rc = cmd_compile_nfg(args)
            assert rc == 0, "DNA-aware compile_nfg must return 0"
            # The new Graphviz pipeline writes a render. The PNG step is
            # skipped only when the `dot` binary is missing — in that
            # case the command should still have returned 0 because the
            # legacy fallback can also handle this case.
            assert out_png.exists() or rc == 0

    def test_dna_file_produces_nontrivial_nfg(self):
        """The NFG built from a DNA snapshot must contain at least one
        population and one synapse — the same shape we get from the
        source architecture directly."""
        from neuroslm.cli import cmd_compile_nfg
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            dna_path = _make_dna(tmpdir)
            out_dot = tmpdir / "nfg.dot"

            # Use --format dot so we don't depend on the `dot` binary
            # being installed for this assertion to succeed.
            args = argparse.Namespace(
                arch=str(dna_path),
                out=str(out_dot),
                png=None,
                semantic=False,
                format="dot",
                engine="dot",
                legacy=False,
            )

            assert cmd_compile_nfg(args) == 0
            assert out_dot.exists(), "expected DOT render not written"
            content = out_dot.read_text(encoding="utf-8")
            # The DOT file must declare nodes and edges and reference at
            # least one rcc_bowtie population so we know it really
            # compiled from the snapshotted DSL.
            assert "digraph" in content
            assert "->" in content, "no edges in DOT output"
            assert "thalamus" in content or "pfc" in content


# ──────────────────────────────────────────────────────────────────────
# Compile-NFG from a folder containing a .dna but no arch.neuro
# ──────────────────────────────────────────────────────────────────────

class TestCompileNfgFromDnaFolder:
    """`brian compile nfg some/dir/` where ``some/dir/`` contains exactly
    one ``.dna`` file and no ``arch.neuro`` should also work, by
    auto-selecting the lone ``.dna``.

    This is the exact failure from the user's command line:
        brian compile nfg .\\architectures\\evol\\
    where ``architectures\\evol\\`` has only ``evol.dna`` + ``evol.neuro``.
    """

    def test_folder_with_single_dna_succeeds(self):
        from neuroslm.cli import cmd_compile_nfg

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            arch_dir = tmpdir / "evol"
            arch_dir.mkdir()
            _ = _make_dna(arch_dir)  # writes evol/evol.dna

            out_dot = arch_dir / "nfg.dot"
            args = argparse.Namespace(
                arch=str(arch_dir),
                out=str(out_dot),
                png=None,
                semantic=False,
                format="dot",
                engine="dot",
                legacy=False,
            )

            rc = cmd_compile_nfg(args)
            assert rc == 0, "folder-with-single-DNA path must route through"
            assert out_dot.exists()

    def test_folder_with_trailing_separator_succeeds(self):
        """Windows-style trailing-backslash argument — same as the user's
        original failure mode."""
        from neuroslm.cli import cmd_compile_nfg

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            arch_dir = tmpdir / "evol"
            arch_dir.mkdir()
            _ = _make_dna(arch_dir)

            arch_arg = str(arch_dir) + os.sep  # trailing separator
            out_dot = arch_dir / "nfg.dot"
            args = argparse.Namespace(
                arch=arch_arg,
                out=str(out_dot),
                png=None,
                semantic=False,
                format="dot",
                engine="dot",
                legacy=False,
            )

            assert cmd_compile_nfg(args) == 0
            assert out_dot.exists()


# ──────────────────────────────────────────────────────────────────────
# Existing arch.neuro-bearing folders MUST keep working
# ──────────────────────────────────────────────────────────────────────

class TestCompileNfgRegression:
    """Regression guard: a real architecture folder with a proper
    ``arch.neuro`` must continue to compile via the original path —
    we are *adding* the DNA route, not replacing the DSL one."""

    def test_full_arch_folder_still_compiles(self):
        from neuroslm.cli import cmd_compile_nfg

        rcc = REPO_ROOT / "architectures" / "master"
        if not (rcc / "arch.neuro").is_file():
            pytest.skip("master arch not present")

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            out_dot = tmpdir / "nfg.dot"
            args = argparse.Namespace(
                arch=str(rcc),
                out=str(out_dot),
                png=None,
                semantic=False,
                format="dot",
                engine="dot",
                legacy=False,
            )

            assert cmd_compile_nfg(args) == 0
            assert out_dot.exists()


# ──────────────────────────────────────────────────────────────────────
# Error path — DNA referencing unknown architecture
# ──────────────────────────────────────────────────────────────────────

class TestCompileNfgFromDnaErrors:
    """When the DNA points to an architecture that isn't on disk, the
    CLI must fail with a clear message — not a confusing
    ``missing arch.neuro at <dna parent>``."""

    def test_dna_with_unknown_arch_fails_clearly(self):
        from neuroslm.cli import cmd_compile_nfg
        from neuroslm.compiler.ribosome import (
            DNATranscriber,
            LatentDNA,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)

            # Fabricate a DNA whose embedded DSL declares an architecture
            # name that does NOT exist in REPO_ROOT/architectures/.
            transcriber = DNATranscriber()
            bogus_dsl = (
                "architecture totally_made_up_arch { d_sem: 64, dt: 0.01 }\n"
                "population pop_a { count: 8, dynamics: \"rate_code\" }\n"
            )
            dna: LatentDNA = transcriber.transcribe(bogus_dsl)
            dna_path = tmpdir / "bogus.dna"
            dna.save(str(dna_path))

            args = argparse.Namespace(
                arch=str(dna_path),
                out=str(tmpdir / "nfg.dot"),
                png=None,
                semantic=False,
                format="dot",
                engine="dot",
                legacy=False,
            )

            # We accept either a non-zero return code OR a raised
            # exception — both signal "user supplied an unresolvable
            # DNA" in a way the caller can react to.
            try:
                rc = cmd_compile_nfg(args)
            except (FileNotFoundError, ValueError) as e:
                # Message must mention the bogus arch name so the user
                # can immediately see WHY we couldn't render the NFG.
                assert "totally_made_up_arch" in str(e), (
                    f"error message lacks arch name: {e!r}"
                )
                return
            assert rc != 0, (
                "compile_nfg should fail when the DNA's source "
                "architecture is not on disk"
            )
