"""Contracts for ``neuroslm.compiler.run_workspace.prepare_run_workspace``.

The workspace helper standardises where DSL / DNA inputs get unfolded
before a training run, so the harness, NFG renderer, and Hypergraph IR
all read from the same canonical location:

    ``<repo>/.neuro/arch/temp/arch.neuro``    (+ modules/, lib/)

It accepts EITHER a ``--dna`` snapshot or an ``--arch`` folder (or a
.neuro file inside one). The output is always the same shape so
downstream code only has one path to handle.

Contracts pinned here:

  1. ``--dna <path>``  → unfold the DNA's modular tree under
                          ``.neuro/arch/temp/`` and return a
                          :class:`RunWorkspace` describing the result.
  2. ``--arch <folder>`` → copy ``arch.neuro`` + ``modules/`` + ``lib/``
                            into ``.neuro/arch/temp/`` (so the run
                            never reads from the source tree directly —
                            avoids accidental in-place writes from
                            checkpoint hooks).
  3. ``--arch <file>.neuro`` → same as folder, with the file's parent
                                directory as the arch root.
  4. ``--arch <name>`` (no slash, no suffix) → resolve to
                         ``architectures/<name>/`` then apply (2).
  5. The returned ``RunWorkspace`` exposes:
        - ``arch_root: Path``       — ``.neuro/arch/temp/``
        - ``arch_neuro: Path``      — ``.neuro/arch/temp/arch.neuro``
        - ``hypergraph_ir: HypergraphIR`` — compiled from the unfolded tree
        - ``training_config: TrainingConfig`` — parsed
        - ``source_kind: str``      — ``"dna"`` or ``"arch"``
        - ``source_path: Path``     — the original input the user passed
  6. The temp folder is cleared on every call (no stale modules from
     a previous run leaking in).
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_workspace(tmp_path, monkeypatch):
    """Run the workspace helper inside a temp dir so .neuro/arch/temp/
    doesn't trample the developer's working tree during tests."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def evol_dna_copy(isolated_workspace):
    """Make a copy of the canonical DNA inside the isolated workspace.

    Fixture name preserved for git-log continuity — the test set used to
    target ``dna/evol/arch.dna`` before the 2026-06-15 cleanup that
    retired ``dna/evol/``. The canonical DNA is now ``dna/master/arch.dna``
    and lives at ``<repo>/dna/master/``."""
    src = REPO_ROOT / "dna" / "master" / "arch.dna"
    if not src.is_file():
        pytest.skip(f"{src} not present — run the recompile first")
    dst_dir = isolated_workspace / "dna" / "master"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "arch.dna"
    dst.write_bytes(src.read_bytes())
    return dst


@pytest.fixture
def evol_arch_copy(isolated_workspace):
    """Make a copy of architectures/master/ inside the isolated workspace.

    Fixture name preserved for git-log continuity — used to target
    ``architectures/evol/`` before the 2026-06-15 cleanup."""
    import shutil
    src = REPO_ROOT / "architectures" / "master"
    if not src.is_dir():
        pytest.skip(f"{src} not present")
    dst = isolated_workspace / "architectures" / "master"
    shutil.copytree(src, dst)
    return dst


# ──────────────────────────────────────────────────────────────────────
# Contract 1 — public API surface
# ──────────────────────────────────────────────────────────────────────


class TestImports:
    def test_prepare_run_workspace_is_importable(self):
        from neuroslm.compiler.run_workspace import (  # noqa: F401
            prepare_run_workspace,
            RunWorkspace,
        )

    def test_default_temp_path_is_under_dot_neuro(self):
        from neuroslm.compiler.run_workspace import DEFAULT_WORKSPACE_DIR
        # We don't compare against an absolute path (it's relative to cwd
        # when the helper is called) but the shape must be ".neuro/arch/temp".
        s = str(DEFAULT_WORKSPACE_DIR).replace("\\", "/")
        assert s.endswith(".neuro/arch/temp"), (
            f"expected DEFAULT_WORKSPACE_DIR to end with '.neuro/arch/temp', "
            f"got {DEFAULT_WORKSPACE_DIR!r}"
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 2 — DNA mode
# ──────────────────────────────────────────────────────────────────────


class TestDnaMode:
    def test_unfolds_dna_to_default_workspace(self, evol_dna_copy):
        from neuroslm.compiler.run_workspace import prepare_run_workspace

        ws = prepare_run_workspace(dna=str(evol_dna_copy))
        # Workspace must contain arch.neuro
        assert ws.arch_neuro.is_file(), (
            f"expected arch.neuro under {ws.arch_root}, none found"
        )
        # Default location is .neuro/arch/temp/ under cwd
        assert ws.arch_root.resolve() == \
            (Path.cwd() / ".neuro" / "arch" / "temp").resolve()
        assert ws.source_kind == "dna"
        assert ws.source_path.resolve() == evol_dna_copy.resolve()

    def test_unfolded_dna_carries_experts_block(self, evol_dna_copy):
        """End-to-end: unfolded arch.neuro must have the MoE roster
        (mirrors test_evol_dna_has_experts.py at the workspace layer)."""
        from neuroslm.compiler.run_workspace import prepare_run_workspace

        ws = prepare_run_workspace(dna=str(evol_dna_copy))
        body = ws.arch_neuro.read_text(encoding="utf-8")
        # Must have an experts: [{...}] block somewhere (the workspace
        # contract is "unfold faithfully"; the roster is the canonical
        # cargo).
        import re
        m = re.search(r"experts:\s*\[\s*\{", body)
        assert m is not None, (
            "unfolded arch.neuro is missing the MoE experts: [{...}] block; "
            "the DNA either is stale or the unfolder dropped the field"
        )

    def test_compiles_hypergraph_ir(self, evol_dna_copy):
        from neuroslm.compiler.run_workspace import prepare_run_workspace
        from neuroslm.compiler.hypergraph_ir import HypergraphIR

        ws = prepare_run_workspace(dna=str(evol_dna_copy))
        assert isinstance(ws.hypergraph_ir, HypergraphIR)
        # The lifted IR must have at least one node (the architecture's
        # populations); zero nodes means the lift saw an empty source.
        assert len(ws.hypergraph_ir.nodes) > 0, (
            "hypergraph IR has zero nodes — workspace unfolded an empty "
            "or unparseable arch.neuro"
        )

    def test_loads_training_config(self, evol_dna_copy):
        from neuroslm.compiler.run_workspace import prepare_run_workspace

        ws = prepare_run_workspace(dna=str(evol_dna_copy))
        # The MoE roster should be loaded.
        assert ws.training_config.multi_cortex.experts is not None
        assert len(ws.training_config.multi_cortex.experts) >= 1


# ──────────────────────────────────────────────────────────────────────
# Contract 3 — ARCH mode
# ──────────────────────────────────────────────────────────────────────


class TestArchMode:
    def test_copies_arch_folder(self, evol_arch_copy):
        from neuroslm.compiler.run_workspace import prepare_run_workspace

        # Pass the real repo as ``repo_root`` so ``@lib/...`` imports
        # resolve against the canonical shared library (the isolated
        # workspace fixture deliberately has no ``pyproject.toml``).
        ws = prepare_run_workspace(
            arch=str(evol_arch_copy),
            repo_root=REPO_ROOT,
        )
        assert ws.arch_neuro.is_file()
        # End-to-end proof the multifile lift succeeded: a non-empty
        # IR means every ``@lib/modules/...`` import resolved against
        # the real repo's ``lib/`` (whether or not there's a local
        # ``modules/`` next to arch.neuro — master uses ``@lib/modules/``
        # exclusively, current/evol may also have a local ``modules/``).
        assert len(ws.hypergraph_ir.nodes) > 0, (
            "workspace's hypergraph IR has zero nodes — arch.neuro's "
            "module imports failed to resolve"
        )
        assert ws.source_kind == "arch"

    def test_resolves_arch_name(self, isolated_workspace):
        """``--arch=evol`` (a bare name) should resolve to
        ``architectures/evol/`` relative to CWD."""
        import shutil
        from neuroslm.compiler.run_workspace import prepare_run_workspace

        # Stage architectures/evol/ in the isolated cwd
        src = REPO_ROOT / "architectures" / "evol"
        if not src.is_dir():
            pytest.skip("evol arch not present")
        shutil.copytree(src, isolated_workspace / "architectures" / "evol")

        ws = prepare_run_workspace(arch="evol", repo_root=REPO_ROOT)
        assert ws.arch_neuro.is_file()

    def test_clears_temp_between_runs(self, evol_arch_copy):
        from neuroslm.compiler.run_workspace import prepare_run_workspace

        ws = prepare_run_workspace(
            arch=str(evol_arch_copy),
            repo_root=REPO_ROOT,
        )
        # Leave a stray file in the temp dir
        stray = ws.arch_root / "stray.txt"
        stray.write_text("should be cleaned next run", encoding="utf-8")
        assert stray.is_file()
        # Second run on same input — stray must be gone
        ws2 = prepare_run_workspace(
            arch=str(evol_arch_copy),
            repo_root=REPO_ROOT,
        )
        assert not stray.is_file(), (
            "prepare_run_workspace must clean the temp dir on each call "
            "to avoid stale modules leaking between runs"
        )
        assert ws2.arch_neuro.is_file()


# ──────────────────────────────────────────────────────────────────────
# Contract 4 — validation
# ──────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_rejects_neither_dna_nor_arch(self):
        from neuroslm.compiler.run_workspace import prepare_run_workspace
        with pytest.raises(ValueError, match=r"(dna|arch)"):
            prepare_run_workspace()

    def test_rejects_both_dna_and_arch(self, evol_dna_copy, evol_arch_copy):
        from neuroslm.compiler.run_workspace import prepare_run_workspace
        with pytest.raises(ValueError, match=r"(both|either|one)"):
            prepare_run_workspace(
                dna=str(evol_dna_copy),
                arch=str(evol_arch_copy),
            )

    def test_rejects_missing_dna(self, isolated_workspace):
        from neuroslm.compiler.run_workspace import prepare_run_workspace
        with pytest.raises(FileNotFoundError):
            prepare_run_workspace(dna="nope/does-not-exist.dna")

    def test_rejects_missing_arch(self, isolated_workspace):
        from neuroslm.compiler.run_workspace import prepare_run_workspace
        with pytest.raises(FileNotFoundError):
            prepare_run_workspace(arch="nope-does-not-exist")
