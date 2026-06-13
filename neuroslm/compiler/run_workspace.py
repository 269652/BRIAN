# -*- coding: utf-8 -*-
"""Run workspace: unpack a DNA snapshot or architecture folder into a
single, predictable location so every downstream consumer (harness,
NFG renderer, hypergraph IR, evolution heat-overlay) reads from the
same tree.

The canonical workspace is ``./.neuro/arch/temp/``. It holds:

    .neuro/arch/temp/
        arch.neuro              # top-level DSL
        modules/                # per-region modules
        lib/                    # shared mechanics

This module is intentionally small and side-effect-free except for the
single ``prepare_run_workspace`` entry-point that writes the temp tree
and returns a :class:`RunWorkspace` describing it. The contract is
pinned by ``tests/compiler/test_run_workspace.py`` — never weaken any
of those guarantees without first updating the test.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from neuroslm.compiler.hypergraph_ir import HypergraphIR
    from neuroslm.dsl.training_config import TrainingConfig


# Default workspace location, relative to whatever directory the helper
# is called from. We use a string-style relative path so the workspace
# stays portable across machines.
DEFAULT_WORKSPACE_DIR: Path = Path(".neuro") / "arch" / "temp"


__all__ = [
    "DEFAULT_WORKSPACE_DIR",
    "RunWorkspace",
    "prepare_run_workspace",
]


@dataclass(frozen=True)
class RunWorkspace:
    """Description of a prepared run workspace.

    Returned from :func:`prepare_run_workspace`. Holds:

      ``arch_root``         — the workspace directory containing the
                              unfolded arch.neuro and its modules.
      ``arch_neuro``        — path to the top-level ``arch.neuro``.
      ``hypergraph_ir``     — :class:`HypergraphIR` lifted from the
                              unfolded multi-file tree (single source
                              of truth for NFG + evolution overlays).
      ``training_config``   — parsed :class:`TrainingConfig` from the
                              same source.
      ``source_kind``       — ``"dna"`` or ``"arch"``.
      ``source_path``       — the exact input the user passed
                              (``.dna`` file, arch folder, or .neuro file).
    """
    arch_root: Path
    arch_neuro: Path
    hypergraph_ir: "HypergraphIR"
    training_config: "TrainingConfig"
    source_kind: str
    source_path: Path


def prepare_run_workspace(
    *,
    dna: Optional[str] = None,
    arch: Optional[str] = None,
    workspace_dir: Optional[Path] = None,
) -> RunWorkspace:
    """Unpack a DNA snapshot or architecture folder into ``.neuro/arch/temp/``.

    Exactly one of ``dna`` or ``arch`` must be provided.

    Parameters
    ----------
    dna : str, optional
        Path to a ``.dna`` snapshot (typically
        ``dna/evol/arch.dna``). The DNA is unfolded with
        :class:`~neuroslm.compiler.ribosome.RibosomeCompiler.unfold_file`
        — preserving the full modular tree (modules/, lib/) — so the
        run sees an identical layout to the source architecture.
    arch : str, optional
        Either:
          * An architecture folder containing ``arch.neuro``
            (e.g. ``architectures/evol``)
          * A path to an ``arch.neuro`` file (we use its parent)
          * A bare architecture name (``evol``) which resolves to
            ``./architectures/<name>/`` relative to the current
            working directory.
    workspace_dir : Path, optional
        Override the default ``.neuro/arch/temp`` location. Mostly
        useful for tests; production callers should leave this unset.

    Returns
    -------
    RunWorkspace
        Describes the prepared workspace; see the dataclass docstring.

    Raises
    ------
    ValueError
        Neither or both of ``dna`` / ``arch`` were provided.
    FileNotFoundError
        The requested DNA file or architecture path doesn't exist.
    """
    # ── argument validation ────────────────────────────────────────
    if dna is None and arch is None:
        raise ValueError(
            "prepare_run_workspace requires exactly one of `dna` or `arch`"
        )
    if dna is not None and arch is not None:
        raise ValueError(
            "prepare_run_workspace accepts exactly one of `dna` or `arch`, "
            "not both"
        )

    target = Path(workspace_dir) if workspace_dir else DEFAULT_WORKSPACE_DIR
    target = target.resolve()

    # ── clear stale temp dir from previous runs ───────────────────
    # This is intentional: leftover module files would silently
    # contribute to the lifted HypergraphIR even after they were
    # removed from the source — exactly the kind of stale-cache bug
    # that masks regressions.
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=False)

    # ── unpack ─────────────────────────────────────────────────────
    if dna is not None:
        source_path = Path(dna).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"DNA file not found: {source_path}")
        source_kind = "dna"
        _unpack_dna(source_path, target)
    else:
        assert arch is not None  # narrowed by the early validation above
        source_path = _resolve_arch_argument(arch)
        source_kind = "arch"
        _copy_arch_tree(source_path, target)

    arch_neuro = target / "arch.neuro"
    if not arch_neuro.is_file():
        # The unpack/copy step should always produce this; defensive
        # error keeps the failure mode obvious if something upstream
        # silently drops the file.
        raise FileNotFoundError(
            f"workspace preparation finished but {arch_neuro} is missing"
        )

    # ── lift to HypergraphIR (single source of truth) ─────────────
    # Lazy imports keep the top of this module cheap to load — the
    # helper is used in a CLI hot path.
    from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
    from neuroslm.dsl.training_config import load_training_config_from_arch

    hypergraph_ir = lift_arch_to_hypergraph(target)
    training_config = load_training_config_from_arch(target)

    return RunWorkspace(
        arch_root=target,
        arch_neuro=arch_neuro,
        hypergraph_ir=hypergraph_ir,
        training_config=training_config,
        source_kind=source_kind,
        source_path=source_path,
    )


# ──────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────


def _resolve_arch_argument(arch: str) -> Path:
    """Resolve the user-supplied ``arch`` to a concrete directory.

    Accepts:
      * Existing directory containing ``arch.neuro``.
      * Existing ``.neuro`` file (we return its parent).
      * Bare name like ``evol`` (resolved to ``./architectures/evol/``).
    """
    p = Path(arch)
    # Case 1: explicit directory
    if p.is_dir():
        if not (p / "arch.neuro").is_file():
            raise FileNotFoundError(
                f"directory {p} has no arch.neuro"
            )
        return p.resolve()
    # Case 2: explicit .neuro file (use its parent)
    if p.is_file() and p.suffix == ".neuro":
        return p.parent.resolve()
    # Case 3: bare name — look under ./architectures/<name>/
    if "/" not in arch and "\\" not in arch and not p.suffix:
        candidate = Path("architectures") / arch
        if candidate.is_dir() and (candidate / "arch.neuro").is_file():
            return candidate.resolve()
        raise FileNotFoundError(
            f"architecture {arch!r} not found at {candidate.resolve()}"
        )
    raise FileNotFoundError(f"arch path not found: {p.resolve()}")


def _unpack_dna(dna_path: Path, target: Path) -> None:
    """Unfold ``dna_path`` into ``target`` preserving the modular tree.

    Uses :meth:`RibosomeCompiler.unfold_file`, which already knows how to
    write back ``modules/*.neuro`` and ``lib/*.neuro`` alongside the
    top-level ``arch.neuro``.
    """
    from neuroslm.compiler.ribosome import RibosomeCompiler

    out_arch_neuro = target / "arch.neuro"
    RibosomeCompiler().unfold_file(str(dna_path), str(out_arch_neuro))


def _copy_arch_tree(src_root: Path, target: Path) -> None:
    """Mirror ``src_root`` into ``target``, copying only the files the
    DSL multifile resolver needs (arch.neuro + modules/ + lib/).

    Top-level ``fitness.json`` / ``fitness.neuro`` are copied too so
    fitness gating still works.
    """
    # arch.neuro is mandatory; resolver fails without it
    shutil.copyfile(src_root / "arch.neuro", target / "arch.neuro")

    for sub in ("modules", "lib"):
        sub_src = src_root / sub
        if sub_src.is_dir():
            shutil.copytree(sub_src, target / sub)

    # Optional sidecars — copy if present, skip silently otherwise.
    for name in ("fitness.json", "fitness.neuro"):
        f = src_root / name
        if f.is_file():
            shutil.copyfile(f, target / name)
