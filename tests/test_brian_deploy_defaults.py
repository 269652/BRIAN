# -*- coding: utf-8 -*-
"""TDD: ``brian dna compile`` consumes ``brian.toml``.

Two foot-guns this contract closes:

2. **DNA two-file split.** ``brian dna compile master`` writes to
   ``architectures/master/evolution.dna`` by default — but the
   deploy reads ``brian.toml [current].dna`` which points at
   ``./dna/evol/arch.dna``. These can (and on 2026-06-14, did)
   diverge: a stale ``dna/evol/arch.dna`` shipped a wasted-compute
   gpt2 deploy after the arch had been swapped to ``smollm2_360m``
   and the wrong DNA file was recompiled. The fix: ``brian dna
   compile`` with NO positional arg reads ``[current].arch`` and
   writes directly to ``[current].dna``, making the canonical
   one-command refresh foolproof.

Contracts pinned here
─────────────────────
  F. ``brian dna compile`` with no positional arg reads
     ``cfg.arch`` and compiles it.
  G. ``brian dna compile`` (no positional, no ``--output``) writes to
     ``cfg.dna`` (the deploy-targeted path) — closing the two-file
     split. When ``cfg.dna`` is empty, fall back to
     ``architectures/<arch>/evolution.dna`` so the command still
     succeeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_brian_toml(tmp_path: Path, monkeypatch):
    """Pin ``load_project_config`` to a tmp ``brian.toml`` for the
    duration of the test, regardless of where pytest was invoked
    from. Returns a callable ``write(toml_str)`` that drops a
    ``brian.toml`` into ``tmp_path`` and re-runs the loader."""
    # Ensure no stale env vars leak in. ``BRANCH`` / ``STEPS`` /
    # ``ARCH`` are the runtime ones consumed by ``_deploy_train.py``;
    # if a dev shell exports any of them (CI box, local tinkering),
    # ``os.environ.copy()`` inside ``_deploy_dsl`` / ``_deploy_dna``
    # would inherit them and silently mask the brian.toml fallback
    # the test is trying to verify.
    for k in (
        "BRIAN_ARCH", "BRIAN_DNA",
        "BRIAN_DEFAULT_STEPS", "BRIAN_DEFAULT_PRESET",
        "BRIAN_DEFAULT_HARDWARE", "BRIAN_DEFAULT_BRANCH",
        "BRANCH", "STEPS", "ARCH", "BRIAN_SOURCE_DNA",
    ):
        monkeypatch.delenv(k, raising=False)

    # Make load_project_config (called WITHOUT start=) discover our
    # tmp brian.toml by patching the module-level default start dir.
    import neuroslm.project_config as pc
    monkeypatch.setattr(pc, "_default_start_dir", lambda: tmp_path)

    def write(toml: str) -> None:
        (tmp_path / "brian.toml").write_text(toml, encoding="utf-8")

    return write


# ─────────────────────────────────────────────────────────────────────
# F + G: brian dna compile reads brian.toml
# ─────────────────────────────────────────────────────────────────────


class TestDnaCompileFromBrianToml:
    """``brian dna compile`` (no positional, no --output) reads
    ``[current].arch`` AND writes to ``[current].dna``. This closes
    the two-file split that caused the 2026-06-14 wasted-compute
    deploy 40951692."""

    def _build_args(self, **overrides) -> argparse.Namespace:
        ns = argparse.Namespace(
            dna_cmd="compile",
            arch=None,         # no positional
            output=None,       # no --output
            dna=None,
        )
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    def _make_minimal_arch(self, arch_dir: Path) -> None:
        """Drop a tiny but compilable arch.neuro into ``arch_dir``."""
        arch_dir.mkdir(parents=True, exist_ok=True)
        (arch_dir / "arch.neuro").write_text(
            "architecture default_test {\n"
            "    d_sem: 64,\n"
            "    dt: 0.01\n"
            "}\n",
            encoding="utf-8",
        )

    def test_no_args_uses_current_arch_and_writes_to_current_dna(
        self, tmp_path, isolated_brian_toml, monkeypatch
    ):
        """``brian dna compile`` (no args) compiles ``[current].arch``
        and writes to ``[current].dna``."""
        # Set up an arch under tmp_path with a real (compilable) DSL.
        arch_dir = tmp_path / "architectures" / "test_default_arch"
        self._make_minimal_arch(arch_dir)
        # Target DNA path lives under tmp_path/dna/ so the test is
        # hermetic.
        target_dna = tmp_path / "dna" / "evol" / "arch.dna"
        isolated_brian_toml(
            '[current]\n'
            f'arch = "{arch_dir.relative_to(tmp_path).as_posix()}"\n'
            f'dna  = "{target_dna.relative_to(tmp_path).as_posix()}"\n'
        )
        # cli.cmd_dna calls _resolve_arch — patch it to return the
        # absolute path we just created, regardless of repo root.
        import neuroslm.cli as cli_mod
        monkeypatch.setattr(
            cli_mod, "_resolve_arch",
            lambda name: str((tmp_path / name).resolve()),
        )
        # Use a fake RibosomeCompiler that just writes a marker so the
        # test doesn't depend on the full compiler pipeline.
        from neuroslm.compiler import ribosome as ribo_mod

        class _StubCompiler:
            def compile_file(self, src, dst):
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                Path(dst).write_text(
                    "STUB DNA from " + str(src), encoding="utf-8"
                )
        monkeypatch.setattr(ribo_mod, "RibosomeCompiler", _StubCompiler)

        from neuroslm.cli import cmd_dna
        rc = cmd_dna(self._build_args())
        assert rc == 0, (
            "brian dna compile (no positional, no --output) should "
            "succeed when brian.toml [current].arch is set"
        )
        assert target_dna.is_file(), (
            f"DNA must be written to brian.toml [current].dna "
            f"({target_dna}), but the file does not exist. The compile "
            f"either wrote to the wrong path or failed silently."
        )
        contents = target_dna.read_text(encoding="utf-8")
        assert "STUB DNA from" in contents
        assert "test_default_arch" in contents, (
            "DNA must be compiled FROM the brian.toml-pointed arch, "
            f"got: {contents[:200]!r}"
        )

    def test_no_args_falls_back_to_arch_evolution_when_current_dna_empty(
        self, tmp_path, isolated_brian_toml, monkeypatch
    ):
        """When ``[current].dna = ""`` AND no ``--output``, fall
        back to ``architectures/<arch>/evolution.dna`` so the command
        still has a deterministic target. Don't refuse to run."""
        arch_dir = tmp_path / "architectures" / "test_no_dna_arch"
        self._make_minimal_arch(arch_dir)
        isolated_brian_toml(
            '[current]\n'
            f'arch = "{arch_dir.relative_to(tmp_path).as_posix()}"\n'
            'dna  = ""\n'
        )
        import neuroslm.cli as cli_mod
        monkeypatch.setattr(
            cli_mod, "_resolve_arch",
            lambda name: str((tmp_path / name).resolve()),
        )
        from neuroslm.compiler import ribosome as ribo_mod

        class _StubCompiler:
            def compile_file(self, src, dst):
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                Path(dst).write_text("stub", encoding="utf-8")
        monkeypatch.setattr(ribo_mod, "RibosomeCompiler", _StubCompiler)

        from neuroslm.cli import cmd_dna
        rc = cmd_dna(self._build_args())
        assert rc == 0
        fallback = arch_dir / "evolution.dna"
        assert fallback.is_file(), (
            f"When [current].dna is empty, fall back to "
            f"architectures/<arch>/evolution.dna, got nothing at {fallback}"
        )

    def test_positional_arch_preserves_legacy_behaviour(
        self, tmp_path, isolated_brian_toml, monkeypatch
    ):
        """Regression guard: ``brian dna compile <arch>`` (positional,
        no --output) still writes to ``architectures/<arch>/evolution.dna``,
        NOT to ``[current].dna``. This keeps the existing API for the
        "compile some other arch, leave deploy alone" workflow."""
        # brian.toml points at arch A with a custom dna target.
        arch_a = tmp_path / "architectures" / "arch_a"
        arch_b = tmp_path / "architectures" / "arch_b"
        self._make_minimal_arch(arch_a)
        self._make_minimal_arch(arch_b)
        current_dna = tmp_path / "dna" / "current.dna"
        isolated_brian_toml(
            '[current]\n'
            f'arch = "{arch_a.relative_to(tmp_path).as_posix()}"\n'
            f'dna  = "{current_dna.relative_to(tmp_path).as_posix()}"\n'
        )
        import neuroslm.cli as cli_mod
        monkeypatch.setattr(
            cli_mod, "_resolve_arch",
            lambda name: str((tmp_path / "architectures" / name).resolve()),
        )
        from neuroslm.compiler import ribosome as ribo_mod

        class _StubCompiler:
            def compile_file(self, src, dst):
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                Path(dst).write_text("stub from " + str(src), encoding="utf-8")
        monkeypatch.setattr(ribo_mod, "RibosomeCompiler", _StubCompiler)

        # Compile arch_b (the OTHER arch, not the current one).
        from neuroslm.cli import cmd_dna
        rc = cmd_dna(self._build_args(arch="arch_b"))
        assert rc == 0
        # Legacy default: <arch>/evolution.dna
        assert (arch_b / "evolution.dna").is_file(), (
            "Positional arch must write to architectures/<arch>/evolution.dna"
        )
        # Crucial: [current].dna MUST NOT be touched when compiling
        # a non-current arch. Otherwise users invoking the legacy
        # form would silently retarget their deploy.
        assert not current_dna.exists(), (
            f"Positional arch must NOT touch [current].dna "
            f"({current_dna}). The legacy form is for compiling "
            f"arbitrary arches without retargeting the deploy."
        )
