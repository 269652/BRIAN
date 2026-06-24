# -*- coding: utf-8 -*-
"""TDD: ``brian deploy --dna`` must route through the canonical
``prepare_run_workspace`` pipeline before dispatching to vast.ai.

User-stated invariant (2026-06-12):
    "When I deploy from dna it should first compile it to DSL and use
     the DSL [unpacked] into Hypergraph IR and that should be used by
     the deployment ... it should NOT directly train from dna."

Today there are THREE separate DNA→DSL unfolds in the deploy path:

  1. ``_deploy_train.py`` (local)            — RibosomeCompiler to extract
                                                 ``architecture <name>`` for
                                                 the vast.ai offer search.
  2. ``scripts/vast_train_dna_loop.sh`` (vast box) — RibosomeCompiler again to
                                                       read STEPS/BATCH/SEQ_LEN.
  3. ``neuroslm/train_dsl.py`` (vast box)    — RibosomeCompiler a THIRD time
                                                 inside the training process.

Each one is bespoke and they can drift apart. The canonical
``prepare_run_workspace()`` helper already exists and is used correctly by
``brian train --dna`` — these contracts pin ``brian deploy --dna`` to use
the same single source of truth.

The torch-DLL load failure observed today in ``.venv-2`` is a *symptom* of
the duplication: ``_deploy_train.py`` doesn't need torch at all, but
RibosomeCompiler transitively imports it (``compiler/ribosome.py:17``).
Removing that import fixes the symptom AND the architectural drift in one
step. The .venv-2 routing becomes redundant.

Contracts pinned here
─────────────────────
  A. ``cmd_deploy`` with ``--dna`` MUST call ``prepare_run_workspace()``
     before invoking the deploy subprocess (fail-fast on bad DNA).
  B. If workspace prep fails, deploy returns non-zero BEFORE any vast.ai
     network call (never pay for provisioning when compilation is broken).
  C. After successful prep, the subprocess environment includes
     ``ARCH=<workspace_arch_root>`` so ``_deploy_train.py`` reads the
     canonical workspace, not the raw DNA.
  D. ``_deploy_train.py`` source must not reference ``RibosomeCompiler``
     (the workspace is pre-compiled by ``cmd_deploy``).
  E. ``scripts/vast_train_dna_loop.sh`` must invoke
     ``prepare_run_workspace`` and train via ``--arch .neuro/arch/temp``
     (NOT via ``--dna``), so the on-box pipeline matches local.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


class _FakeHypergraphIR:
    """Stand-in for HypergraphIR — only the bits cli logs on success."""
    def __init__(self):
        self.nodes = []
        self.hyperedges = []


class _FakeWorkspace:
    """Minimal stand-in for RunWorkspace — only fields the deploy
    path needs."""
    def __init__(self, arch_root: Path):
        self.arch_root = arch_root
        self.arch_neuro = arch_root / "arch.neuro"
        self.source_kind = "dna"
        self.source_path = arch_root / "_fake.dna"
        # cli._deploy_dna prints len(workspace.hypergraph_ir.nodes) and
        # len(workspace.hypergraph_ir.hyperedges) for operator telemetry.
        # Match the real RunWorkspace surface so those prints succeed.
        self.hypergraph_ir = _FakeHypergraphIR()
        # The real RunWorkspace also has training_config; we leave it
        # unset so any attempt to read training hyperparameters from
        # the stub (which shouldn't happen in cli.cmd_deploy) surfaces
        # as AttributeError immediately.


@pytest.fixture
def fake_workspace(tmp_path: Path) -> _FakeWorkspace:
    ws = tmp_path / ".neuro" / "arch" / "temp"
    ws.mkdir(parents=True)
    (ws / "arch.neuro").write_text(
        "architecture deploy_test { d_sem: 64 }\n",
        encoding="utf-8",
    )
    return _FakeWorkspace(arch_root=ws)


@pytest.fixture
def capture_subprocess(monkeypatch):
    """Replace ``subprocess.call`` so the test never actually launches
    ``_deploy_train.py``. Returns the list of recorded calls."""
    calls: list[dict[str, Any]] = []

    def _fake_call(args, *, cwd=None, env=None, **kwargs):
        calls.append({"args": list(args), "cwd": cwd, "env": dict(env or {})})
        return 0

    # cli.py uses ``subprocess.call`` via ``import subprocess`` then
    # ``subprocess.call(...)``. Patch the module attribute it actually
    # reads at call time.
    import subprocess
    monkeypatch.setattr(subprocess, "call", _fake_call)
    return calls


# ─────────────────────────────────────────────────────────────────────
# Contract A — deploy compiles DNA locally before any vast.ai call
# ─────────────────────────────────────────────────────────────────────


class TestDeployFromDnaUsesCanonicalPipeline:

    def test_deploy_dna_calls_prepare_run_workspace_locally(
            self, monkeypatch, fake_workspace, capture_subprocess):
        """``_deploy_dna`` must invoke ``prepare_run_workspace(dna=...)``
        before any subprocess dispatch. The DNA path passed by the
        user flows through unchanged."""
        recorded_calls: list[dict[str, Any]] = []

        def _fake_prepare(*, dna=None, arch=None, workspace_dir=None):
            recorded_calls.append({"dna": dna, "arch": arch})
            return fake_workspace

        # Patch at the import site cli.py uses — must match the actual
        # import in the deploy code path. We accept either symbol-import
        # (``from neuroslm.compiler.run_workspace import prepare_run_workspace``)
        # or module-import; patch the source-of-truth module attribute so
        # both forms see the fake.
        import neuroslm.compiler.run_workspace as rw_mod
        monkeypatch.setattr(rw_mod, "prepare_run_workspace", _fake_prepare)

        from neuroslm.cli import _deploy_dna
        rc = _deploy_dna(
            dna_path="dna/evol/arch.dna",
            steps=42,
            branch=None,
            extra_env={},
            ood_every=0,
        )

        assert rc == 0
        assert len(recorded_calls) == 1, (
            f"expected exactly one prepare_run_workspace call, got "
            f"{len(recorded_calls)}: {recorded_calls!r}")
        assert recorded_calls[0]["dna"] == "dna/evol/arch.dna"
        # arch must NOT be passed when dna is — that's the
        # prepare_run_workspace contract (exactly one of dna/arch).
        assert recorded_calls[0]["arch"] is None

    def test_deploy_dna_aborts_when_workspace_prep_fails(
            self, monkeypatch, capture_subprocess):
        """Contract B: if DNA compilation fails (bad DNA, missing modules,
        etc), the deploy MUST abort with non-zero BEFORE any vast.ai
        network call. The user never pays for provisioning when their
        DNA is broken."""

        def _failing_prepare(**_kwargs):
            raise FileNotFoundError("simulated: DNA snapshot missing")

        import neuroslm.compiler.run_workspace as rw_mod
        monkeypatch.setattr(rw_mod, "prepare_run_workspace", _failing_prepare)

        from neuroslm.cli import _deploy_dna
        rc = _deploy_dna(
            dna_path="does/not/exist.dna",
            steps=10,
            branch=None,
            extra_env={},
            ood_every=0,
        )

        assert rc != 0, (
            "deploy must exit non-zero when workspace preparation fails")
        assert len(capture_subprocess) == 0, (
            "no subprocess (and therefore no vast.ai call) may run when "
            f"workspace prep fails — got {len(capture_subprocess)} subprocess "
            f"call(s): {capture_subprocess!r}")

    def test_deploy_dna_passes_workspace_arch_path_to_subprocess(
            self, monkeypatch, fake_workspace, capture_subprocess):
        """Contract C: after successful prep, ``_deploy_train.py`` is
        invoked with ``ARCH=<workspace_arch_root>`` in its environment.
        That's how the deploy script reads the canonical, pre-compiled
        tree instead of the raw DNA."""
        import neuroslm.compiler.run_workspace as rw_mod
        monkeypatch.setattr(
            rw_mod, "prepare_run_workspace",
            lambda **kw: fake_workspace)

        from neuroslm.cli import _deploy_dna
        rc = _deploy_dna(
            dna_path="dna/evol/arch.dna",
            steps=100,
            branch=None,
            extra_env={},
            ood_every=0,
        )

        assert rc == 0
        assert len(capture_subprocess) == 1, (
            f"expected one subprocess call, got {len(capture_subprocess)}")
        env = capture_subprocess[0]["env"]
        arch_env = env.get("ARCH", "")
        # Normalise OS path separators so the test passes on Windows
        # (backslash) and Linux (forward slash) alike.
        assert arch_env, (
            f"ARCH env var must be set to the workspace path, "
            f"got env={env!r}")
        assert arch_env.replace("\\", "/").endswith(
            ".neuro/arch/temp"), (
            f"ARCH must point at the canonical workspace, "
            f"got ARCH={arch_env!r}")


# ─────────────────────────────────────────────────────────────────────
# Contract D — deploy script is torch-free (no RibosomeCompiler)
# ─────────────────────────────────────────────────────────────────────


class TestDeployTrainScriptHasNoRibosomeCompilerImport:
    """``_deploy_train.py`` must not import ``RibosomeCompiler``. With
    the canonical pipeline doing the DNA→DSL→IR compile in
    ``cmd_deploy``, the deploy script only needs the prepared workspace
    path. This eliminates the torch transitive import (RibosomeCompiler
    imports torch at module load) and the .venv-2 routing dance."""

    def test_deploy_train_script_does_not_reference_ribosome(self):
        deploy_src = (REPO_ROOT / "_deploy_train.py").read_text(
            encoding="utf-8")
        # Grep test, not import test — running ``import _deploy_train``
        # would actually execute the vast.ai offer search.
        assert "RibosomeCompiler" not in deploy_src, (
            "_deploy_train.py must not import RibosomeCompiler — "
            "the canonical pipeline (cli.cmd_deploy → "
            "prepare_run_workspace) pre-compiles the workspace, so "
            "the deploy script only reads training_config from the "
            "prepared arch tree (which is torch-free).")
        assert "ribosome" not in deploy_src.lower(), (
            "_deploy_train.py must not reference the ribosome module at "
            "all — that pulls torch transitively and breaks deploy in "
            "any environment without a working torch install.")


# ─────────────────────────────────────────────────────────────────────
# Contract E — vast.ai bash wrapper uses the canonical pipeline too
# ─────────────────────────────────────────────────────────────────────


class TestVastBashWrapperUsesCanonicalPipeline:
    """The vast.ai box must run the SAME pipeline as the local prep.
    Otherwise we're back to two separate DNA-handling code paths and
    the architectural problem returns through the back door."""

    def test_bash_wrapper_invokes_prepare_run_workspace(self):
        bash_src = (REPO_ROOT / "scripts" / "vast_train_dna_loop.sh").read_text(
            encoding="utf-8")
        assert "prepare_run_workspace" in bash_src, (
            "scripts/vast_train_dna_loop.sh must call "
            "prepare_run_workspace on the vast.ai box (typically via "
            "``python -m neuroslm.compiler.run_workspace ...`` or an "
            "inline ``python - <<PY`` heredoc). The on-box ad-hoc "
            "RibosomeCompiler unfold has been retired.")

    def test_bash_wrapper_trains_from_workspace_not_raw_dna(self):
        """After the on-box workspace prep, the actual training
        invocation must read the prepared workspace (``--arch
        .neuro/arch/temp``) — not the raw DNA file (``--dna``)."""
        bash_src = (REPO_ROOT / "scripts" / "vast_train_dna_loop.sh").read_text(
            encoding="utf-8")
        # The canonical workspace path. Bash uses forward slashes on
        # Linux even when paths contain ``.neuro``.
        assert ".neuro/arch/temp" in bash_src, (
            "vast_train_dna_loop.sh must reference the canonical "
            "workspace path .neuro/arch/temp (the destination of "
            "prepare_run_workspace).")
        # The train invocation must use --arch with that path.
        # We look for the substring '--arch' followed somewhere on
        # the same line or nearby by .neuro/arch/temp.
        import re
        match = re.search(
            r"--arch\s+[\"']?[^\s\"']*\.neuro[/\\]arch[/\\]temp",
            bash_src,
        )
        assert match is not None, (
            "vast_train_dna_loop.sh must launch train_dsl with "
            "``--arch .neuro/arch/temp`` (the prepared workspace), "
            "not ``--dna <raw>``. Search of bash source for "
            "'--arch ... .neuro/arch/temp' returned no match.")


