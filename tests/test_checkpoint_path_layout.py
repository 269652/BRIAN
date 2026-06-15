# -*- coding: utf-8 -*-
"""Regression tests for checkpoint save/load + per-run directory layout.

Captures three production bugs that all bit H24 (run 41031063,
2026-06-15) silently in the existing test suite:

  * **Bug A — state_dict shape drift.** The harness's
    ``_genetics_orch`` and ``_transmitter_sys`` submodules are built
    *lazily* on the first ``_step_genetics_pre`` call. The existing
    ``test_save_load_round_trip`` (``tests/test_brian_harness.py``)
    never trains, so it never triggers the lazy build, so its saved
    checkpoint never contains those keys. A *real* training run does,
    so resume from a real checkpoint blows up with
    ``Unexpected key(s) in state_dict: "_genetics_orch.*"``.

  * **Bug B — flat checkpoint dir.** Every run dropped its files into
    the same flat ``lfs_checkpoints/`` directory with only the run
    timestamp embedded in the filename
    (``dsl_arch_20260615-081833_step3000.pt``). Two side effects:
    (1) no per-git-commit / per-arch grouping, so you can't tell at a
    glance which trunk produced which checkpoint, and
    (2) ``_maybe_resume`` is forced to glob the whole directory
    looking for *any* compatible checkpoint, which makes a fresh DNA
    revert grab the previous DNA's last checkpoint by accident.

  * **Bug C — resume globber only sees flat layout.** Even with a new
    per-run subdir layout, ``_maybe_resume`` must keep finding the
    old flat-layout checkpoints (back-compat for already-deployed
    runs) AND descend into the new per-run subdirs.

The tests run on CPU with a tiny d_sem so the round-trip is fast.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import torch

from neuroslm.dsl.codegen import CodeGenerator
from neuroslm.dsl.multifile import compile_folder
from neuroslm.dsl.training_config import TrainingConfig
from neuroslm.harness import BRIANHarness


_ARCH_ROOT = Path(__file__).resolve().parent.parent / "architectures" / "master"


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _compiled_circuit_factory():
    """Module-scoped circuit IR (compile is slow); per-test circuits are
    constructed from it so each test gets a *fresh* submodule tree."""
    ir = compile_folder(_ARCH_ROOT)
    Cls = CodeGenerator(ir, module_name="CkptTestCircuit").compile_to_module()
    return lambda: Cls(d_sem=64)


def _fresh_harness(circuit_factory, *, enable_genetics: bool):
    """Build a harness from a fresh circuit + ``TrainingConfig`` clone.

    When ``enable_genetics=True`` we flip the config so the harness's
    ``_step_genetics_pre`` actually builds the orchestrator on first
    ``train_step``.
    """
    cfg = TrainingConfig()
    cfg.genetics.enabled = enable_genetics
    cfg.grad_accum_steps = 1   # one optimizer step per train_step call
    return BRIANHarness(
        circuit=circuit_factory(),
        vocab_size=512,
        d_sem=64,
        training_config=cfg,
    )


def _one_train_step(h: BRIANHarness) -> None:
    """Run a single ``train_step`` so any lazy submodules get built."""
    ids = torch.randint(0, 512, (2, 16))
    targets = torch.randint(0, 512, (2, 16))
    h.train_step(ids, targets)


# ── Bug A — state_dict round-trip after a real train_step ────────────


class TestStateDictRoundTripAfterTraining:
    """If a freshly-built harness can't load a checkpoint produced by a
    trained-then-saved harness of the same config, resume from any
    real production run is broken. H24 (run 41031063, 2026-06-15)
    hit exactly this case for `_genetics_orch.*` and
    `_transmitter_sys.*` keys."""

    def test_round_trip_with_genetics_disabled(
            self, _compiled_circuit_factory, tmp_path):
        # Sanity: when genetics is off, the orchestrator is never built
        # and round-trip is trivial. This is the existing test's regime
        # and stays the safety net for that path.
        h1 = _fresh_harness(_compiled_circuit_factory, enable_genetics=False)
        _one_train_step(h1)

        ckpt = tmp_path / "no_genetics.pt"
        h1.save_checkpoint(str(ckpt), step=7)

        h2 = _fresh_harness(_compiled_circuit_factory, enable_genetics=False)
        step = h2.load_checkpoint(str(ckpt))
        assert step == 7

        sd1, sd2 = h1.state_dict(), h2.state_dict()
        assert set(sd1.keys()) == set(sd2.keys())
        # `_genetics_orch.*` MUST NOT appear when disabled
        assert not any(k.startswith("_genetics_orch.") for k in sd1)

    def test_round_trip_with_genetics_enabled(
            self, _compiled_circuit_factory, tmp_path):
        # THE REGRESSION TEST FOR BUG A.
        # Production runs (rcc_bowtie_30m_p4 preset) enable genetics;
        # the orchestrator is built lazily on first train_step. Without
        # the load-side rebuild, h2.load_checkpoint(...) raises
        # `Unexpected key(s) in state_dict`.
        h1 = _fresh_harness(_compiled_circuit_factory, enable_genetics=True)
        _one_train_step(h1)
        assert h1._genetics_orch is not None, (
            "precondition: training MUST build the orchestrator lazily"
        )

        ckpt = tmp_path / "with_genetics.pt"
        h1.save_checkpoint(str(ckpt), step=42)

        # Verify the saved file actually contains the lazy keys
        # (otherwise our regression target wouldn't exist).
        payload = torch.load(ckpt, weights_only=False, map_location="cpu")
        lazy_keys = [k for k in payload["model"]
                     if k.startswith("_genetics_orch.")
                     or k.startswith("_transmitter_sys.")]
        assert lazy_keys, (
            "precondition: save_checkpoint MUST include lazily-built "
            "submodules — without these in the file, there's nothing "
            "for load_checkpoint to rebuild on resume."
        )

        # The actual regression: a fresh harness must absorb those keys
        # without hand-holding from the caller.
        h2 = _fresh_harness(_compiled_circuit_factory, enable_genetics=True)
        assert h2._genetics_orch is None, (
            "precondition: a freshly-built harness has NOT yet trained, "
            "so the orchestrator is still None — this is the gap that "
            "load_checkpoint must bridge."
        )

        step = h2.load_checkpoint(str(ckpt))   # MUST NOT raise
        assert step == 42

        # After load, the lazy submodules MUST exist with matching
        # parameters (parameter-by-parameter, not just key-by-key).
        assert h2._genetics_orch is not None
        assert h2._transmitter_sys is not None
        sd1 = h1.state_dict()
        sd2 = h2.state_dict()
        assert set(sd1.keys()) == set(sd2.keys())
        for k in lazy_keys:
            assert torch.allclose(sd1[k], sd2[k]), (
                f"lazy submodule key {k} did not round-trip"
            )


# ── Bug B — per-run subdirectory layout ──────────────────────────────


class TestCheckpointDirLayout:
    """The new layout must:

      * group every artefact of a single run under one subdirectory,
      * name that subdirectory deterministically from
        ``<UTC_RUN_ID>_<GIT_SHORT>_<ARCH_LABEL>`` so
        - per-day grouping works (lexicographic sort by prefix),
        - per-git-commit traceability is baked in,
        - per-arch experiments don't collide on the same day,
      * leave the legacy flat filename emitter exposed for back-compat
        tests that still target ``lfs_checkpoints/dsl_arch_*.pt``.
    """

    def test_run_dir_name_has_all_three_components(self):
        from neuroslm.train_dsl import build_run_dir_name
        name = build_run_dir_name(
            run_id="20260615-081833",
            git_short="6b30c8a",
            arch_label="h24-cfd-3k",
        )
        # All three components must appear, separated by underscores
        # (the same separator log_pusher.sh uses).
        assert "20260615-081833" in name
        assert "6b30c8a" in name
        assert "h24-cfd-3k" in name
        # Run-id prefix MUST come first so `ls lfs_checkpoints/` sorts
        # chronologically (mirrors logs/vast/ sort semantics).
        assert name.startswith("20260615-081833")

    def test_run_dir_name_is_filesystem_safe(self):
        from neuroslm.train_dsl import build_run_dir_name
        # Real-world labels can contain slashes / spaces (the deploy
        # label generator strips most, but the contract has to be
        # defensive). Slashes MUST be sanitised.
        name = build_run_dir_name(
            run_id="20260615-081833",
            git_short="6b30c8a",
            arch_label="some/weird label",
        )
        assert "/" not in name
        assert "\\" not in name

    def test_save_checkpoint_writes_into_run_subdir(
            self, _compiled_circuit_factory, tmp_path,
            monkeypatch):
        # The save_checkpoint emitter MUST drop into
        # ``<ckpt_dir>/<run_dir>/step<N>.pt`` when given the new layout.
        from neuroslm.train_dsl import (
            build_run_dir_name, checkpoint_path_for_step,
        )
        ckpt_root = tmp_path / "lfs_checkpoints"
        ckpt_root.mkdir()

        run_dir_name = build_run_dir_name(
            run_id="20260615-081833",
            git_short="6b30c8a",
            arch_label="h24-cfd-3k",
        )
        out_path = checkpoint_path_for_step(
            ckpt_root, run_dir_name, step=1000)

        # Parent dir is the run subdir; the file itself is `step<N>.pt`.
        assert out_path.parent == ckpt_root / run_dir_name
        assert out_path.name == "step1000.pt"

        # Save actually works there (mkdir-p semantics).
        h = _fresh_harness(_compiled_circuit_factory, enable_genetics=False)
        h.save_checkpoint(str(out_path), step=1000)
        assert out_path.exists()


# ── Bug C — _maybe_resume must traverse both layouts ─────────────────


class TestMaybeResumeDiscovery:
    """``_maybe_resume`` must find the highest-step checkpoint across:

      * legacy flat layout
        ``lfs_checkpoints/dsl_arch_<RUN_ID>_step<N>.pt`` (H21–H23 runs)
      * legacy stamp-less layout
        ``lfs_checkpoints/dsl_arch_step<N>.pt`` (very old runs)
      * **new per-run subdir layout**
        ``lfs_checkpoints/<RUN_DIR>/step<N>.pt`` (H24+)
    """

    def _write_dummy_ckpt(self, path: Path, step: int) -> None:
        """Write a minimal ``.pt`` that ``_checkpoint_step`` can parse
        without ``load_checkpoint`` ever needing to deserialise it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"step": step, "model": {}}, path)

    def test_flat_legacy_layout_still_discovered(
            self, tmp_path):
        from neuroslm.train_dsl import _find_resume_candidates
        ckpt_dir = tmp_path / "lfs_checkpoints"
        ckpt_dir.mkdir()
        self._write_dummy_ckpt(
            ckpt_dir / "dsl_arch_20260101-000000_step100.pt", 100)
        self._write_dummy_ckpt(
            ckpt_dir / "dsl_arch_step50.pt", 50)

        candidates = _find_resume_candidates(ckpt_dir)
        steps = sorted(s for s, _ in candidates)
        assert steps == [50, 100]

    def test_per_run_subdir_layout_discovered(self, tmp_path):
        from neuroslm.train_dsl import _find_resume_candidates
        ckpt_dir = tmp_path / "lfs_checkpoints"
        run_dir = ckpt_dir / "20260615-081833_6b30c8a_h24-cfd-3k"
        self._write_dummy_ckpt(run_dir / "step1000.pt", 1000)
        self._write_dummy_ckpt(run_dir / "step2000.pt", 2000)

        candidates = _find_resume_candidates(ckpt_dir)
        steps = sorted(s for s, _ in candidates)
        assert steps == [1000, 2000]

    def test_mixed_layouts_pick_highest_step(self, tmp_path):
        # A repo mid-migration has BOTH layouts side-by-side. Resume
        # must always grab the truly highest step regardless of where
        # it lives.
        from neuroslm.train_dsl import _find_resume_candidates
        ckpt_dir = tmp_path / "lfs_checkpoints"
        self._write_dummy_ckpt(
            ckpt_dir / "dsl_arch_20260101-000000_step100.pt", 100)
        run_dir = ckpt_dir / "20260615-081833_6b30c8a_h24-cfd-3k"
        self._write_dummy_ckpt(run_dir / "step2500.pt", 2500)

        candidates = _find_resume_candidates(ckpt_dir)
        # Highest step wins
        top_step, top_path = max(candidates, key=lambda x: x[0])
        assert top_step == 2500
        assert top_path.name == "step2500.pt"
        # And it came from the per-run subdir
        assert top_path.parent == run_dir


# ── Bug D — _deploy_train.py checkpoint push glob ────────────────────


class TestDeployPushGlob:
    """The ``_deploy_train.py`` post-training push step iterates
    ``lfs_checkpoints/dsl_arch_*.pt``. Under the new per-run subdir
    layout this glob misses every new checkpoint and the run's
    artefacts are stranded on the (about-to-self-destroy) vast box.

    The push glob MUST be a recursive ``**/*.pt`` pattern so both
    layouts are covered.
    """

    def test_push_glob_pattern_is_recursive(self):
        # Static-text assertion: the deploy script must contain a
        # recursive glob (or an equivalent shell expansion) so future
        # changes don't silently regress to the flat pattern.
        deploy_script = (
            Path(__file__).resolve().parent.parent / "_deploy_train.py"
        ).read_text(encoding="utf-8")
        # Must mention `lfs_checkpoints` and either `**` (glob recursion)
        # or `find` (shell traversal). The flat `dsl_arch_*.pt` pattern
        # alone is no longer sufficient.
        assert "lfs_checkpoints" in deploy_script
        # Accept either: bash `find lfs_checkpoints -name '*.pt'` OR
        # bash globstar `lfs_checkpoints/**/*.pt`. Reject the
        # legacy flat-only `lfs_checkpoints/dsl_arch_*.pt` if that's
        # the ONLY thing present.
        has_recursive = (
            re.search(r"lfs_checkpoints/\*\*/\*\.pt", deploy_script)
            is not None
            or re.search(
                r"find\s+lfs_checkpoints.*-name\s+['\"]?\*\.pt",
                deploy_script,
            ) is not None
        )
        assert has_recursive, (
            "_deploy_train.py push glob is flat-only — new per-run "
            "subdir checkpoints will never be pushed. Switch to "
            "`find lfs_checkpoints -name '*.pt'` or "
            "`lfs_checkpoints/**/*.pt` with `shopt -s globstar`."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
