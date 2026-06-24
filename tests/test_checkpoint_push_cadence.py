# -*- coding: utf-8 -*-
"""Regression tests for checkpoint push cadence + config plumbing.

Captures the H24 (run 41031063, 2026-06-15) checkpoint loss: the box
self-destroyed before the ``_deploy_train.py`` end-of-training push,
and because ``neuroslm/train_dsl.py`` (unlike the legacy
``neuroslm/train.py``) had no per-save push, all step-3000 artefacts
died with the instance.

Contract:

  1. ``neuroslm/checkpoint_push.py`` exposes ``push_checkpoint_to_lfs``
     as a pure subprocess wrapper (no torch import) — safe to call
     from the DSL training loop.

  2. ``train_dsl.py`` accepts ``--push_every`` and invokes the push
     after every save whose step is divisible by it. ``--push_every 0``
     disables per-save push (legacy behaviour).

  3. ``ProjectConfig`` exposes ``default_log_every``,
     ``default_save_every``, ``default_push_every`` from
     ``[defaults]`` (defaulting to 20 / 500 / 500 so the H24 push
     gap closes automatically).

  4. ``brian deploy`` propagates those three knobs to
     ``_deploy_train.py`` as ``LOG_EVERY`` / ``SAVE_EVERY`` /
     ``PUSH_EVERY`` env vars so the on-box ``vast_train_dsl_loop.sh``
     can forward them to ``train_dsl.py --push_every``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────
# 1. neuroslm/checkpoint_push.py — standalone push helper
# ─────────────────────────────────────────────────────────────────────


class TestCheckpointPushModule:
    """The push helper must be importable without pulling torch in."""

    def test_module_importable(self):
        # If this raises ImportError the helper hasn't been extracted
        # yet — make_green by creating neuroslm/checkpoint_push.py.
        from neuroslm import checkpoint_push  # noqa: F401

    def test_push_function_exposed(self):
        from neuroslm.checkpoint_push import push_checkpoint_to_lfs
        assert callable(push_checkpoint_to_lfs)

    def test_push_runs_git_add_commit_push(self, tmp_path, monkeypatch):
        """A single call should issue ``git add``, ``git commit``, and
        ``git push`` — the three commands that actually upload the LFS
        object. We patch ``subprocess.run`` and just assert the
        sequence so the test doesn't need a real git repo."""
        from neuroslm import checkpoint_push as cp

        recorded: List[List[str]] = []

        def _fake_run(args, *a, **kw):
            recorded.append(list(args))
            # Mimic the success path for all calls.
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="", stderr=""
            )

        monkeypatch.setattr(subprocess, "run", _fake_run)
        # The function does ``shutil.copy2`` if src != dst; with the
        # ckpt living *inside* lfs_checkpoints already this is a no-op
        # path that doesn't touch disk beyond mkdir.
        ckpt = tmp_path / "lfs_checkpoints" / "run-abc" / "step1000.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"\x00")  # any bytes; subprocess.run is faked

        cp.push_checkpoint_to_lfs(str(ckpt), repo_root=str(tmp_path))

        verbs = [a[1] for a in recorded if len(a) >= 2 and a[0] == "git"]
        assert "add" in verbs, f"expected `git add`, got {verbs}"
        assert "commit" in verbs, f"expected `git commit`, got {verbs}"
        assert "push" in verbs, f"expected `git push`, got {verbs}"


# ─────────────────────────────────────────────────────────────────────
# 2. train_dsl.py — --push_every plumbing
# ─────────────────────────────────────────────────────────────────────


class TestTrainDslPushEveryFlag:
    """``--push_every`` is a real CLI argument and propagates to the
    train loop."""

    def test_push_every_arg_registered(self):
        # Parse the train_dsl arg-parser and verify --push_every exists.
        # We can't easily reach the parser without running argparse on
        # an arg list, so we use the module's _build_parser helper if
        # present, otherwise grep the source.
        src = (REPO_ROOT / "neuroslm" / "train_dsl.py").read_text(
            encoding="utf-8"
        )
        assert '--push_every' in src, (
            "neuroslm/train_dsl.py must declare a --push_every CLI flag "
            "so the on-box vast_train_dsl_loop.sh can forward "
            "$PUSH_EVERY to it. Defaults to 0 (off) to preserve "
            "legacy behaviour for local-dev runs."
        )

    def test_save_site_calls_push_helper(self):
        """The periodic-save block must invoke a push helper. We
        grep for the import / call so the test doesn't need a real
        train run.

        2026-06-15: relaxed to accept either the legacy direct
        ``push_checkpoint_to_lfs`` call OR the new
        :func:`push_checkpoint` dispatcher (see
        ``tests/test_checkpoint_push_hf.py`` which pins the
        dispatcher contract). The dispatcher routes to HF Hub by
        default, with LFS still reachable via
        ``--push_backend lfs`` / ``CHECKPOINT_PUSH_BACKEND=lfs``.
        """
        src = (REPO_ROOT / "neuroslm" / "train_dsl.py").read_text(
            encoding="utf-8"
        )
        assert (
            "push_checkpoint_to_lfs" in src
            or "push_checkpoint" in src
        ), (
            "neuroslm/train_dsl.py must import + call a push helper "
            "(``push_checkpoint`` dispatcher or "
            "``push_checkpoint_to_lfs``) from "
            "neuroslm.checkpoint_push after each periodic save when "
            "--push_every > 0."
        )


# ─────────────────────────────────────────────────────────────────────
# 3. ProjectConfig — log_every / save_every / push_every fields
# ─────────────────────────────────────────────────────────────────────


class TestProjectConfigCadenceFields:
    """``brian.toml [defaults]`` exposes the three cadence knobs."""

    def test_defaults_when_file_missing(self, tmp_path):
        """A folder without ``brian.toml`` returns sensible defaults
        that match the historic pre-H24 behaviour (push every save,
        500-step cadence)."""
        from neuroslm.project_config import load_project_config
        cfg = load_project_config(start=tmp_path)
        # log_every default = 20 (matches LOG_EVERY in
        # vast_train_dsl_loop.sh + train_dsl.py argparse default).
        assert cfg.default_log_every == 20
        # save_every default = 500 — restores the cadence the user
        # remembers from pre-H24 runs (was raised to 1000 silently;
        # H24 lost everything because nothing pushed).
        assert cfg.default_save_every == 500
        # push_every default = 500 — every save pushes by default so
        # an instance crash never strands all checkpoints again.
        assert cfg.default_push_every == 500

    def test_parses_defaults_section(self, tmp_path):
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            '[current]\n'
            'arch = "architectures/master"\n'
            '[defaults]\n'
            'log_every  = 50\n'
            'save_every = 250\n'
            'push_every = 1000\n',
            encoding="utf-8",
        )
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_log_every == 50
        assert cfg.default_save_every == 250
        assert cfg.default_push_every == 1000

    def test_env_var_overrides(self, tmp_path, monkeypatch):
        """``BRIAN_DEFAULT_LOG_EVERY`` etc. override the file."""
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            '[defaults]\n'
            'log_every  = 50\n'
            'save_every = 250\n'
            'push_every = 1000\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("BRIAN_DEFAULT_LOG_EVERY",  "7")
        monkeypatch.setenv("BRIAN_DEFAULT_SAVE_EVERY", "11")
        monkeypatch.setenv("BRIAN_DEFAULT_PUSH_EVERY", "13")
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_log_every == 7
        assert cfg.default_save_every == 11
        assert cfg.default_push_every == 13


# ─────────────────────────────────────────────────────────────────────
# 5. _deploy_train.py + vast_train_dsl_loop.sh — wired through
# ─────────────────────────────────────────────────────────────────────


class TestOnBoxWiring:
    """The downstream box-side scripts must consume the env vars."""

    def test_deploy_train_reads_cadence_env(self):
        """``_deploy_train.py`` must read all three env vars (so they
        get baked into ONSTART)."""
        src = (REPO_ROOT / "_deploy_train.py").read_text(encoding="utf-8")
        assert 'LOG_EVERY' in src, (
            "_deploy_train.py must read LOG_EVERY from the env so "
            "it can ship the value to the box."
        )
        assert 'SAVE_EVERY' in src, (
            "_deploy_train.py must read SAVE_EVERY from the env."
        )
        assert 'PUSH_EVERY' in src, (
            "_deploy_train.py must read PUSH_EVERY from the env."
        )

    def test_vast_loop_forwards_push_every(self):
        """``vast_train_dsl_loop.sh`` must forward PUSH_EVERY to
        ``train_dsl.py --push_every``."""
        src = (REPO_ROOT / "scripts" / "vast_train_dsl_loop.sh").read_text(
            encoding="utf-8"
        )
        assert "PUSH_EVERY" in src, (
            "vast_train_dsl_loop.sh must read PUSH_EVERY env var."
        )
        assert "--push_every" in src, (
            "vast_train_dsl_loop.sh must forward --push_every to "
            "neuroslm.train_dsl."
        )
