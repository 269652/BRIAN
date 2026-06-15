# -*- coding: utf-8 -*-
"""HuggingFace Hub checkpoint push — contracts.

Captures the 2026-06-15 switch from Git LFS to HuggingFace Hub for
periodic checkpoint upload. Surfaced by run 41063959 which hung at
exactly step 500 — the LFS push raced the background ``log_pusher.sh``
and blocked the training loop on a rebase + 569 MB git push that never
returned within the 600 s timeout. HF Hub has a sane upload API,
generous bandwidth, no rebase race, and ``upload_file`` is a single
sync HTTP call.

Contract:

  1. ``neuroslm.checkpoint_push.push_checkpoint_to_hf`` exists, is a
     thin wrapper over ``huggingface_hub.upload_file``, and inherits
     the legacy "never raise" contract (errors are printed + swallowed).

  2. The push **preserves the per-run subdir layout** as
     ``path_in_repo``::

         lfs_checkpoints/<RUN_DIR>/step<N>.pt
            ▼
         checkpoints/<RUN_DIR>/step<N>.pt   (on the HF repo)

     so the HF repo can hold multiple concurrent runs without name
     collisions, mirroring the on-disk layout.

  3. The ``.mem`` sidecar is pushed alongside the ``.pt`` when present.

  4. Authentication chain:
     - explicit ``token`` arg              → wins
     - ``HF_TOKEN`` env var                → second
     - cached token from ``huggingface_hub.HfFolder``  → third
     - none of the above → skip with a clear message, don't crash.

  5. ``neuroslm.checkpoint_push.push_checkpoint`` is a backend
     dispatcher with HF as the default. ``backend="lfs"`` falls back
     to the legacy Git LFS code path; ``backend="none"`` is a no-op.
     The ``CHECKPOINT_PUSH_BACKEND`` env var overrides the default.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def ckpt(tmp_path: Path) -> Path:
    """A fake .pt file inside a per-run subdir, matching the H24+
    on-box layout ``lfs_checkpoints/<RUN_DIR>/step<N>.pt``."""
    p = tmp_path / "lfs_checkpoints" / "run-20260615_abc1234_arch" / "step500.pt"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"\x00" * 128)
    return p


@pytest.fixture
def fake_hf(monkeypatch):
    """Inject a fake ``huggingface_hub`` module that records every
    ``upload_file`` call without hitting the network.

    Returns the list of recorded call-kwargs so individual tests can
    assert against them.
    """
    calls: List[Dict[str, Any]] = []

    def _fake_upload_file(**kwargs):
        calls.append(dict(kwargs))
        # huggingface_hub returns a CommitInfo; we return a string
        # placeholder — push_checkpoint_to_hf must not depend on the
        # return value beyond "did not raise".
        return f"https://huggingface.co/{kwargs.get('repo_id')}/blob/main/{kwargs.get('path_in_repo')}"

    class _FakeHfFolder:
        _cached: str | None = None

        @classmethod
        def get_token(cls) -> str | None:
            return cls._cached

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.upload_file = _fake_upload_file      # type: ignore[attr-defined]
    fake_module.HfFolder = _FakeHfFolder              # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    # Also clear HF_TOKEN by default; individual tests opt-in.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.delenv("CHECKPOINT_PUSH_BACKEND", raising=False)
    return calls, _FakeHfFolder


# ─────────────────────────────────────────────────────────────────────
# 1. push_checkpoint_to_hf — single-file HF upload
# ─────────────────────────────────────────────────────────────────────


class TestPushCheckpointToHF:
    """``push_checkpoint_to_hf`` is the new canonical periodic push."""

    def test_function_exists(self):
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        assert callable(push_checkpoint_to_hf)

    def test_calls_upload_file_with_correct_repo_id_and_path(
            self, ckpt, fake_hf, monkeypatch):
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        calls, _ = fake_hf
        monkeypatch.setenv("HF_TOKEN", "hf_dummy_token_for_test")

        push_checkpoint_to_hf(
            str(ckpt), repo_id="moritzroessler/BRIAN",
            repo_root=str(ckpt.parent.parent.parent),
        )

        assert len(calls) >= 1, (
            f"expected at least 1 upload_file call, got {len(calls)}; "
            f"the function may have errored silently — check stdout"
        )
        # Find the .pt upload (the test creates only the .pt, no .mem).
        pt_calls = [c for c in calls if c["path_in_repo"].endswith(".pt")]
        assert len(pt_calls) == 1, (
            f"expected exactly one .pt upload, got {len(pt_calls)} "
            f"({[c['path_in_repo'] for c in calls]})"
        )
        call = pt_calls[0]
        assert call["repo_id"] == "moritzroessler/BRIAN"
        assert call["repo_type"] == "model"
        assert call["path_or_fileobj"] == str(ckpt)

    def test_path_in_repo_preserves_run_subdir(
            self, ckpt, fake_hf, monkeypatch):
        """The per-run subdir name + step file name must survive the
        round-trip so the HF repo mirrors the on-disk layout."""
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        calls, _ = fake_hf
        monkeypatch.setenv("HF_TOKEN", "hf_dummy_token")

        push_checkpoint_to_hf(
            str(ckpt), repo_id="moritzroessler/BRIAN",
            repo_root=str(ckpt.parent.parent.parent),
        )

        # Expected path_in_repo: checkpoints/<RUN_DIR>/step500.pt
        # (we don't pin the "checkpoints/" prefix exactly — accept any
        # leading directory — but we DO pin the RUN_DIR/stepN.pt tail
        # because that's what makes per-run isolation work on HF.)
        pt_call = [c for c in calls if c["path_in_repo"].endswith(".pt")][0]
        path_in_repo = pt_call["path_in_repo"]
        assert "run-20260615_abc1234_arch/step500.pt" in path_in_repo, (
            f"path_in_repo {path_in_repo!r} must preserve the "
            f"per-run subdir + step file; otherwise concurrent runs "
            f"collide on the HF repo."
        )

    def test_uploads_mem_sidecar_when_present(
            self, ckpt, fake_hf, monkeypatch):
        """The .mem sidecar (genetics overlay state) must travel with
        the .pt to keep the checkpoint loadable."""
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        calls, _ = fake_hf
        monkeypatch.setenv("HF_TOKEN", "hf_dummy_token")

        mem_path = ckpt.with_suffix(".mem")
        mem_path.write_bytes(b"\x01" * 64)

        push_checkpoint_to_hf(
            str(ckpt), repo_id="moritzroessler/BRIAN",
            repo_root=str(ckpt.parent.parent.parent),
        )

        suffixes = {Path(c["path_in_repo"]).suffix for c in calls}
        assert ".pt"  in suffixes, f"missing .pt upload; got {suffixes}"
        assert ".mem" in suffixes, f"missing .mem sidecar upload; got {suffixes}"

    def test_explicit_token_arg_wins_over_env(
            self, ckpt, fake_hf, monkeypatch):
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        calls, _ = fake_hf
        monkeypatch.setenv("HF_TOKEN", "env_token_should_lose")

        push_checkpoint_to_hf(
            str(ckpt), repo_id="moritzroessler/BRIAN",
            token="explicit_arg_token",
            repo_root=str(ckpt.parent.parent.parent),
        )

        assert calls, "no upload happened"
        assert calls[0]["token"] == "explicit_arg_token"

    def test_HF_TOKEN_env_var_used_when_no_arg(
            self, ckpt, fake_hf, monkeypatch):
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        calls, _ = fake_hf
        monkeypatch.setenv("HF_TOKEN", "env_token_wins")

        push_checkpoint_to_hf(
            str(ckpt), repo_id="moritzroessler/BRIAN",
            repo_root=str(ckpt.parent.parent.parent),
        )

        assert calls, "no upload happened despite HF_TOKEN being set"
        assert calls[0]["token"] == "env_token_wins"

    def test_cached_hf_folder_token_used_when_env_missing(
            self, ckpt, fake_hf, monkeypatch):
        """When HF_TOKEN env is absent but ``~/.huggingface/token``
        is populated (vast_bootstrap.sh writes one), upload_file is
        still invoked. ``huggingface_hub`` itself reads that cache
        when ``token`` is None, so we expect ``token=None`` to be
        passed through cleanly."""
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        calls, fake_folder = fake_hf
        fake_folder._cached = "cached_token_from_disk"

        push_checkpoint_to_hf(
            str(ckpt), repo_id="moritzroessler/BRIAN",
            repo_root=str(ckpt.parent.parent.parent),
        )

        assert calls, (
            "upload_file was not called even though a cached token "
            "is available — push_checkpoint_to_hf must trust the "
            "huggingface_hub credential cache."
        )

    def test_skipped_when_no_auth_at_all(
            self, ckpt, fake_hf, monkeypatch, capsys):
        """No HF_TOKEN env, no cached token — skip cleanly with a
        message; do NOT call upload_file (which would raise a
        confusing 401)."""
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        calls, fake_folder = fake_hf
        fake_folder._cached = None  # ensure cache is empty

        push_checkpoint_to_hf(
            str(ckpt), repo_id="moritzroessler/BRIAN",
            repo_root=str(ckpt.parent.parent.parent),
        )

        assert not calls, (
            f"upload_file should NOT be called when no auth is "
            f"available; got {len(calls)} call(s)"
        )
        out = capsys.readouterr().out
        assert "skip" in out.lower() or "no" in out.lower(), (
            f"expected a clear skip message in stdout, got: {out!r}"
        )

    def test_errors_are_swallowed(self, ckpt, fake_hf, monkeypatch, capsys):
        """A failure inside upload_file must NOT propagate — it must
        be printed and the function must return normally. This is the
        same contract as push_checkpoint_to_lfs."""
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        calls, _ = fake_hf
        monkeypatch.setenv("HF_TOKEN", "tok")

        # Replace upload_file with one that raises
        import huggingface_hub  # the fake we just injected
        def _raises(**kw):
            raise RuntimeError("simulated 503 from HF Hub")
        huggingface_hub.upload_file = _raises  # type: ignore[assignment]

        # Must NOT raise
        push_checkpoint_to_hf(
            str(ckpt), repo_id="moritzroessler/BRIAN", repo_root=str(ckpt.parent.parent.parent),
        )

        out = capsys.readouterr().out
        assert "failed" in out.lower() or "error" in out.lower() or "⚠" in out, (
            f"expected an error message in stdout, got: {out!r}"
        )

    def test_default_repo_id_is_moritzroessler_BRIAN(
            self, ckpt, fake_hf, monkeypatch):
        """When neither ``repo_id`` arg nor ``HF_REPO_ID`` env is set,
        the helper uses ``moritzroessler/BRIAN`` — the canonical
        checkpoint repo for this workspace."""
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        calls, _ = fake_hf
        monkeypatch.setenv("HF_TOKEN", "tok")
        monkeypatch.delenv("HF_REPO_ID", raising=False)

        push_checkpoint_to_hf(
            str(ckpt), repo_root=str(ckpt.parent.parent.parent),
        )

        assert calls, "no upload happened"
        assert calls[0]["repo_id"] == "moritzroessler/BRIAN", (
            f"default repo_id must be moritzroessler/BRIAN, got "
            f"{calls[0]['repo_id']!r}"
        )

    def test_HF_REPO_ID_env_overrides_default(
            self, ckpt, fake_hf, monkeypatch):
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        calls, _ = fake_hf
        monkeypatch.setenv("HF_TOKEN", "tok")
        monkeypatch.setenv("HF_REPO_ID", "someorg/somerepo")

        push_checkpoint_to_hf(
            str(ckpt), repo_root=str(ckpt.parent.parent.parent),
        )

        assert calls and calls[0]["repo_id"] == "someorg/somerepo"


# ─────────────────────────────────────────────────────────────────────
# 2. push_checkpoint dispatcher
# ─────────────────────────────────────────────────────────────────────


class TestPushCheckpointDispatcher:
    """The dispatcher picks HF (default), LFS, or none based on
    backend kwarg / env var."""

    def test_dispatcher_exists(self):
        from neuroslm.checkpoint_push import push_checkpoint
        assert callable(push_checkpoint)

    def test_routes_to_hf_by_default(self, ckpt, fake_hf, monkeypatch):
        from neuroslm import checkpoint_push as cp
        calls, _ = fake_hf
        monkeypatch.setenv("HF_TOKEN", "tok")

        # No backend kwarg, no CHECKPOINT_PUSH_BACKEND env → HF wins.
        cp.push_checkpoint(
            str(ckpt), repo_root=str(ckpt.parent.parent.parent),
        )

        assert calls, (
            "default dispatcher path should call HF upload_file; "
            "got no upload — backend may not be defaulting to 'hf'"
        )

    def test_routes_to_lfs_when_requested(self, ckpt, monkeypatch):
        from neuroslm import checkpoint_push as cp

        invoked = {"lfs": False}

        def _fake_lfs(path, repo_root=None):
            invoked["lfs"] = True

        monkeypatch.setattr(cp, "push_checkpoint_to_lfs", _fake_lfs)

        cp.push_checkpoint(
            str(ckpt), backend="lfs",
            repo_root=str(ckpt.parent.parent.parent),
        )

        assert invoked["lfs"], (
            "backend='lfs' must dispatch to push_checkpoint_to_lfs"
        )

    def test_routes_to_none_is_no_op(self, ckpt, fake_hf, monkeypatch):
        from neuroslm import checkpoint_push as cp
        calls, _ = fake_hf
        monkeypatch.setenv("HF_TOKEN", "tok")

        cp.push_checkpoint(
            str(ckpt), backend="none",
            repo_root=str(ckpt.parent.parent.parent),
        )

        assert not calls, (
            "backend='none' must be a no-op; got "
            f"{len(calls)} HF upload(s)"
        )

    def test_env_var_overrides_default(self, ckpt, monkeypatch):
        """``CHECKPOINT_PUSH_BACKEND=lfs`` flips the default."""
        from neuroslm import checkpoint_push as cp

        invoked = {"lfs": False}

        def _fake_lfs(path, repo_root=None):
            invoked["lfs"] = True

        monkeypatch.setattr(cp, "push_checkpoint_to_lfs", _fake_lfs)
        monkeypatch.setenv("CHECKPOINT_PUSH_BACKEND", "lfs")

        cp.push_checkpoint(
            str(ckpt),  # no explicit backend
            repo_root=str(ckpt.parent.parent.parent),
        )

        assert invoked["lfs"], (
            "CHECKPOINT_PUSH_BACKEND=lfs must flip the dispatcher "
            "default to LFS"
        )

    def test_explicit_backend_arg_wins_over_env(
            self, ckpt, fake_hf, monkeypatch):
        """Explicit ``backend='hf'`` beats ``CHECKPOINT_PUSH_BACKEND=lfs``."""
        from neuroslm import checkpoint_push as cp
        calls, _ = fake_hf
        monkeypatch.setenv("HF_TOKEN", "tok")
        monkeypatch.setenv("CHECKPOINT_PUSH_BACKEND", "lfs")

        cp.push_checkpoint(
            str(ckpt), backend="hf",
            repo_root=str(ckpt.parent.parent.parent),
        )

        assert calls, "explicit backend='hf' was ignored"


# ─────────────────────────────────────────────────────────────────────
# 3. train_dsl.py call-sites switched over
# ─────────────────────────────────────────────────────────────────────


class TestTrainDslUsesDispatcher:
    """The 4 push call-sites in ``train_dsl.py`` must call the
    dispatcher (not the LFS function directly), so the backend can
    be picked per-run via env/CLI without re-editing the trainer."""

    def test_dispatcher_imported(self):
        src = (REPO_ROOT / "neuroslm" / "train_dsl.py").read_text(
            encoding="utf-8"
        )
        assert "push_checkpoint" in src, (
            "neuroslm/train_dsl.py must import the push_checkpoint "
            "dispatcher (or push_checkpoint_to_hf) from "
            "neuroslm.checkpoint_push so the backend is selectable "
            "without editing the trainer."
        )

    def test_push_backend_cli_flag_exists(self):
        """``--push_backend hf|lfs|none`` so the on-box scripts can
        pick the backend at launch time."""
        src = (REPO_ROOT / "neuroslm" / "train_dsl.py").read_text(
            encoding="utf-8"
        )
        assert "--push_backend" in src, (
            "neuroslm/train_dsl.py must register a --push_backend "
            "CLI flag (choices: hf|lfs|none)"
        )


# ─────────────────────────────────────────────────────────────────────
# 4. brian.toml [defaults] new fields
# ─────────────────────────────────────────────────────────────────────


class TestProjectConfigPushBackend:
    """``brian.toml [defaults]`` exposes ``push_backend`` and
    ``hf_repo_id`` so the workspace can be retargeted in one file."""

    def test_default_push_backend_is_hf(self, tmp_path):
        from neuroslm.project_config import load_project_config
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_push_backend == "hf", (
            f"new default must be 'hf' (post-LFS-hang fix), got "
            f"{cfg.default_push_backend!r}"
        )

    def test_default_hf_repo_id_is_moritzroessler_BRIAN(self, tmp_path):
        from neuroslm.project_config import load_project_config
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_hf_repo_id == "moritzroessler/BRIAN", (
            f"default HF repo must be moritzroessler/BRIAN, got "
            f"{cfg.default_hf_repo_id!r}"
        )

    def test_parses_push_backend_from_toml(self, tmp_path):
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            '[defaults]\n'
            'push_backend = "lfs"\n'
            'hf_repo_id   = "myorg/myrepo"\n',
            encoding="utf-8",
        )
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_push_backend == "lfs"
        assert cfg.default_hf_repo_id == "myorg/myrepo"


# ─────────────────────────────────────────────────────────────────────
# 5. _deploy_train.py + on-box scripts propagate the new envs
# ─────────────────────────────────────────────────────────────────────


class TestDeployPropagatesHFEnv:
    """``_deploy_train.py`` must propagate ``HF_TOKEN``,
    ``CHECKPOINT_PUSH_BACKEND``, and ``HF_REPO_ID`` to the box."""

    def test_HF_TOKEN_propagated_from_env(self):
        """The deploy script must NOT hardcode ``HF_TOKEN=''`` — it
        must thread the real env var through so the box can push."""
        src = (REPO_ROOT / "_deploy_train.py").read_text(encoding="utf-8")
        # The fix: ``HF_TOKEN='{HF_TOKEN}'`` not ``HF_TOKEN=''``
        assert "HF_TOKEN=''" not in src, (
            "_deploy_train.py must NOT hardcode HF_TOKEN to empty; "
            "thread the real os.environ['HF_TOKEN'] through so the "
            "on-box checkpoint pusher can authenticate to HF Hub."
        )
        assert "HF_TOKEN" in src, (
            "_deploy_train.py must still reference HF_TOKEN to "
            "propagate it to the box."
        )

    def test_checkpoint_push_backend_env_propagated(self):
        src = (REPO_ROOT / "_deploy_train.py").read_text(encoding="utf-8")
        assert "CHECKPOINT_PUSH_BACKEND" in src, (
            "_deploy_train.py must export CHECKPOINT_PUSH_BACKEND so "
            "the on-box trainer picks the configured backend."
        )

    def test_HF_REPO_ID_env_propagated(self):
        src = (REPO_ROOT / "_deploy_train.py").read_text(encoding="utf-8")
        assert "HF_REPO_ID" in src, (
            "_deploy_train.py must export HF_REPO_ID."
        )

    def test_vast_loop_forwards_push_backend(self):
        src = (REPO_ROOT / "scripts" / "vast_train_dsl_loop.sh").read_text(
            encoding="utf-8"
        )
        assert "PUSH_BACKEND" in src or "push_backend" in src.lower(), (
            "vast_train_dsl_loop.sh must read CHECKPOINT_PUSH_BACKEND "
            "(or PUSH_BACKEND) and forward it as --push_backend."
        )
