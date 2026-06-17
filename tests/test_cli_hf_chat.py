# -*- coding: utf-8 -*-
"""TDD tests for the new ``brian hf``, ``brian chat``, and ``brian
deploy --resume/--latest`` CLI surfaces.

Strategy:

* Parser-level tests use ``_build_parser()`` directly so they don't
  spawn a subprocess and they don't need any HF / torch deps.

* Handler-level tests monkey-patch the underlying functions
  (``find_latest_checkpoint``, ``download_checkpoint``,
  ``run_chat_daemon``, ``list_repo_checkpoints``) so the assertions
  are about wiring + flag plumbing, not about HF behaviour (those
  tests live in ``test_hf_checkpoints.py``).

* Subprocess tests (mirroring ``test_cli_commands.py``) cover the
  ``--help`` exit code on each new command.
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────
# Parser-level tests — argparse wiring + flag presence
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def parser():
    """The full ``brian`` parser, built once per test."""
    from neuroslm.cli import _build_parser
    return _build_parser()


class TestParserHasNewCommands:

    def test_hf_subparser_exists(self, parser):
        """``brian hf --help`` must not exit 2."""
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["hf", "--help"])
        assert excinfo.value.code == 0  # 0 = help printed successfully

    def test_chat_subparser_exists(self, parser):
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["chat", "--help"])
        assert excinfo.value.code == 0

    def test_hf_has_three_subcommands(self, parser):
        for cmd in ["list", "pull", "latest"]:
            with pytest.raises(SystemExit) as excinfo:
                parser.parse_args(["hf", cmd, "--help"])
            assert excinfo.value.code == 0, \
                f"`brian hf {cmd} --help` should print help"


class TestHfListFlags:

    def test_default_no_args(self, parser):
        args = parser.parse_args(["hf", "list"])
        assert args.cmd == "hf"
        assert args.hf_cmd == "list"
        assert args.repo is None
        assert args.prefix is None
        assert args.limit == 20  # default

    def test_repo_and_prefix(self, parser):
        args = parser.parse_args([
            "hf", "list",
            "--repo", "alice/bob",
            "--prefix", "run-X",
            "--limit", "5",
        ])
        assert args.repo == "alice/bob"
        assert args.prefix == "run-X"
        assert args.limit == 5


class TestHfPullFlags:

    def test_path_positional(self, parser):
        args = parser.parse_args([
            "hf", "pull", "checkpoints/run-A/step5000.pt",
        ])
        assert args.target == "checkpoints/run-A/step5000.pt"
        assert args.latest is False
        assert args.force is False

    def test_latest_flag(self, parser):
        args = parser.parse_args(["hf", "pull", "--latest"])
        assert args.latest is True
        assert args.target is None

    def test_force_and_out(self, parser):
        args = parser.parse_args([
            "hf", "pull", "checkpoints/x/step1.pt",
            "--force", "--out", "/tmp/here",
        ])
        assert args.force is True
        assert args.out == "/tmp/here"


class TestHfLatestFlags:

    def test_no_args(self, parser):
        args = parser.parse_args(["hf", "latest"])
        assert args.hf_cmd == "latest"
        assert args.repo is None
        assert args.prefix is None

    def test_with_prefix(self, parser):
        args = parser.parse_args(["hf", "latest", "--prefix", "run-Z"])
        assert args.prefix == "run-Z"


class TestChatFlags:

    def test_no_args_defaults(self, parser):
        args = parser.parse_args(["chat"])
        assert args.cmd == "chat"
        assert args.ckpt is None
        assert args.latest is False
        assert args.device == "cpu"
        assert args.temperature == pytest.approx(0.8)
        assert args.top_k == 40
        assert args.max_new_tokens == 96
        assert args.thought_tokens == 32
        assert args.thought_period == pytest.approx(12.0)
        assert args.idle_threshold == pytest.approx(6.0)
        assert args.no_color is False
        assert args.no_thoughts is False

    def test_positional_ckpt(self, parser):
        args = parser.parse_args(["chat", "lfs_checkpoints/x.pt"])
        assert args.ckpt == "lfs_checkpoints/x.pt"

    def test_latest_pulls_from_hf(self, parser):
        args = parser.parse_args(["chat", "--latest", "--repo", "x/y"])
        assert args.latest is True
        assert args.repo == "x/y"

    def test_no_color_and_no_thoughts(self, parser):
        args = parser.parse_args([
            "chat", "--no-color", "--no-thoughts"])
        assert args.no_color is True
        assert args.no_thoughts is True

    def test_device_choices(self, parser):
        # cuda is a valid choice
        args = parser.parse_args(["chat", "--device", "cuda"])
        assert args.device == "cuda"
        # invalid device raises
        with pytest.raises(SystemExit):
            parser.parse_args(["chat", "--device", "tpu"])


class TestDeployResumeFlags:

    def test_resume_path(self, parser):
        args = parser.parse_args([
            "deploy", "--resume", "lfs_checkpoints/run-A/step5000.pt",
        ])
        assert args.resume == "lfs_checkpoints/run-A/step5000.pt"
        assert args.latest is False

    def test_resume_hf_uri(self, parser):
        args = parser.parse_args([
            "deploy", "--resume",
            "hf://moritzroessler/BRIAN/checkpoints/run-A/step5000.pt",
        ])
        assert args.resume.startswith("hf://")

    def test_latest_flag(self, parser):
        args = parser.parse_args([
            "deploy", "--latest", "--hf-prefix", "run-X",
        ])
        assert args.latest is True
        assert args.hf_prefix == "run-X"

    def test_hf_repo_override(self, parser):
        args = parser.parse_args([
            "deploy", "--latest", "--hf-repo", "alice/bob",
        ])
        assert args.hf_repo == "alice/bob"


# ─────────────────────────────────────────────────────────────────────
# Handler-level tests — verify the dispatch + wiring
# ─────────────────────────────────────────────────────────────────────


class TestCmdHfDispatch:
    """``cmd_hf`` routes to ``_hf_list/_hf_pull/_hf_latest`` correctly."""

    def test_list_dispatches(self, monkeypatch, capsys):
        from neuroslm import cli
        called = {}
        def fake_list_repo_checkpoints(**kw):
            called.update(kw)
            return []
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.list_repo_checkpoints",
            fake_list_repo_checkpoints,
        )
        args = argparse.Namespace(
            hf_cmd="list", repo="x/y", prefix=None, limit=20)
        rc = cli.cmd_hf(args)
        # No checkpoints → exit 1
        assert rc == 1
        assert called["repo_id"] == "x/y"

    def test_latest_dispatches(self, monkeypatch, capsys):
        from neuroslm import cli
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.find_latest_checkpoint",
            lambda **kw: None,  # empty repo
        )
        args = argparse.Namespace(hf_cmd="latest", repo=None, prefix=None)
        rc = cli.cmd_hf(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "no checkpoints" in out.lower()

    def test_unknown_subcommand_returns_2(self, capsys):
        from neuroslm import cli
        args = argparse.Namespace(hf_cmd="bogus")
        rc = cli.cmd_hf(args)
        assert rc == 2

    def test_latest_prints_full_uri_on_success(self, monkeypatch, capsys):
        from neuroslm import cli
        from neuroslm.hf_checkpoints import CheckpointEntry
        entry = CheckpointEntry(
            path_in_repo="checkpoints/run-A/step5000.pt",
            step=5000, run_dir="run-A", has_mem_sidecar=True,
        )
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.find_latest_checkpoint",
            lambda **kw: entry,
        )
        args = argparse.Namespace(
            hf_cmd="latest", repo="alice/bob", prefix=None)
        rc = cli.cmd_hf(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "hf://alice/bob/checkpoints/run-A/step5000.pt" in out
        assert "5000" in out


class TestCmdChat:
    """``cmd_chat`` resolves the checkpoint via the precedence chain."""

    def test_no_ckpt_no_latest_no_local_fallback_returns_2(
            self, monkeypatch, capsys):
        from neuroslm import cli
        monkeypatch.setattr(cli, "_pick_local_latest_ckpt", lambda: None)
        args = argparse.Namespace(
            ckpt=None, latest=False, repo=None, prefix=None,
            arch=None, device="cpu", temperature=0.8, top_k=40,
            max_new_tokens=96, thought_tokens=32, thought_period=12.0,
            idle_threshold=6.0, no_color=False, no_thoughts=False,
        )
        rc = cli.cmd_chat(args)
        assert rc == 2
        err = capsys.readouterr().err
        assert "no checkpoint" in err.lower()

    def test_explicit_ckpt_calls_run_chat_daemon(
            self, monkeypatch, tmp_path):
        from neuroslm import cli
        # Make the file exist so the is_file() check passes
        ckpt = tmp_path / "fake.pt"
        ckpt.write_bytes(b"\x00")
        called = {}
        def fake_run(**kw):
            called.update(kw)
            return 0
        monkeypatch.setattr(
            "neuroslm.chat_daemon.run_chat_daemon", fake_run)
        args = argparse.Namespace(
            ckpt=str(ckpt), latest=False, repo=None, prefix=None,
            arch=None, device="cpu", temperature=0.8, top_k=40,
            max_new_tokens=96, thought_tokens=32, thought_period=12.0,
            idle_threshold=6.0, no_color=True, no_thoughts=True,
        )
        rc = cli.cmd_chat(args)
        assert rc == 0
        assert called["ckpt_path"] == str(ckpt)
        assert called["no_color"] is True
        assert called["no_thoughts"] is True

    def test_latest_pulls_then_boots(self, monkeypatch, tmp_path):
        from neuroslm import cli
        from neuroslm.hf_checkpoints import CheckpointEntry
        local = tmp_path / "step5000.pt"
        local.write_bytes(b"\x00")
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.find_latest_checkpoint",
            lambda **kw: CheckpointEntry(
                path_in_repo="checkpoints/x/step5000.pt", step=5000),
        )
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.download_checkpoint",
            lambda *a, **kw: local,
        )
        called = {}
        def fake_run(**kw):
            called.update(kw)
            return 0
        monkeypatch.setattr(
            "neuroslm.chat_daemon.run_chat_daemon", fake_run)
        args = argparse.Namespace(
            ckpt=None, latest=True, repo="x/y", prefix=None,
            arch=None, device="cpu", temperature=0.8, top_k=40,
            max_new_tokens=96, thought_tokens=32, thought_period=12.0,
            idle_threshold=6.0, no_color=False, no_thoughts=False,
        )
        rc = cli.cmd_chat(args)
        assert rc == 0
        assert called["ckpt_path"] == str(local)


class TestCmdDeployResume:
    """``cmd_deploy --resume / --latest`` should set RESUME_FROM in
    the env that gets forwarded to vast.ai."""

    def _make_args(self, **overrides):
        defaults = dict(
            arch=None, steps=None, branch=None, scale=None, dna=None,
            label=None, ood=None,
            resume=None, latest=False, hf_repo=None, hf_prefix=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_explicit_resume_sets_env(self, monkeypatch):
        from neuroslm import cli
        captured = {}
        def fake_deploy_dsl(*, steps, branch, extra_env, **kw):
            captured["extra"] = dict(extra_env)
            return 0
        monkeypatch.setattr(cli, "_deploy_dsl", fake_deploy_dsl)
        # Make load_brian_config return defaults
        args = self._make_args(
            resume="lfs_checkpoints/run-A/step5000.pt")
        rc = cli.cmd_deploy(args)
        assert rc == 0
        assert captured["extra"].get("RESUME_FROM") == \
            "lfs_checkpoints/run-A/step5000.pt"

    def test_latest_resolves_to_hf_uri(self, monkeypatch):
        from neuroslm import cli
        from neuroslm.hf_checkpoints import CheckpointEntry
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.find_latest_checkpoint",
            lambda **kw: CheckpointEntry(
                path_in_repo="checkpoints/run-A/step5000.pt", step=5000),
        )
        captured = {}
        def fake_deploy_dsl(*, steps, branch, extra_env, **kw):
            captured["extra"] = dict(extra_env)
            return 0
        monkeypatch.setattr(cli, "_deploy_dsl", fake_deploy_dsl)
        args = self._make_args(latest=True, hf_repo="alice/bob")
        rc = cli.cmd_deploy(args)
        assert rc == 0
        resume = captured["extra"].get("RESUME_FROM", "")
        assert resume.startswith("hf://alice/bob/")
        assert "step5000.pt" in resume

    def test_latest_with_no_remote_ckpts_returns_1(
            self, monkeypatch, capsys):
        from neuroslm import cli
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.find_latest_checkpoint",
            lambda **kw: None,
        )
        args = self._make_args(latest=True)
        rc = cli.cmd_deploy(args)
        assert rc == 1
        err = capsys.readouterr().err
        assert "no checkpoints" in err.lower()

    def test_no_resume_no_latest_no_env_var(self, monkeypatch):
        from neuroslm import cli
        captured = {}
        def fake_deploy_dsl(*, steps, branch, extra_env, **kw):
            captured["extra"] = dict(extra_env)
            return 0
        monkeypatch.setattr(cli, "_deploy_dsl", fake_deploy_dsl)
        args = self._make_args()
        rc = cli.cmd_deploy(args)
        assert rc == 0
        # Without --resume or --latest, RESUME_FROM should NOT be set
        assert "RESUME_FROM" not in captured["extra"]


# ─────────────────────────────────────────────────────────────────────
# Subprocess-level smoke tests (mirror test_cli_commands.py)
# ─────────────────────────────────────────────────────────────────────


class TestSubprocessHelp:
    """``--help`` exit code 0 for each new subcommand. Each spawns a
    real Python so this is a final integration check."""

    def _run(self, *cli_args, timeout=10):
        return subprocess.run(
            [sys.executable, "-m", "neuroslm.cli", *cli_args],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
            timeout=timeout,
        )

    def test_hf_help(self):
        r = self._run("hf", "--help")
        assert r.returncode == 0
        assert "list" in r.stdout
        assert "pull" in r.stdout
        assert "latest" in r.stdout

    def test_chat_help(self):
        r = self._run("chat", "--help")
        assert r.returncode == 0
        assert "--latest" in r.stdout
        assert "--thought-period" in r.stdout

    def test_deploy_help_lists_resume_flags(self):
        r = self._run("deploy", "--help")
        assert r.returncode == 0
        assert "--resume" in r.stdout
        assert "--latest" in r.stdout
        assert "--hf-repo" in r.stdout
        assert "--hf-prefix" in r.stdout

    def test_hf_list_help(self):
        r = self._run("hf", "list", "--help")
        assert r.returncode == 0
        assert "--repo" in r.stdout
        assert "--prefix" in r.stdout

    def test_hf_pull_help(self):
        r = self._run("hf", "pull", "--help")
        assert r.returncode == 0
        assert "--latest" in r.stdout
        assert "--out" in r.stdout
        assert "--force" in r.stdout

    def test_hf_latest_help(self):
        r = self._run("hf", "latest", "--help")
        assert r.returncode == 0
