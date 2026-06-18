# -*- coding: utf-8 -*-
"""TDD tests for ``brian checkpoints`` command.

Covers: argparse wiring, _ckpts_list, _ckpts_download, _ckpts_use,
_ckpts_active, cmd_chat .neuro/checkpoint.ln integration, and
inspect_checkpoint_metadata from hf_checkpoints.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def parser():
    from neuroslm.cli import _build_parser
    return _build_parser()


# ── A: Parser-level wiring ─────────────────────────────────────────

class TestCheckpointsSubparser:

    def test_checkpoints_help_exits_0(self, parser):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["checkpoints", "--help"])
        assert exc.value.code == 0

    def test_list_subcommand_exists(self, parser):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["checkpoints", "list", "--help"])
        assert exc.value.code == 0

    def test_download_subcommand_exists(self, parser):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["checkpoints", "download", "--help"])
        assert exc.value.code == 0

    def test_use_subcommand_exists(self, parser):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["checkpoints", "use", "--help"])
        assert exc.value.code == 0

    def test_active_subcommand_exists(self, parser):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["checkpoints", "active", "--help"])
        assert exc.value.code == 0

    def test_list_default_flags(self, parser):
        args = parser.parse_args(["checkpoints", "list"])
        assert args.ckpts_cmd == "list"
        assert args.limit == 20
        assert args.repo is None
        assert args.prefix is None

    def test_list_limit_flag(self, parser):
        args = parser.parse_args(["checkpoints", "list", "--limit", "5"])
        assert args.limit == 5

    def test_list_repo_prefix(self, parser):
        args = parser.parse_args([
            "checkpoints", "list", "--repo", "alice/bob", "--prefix", "run-X",
        ])
        assert args.repo == "alice/bob"
        assert args.prefix == "run-X"

    def test_download_latest_flag(self, parser):
        args = parser.parse_args(["checkpoints", "download", "--latest"])
        assert args.latest is True
        assert args.target is None

    def test_download_target_positional(self, parser):
        args = parser.parse_args(["checkpoints", "download", "5000"])
        assert args.target == "5000"
        assert args.latest is False

    def test_download_activate_default_true(self, parser):
        args = parser.parse_args(["checkpoints", "download", "--latest"])
        assert args.activate is True

    def test_download_no_activate_flag(self, parser):
        args = parser.parse_args(["checkpoints", "download", "--latest", "--no-activate"])
        assert args.activate is False

    def test_use_positional(self, parser):
        args = parser.parse_args([
            "checkpoints", "use", "checkpoints/run/step5000.pt",
        ])
        assert args.path == "checkpoints/run/step5000.pt"


# ── B: _ckpts_list handler ─────────────────────────────────────────

class TestCkptsList:

    def test_list_calls_list_repo_checkpoints(self, monkeypatch, capsys):
        from neuroslm import cli
        from neuroslm.hf_checkpoints import CheckpointEntry

        entries = [
            CheckpointEntry("checkpoints/run-A/step10000.pt",
                            step=10000, run_dir="run-A", size=450_000_000),
            CheckpointEntry("checkpoints/run-A/step5000.pt",
                            step=5000, run_dir="run-A", size=450_000_000),
        ]
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.list_repo_checkpoints",
            lambda **kw: entries,
        )
        args = argparse.Namespace(
            ckpts_cmd="list", repo=None, prefix=None, limit=20)
        rc = cli.cmd_checkpoints(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "10000" in out
        assert "5000" in out

    def test_list_shows_run_dir(self, monkeypatch, capsys):
        from neuroslm import cli
        from neuroslm.hf_checkpoints import CheckpointEntry

        entries = [
            CheckpointEntry("checkpoints/run-A/step10000.pt",
                            step=10000, run_dir="run-A", size=0),
        ]
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.list_repo_checkpoints",
            lambda **kw: entries,
        )
        args = argparse.Namespace(
            ckpts_cmd="list", repo=None, prefix=None, limit=20)
        cli.cmd_checkpoints(args)
        out = capsys.readouterr().out
        assert "run-A" in out

    def test_list_marks_active_checkpoint(self, monkeypatch, tmp_path, capsys):
        from neuroslm import cli
        from neuroslm.hf_checkpoints import CheckpointEntry

        entries = [
            CheckpointEntry("checkpoints/run-A/step10000.pt",
                            step=10000, run_dir="run-A", size=0),
            CheckpointEntry("checkpoints/run-A/step5000.pt",
                            step=5000, run_dir="run-A", size=0),
        ]
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.list_repo_checkpoints",
            lambda **kw: entries,
        )
        (tmp_path / ".neuro").mkdir()
        (tmp_path / ".neuro" / "checkpoint.ln").write_text(
            "checkpoints/run-A/step10000.pt")
        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

        args = argparse.Namespace(
            ckpts_cmd="list", repo=None, prefix=None, limit=20)
        cli.cmd_checkpoints(args)
        out = capsys.readouterr().out
        assert "●" in out

    def test_list_no_checkpoints_returns_1(self, monkeypatch, capsys):
        from neuroslm import cli

        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.list_repo_checkpoints",
            lambda **kw: [],
        )
        args = argparse.Namespace(
            ckpts_cmd="list", repo=None, prefix=None, limit=20)
        rc = cli.cmd_checkpoints(args)
        assert rc == 1


# ── C: _ckpts_download handler ─────────────────────────────────────

class TestCkptsDownload:

    def _make_entry(self, step=7500, run_dir="run-A"):
        from neuroslm.hf_checkpoints import CheckpointEntry
        return CheckpointEntry(
            f"checkpoints/{run_dir}/step{step}.pt",
            step=step, run_dir=run_dir, size=0)

    def test_download_latest_returns_0(self, monkeypatch, tmp_path):
        from neuroslm import cli

        entry = self._make_entry(7500)
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.find_latest_checkpoint",
            lambda **kw: entry)
        fake_local = tmp_path / "checkpoints" / "run-A" / "step7500.pt"
        fake_local.parent.mkdir(parents=True)
        fake_local.write_bytes(b"\x00")
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.download_checkpoint",
            lambda *a, **kw: fake_local)
        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
        (tmp_path / ".neuro").mkdir()

        args = argparse.Namespace(
            ckpts_cmd="download", target=None, latest=True,
            repo=None, prefix=None, activate=True)
        rc = cli.cmd_checkpoints(args)
        assert rc == 0

    def test_download_latest_writes_neuro_checkpoint_ln(self, monkeypatch, tmp_path):
        from neuroslm import cli

        entry = self._make_entry(7500)
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.find_latest_checkpoint",
            lambda **kw: entry)
        fake_local = tmp_path / "checkpoints" / "run-A" / "step7500.pt"
        fake_local.parent.mkdir(parents=True)
        fake_local.write_bytes(b"\x00")
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.download_checkpoint",
            lambda *a, **kw: fake_local)
        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
        (tmp_path / ".neuro").mkdir()

        args = argparse.Namespace(
            ckpts_cmd="download", target=None, latest=True,
            repo=None, prefix=None, activate=True)
        cli.cmd_checkpoints(args)

        ln_file = tmp_path / ".neuro" / "checkpoint.ln"
        assert ln_file.is_file()
        assert "step7500.pt" in ln_file.read_text()

    def test_download_writes_relative_path_in_checkpoint_ln(self, monkeypatch, tmp_path):
        from neuroslm import cli

        entry = self._make_entry(5000, "run-B")
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.find_latest_checkpoint",
            lambda **kw: entry)
        fake_local = tmp_path / "checkpoints" / "run-B" / "step5000.pt"
        fake_local.parent.mkdir(parents=True)
        fake_local.write_bytes(b"\x00")
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.download_checkpoint",
            lambda *a, **kw: fake_local)
        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
        (tmp_path / ".neuro").mkdir()

        args = argparse.Namespace(
            ckpts_cmd="download", target=None, latest=True,
            repo=None, prefix=None, activate=True)
        cli.cmd_checkpoints(args)

        content = (tmp_path / ".neuro" / "checkpoint.ln").read_text().strip()
        # Must be a relative path (not starting with the tmp_path absolute prefix)
        assert not content.startswith(str(tmp_path))
        # Must resolve back to the downloaded file
        assert (tmp_path / content) == fake_local

    def test_download_without_activate_does_not_write_checkpoint_ln(
            self, monkeypatch, tmp_path):
        from neuroslm import cli

        entry = self._make_entry(7500)
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.find_latest_checkpoint",
            lambda **kw: entry)
        fake_local = tmp_path / "checkpoints" / "run-A" / "step7500.pt"
        fake_local.parent.mkdir(parents=True)
        fake_local.write_bytes(b"\x00")
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.download_checkpoint",
            lambda *a, **kw: fake_local)
        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
        (tmp_path / ".neuro").mkdir()

        args = argparse.Namespace(
            ckpts_cmd="download", target=None, latest=True,
            repo=None, prefix=None, activate=False)
        cli.cmd_checkpoints(args)

        assert not (tmp_path / ".neuro" / "checkpoint.ln").exists()

    def test_download_by_step_number(self, monkeypatch, tmp_path):
        from neuroslm import cli
        from neuroslm.hf_checkpoints import CheckpointEntry

        entries = [
            CheckpointEntry("checkpoints/run-A/step10000.pt",
                            step=10000, run_dir="run-A", size=0),
            CheckpointEntry("checkpoints/run-A/step5000.pt",
                            step=5000, run_dir="run-A", size=0),
        ]
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.list_repo_checkpoints",
            lambda **kw: entries)
        fake_local = tmp_path / "checkpoints" / "run-A" / "step5000.pt"
        fake_local.parent.mkdir(parents=True)
        fake_local.write_bytes(b"\x00")

        pulled = {}

        def fake_download(path_in_repo, **kw):
            pulled["path"] = path_in_repo
            return fake_local

        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.download_checkpoint", fake_download)
        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
        (tmp_path / ".neuro").mkdir()

        args = argparse.Namespace(
            ckpts_cmd="download", target="5000", latest=False,
            repo=None, prefix=None, activate=False)
        rc = cli.cmd_checkpoints(args)
        assert rc == 0
        assert "step5000.pt" in pulled["path"]

    def test_download_no_checkpoints_at_step_returns_1(self, monkeypatch, tmp_path, capsys):
        from neuroslm import cli

        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.list_repo_checkpoints",
            lambda **kw: [])
        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

        args = argparse.Namespace(
            ckpts_cmd="download", target="9999", latest=False,
            repo=None, prefix=None, activate=False)
        rc = cli.cmd_checkpoints(args)
        assert rc == 1

    def test_download_no_target_no_latest_returns_2(self, monkeypatch, tmp_path, capsys):
        from neuroslm import cli

        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

        args = argparse.Namespace(
            ckpts_cmd="download", target=None, latest=False,
            repo=None, prefix=None, activate=False)
        rc = cli.cmd_checkpoints(args)
        assert rc == 2

    def test_download_passes_checkpoints_dest_dir(self, monkeypatch, tmp_path):
        """download_checkpoint must receive dest_dir=<repo>/checkpoints/."""
        from neuroslm import cli

        entry = self._make_entry(7500)
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.find_latest_checkpoint",
            lambda **kw: entry)
        captured = {}

        def fake_download(path_in_repo, *, repo_id=None, dest_dir=None, **kw):
            captured["dest_dir"] = dest_dir
            return None  # simulate failure → rc=1

        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.download_checkpoint", fake_download)
        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
        (tmp_path / ".neuro").mkdir()

        args = argparse.Namespace(
            ckpts_cmd="download", target=None, latest=True,
            repo=None, prefix=None, activate=False)
        cli.cmd_checkpoints(args)
        assert captured["dest_dir"] is not None
        # dest_dir must be under the repo root's checkpoints/ folder
        assert "checkpoints" in str(captured["dest_dir"])


# ── D: _ckpts_use / _ckpts_active handlers ─────────────────────────

class TestCkptsUseActive:

    def test_use_writes_checkpoint_ln(self, monkeypatch, tmp_path):
        from neuroslm import cli

        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
        (tmp_path / ".neuro").mkdir()
        fake_pt = tmp_path / "checkpoints" / "run-A" / "step5000.pt"
        fake_pt.parent.mkdir(parents=True)
        fake_pt.write_bytes(b"\x00")

        args = argparse.Namespace(ckpts_cmd="use", path=str(fake_pt))
        rc = cli.cmd_checkpoints(args)
        assert rc == 0
        content = (tmp_path / ".neuro" / "checkpoint.ln").read_text().strip()
        assert "step5000.pt" in content

    def test_use_nonexistent_path_returns_2(self, monkeypatch, tmp_path, capsys):
        from neuroslm import cli

        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
        (tmp_path / ".neuro").mkdir()

        args = argparse.Namespace(
            ckpts_cmd="use", path="/nonexistent/step5000.pt")
        rc = cli.cmd_checkpoints(args)
        assert rc == 2

    def test_active_prints_checkpoint_ln(self, monkeypatch, tmp_path, capsys):
        from neuroslm import cli

        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
        (tmp_path / ".neuro").mkdir()
        (tmp_path / ".neuro" / "checkpoint.ln").write_text(
            "checkpoints/run-A/step5000.pt\n")

        args = argparse.Namespace(ckpts_cmd="active")
        rc = cli.cmd_checkpoints(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "step5000.pt" in out

    def test_active_no_checkpoint_ln_returns_1(self, monkeypatch, tmp_path, capsys):
        from neuroslm import cli

        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
        (tmp_path / ".neuro").mkdir()

        args = argparse.Namespace(ckpts_cmd="active")
        rc = cli.cmd_checkpoints(args)
        assert rc == 1


# ── E: cmd_chat reads .neuro/checkpoint.ln (hop 3.5) ──────────────

class TestChatReadsNeuroCheckpointLn:

    def test_chat_uses_neuro_checkpoint_ln_before_hf_hop(
            self, monkeypatch, tmp_path):
        from neuroslm import cli

        ckpt_file = tmp_path / "checkpoints" / "run-A" / "step5000.pt"
        ckpt_file.parent.mkdir(parents=True)
        ckpt_file.write_bytes(b"\x00")
        (tmp_path / ".neuro").mkdir()
        (tmp_path / ".neuro" / "checkpoint.ln").write_text(
            "checkpoints/run-A/step5000.pt")
        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

        called = {}

        def fake_run(**kw):
            called.update(kw)
            return 0

        monkeypatch.setattr("neuroslm.chat_daemon.run_chat_daemon", fake_run)
        monkeypatch.setattr(cli, "_pick_local_latest_ckpt", lambda: None)

        args = argparse.Namespace(
            ckpt=None, latest=False, repo=None, prefix=None,
            arch=None, device="cpu", temperature=0.8, top_k=40,
            max_new_tokens=96, thought_tokens=32, thought_period=12.0,
            idle_threshold=6.0, no_color=False, no_thoughts=False,
        )
        rc = cli.cmd_chat(args)
        assert rc == 0
        assert called["ckpt_path"] == str(ckpt_file)

    def test_chat_skips_neuro_ln_if_explicit_ckpt(self, monkeypatch, tmp_path):
        from neuroslm import cli

        explicit = tmp_path / "explicit.pt"
        explicit.write_bytes(b"\x00")
        other = tmp_path / "checkpoints" / "run-A" / "step5000.pt"
        other.parent.mkdir(parents=True)
        other.write_bytes(b"\x00")
        (tmp_path / ".neuro").mkdir()
        (tmp_path / ".neuro" / "checkpoint.ln").write_text(
            "checkpoints/run-A/step5000.pt")
        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

        called = {}

        def fake_run(**kw):
            called.update(kw)
            return 0

        monkeypatch.setattr("neuroslm.chat_daemon.run_chat_daemon", fake_run)

        args = argparse.Namespace(
            ckpt=str(explicit), latest=False, repo=None, prefix=None,
            arch=None, device="cpu", temperature=0.8, top_k=40,
            max_new_tokens=96, thought_tokens=32, thought_period=12.0,
            idle_threshold=6.0, no_color=False, no_thoughts=False,
        )
        rc = cli.cmd_chat(args)
        assert rc == 0
        assert called["ckpt_path"] == str(explicit)

    def test_chat_neuro_ln_missing_falls_through(self, monkeypatch, tmp_path):
        """When .neuro/checkpoint.ln doesn't exist, cmd_chat continues to HF hop."""
        from neuroslm import cli

        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(cli, "_pick_local_latest_ckpt", lambda: None)
        (tmp_path / ".neuro").mkdir()  # no checkpoint.ln

        args = argparse.Namespace(
            ckpt=None, latest=False, repo=None, prefix=None,
            arch=None, device="cpu", temperature=0.8, top_k=40,
            max_new_tokens=96, thought_tokens=32, thought_period=12.0,
            idle_threshold=6.0, no_color=False, no_thoughts=False,
        )
        rc = cli.cmd_chat(args)
        # HF hop will fail (no token), local fallback returns None → rc=2
        assert rc == 2


# ── F: inspect_checkpoint_metadata ────────────────────────────────

class TestInspectCheckpointMetadata:

    def _make_fake_ckpt(self, tmp_path, step=5000, ppl=None, ood_ppl=None):
        torch = pytest.importorskip("torch")
        model = {"weight": torch.zeros(10, 20), "bias": torch.zeros(20)}
        extra = {}
        if ppl is not None:
            extra["ppl"] = ppl
        if ood_ppl is not None:
            extra["ood_ppl"] = ood_ppl
        payload = {
            "step": step,
            "model": model,
            "vocab_size": 50257,
            "d_sem": 256,
            "extra": extra if extra else None,
        }
        path = tmp_path / f"step{step}.pt"
        torch.save(payload, str(path))
        return path

    def test_extracts_step_and_params(self, tmp_path):
        from neuroslm.hf_checkpoints import inspect_checkpoint_metadata
        path = self._make_fake_ckpt(tmp_path, step=3000)
        meta = inspect_checkpoint_metadata(path)
        assert meta["step"] == 3000
        # weight: 10*20=200, bias: 20 → total 220
        assert meta["params"] == 220

    def test_extracts_ppl_and_ood_ppl(self, tmp_path):
        from neuroslm.hf_checkpoints import inspect_checkpoint_metadata
        path = self._make_fake_ckpt(tmp_path, step=5000, ppl=23.6, ood_ppl=155.0)
        meta = inspect_checkpoint_metadata(path)
        assert meta["ppl"] == pytest.approx(23.6)
        assert meta["ood_ppl"] == pytest.approx(155.0)

    def test_returns_none_for_missing_ppl(self, tmp_path):
        from neuroslm.hf_checkpoints import inspect_checkpoint_metadata
        path = self._make_fake_ckpt(tmp_path, step=1000)
        meta = inspect_checkpoint_metadata(path)
        assert meta["ppl"] is None
        assert meta["ood_ppl"] is None

    def test_returns_model_hash_string(self, tmp_path):
        from neuroslm.hf_checkpoints import inspect_checkpoint_metadata
        path = self._make_fake_ckpt(tmp_path, step=7500)
        meta = inspect_checkpoint_metadata(path)
        assert isinstance(meta["model_hash"], str)
        assert len(meta["model_hash"]) == 12  # first 12 hex chars of SHA-256

    def test_hash_is_deterministic(self, tmp_path):
        from neuroslm.hf_checkpoints import inspect_checkpoint_metadata
        path = self._make_fake_ckpt(tmp_path, step=2000)
        h1 = inspect_checkpoint_metadata(path)["model_hash"]
        h2 = inspect_checkpoint_metadata(path)["model_hash"]
        assert h1 == h2


# ── G: HF token bootstrap from .env ──────────────────────────────

class TestCkptsBootstrapsHFToken:
    """_ckpts_list / _ckpts_download must call bootstrap_secrets so that
    HF_TOKEN from .env is picked up before the Hub API call is made."""

    def test_list_calls_bootstrap_secrets_with_hf_token(self, monkeypatch, capsys):
        from neuroslm import cli

        bootstrapped: list[str] = []

        def fake_bootstrap(names, **kw):
            bootstrapped.extend(names)

        monkeypatch.setattr(
            "neuroslm.utils.secrets.bootstrap_secrets", fake_bootstrap)
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.list_repo_checkpoints",
            lambda **kw: [])

        args = argparse.Namespace(
            ckpts_cmd="list", repo=None, prefix=None, limit=20)
        cli.cmd_checkpoints(args)

        assert "HF_TOKEN" in bootstrapped, (
            "_ckpts_list must call bootstrap_secrets(['HF_TOKEN', ...]) "
            "so .env / Colab / Kaggle tokens are resolved before the HF call"
        )

    def test_download_calls_bootstrap_secrets_with_hf_token(
            self, monkeypatch, tmp_path, capsys):
        from neuroslm import cli
        from neuroslm.hf_checkpoints import CheckpointEntry

        bootstrapped: list[str] = []

        def fake_bootstrap(names, **kw):
            bootstrapped.extend(names)

        monkeypatch.setattr(
            "neuroslm.utils.secrets.bootstrap_secrets", fake_bootstrap)

        entry = CheckpointEntry("checkpoints/run-A/step7500.pt",
                                step=7500, run_dir="run-A", size=0)
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.find_latest_checkpoint",
            lambda **kw: entry)
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.download_checkpoint",
            lambda *a, **kw: None)  # download fails → rc=1, but bootstrap ran
        monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

        args = argparse.Namespace(
            ckpts_cmd="download", target=None, latest=True,
            repo=None, prefix=None, activate=False)
        cli.cmd_checkpoints(args)

        assert "HF_TOKEN" in bootstrapped


# ── H: .gitignore has checkpoints/ ────────────────────────────────

class TestGitignore:

    def test_checkpoints_dir_is_gitignored(self):
        gitignore = REPO_ROOT / ".gitignore"
        assert gitignore.is_file()
        content = gitignore.read_text(encoding="utf-8")
        assert "checkpoints/" in content


# ── H: subprocess --help smoke tests ──────────────────────────────

class TestSubprocessHelp:

    def _run(self, *args, timeout=15):
        return subprocess.run(
            [sys.executable, "-m", "neuroslm.cli", *args],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT), timeout=timeout,
        )

    def test_checkpoints_help(self):
        r = self._run("checkpoints", "--help")
        assert r.returncode == 0
        assert "list" in r.stdout
        assert "download" in r.stdout

    def test_checkpoints_list_help(self):
        r = self._run("checkpoints", "list", "--help")
        assert r.returncode == 0
        assert "--limit" in r.stdout

    def test_checkpoints_download_help(self):
        r = self._run("checkpoints", "download", "--help")
        assert r.returncode == 0
        assert "--latest" in r.stdout

    def test_checkpoints_use_help(self):
        r = self._run("checkpoints", "use", "--help")
        assert r.returncode == 0

    def test_checkpoints_active_help(self):
        r = self._run("checkpoints", "active", "--help")
        assert r.returncode == 0
