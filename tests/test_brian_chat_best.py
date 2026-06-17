# -*- coding: utf-8 -*-
"""TDD: ``brian chat`` defaults to the best-run checkpoint.

Contract being pinned:

* When invoked with no checkpoint args, ``brian chat`` reads
  ``.brian/checkpoint.ln`` (auto-written by ``brian best update``) and
  downloads that HF checkpoint before booting the daemon.

* ``--pt PATH_OR_URI`` is an explicit override (named-flag alias for
  the positional ``ckpt``). Both shapes must work and ``--pt`` wins
  when both are given.

* Precedence chain (top wins):
    1. ``--pt PATH_OR_URI``     explicit named flag (NEW)
    2. positional ``ckpt``      explicit positional (existing)
    3. ``--latest``             HF Hub highest-step (existing)
    4. ``.brian/checkpoint.ln`` best-run pointer (NEW default)
    5. local ``lfs_checkpoints/`` highest-step (existing fallback)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _make_chat_args(**overrides):
    """Build a Namespace with every flag the chat dispatcher reads.

    Each new flag added to the chat subparser MUST also appear here
    or the test will crash with AttributeError instead of failing
    cleanly.
    """
    defaults = dict(
        ckpt=None, pt=None, latest=False, best=False, no_best=False,
        repo=None, prefix=None, arch=None, device="cpu",
        temperature=0.8, top_k=40, max_new_tokens=96,
        thought_tokens=32, thought_period=12.0, idle_threshold=6.0,
        no_color=True, no_thoughts=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ─────────────────────────────────────────────────────────────────────
# 1. --pt PATH flag exists and is honoured
# ─────────────────────────────────────────────────────────────────────


class TestPtFlag:
    """``--pt PATH`` is a named-flag alias for the positional ``ckpt``."""

    def test_pt_flag_in_parser(self):
        """The argparse parser must expose ``--pt``."""
        from neuroslm.cli import _build_parser
        parser = _build_parser()
        # --pt should accept a string
        args = parser.parse_args(["chat", "--pt", "lfs_checkpoints/x.pt"])
        assert args.pt == "lfs_checkpoints/x.pt"

    def test_pt_flag_calls_run_chat_daemon_with_that_path(
            self, monkeypatch, tmp_path):
        from neuroslm import cli
        ckpt = tmp_path / "explicit.pt"
        ckpt.write_bytes(b"\x00")
        captured = {}
        def fake_run(**kw):
            captured.update(kw)
            return 0
        monkeypatch.setattr(
            "neuroslm.chat_daemon.run_chat_daemon", fake_run)
        rc = cli.cmd_chat(_make_chat_args(pt=str(ckpt)))
        assert rc == 0
        assert captured["ckpt_path"] == str(ckpt)

    def test_pt_flag_wins_over_positional(self, monkeypatch, tmp_path):
        """If both ``ckpt`` and ``--pt`` are given, ``--pt`` is the
        explicit user request — it should win."""
        from neuroslm import cli
        positional = tmp_path / "positional.pt"
        explicit = tmp_path / "explicit.pt"
        positional.write_bytes(b"\x00")
        explicit.write_bytes(b"\x00")
        captured = {}
        def fake_run(**kw):
            captured.update(kw)
            return 0
        monkeypatch.setattr(
            "neuroslm.chat_daemon.run_chat_daemon", fake_run)
        rc = cli.cmd_chat(_make_chat_args(
            ckpt=str(positional), pt=str(explicit)))
        assert rc == 0
        assert captured["ckpt_path"] == str(explicit)

    def test_pt_flag_accepts_hf_uri(self, monkeypatch, tmp_path):
        """``--pt hf://...`` should trigger HF download + boot."""
        from neuroslm import cli
        local = tmp_path / "downloaded.pt"
        local.write_bytes(b"\x00")
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.parse_hf_uri",
            lambda uri: ("alice/bob", "checkpoints/run-X/step5000.pt"))
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.download_checkpoint",
            lambda path_in_repo, **kw: local)
        captured = {}
        def fake_run(**kw):
            captured.update(kw)
            return 0
        monkeypatch.setattr(
            "neuroslm.chat_daemon.run_chat_daemon", fake_run)
        rc = cli.cmd_chat(_make_chat_args(
            pt="hf://alice/bob/checkpoints/run-X/step5000.pt"))
        assert rc == 0
        assert captured["ckpt_path"] == str(local)


# ─────────────────────────────────────────────────────────────────────
# 2. Default behaviour: pull best-run checkpoint from .brian/checkpoint.ln
# ─────────────────────────────────────────────────────────────────────


class TestDefaultBestPull:
    """When no checkpoint arg is given, default is to use the
    best-run checkpoint from ``.brian/checkpoint.ln`` (HF URL)."""

    def test_no_args_reads_checkpoint_ln_when_present(
            self, monkeypatch, tmp_path):
        """Default mode: read .brian/checkpoint.ln → HF URL → download
        → boot. Tests should not touch the real repo's .brian/ dir."""
        from neuroslm import cli
        from neuroslm import log_refs
        # Stub repo root + checkpoint.ln read
        monkeypatch.setattr(
            "neuroslm.log_refs.read_checkpoint_url",
            lambda root: "hf://alice/bob/checkpoints/run-Best/step9000.pt")
        local = tmp_path / "step9000.pt"
        local.write_bytes(b"\x00")
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.parse_hf_uri",
            lambda uri: ("alice/bob", "checkpoints/run-Best/step9000.pt"))
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.download_checkpoint",
            lambda path_in_repo, **kw: local)
        # Make _pick_local_latest_ckpt return something so we can verify
        # the best-pull won (and not the fallback).
        monkeypatch.setattr(
            cli, "_pick_local_latest_ckpt",
            lambda: str(tmp_path / "old_local.pt"))
        captured = {}
        def fake_run(**kw):
            captured.update(kw)
            return 0
        monkeypatch.setattr(
            "neuroslm.chat_daemon.run_chat_daemon", fake_run)
        rc = cli.cmd_chat(_make_chat_args())
        assert rc == 0
        # Must boot the BEST checkpoint, not the local fallback
        assert captured["ckpt_path"] == str(local)

    def test_no_checkpoint_ln_falls_back_to_local(
            self, monkeypatch, tmp_path):
        """Missing/empty .brian/checkpoint.ln → fall through to
        local-latest, then the not-found error."""
        from neuroslm import cli
        monkeypatch.setattr(
            "neuroslm.log_refs.read_checkpoint_url", lambda root: None)
        local = tmp_path / "old_local.pt"
        local.write_bytes(b"\x00")
        monkeypatch.setattr(
            cli, "_pick_local_latest_ckpt", lambda: str(local))
        captured = {}
        def fake_run(**kw):
            captured.update(kw)
            return 0
        monkeypatch.setattr(
            "neuroslm.chat_daemon.run_chat_daemon", fake_run)
        rc = cli.cmd_chat(_make_chat_args())
        assert rc == 0
        assert captured["ckpt_path"] == str(local)

    def test_no_best_no_local_returns_2(self, monkeypatch, capsys):
        """No best pointer + no local → explicit failure (rc=2)."""
        from neuroslm import cli
        monkeypatch.setattr(
            "neuroslm.log_refs.read_checkpoint_url", lambda root: None)
        monkeypatch.setattr(
            cli, "_pick_local_latest_ckpt", lambda: None)
        rc = cli.cmd_chat(_make_chat_args())
        assert rc == 2
        err = capsys.readouterr().err
        assert "no checkpoint" in err.lower()


# ─────────────────────────────────────────────────────────────────────
# 3. Precedence chain
# ─────────────────────────────────────────────────────────────────────


class TestPrecedenceChain:
    """Full precedence ordering: pt > positional > latest > best > local."""

    def test_pt_overrides_best(self, monkeypatch, tmp_path):
        """Even when checkpoint.ln points at one ckpt, ``--pt`` wins."""
        from neuroslm import cli
        explicit = tmp_path / "explicit.pt"
        explicit.write_bytes(b"\x00")
        # checkpoint.ln points at a DIFFERENT path that won't be used
        monkeypatch.setattr(
            "neuroslm.log_refs.read_checkpoint_url",
            lambda root: "hf://x/y/checkpoints/should-not-be-used/step1.pt")
        captured = {}
        def fake_run(**kw):
            captured.update(kw)
            return 0
        monkeypatch.setattr(
            "neuroslm.chat_daemon.run_chat_daemon", fake_run)
        rc = cli.cmd_chat(_make_chat_args(pt=str(explicit)))
        assert rc == 0
        assert captured["ckpt_path"] == str(explicit)

    def test_positional_overrides_best(self, monkeypatch, tmp_path):
        """Positional ckpt wins over the best-run default too."""
        from neuroslm import cli
        positional = tmp_path / "positional.pt"
        positional.write_bytes(b"\x00")
        monkeypatch.setattr(
            "neuroslm.log_refs.read_checkpoint_url",
            lambda root: "hf://x/y/checkpoints/best/step9000.pt")
        captured = {}
        def fake_run(**kw):
            captured.update(kw)
            return 0
        monkeypatch.setattr(
            "neuroslm.chat_daemon.run_chat_daemon", fake_run)
        rc = cli.cmd_chat(_make_chat_args(ckpt=str(positional)))
        assert rc == 0
        assert captured["ckpt_path"] == str(positional)

    def test_latest_overrides_best(self, monkeypatch, tmp_path):
        """``--latest`` is an explicit user request — wins over best
        too (best is for *quality*, latest is for *recency*)."""
        from neuroslm import cli
        from neuroslm.hf_checkpoints import CheckpointEntry
        latest_local = tmp_path / "latest.pt"
        latest_local.write_bytes(b"\x00")
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.find_latest_checkpoint",
            lambda **kw: CheckpointEntry(
                path_in_repo="checkpoints/x/step9999.pt", step=9999))
        monkeypatch.setattr(
            "neuroslm.hf_checkpoints.download_checkpoint",
            lambda *a, **kw: latest_local)
        # checkpoint.ln should NOT be consulted at all when --latest is set
        monkeypatch.setattr(
            "neuroslm.log_refs.read_checkpoint_url",
            lambda root: pytest.fail(
                "read_checkpoint_url should not be called with --latest"))
        captured = {}
        def fake_run(**kw):
            captured.update(kw)
            return 0
        monkeypatch.setattr(
            "neuroslm.chat_daemon.run_chat_daemon", fake_run)
        rc = cli.cmd_chat(_make_chat_args(latest=True))
        assert rc == 0
        assert captured["ckpt_path"] == str(latest_local)

    def test_no_best_flag_opts_out(self, monkeypatch, tmp_path):
        """``--no-best`` disables the auto-best lookup and falls
        straight to local fallback (escape hatch for offline use)."""
        from neuroslm import cli
        local = tmp_path / "local.pt"
        local.write_bytes(b"\x00")
        monkeypatch.setattr(
            "neuroslm.log_refs.read_checkpoint_url",
            lambda root: pytest.fail(
                "read_checkpoint_url should not be called with --no-best"))
        monkeypatch.setattr(
            cli, "_pick_local_latest_ckpt", lambda: str(local))
        captured = {}
        def fake_run(**kw):
            captured.update(kw)
            return 0
        monkeypatch.setattr(
            "neuroslm.chat_daemon.run_chat_daemon", fake_run)
        rc = cli.cmd_chat(_make_chat_args(no_best=True))
        assert rc == 0
        assert captured["ckpt_path"] == str(local)


# ─────────────────────────────────────────────────────────────────────
# 4. Parser flag presence
# ─────────────────────────────────────────────────────────────────────


class TestChatParserFlags:

    def test_pt_flag_default_none(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(["chat"])
        assert args.pt is None

    def test_no_best_flag_default_false(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(["chat"])
        assert args.no_best is False

    def test_no_best_flag_can_be_set(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(["chat", "--no-best"])
        assert args.no_best is True
