# -*- coding: utf-8 -*-
"""TDD: ``brian logs`` works for destroyed instances + ``--latest`` flag.

User-stated invariants (2026-06-14, captured after H22 forensic dig):
    1. "Make it so that the brian logs command also works when the
        instance is destroyed... it should execute git fetch then git
        pull if necessary; then return the synced log."
    2. "Make it so that brian logs --latest returns the logs of the
        last instance that was running or is running. Output that
        command in the help message of brian logs."

Two foot-guns this contract closes:

1. **Destroyed-instance log access.** Today ``brian logs <id>`` only
   shells out to ``scripts/vast.sh logs <id>``, which fails with
   "instance not found" the moment vast.ai tears the container down.
   But ``scripts/log_pusher.sh`` running on the instance already
   pushed snapshots of the training log to ``logs/vast/`` while the
   instance was alive. So the local clone has the data —
   ``brian logs <id>`` just needs to know to look there when the
   vast API fails. Plus a ``git fetch && git pull`` so the user
   sees logs pushed from other workstations or from a more recent
   instance death they didn't witness.

2. **No "show me the most recent run" verb.** Today the only way to
   answer "what happened in the most recent training run?" is to
   ``ls -lt logs/vast/`` and ``cat`` the top result. The user wants
   ``brian logs --latest`` as the one-liner — strictly local, no
   vast API call at all, just file-mtime sort.

Contracts pinned here
─────────────────────
  A. ``brian logs --latest`` reads the newest file in ``logs/vast/``
     by mtime and prints it to stdout. No vast API call.
  B. ``brian logs --latest`` works with the new filename format
     (``<utc>_<container>_<arch>_<params>_<label>_stepNofN.log``)
     not just the old ``<id>__neuroslm-full.log``.
  C. ``brian logs`` (no positional, no ``--latest``) prints help
     describing both forms and exits non-zero.
  D. ``brian logs <id>`` with a destroyed instance falls back to the
     local file ``logs/vast/<id>__neuroslm-full.log`` if it exists.
  E. ``brian logs <id>`` with a destroyed instance AND no local
     file triggers ``git fetch && git pull`` then retries the local
     lookup once.
  F. ``brian logs --help`` mentions ``--latest`` in its description.
  G. ``_find_latest_log_file(log_dir)`` helper returns the newest
     ``.log`` by mtime, or ``None`` if directory is empty/missing.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_log_dir(tmp_path: Path, monkeypatch) -> Path:
    """Create ``logs/vast/`` inside tmp_path and chdir into tmp_path.

    Yields the absolute Path to ``logs/vast/`` so tests can populate
    it. The chdir is important because ``cmd_logs`` resolves
    ``logs/vast`` as a relative path.
    """
    log_dir = tmp_path / "logs" / "vast"
    log_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return log_dir


def _write_log(log_dir: Path, name: str, content: str, mtime: float | None = None) -> Path:
    """Write a log file and optionally set its mtime."""
    p = log_dir / name
    p.write_text(content, encoding="utf-8")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


# ─────────────────────────────────────────────────────────────────────
# G. Helper: _find_latest_log_file
# ─────────────────────────────────────────────────────────────────────


class TestFindLatestLogFile:
    """Unit tests for the new ``_find_latest_log_file`` helper.

    Per CLAUDE.md §1b, this helper exists so every caller that wants
    "newest log" uses the same mtime-sort logic instead of inlining
    glob + sort N times.
    """

    def test_returns_none_when_directory_missing(self, tmp_path: Path):
        from neuroslm.cli import _find_latest_log_file
        missing = tmp_path / "does_not_exist"
        assert _find_latest_log_file(missing) is None

    def test_returns_none_when_directory_empty(self, tmp_path: Path):
        from neuroslm.cli import _find_latest_log_file
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _find_latest_log_file(empty) is None

    def test_returns_newest_by_mtime(self, tmp_path: Path):
        from neuroslm.cli import _find_latest_log_file
        d = tmp_path / "logs"
        d.mkdir()
        now = time.time()
        _write_log(d, "old.log",    "old content",    mtime=now - 1000)
        _write_log(d, "middle.log", "middle content", mtime=now - 500)
        newest = _write_log(d, "newest.log", "newest content", mtime=now - 1)
        assert _find_latest_log_file(d) == newest

    def test_matches_new_format_filename(self, tmp_path: Path):
        """Regression: ``_scan_recent_destroyed`` uses glob
        ``*__neuroslm-*.log`` which misses the new format
        ``<utc>_<container>_<arch>_<params>_<label>_stepNofN.log``.
        ``_find_latest_log_file`` must match BOTH formats.
        """
        from neuroslm.cli import _find_latest_log_file
        d = tmp_path / "logs"
        d.mkdir()
        now = time.time()
        _write_log(d, "37847622__neuroslm-full.log", "old format", mtime=now - 100)
        new_fmt = _write_log(
            d,
            "20260614T184807Z_31cf84a0b3c6_arch_1127M_h22-smollm2-dna-arch_step7800of10k.log",
            "new format",
            mtime=now - 1,
        )
        assert _find_latest_log_file(d) == new_fmt

    def test_ignores_non_log_files(self, tmp_path: Path):
        from neuroslm.cli import _find_latest_log_file
        d = tmp_path / "logs"
        d.mkdir()
        now = time.time()
        log = _write_log(d, "real.log", "log content", mtime=now - 100)
        # Touch a non-log file later — should be ignored
        notes = d / "notes.txt"
        notes.write_text("scratch", encoding="utf-8")
        os.utime(notes, (now, now))
        assert _find_latest_log_file(d) == log


# ─────────────────────────────────────────────────────────────────────
# A + B. brian logs --latest
# ─────────────────────────────────────────────────────────────────────


class TestLogsLatest:
    """``brian logs --latest`` prints the newest log file's content."""

    def test_latest_prints_newest_log_content(
        self, fake_log_dir: Path, capsys, monkeypatch
    ):
        from neuroslm.cli import cmd_logs
        now = time.time()
        _write_log(fake_log_dir, "old.log", "STALE LOG\n", mtime=now - 1000)
        _write_log(
            fake_log_dir,
            "20260614T184807Z_31cf84a0b3c6_arch_1127M_h22_step7800of10k.log",
            "[train_dsl] boot @ 2026-06-14\nstep 7800 | loss 4.20\n",
            mtime=now - 1,
        )
        ns = argparse.Namespace(instance_id=None, latest=True)
        rc = cmd_logs(ns)
        assert rc == 0
        out = capsys.readouterr().out
        assert "step 7800" in out
        assert "STALE LOG" not in out

    def test_latest_makes_no_vast_api_call(
        self, fake_log_dir: Path, capsys, monkeypatch
    ):
        """``--latest`` is a strictly local operation — it must NOT
        shell out to vast.ai.
        """
        from neuroslm import cli
        now = time.time()
        _write_log(fake_log_dir, "only.log", "local content\n", mtime=now - 1)
        spawned: List[List[str]] = []

        def _trap_run(argv, *, env=None):
            spawned.append(list(argv))
            return 0

        monkeypatch.setattr(cli, "_run", _trap_run)
        ns = argparse.Namespace(instance_id=None, latest=True)
        rc = cli.cmd_logs(ns)
        assert rc == 0
        # No subprocess.call (_run) should have happened
        assert spawned == [], f"--latest leaked to subprocess: {spawned}"

    def test_latest_with_empty_log_dir_returns_nonzero(
        self, fake_log_dir: Path, capsys
    ):
        from neuroslm.cli import cmd_logs
        # fake_log_dir exists but is empty
        ns = argparse.Namespace(instance_id=None, latest=True)
        rc = cmd_logs(ns)
        assert rc != 0
        out = capsys.readouterr().out + capsys.readouterr().err
        # User-facing hint, not a stack trace
        assert "no log" in out.lower() or "empty" in out.lower()


# ─────────────────────────────────────────────────────────────────────
# C. brian logs with no arg + no --latest prints help
# ─────────────────────────────────────────────────────────────────────


class TestLogsNoArgs:
    """``brian logs`` with neither positional nor ``--latest`` is a
    user mistake → print help, exit non-zero.
    """

    def test_no_args_prints_helpful_message(
        self, fake_log_dir: Path, capsys
    ):
        from neuroslm.cli import cmd_logs
        ns = argparse.Namespace(instance_id=None, latest=False)
        rc = cmd_logs(ns)
        assert rc != 0
        out = capsys.readouterr().out + capsys.readouterr().err
        assert "--latest" in out or "instance_id" in out


# ─────────────────────────────────────────────────────────────────────
# D + E. brian logs <id> for destroyed instances
# ─────────────────────────────────────────────────────────────────────


class TestLogsDestroyedFallback:
    """When vast.ai reports the instance is gone, fall back to the
    locally-pushed log snapshot.
    """

    def test_destroyed_instance_falls_back_to_local_log(
        self, fake_log_dir: Path, capsys, monkeypatch
    ):
        """vast.sh logs <id> fails → look in logs/vast/<id>__*.log."""
        from neuroslm import cli
        # Create the pushed log file
        instance_id = "40952126"
        _write_log(
            fake_log_dir,
            f"{instance_id}__neuroslm-full.log",
            f"[train_dsl] boot for {instance_id}\nstep 7800 | loss 4.20\n",
        )

        def _fail_run(argv, *, env=None):
            # Simulate "scripts/vast.sh logs <id>" failing with nonzero
            return 1

        monkeypatch.setattr(cli, "_run", _fail_run)
        ns = argparse.Namespace(instance_id=instance_id, latest=False)
        rc = cli.cmd_logs(ns)
        assert rc == 0, "fallback to local file must succeed"
        out = capsys.readouterr().out
        assert f"boot for {instance_id}" in out
        assert "step 7800" in out

    def test_destroyed_no_local_log_triggers_git_pull(
        self, fake_log_dir: Path, capsys, monkeypatch
    ):
        """When vast.sh fails AND no local file exists, attempt
        ``git fetch && git pull`` and retry the lookup.
        """
        from neuroslm import cli
        instance_id = "40999999"
        spawned: List[List[str]] = []

        def _trap_run(argv, *, env=None, cwd=None):
            spawned.append(list(argv))
            # vast.sh fails; git fetch + pull succeed (and simulate
            # pull writing the log file as a side effect).
            if "vast.sh" in " ".join(argv):
                return 1
            if "git" in argv[0] and "pull" in argv:
                _write_log(
                    fake_log_dir,
                    f"{instance_id}__neuroslm-full.log",
                    f"[train_dsl] pulled log for {instance_id}\n",
                )
                return 0
            return 0

        monkeypatch.setattr(cli, "_run", _trap_run)
        ns = argparse.Namespace(instance_id=instance_id, latest=False)
        rc = cli.cmd_logs(ns)
        # After pull, the file exists → print succeeds
        assert rc == 0, f"expected success after git-pull retry, spawned={spawned}"
        # Verify git fetch + git pull actually got invoked
        joined = [" ".join(c) for c in spawned]
        assert any("git" in s and "fetch" in s for s in joined), \
            f"expected git fetch, got: {joined}"
        assert any("git" in s and "pull"  in s for s in joined), \
            f"expected git pull, got: {joined}"
        out = capsys.readouterr().out
        assert f"pulled log for {instance_id}" in out

    def test_destroyed_no_local_no_pulled_log_gives_hint(
        self, fake_log_dir: Path, capsys, monkeypatch
    ):
        """vast.sh fails, git pull doesn't surface the log → user
        gets a helpful message pointing at ``--latest`` and exits
        non-zero.
        """
        from neuroslm import cli
        instance_id = "40999999"

        def _trap_run(argv, *, env=None, cwd=None):
            if "vast.sh" in " ".join(argv):
                return 1
            return 0  # git fetch/pull succeed but produce no new file

        monkeypatch.setattr(cli, "_run", _trap_run)
        ns = argparse.Namespace(instance_id=instance_id, latest=False)
        rc = cli.cmd_logs(ns)
        assert rc != 0
        out = capsys.readouterr().out + capsys.readouterr().err
        assert "--latest" in out, "hint should mention --latest"


# ─────────────────────────────────────────────────────────────────────
# F. Help text mentions --latest
# ─────────────────────────────────────────────────────────────────────


class TestLogsHelpText:
    """The ``brian logs --help`` output must advertise ``--latest``."""

    def test_help_mentions_latest(self, capsys):
        from neuroslm.cli import _build_parser
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["logs", "--help"])
        out = capsys.readouterr().out
        assert "--latest" in out


# ── brian logs <id> -f — follow mode ─────────────────────────────────

class TestNewLogLines:
    """Which lines of a fresh tail snapshot are new since the last."""

    def test_first_snapshot_is_all_new(self):
        from neuroslm.cli import _new_log_lines
        assert _new_log_lines([], ["a", "b"]) == ["a", "b"]

    def test_growing_log_yields_only_the_tail(self):
        from neuroslm.cli import _new_log_lines
        assert _new_log_lines(["a", "b"], ["a", "b", "c", "d"]) == ["c", "d"]

    def test_identical_snapshot_yields_nothing(self):
        from neuroslm.cli import _new_log_lines
        assert _new_log_lines(["a", "b"], ["a", "b"]) == []

    def test_rolled_tail_window(self):
        from neuroslm.cli import _new_log_lines
        prev = [f"l{i}" for i in range(1, 31)]
        cur = [f"l{i}" for i in range(11, 36)]
        assert _new_log_lines(prev, cur) == [f"l{i}" for i in range(31, 36)]

    def test_partially_written_last_line_is_reprinted_complete(self):
        from neuroslm.cli import _new_log_lines
        assert _new_log_lines(["a", "b", "par"],
                              ["a", "b", "partial done", "c"]) \
            == ["partial done", "c"]

    def test_no_overlap_marks_the_gap(self):
        from neuroslm.cli import _new_log_lines
        out = _new_log_lines(["x1", "x2", "x3"], ["y1", "y2"])
        assert "gap" in out[0].lower()
        assert out[1:] == ["y1", "y2"]


class TestFollowLogs:
    def _drive(self, snapshots, pattern=None):
        from neuroslm.cli import _follow_logs
        it = iter(snapshots)
        printed, sleeps = [], []
        rc = _follow_logs("42", interval=5, pattern=pattern,
                          fetch=lambda: next(it), sleep=sleeps.append,
                          out=printed.append)
        return rc, printed, sleeps

    def test_prints_each_line_once_and_stops_when_instance_is_gone(self):
        rc, printed, sleeps = self._drive([
            (0, "a\nb"), (0, "a\nb\nc"), (0, "a\nb\nc"),
            (1, ""), (0, ""), (1, "")])
        assert rc == 0
        body = [l for l in printed if not l.startswith("#")]
        assert body == ["a", "b", "c"]
        assert "no longer" in printed[-1]
        assert sleeps and all(s == 5 for s in sleeps)

    def test_a_transient_miss_does_not_stop_or_reprint(self):
        rc, printed, _ = self._drive([
            (0, "a"), (1, ""), (0, "a\nb"), (1, ""), (1, ""), (1, "")])
        assert [l for l in printed if not l.startswith("#")] == ["a", "b"]

    def test_filter_applies_to_new_lines(self):
        _, printed, _ = self._drive(
            [(0, "WS[R=0.3]\nnoise\nWS[R=0.1]"), (1, ""), (1, ""), (1, "")],
            pattern=r"WS\[")
        assert [l for l in printed if not l.startswith("#")] == \
            ["WS[R=0.3]", "WS[R=0.1]"]

    def test_default_output_survives_a_cp1252_console(self, monkeypatch):
        """Live smoke on Windows: the log's '──', 'γ', 'Φ' crashed
        print() on a cp1252 console. Unencodable chars degrade, the
        follow loop keeps running."""
        import io
        from neuroslm.cli import _follow_logs
        buf = io.BytesIO()
        monkeypatch.setattr(sys, "stdout",
                            io.TextIOWrapper(buf, encoding="cp1252"))
        snaps = iter([(0, "── WS[R=0.3 γ=0.8] Φ"), (1, ""), (1, ""), (1, "")])
        assert _follow_logs("42", interval=0, fetch=lambda: next(snaps),
                            sleep=lambda s: None) == 0
        sys.stdout.flush()
        assert b"WS[R=0.3" in buf.getvalue()

    def test_ctrl_c_exits_cleanly(self):
        from neuroslm.cli import _follow_logs

        def _interrupt(_):
            raise KeyboardInterrupt
        assert _follow_logs("42", interval=1, fetch=lambda: (0, "a"),
                            sleep=_interrupt, out=lambda s: None) == 0


class TestFollowCli:
    def test_flags_parse(self):
        from neuroslm.cli import _build_parser
        a = _build_parser().parse_args(["logs", "123", "-f"])
        assert a.follow is True and a.interval == 10.0 and a.filter is None
        a = _build_parser().parse_args(
            ["logs", "123", "--follow", "--interval", "3", "--filter", "WS"])
        assert a.interval == 3.0 and a.filter == "WS"

    def test_follow_dispatches_without_a_one_shot_dump(self, monkeypatch):
        import neuroslm.cli as cli
        calls = []
        monkeypatch.setattr(cli, "_follow_logs",
                            lambda iid, interval, pattern: calls.append(
                                (iid, interval, pattern)) or 0)
        monkeypatch.setattr(cli, "_run", lambda *a, **k: pytest.fail(
            "follow mode must not run the one-shot dump"))
        a = cli._build_parser().parse_args(["logs", "123", "-f",
                                            "--filter", "WS"])
        assert a.func(a) == 0
        assert calls == [("123", 10.0, "WS")]
