# -*- coding: utf-8 -*-
"""TDD: ``brian update-readme`` auto-refreshes the best-run pointer
before rendering so ``${LOG_TAIL:best:N}`` is never stale.

Contract being pinned:

* ``brian update-readme`` (no flags) calls
  ``log_refs.update_best_run_pointer`` BEFORE invoking the renderer
  so a fresh run that just landed under ``logs/`` is picked up on
  the next README regen — no manual ``brian best update`` needed.

* ``brian update-readme --no-best-update`` skips that auto-call.
  Needed by tests and by environments where the workspace's ``logs/``
  is unstable (mid-pull, CI artefacts).

* Failure of ``update_best_run_pointer`` is non-fatal: the renderer
  still runs even if no logs qualify. The ``${LOG_TAIL:best:N}``
  macro itself falls back to ``*(log not available)*``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_update_readme_args(**overrides):
    """Build a Namespace with every flag cmd_update_readme reads."""
    defaults = dict(check=False, no_best_update=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ─────────────────────────────────────────────────────────────────────
# 1. Auto-best-update default behaviour
# ─────────────────────────────────────────────────────────────────────


class TestAutoBestUpdate:
    """Default mode calls update_best_run_pointer before render."""

    def test_update_readme_calls_update_best_run_pointer(self, monkeypatch):
        """Default ``brian update-readme`` should refresh the best
        pointer before rendering."""
        from neuroslm import cli
        called = {"best": 0, "render": 0, "order": []}

        def fake_update_best(root, log_dir=None, metric="gap_ratio"):
            called["best"] += 1
            called["order"].append("best")
            return Path("logs/fake_best.log")

        def fake_render(template_path, metrics_path, output_path=None,
                         **kw):
            called["render"] += 1
            called["order"].append("render")
            return ("rendered", True)

        monkeypatch.setattr(
            "neuroslm.log_refs.update_best_run_pointer", fake_update_best)
        monkeypatch.setattr(
            "neuroslm.readme_renderer_v2.render_readme", fake_render)
        # Also stub out arch exports (depends on architectures/master/arch.neuro)
        monkeypatch.setattr(
            "neuroslm.arch_exports.collect_arch_exports",
            lambda p: {})
        monkeypatch.setattr(
            "neuroslm.arch_exports.write_neuro_exports",
            lambda exp, d: None)

        rc = cli.cmd_update_readme(_make_update_readme_args())
        assert rc == 0
        assert called["best"] == 1, \
            "update_best_run_pointer must be called once by default"
        assert called["render"] == 1
        # Order matters: best update MUST happen first
        assert called["order"] == ["best", "render"], \
            f"best update must precede render; got {called['order']}"

    def test_no_best_update_flag_skips(self, monkeypatch):
        """``--no-best-update`` skips the auto best-pointer refresh."""
        from neuroslm import cli
        called = {"best": 0, "render": 0}

        def fake_update_best(*a, **kw):
            called["best"] += 1
            return None

        def fake_render(*a, **kw):
            called["render"] += 1
            return ("rendered", True)

        monkeypatch.setattr(
            "neuroslm.log_refs.update_best_run_pointer", fake_update_best)
        monkeypatch.setattr(
            "neuroslm.readme_renderer_v2.render_readme", fake_render)
        monkeypatch.setattr(
            "neuroslm.arch_exports.collect_arch_exports",
            lambda p: {})
        monkeypatch.setattr(
            "neuroslm.arch_exports.write_neuro_exports",
            lambda exp, d: None)

        rc = cli.cmd_update_readme(
            _make_update_readme_args(no_best_update=True))
        assert rc == 0
        assert called["best"] == 0, \
            "--no-best-update must skip the best-pointer call"
        assert called["render"] == 1

    def test_best_update_failure_is_non_fatal(self, monkeypatch, capsys):
        """If update_best_run_pointer raises, the render still runs.
        The hook is best-effort, not a gating dependency."""
        from neuroslm import cli

        def boom(*a, **kw):
            raise RuntimeError("logs dir busy")

        called = {"render": 0}
        def fake_render(*a, **kw):
            called["render"] += 1
            return ("rendered", True)

        monkeypatch.setattr(
            "neuroslm.log_refs.update_best_run_pointer", boom)
        monkeypatch.setattr(
            "neuroslm.readme_renderer_v2.render_readme", fake_render)
        monkeypatch.setattr(
            "neuroslm.arch_exports.collect_arch_exports",
            lambda p: {})
        monkeypatch.setattr(
            "neuroslm.arch_exports.write_neuro_exports",
            lambda exp, d: None)

        rc = cli.cmd_update_readme(_make_update_readme_args())
        # Render still runs, exit code is still 0
        assert rc == 0
        assert called["render"] == 1
        # Warning printed
        err = capsys.readouterr().err
        assert "best" in err.lower() or "warning" in err.lower()

    def test_best_update_returning_none_is_silent_success(
            self, monkeypatch):
        """No qualifying logs → ``update_best_run_pointer`` returns
        None. That is not an error — render still proceeds."""
        from neuroslm import cli
        called = {"render": 0}
        monkeypatch.setattr(
            "neuroslm.log_refs.update_best_run_pointer",
            lambda root, log_dir=None, metric="gap_ratio": None)

        def fake_render(*a, **kw):
            called["render"] += 1
            return ("rendered", True)
        monkeypatch.setattr(
            "neuroslm.readme_renderer_v2.render_readme", fake_render)
        monkeypatch.setattr(
            "neuroslm.arch_exports.collect_arch_exports",
            lambda p: {})
        monkeypatch.setattr(
            "neuroslm.arch_exports.write_neuro_exports",
            lambda exp, d: None)

        rc = cli.cmd_update_readme(_make_update_readme_args())
        assert rc == 0
        assert called["render"] == 1


# ─────────────────────────────────────────────────────────────────────
# 2. Parser flag presence
# ─────────────────────────────────────────────────────────────────────


class TestUpdateReadmeParserFlags:

    def test_no_best_update_flag_default_false(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(["update-readme"])
        assert args.no_best_update is False

    def test_no_best_update_can_be_set(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(
            ["update-readme", "--no-best-update"])
        assert args.no_best_update is True

    def test_check_and_no_best_update_compose(self):
        from neuroslm.cli import _build_parser
        args = _build_parser().parse_args(
            ["update-readme", "--check", "--no-best-update"])
        assert args.check is True
        assert args.no_best_update is True


# ─────────────────────────────────────────────────────────────────────
# 3. End-to-end: best pointer makes ${LOG_TAIL:best:N} resolve
# ─────────────────────────────────────────────────────────────────────


class TestLogTailBestResolvesAfterAutoUpdate:
    """Integration: after auto-best-update writes .brian/best_run.ln,
    the LOG_TAIL:best:N macro must resolve to the inlined tail
    (not the ``*(log not available)*`` fallback)."""

    def test_log_tail_best_resolves_with_fresh_pointer(self, tmp_path):
        """Write a fresh best_run.ln + a log file, render a template
        with ${LOG_TAIL:best:3}, and verify the tail is inlined."""
        from neuroslm.readme_renderer import resolve_log_macros
        from neuroslm.log_refs import write_ref, BEST_RUN_LN

        log = tmp_path / "logs" / "best.log"
        log.parent.mkdir(parents=True)
        log.write_text(
            "\n".join(f"line {i}" for i in range(20)),
            encoding="utf-8",
        )
        write_ref(
            tmp_path / BEST_RUN_LN,
            Path("logs/best.log"),
            comment="auto: best run",
        )

        out = resolve_log_macros(
            "TAIL HERE:\n${LOG_TAIL:best:3}\nEND",
            {},
            repo_root=tmp_path,
        )
        # Tail must include the last 3 lines and the GitHub link.
        assert "line 19" in out
        assert "line 18" in out
        assert "line 17" in out
        assert "line 16" not in out  # Strictly the last 3
        assert "[`logs/best.log`](logs/best.log)" in out

    def test_log_tail_best_falls_back_when_no_pointer(self, tmp_path):
        """Missing .brian/best_run.ln → macro renders the graceful
        fallback marker, not a literal ``${LOG_TAIL:...}``."""
        from neuroslm.readme_renderer import resolve_log_macros
        out = resolve_log_macros(
            "${LOG_TAIL:best:3}", {}, repo_root=tmp_path)
        assert "${LOG_TAIL" not in out, \
            "macro must always be substituted, never left literal"
        assert "not available" in out.lower()
