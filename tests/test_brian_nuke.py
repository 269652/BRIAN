"""Contracts for ``brian nuke`` — destroy all running instances.

``brian nuke`` is the emergency kill-switch: it destroys every running
neuroslm-labelled vast.ai instance in a single command.  Like ``brian
deploy``, it requires an interactive TTY and a typed confirmation so that
AI agents cannot invoke it autonomously.

Semantics pinned here:
  * Non-TTY stdin (agent/pipe) -> SystemExit non-zero, nothing destroyed.
  * Wrong confirmation word     -> SystemExit non-zero, nothing destroyed.
  * Correct word at a TTY       -> calls destroy --all and returns its exit code.
  * The confirmation word is "nuke" (distinct from "deploy" so the prompts
    are unambiguous).
"""
from __future__ import annotations

import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# cmd_nuke unit tests
# ──────────────────────────────────────────────────────────────────────


class TestCmdNuke:
    def _run_nuke(self, *, tty: bool, user_input: str = "nuke"):
        from neuroslm import cli as cli_mod
        ns = argparse.Namespace()
        with patch.object(sys.stdin, "isatty", return_value=tty), \
             patch("builtins.input", return_value=user_input), \
             patch.object(cli_mod, "cmd_destroy", return_value=0) as mock_destroy:
            rc = cli_mod.cmd_nuke(ns)
        return rc, mock_destroy

    def test_nuke_exists(self):
        """brian nuke must be a callable in the CLI module."""
        from neuroslm import cli as cli_mod
        assert callable(getattr(cli_mod, "cmd_nuke", None)), (
            "neuroslm.cli must expose cmd_nuke"
        )

    def test_nuke_blocks_non_tty(self):
        """Non-TTY stdin must be rejected — agents cannot nuke."""
        with pytest.raises(SystemExit) as exc:
            self._run_nuke(tty=False)
        assert exc.value.code != 0

    def test_nuke_blocks_wrong_word(self):
        """Any input other than 'nuke' must abort."""
        with pytest.raises(SystemExit) as exc:
            self._run_nuke(tty=True, user_input="yes")
        assert exc.value.code != 0

    def test_nuke_blocks_on_deploy_word(self):
        """Typing 'deploy' (the deploy gate word) must also be rejected."""
        with pytest.raises(SystemExit) as exc:
            self._run_nuke(tty=True, user_input="deploy")
        assert exc.value.code != 0

    def test_nuke_blocks_empty(self):
        """Empty input (just Enter) must abort."""
        with pytest.raises(SystemExit) as exc:
            self._run_nuke(tty=True, user_input="")
        assert exc.value.code != 0

    def test_nuke_calls_destroy_all_on_confirm(self):
        """Correct word at a TTY must call cmd_destroy --all."""
        rc, mock_destroy = self._run_nuke(tty=True, user_input="nuke")
        assert mock_destroy.called, "cmd_nuke must call cmd_destroy"
        called_ns = mock_destroy.call_args[0][0]
        assert getattr(called_ns, "all", False) is True, (
            "cmd_nuke must pass all=True to cmd_destroy"
        )

    def test_nuke_returns_destroy_exit_code(self):
        """cmd_nuke forwards the exit code from cmd_destroy."""
        from neuroslm import cli as cli_mod
        ns = argparse.Namespace()
        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch("builtins.input", return_value="nuke"), \
             patch.object(cli_mod, "cmd_destroy", return_value=42):
            rc = cli_mod.cmd_nuke(ns)
        assert rc == 42

    def test_nuke_stderr_explains_tty_rejection(self, capsys):
        """The non-TTY error must be self-explanatory."""
        from neuroslm import cli as cli_mod
        ns = argparse.Namespace()
        with patch.object(sys.stdin, "isatty", return_value=False):
            with pytest.raises(SystemExit):
                cli_mod.cmd_nuke(ns)
        err = capsys.readouterr().err
        assert any(w in err.lower() for w in ("tty", "terminal", "interactive")), (
            f"non-TTY rejection must mention terminal/interactive; got:\n{err}"
        )


# ──────────────────────────────────────────────────────────────────────
# Parser surface — ``brian nuke`` must be a registered subcommand
# ──────────────────────────────────────────────────────────────────────


class TestNukeParser:
    def test_nuke_subcommand_registered(self):
        """``brian nuke`` must parse without error."""
        from neuroslm.cli import _build_parser
        parser = _build_parser()
        ns = parser.parse_args(["nuke"])
        assert ns.func.__name__ == "cmd_nuke", (
            "nuke subcommand must dispatch to cmd_nuke"
        )

    def test_nuke_appears_in_help(self):
        """``brian --help`` must list the nuke subcommand."""
        from neuroslm.cli import _build_parser
        parser = _build_parser()
        helptext = parser.format_help()
        assert "nuke" in helptext, (
            f"top-level --help must mention nuke; got:\n{helptext}"
        )
