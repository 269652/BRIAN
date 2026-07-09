"""Contracts for the mandatory human-confirmation gate on ``brian deploy``.

The gate exists to prevent AI agents, CI pipelines, and piped scripts from
accidentally launching paid cloud instances.  It has two layers:

1. **TTY check** — ``sys.stdin.isatty()`` must be True.  A subprocess call
   from an agent, ``echo y | brian deploy``, or any non-interactive context
   fails here.  No flag can bypass this.

2. **Word confirmation** — the user must type the word ``deploy`` verbatim.
   Anything else (Enter, "y", "yes", wrong word) aborts.

Critically: neither ``--no-verify`` nor ``brian deploy yolo`` bypasses the
gate.  Those flags only skip the *pre-deploy hook*, not the safety gate.
"""
from __future__ import annotations

import argparse
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# _require_human_confirmation unit tests
# ──────────────────────────────────────────────────────────────────────


class TestRequireHumanConfirmation:
    def _call(self, isatty: bool, user_input: str = "deploy", platform="vast",
              steps=10_000):
        from neuroslm import cli as cli_mod
        with patch.object(sys.stdin, "isatty", return_value=isatty), \
             patch("builtins.input", return_value=user_input):
            return cli_mod._require_human_confirmation(platform, steps)

    def test_blocks_when_stdin_not_tty(self):
        """Non-TTY stdin must raise SystemExit — piped/agent calls are blocked."""
        from neuroslm import cli as cli_mod
        with patch.object(sys.stdin, "isatty", return_value=False):
            with pytest.raises(SystemExit) as exc:
                cli_mod._require_human_confirmation("vast", 10_000)
        assert exc.value.code != 0, "must exit non-zero when stdin is not a TTY"

    def test_blocks_when_stdin_not_tty_regardless_of_flags(self):
        """The TTY check is not gated by any flag — no escape hatch exists."""
        from neuroslm import cli as cli_mod
        with patch.object(sys.stdin, "isatty", return_value=False), \
             patch("builtins.input", return_value="deploy"):
            with pytest.raises(SystemExit):
                cli_mod._require_human_confirmation("vast", 10_000)

    def test_proceeds_when_tty_and_correct_word(self):
        """Typing 'deploy' at a real TTY must return normally (no exception)."""
        self._call(isatty=True, user_input="deploy")

    def test_blocks_on_wrong_word(self):
        """Any input other than 'deploy' must abort."""
        from neuroslm import cli as cli_mod
        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch("builtins.input", return_value="yes"):
            with pytest.raises(SystemExit) as exc:
                cli_mod._require_human_confirmation("vast", 10_000)
        assert exc.value.code != 0

    def test_blocks_on_empty_input(self):
        """Just pressing Enter (empty string) must abort."""
        from neuroslm import cli as cli_mod
        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch("builtins.input", return_value=""):
            with pytest.raises(SystemExit):
                cli_mod._require_human_confirmation("vast", 10_000)

    def test_blocks_on_yes(self):
        """'yes' is not the confirmation word — must abort."""
        from neuroslm import cli as cli_mod
        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch("builtins.input", return_value="yes"):
            with pytest.raises(SystemExit):
                cli_mod._require_human_confirmation("vast", 10_000)

    def test_blocks_on_eof(self):
        """EOFError (piped empty stdin reaching end) must abort."""
        from neuroslm import cli as cli_mod
        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch("builtins.input", side_effect=EOFError):
            with pytest.raises(SystemExit):
                cli_mod._require_human_confirmation("vast", 10_000)

    def test_blocks_on_ctrl_c(self):
        """KeyboardInterrupt (Ctrl-C) must abort cleanly."""
        from neuroslm import cli as cli_mod
        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch("builtins.input", side_effect=KeyboardInterrupt):
            with pytest.raises(SystemExit):
                cli_mod._require_human_confirmation("vast", 10_000)

    def test_stderr_message_mentions_tty_when_not_tty(self, capsys):
        """The non-TTY rejection must explain why, so the user understands."""
        from neuroslm import cli as cli_mod
        with patch.object(sys.stdin, "isatty", return_value=False):
            with pytest.raises(SystemExit):
                cli_mod._require_human_confirmation("vast", 10_000)
        err = capsys.readouterr().err
        assert "tty" in err.lower() or "terminal" in err.lower() or "interactive" in err.lower(), (
            f"rejection message must mention TTY/terminal/interactive; got:\n{err}"
        )


# ──────────────────────────────────────────────────────────────────────
# cmd_deploy integration — gate is called and cannot be bypassed by flags
# ──────────────────────────────────────────────────────────────────────


def _fake_cfg():
    cfg = MagicMock()
    cfg.default_platform = "vast"
    cfg.default_steps = 0
    cfg.default_branch = None
    cfg.is_dna_mode = False
    cfg.dna = None
    cfg.arch = None
    cfg.default_ood_every = 0
    cfg.default_log_every = 100
    cfg.default_save_every = 1000
    cfg.default_push_every = 1000
    cfg.default_push_backend = "hf"
    cfg.default_hf_repo_id = "moritzroessler/BRIAN"
    cfg.default_push_optimizer = False
    cfg.default_machine = None
    cfg.default_teamspace = None
    return cfg


def _deploy_ns(**overrides):
    defaults = dict(
        arch=None, steps=None, branch=None, scale=None,
        dna=None, label=None, ood=None, no_verify=False,
        resume=None, latest=False, hf_repo=None, hf_prefix=None,
        platform=None, machine=None, teamspace=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCmdDeployGate:
    def _run_deploy(self, ns, *, tty: bool, user_input: str = "deploy"):
        """Run cmd_deploy with full mocking — only TTY and input vary."""
        from neuroslm import cli as cli_mod
        with patch.object(cli_mod, "_run_hook", return_value=0), \
             patch("neuroslm.project_config.load_project_config",
                   return_value=_fake_cfg()), \
             patch("neuroslm.connectors.get_connector") as mock_gc, \
             patch.object(sys.stdin, "isatty", return_value=tty), \
             patch("builtins.input", return_value=user_input):
            mock_gc.return_value.launch.return_value = 0
            return cli_mod.cmd_deploy(ns)

    def test_gate_is_called_by_cmd_deploy(self):
        """cmd_deploy must invoke _require_human_confirmation."""
        from neuroslm import cli as cli_mod
        with patch.object(cli_mod, "_require_human_confirmation") as gate_mock, \
             patch.object(cli_mod, "_run_hook", return_value=0), \
             patch("neuroslm.project_config.load_project_config",
                   return_value=_fake_cfg()), \
             patch("neuroslm.connectors.get_connector") as mock_gc:
            mock_gc.return_value.launch.return_value = 0
            cli_mod.cmd_deploy(_deploy_ns())
        assert gate_mock.called, "_require_human_confirmation must be called by cmd_deploy"

    def test_non_tty_blocks_deploy(self):
        """Non-TTY stdin (agent/pipe) must abort before the connector runs."""
        with pytest.raises(SystemExit) as exc:
            self._run_deploy(_deploy_ns(), tty=False)
        assert exc.value.code != 0

    def test_wrong_word_blocks_deploy(self):
        """Wrong confirmation word must abort before the connector runs."""
        with pytest.raises(SystemExit) as exc:
            self._run_deploy(_deploy_ns(), tty=True, user_input="yes")
        assert exc.value.code != 0

    def test_correct_word_allows_deploy(self):
        """Correct word at a TTY must allow the deploy to proceed."""
        rc = self._run_deploy(_deploy_ns(), tty=True, user_input="deploy")
        assert rc == 0

    def test_no_verify_does_not_bypass_gate(self):
        """--no-verify skips the hook but must NOT skip the safety gate."""
        with pytest.raises(SystemExit):
            self._run_deploy(_deploy_ns(no_verify=True), tty=False)

    def test_yolo_does_not_bypass_gate(self):
        """``brian deploy yolo`` must NOT bypass the safety gate."""
        with pytest.raises(SystemExit):
            self._run_deploy(_deploy_ns(arch="yolo"), tty=False)


# ──────────────────────────────────────────────────────────────────────
# --latest / _resolve_checkpoint_uri — pins cmd_deploy's checkpoint
# resolution (also shared by `discover checkpoint`'s --latest, see
# tests/test_discover_checkpoint.py) so the refactor into a shared helper
# can't silently change behaviour.
# ──────────────────────────────────────────────────────────────────────


class TestCmdDeployLatestResolution:
    def _run(self, ns):
        from neuroslm import cli as cli_mod
        captured = {}

        def _launch(config):
            captured["config"] = config
            return 0

        with patch.object(cli_mod, "_run_hook", return_value=0), \
             patch.object(cli_mod, "_require_human_confirmation", return_value=None), \
             patch("neuroslm.project_config.load_project_config",
                   return_value=_fake_cfg()), \
             patch("neuroslm.connectors.get_connector") as mock_gc:
            mock_gc.return_value.launch.side_effect = _launch
            rc = cli_mod.cmd_deploy(ns)
        return rc, captured.get("config")

    def test_latest_resolves_to_hf_uri_and_sets_resume_from(self):
        from neuroslm import cli as cli_mod
        from neuroslm.hf_checkpoints import CheckpointEntry
        entry = CheckpointEntry(path_in_repo="checkpoints/run-x/step7000.pt", step=7000)
        with patch("neuroslm.hf_checkpoints.find_latest_checkpoint", return_value=entry):
            rc, config = self._run(_deploy_ns(latest=True))
        assert rc == 0
        assert config.resume_from == "hf://moritzroessler/BRIAN/checkpoints/run-x/step7000.pt"

    def test_latest_with_no_checkpoints_found_aborts(self):
        with patch("neuroslm.hf_checkpoints.find_latest_checkpoint", return_value=None):
            rc, config = self._run(_deploy_ns(latest=True))
        assert rc == 1
        assert config is None, "connector must never launch when --latest finds nothing"

    def test_explicit_resume_takes_precedence_over_latest(self):
        from neuroslm.hf_checkpoints import CheckpointEntry
        entry = CheckpointEntry(path_in_repo="checkpoints/run-x/step7000.pt", step=7000)
        with patch("neuroslm.hf_checkpoints.find_latest_checkpoint", return_value=entry) as m:
            rc, config = self._run(_deploy_ns(resume="lfs_checkpoints/step500.pt", latest=True))
        assert rc == 0
        assert config.resume_from == "lfs_checkpoints/step500.pt"
        m.assert_not_called()

    def test_no_resume_no_latest_leaves_resume_from_unset(self):
        rc, config = self._run(_deploy_ns())
        assert rc == 0
        assert not config.resume_from
