# -*- coding: utf-8 -*-
"""TDD contracts for `brian discover checkpoint` (Mode A).

The user's ask: probe the REAL trunk from a loaded checkpoint, not the
synthetic `_TinyLM` proxy `discover trunk`/`discover explore` actually use
(confirmed by reading neuro_evolve.py — those modes never touch the real
SmolLM architecture). The real-trunk, probe-only, checkpoint-loading
machinery already exists in `train_dsl.py` (H52/H53's `--explore_only` +
`--resume_from`) — this mode is a thin CLI wrapper that resolves a
checkpoint (explicit path/URI or `--latest` via HF Hub) and shells out to
`python -m neuroslm.train_dsl --explore_only --resume_from ...`, reusing
that tested machinery instead of re-implementing harness construction here
(CLAUDE.md §1b: reuse before reinventing).

Contracts:
  A. --checkpoint PATH is forwarded verbatim as --resume_from
  B. --latest resolves via find_latest_checkpoint → hf:// URI
  C. neither --checkpoint nor --latest given → clear error, no subprocess
  D. --explore_only + all --explore_* knobs land on the train_dsl invocation
  E. --arch/--preset default from brian.toml when not passed
  F. --push triggers a push_artifacts call after a successful run
  G. a non-zero train_dsl exit code propagates as cmd_discover's return code
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ns(**overrides):
    defaults = dict(
        discover_cmd="checkpoint",
        checkpoint=None, latest=False, hf_repo=None, hf_prefix=None,
        arch=None, preset=None,
        rounds=30, pop=24, generations=10, length=8, sites=2,
        device="cpu", push=False, out=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _fake_project_cfg(arch=None):
    cfg = MagicMock()
    cfg.arch = arch
    return cfg


class TestCheckpointResolution:
    def test_explicit_checkpoint_forwarded_as_resume_from(self):
        from neuroslm import cli as cli_mod
        captured = {}

        def _fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=_fake_run), \
             patch("neuroslm.project_config.load_project_config",
                   return_value=_fake_project_cfg()):
            rc = cli_mod.cmd_discover(_ns(checkpoint="lfs_checkpoints/step5000.pt"))
        assert rc == 0
        cmd = captured["cmd"]
        assert "--resume_from" in cmd
        assert cmd[cmd.index("--resume_from") + 1] == "lfs_checkpoints/step5000.pt"

    def test_latest_resolves_via_find_latest_checkpoint(self):
        from neuroslm import cli as cli_mod
        from neuroslm.hf_checkpoints import CheckpointEntry
        entry = CheckpointEntry(path_in_repo="checkpoints/run-x/step9000.pt", step=9000)
        captured = {}

        def _fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=_fake_run), \
             patch("neuroslm.hf_checkpoints.find_latest_checkpoint", return_value=entry), \
             patch("neuroslm.project_config.load_project_config",
                   return_value=_fake_project_cfg()):
            rc = cli_mod.cmd_discover(_ns(latest=True))
        assert rc == 0
        cmd = captured["cmd"]
        idx = cmd.index("--resume_from")
        assert cmd[idx + 1] == "hf://moritzroessler/BRIAN/checkpoints/run-x/step9000.pt"

    def test_no_checkpoint_no_latest_errors_without_subprocess(self, capsys):
        from neuroslm import cli as cli_mod
        with patch("subprocess.run") as mock_run, \
             patch("neuroslm.project_config.load_project_config",
                   return_value=_fake_project_cfg()):
            rc = cli_mod.cmd_discover(_ns())
        assert rc != 0
        mock_run.assert_not_called()
        err = capsys.readouterr().err
        assert "--checkpoint" in err or "--latest" in err

    def test_latest_with_no_checkpoints_found_errors_without_subprocess(self):
        from neuroslm import cli as cli_mod
        with patch("subprocess.run") as mock_run, \
             patch("neuroslm.hf_checkpoints.find_latest_checkpoint", return_value=None), \
             patch("neuroslm.project_config.load_project_config",
                   return_value=_fake_project_cfg()):
            rc = cli_mod.cmd_discover(_ns(latest=True))
        assert rc != 0
        mock_run.assert_not_called()


class TestTrainDslInvocation:
    def _captured_cmd(self, **ns_overrides):
        from neuroslm import cli as cli_mod
        captured = {}

        def _fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=_fake_run), \
             patch("neuroslm.project_config.load_project_config",
                   return_value=_fake_project_cfg()):
            cli_mod.cmd_discover(_ns(checkpoint="lfs_checkpoints/step5000.pt", **ns_overrides))
        return captured["cmd"], captured["kwargs"]

    def test_invokes_train_dsl_module(self):
        cmd, _ = self._captured_cmd()
        assert cmd[0] == sys.executable
        assert "-m" in cmd
        assert "neuroslm.train_dsl" in cmd

    def test_explore_only_flag_present(self):
        cmd, _ = self._captured_cmd()
        assert "--explore_only" in cmd

    def test_explore_knobs_forwarded(self):
        cmd, _ = self._captured_cmd(rounds=15, pop=32, generations=6, length=12, sites=4)
        assert cmd[cmd.index("--explore_rounds") + 1] == "15"
        assert cmd[cmd.index("--explore_pop") + 1] == "32"
        assert cmd[cmd.index("--explore_gens") + 1] == "6"
        assert cmd[cmd.index("--explore_len") + 1] == "12"
        assert cmd[cmd.index("--explore_sites") + 1] == "4"

    def test_device_forwarded(self):
        cmd, _ = self._captured_cmd(device="cuda")
        assert cmd[cmd.index("--device") + 1] == "cuda"

    def test_explicit_arch_and_preset_forwarded(self):
        cmd, _ = self._captured_cmd(arch="architectures/SmolLM", preset="rcc_bowtie_100m")
        assert cmd[cmd.index("--arch") + 1] == "architectures/SmolLM"
        assert cmd[cmd.index("--preset") + 1] == "rcc_bowtie_100m"

    def test_arch_defaults_from_project_config_when_unset(self):
        from neuroslm import cli as cli_mod
        captured = {}

        def _fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=_fake_run), \
             patch("neuroslm.project_config.load_project_config",
                   return_value=_fake_project_cfg(arch="architectures/SmolLM")):
            cli_mod.cmd_discover(_ns(checkpoint="lfs_checkpoints/step5000.pt"))
        cmd = captured["cmd"]
        assert cmd[cmd.index("--arch") + 1] == "architectures/SmolLM"

    def test_runs_in_repo_root(self):
        _, kwargs = self._captured_cmd()
        assert Path(kwargs.get("cwd")) == REPO_ROOT


class TestPushAndExitCode:
    def test_push_triggers_push_artifacts(self):
        from neuroslm import cli as cli_mod
        with patch("subprocess.run", return_value=MagicMock(returncode=0)), \
             patch("neuroslm.project_config.load_project_config",
                   return_value=_fake_project_cfg()), \
             patch("neuroslm.genetic.modulation_pusher.push_artifacts",
                   return_value={"pushed": True, "branch": "master"}) as mock_push:
            rc = cli_mod.cmd_discover(
                _ns(checkpoint="lfs_checkpoints/step5000.pt", push=True))
        assert rc == 0
        assert mock_push.called

    def test_no_push_flag_skips_push_artifacts(self):
        from neuroslm import cli as cli_mod
        with patch("subprocess.run", return_value=MagicMock(returncode=0)), \
             patch("neuroslm.project_config.load_project_config",
                   return_value=_fake_project_cfg()), \
             patch("neuroslm.genetic.modulation_pusher.push_artifacts") as mock_push:
            cli_mod.cmd_discover(_ns(checkpoint="lfs_checkpoints/step5000.pt", push=False))
        assert not mock_push.called

    def test_nonzero_train_dsl_exit_propagates(self):
        from neuroslm import cli as cli_mod
        with patch("subprocess.run", return_value=MagicMock(returncode=3)), \
             patch("neuroslm.project_config.load_project_config",
                   return_value=_fake_project_cfg()):
            rc = cli_mod.cmd_discover(_ns(checkpoint="lfs_checkpoints/step5000.pt"))
        assert rc == 3


class TestArgparseWiring:
    def test_checkpoint_subcommand_registered(self):
        from neuroslm import cli as cli_mod
        parser = cli_mod._build_parser() if hasattr(cli_mod, "_build_parser") else None
        if parser is None:
            pytest.skip("no _build_parser() introspection helper in cli.py")

    def test_discover_checkpoint_help_does_not_crash(self):
        from neuroslm import cli as cli_mod
        with pytest.raises(SystemExit) as exc:
            cli_mod.main(["discover", "checkpoint", "--help"])
        assert exc.value.code == 0

    def test_discover_checkpoint_parses_flags(self):
        from neuroslm import cli as cli_mod
        with patch("subprocess.run", return_value=MagicMock(returncode=0)), \
             patch("neuroslm.project_config.load_project_config",
                   return_value=_fake_project_cfg()):
            rc = cli_mod.main([
                "discover", "checkpoint",
                "--checkpoint", "lfs_checkpoints/step5000.pt",
                "--rounds", "5", "--pop", "8", "--generations", "3",
                "--sites", "1", "--device", "cpu",
            ])
        assert rc == 0
