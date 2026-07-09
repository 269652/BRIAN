# -*- coding: utf-8 -*-
"""TDD contracts for deploying `brian discover <mode>` runs to vast.ai.

Distinct from the training connector (`neuroslm.connectors.vast`): discover
jobs are shorter, mode-driven searches (not arch/scale/steps training), and
need a mode-agnostic background pusher so logs + modulations + the search
ledger reach git WHILE the run is in progress, not just at the end. Only
`experts`/`trunk`/`explore` are deployable — the other discover modes finish
in seconds/minutes on the free local Colab GPU already.

Contracts:
  A. DiscoverDeployConfig rejects non-deployable modes
  B. build_discover_onstart() substitutes every placeholder, none survive
  C. discover_args are shell-quoted and appear verbatim in the command line
  D. the mode-agnostic background pusher block is present, sleeping the
     configured push_interval, scoped to logs/modulations/ledger/heatmaps
  E. --push is always appended to the remote discover invocation
  F. self-destroy block present, keyed to the discover label (not training's)
  G. VastDiscoverConnector.launch() calls bash + vast_discover.sh, with
     ONSTART_FILE set in the subprocess env
  H. cmd_deploy_discover calls the human-confirmation gate BEFORE any
     connector/subprocess action — the same anti-agent gate `brian deploy`
     uses, not a weaker bespoke one
  I. cmd_deploy_discover rejects a non-deployable mode with a clear message,
     without ever reaching the confirmation gate or a subprocess call
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def capture_subprocess(monkeypatch):
    calls: List[dict] = []

    def _fake_call(args, *, cwd=None, env=None, **kwargs):
        calls.append({"args": list(args), "cwd": cwd, "env": dict(env or {}), **kwargs})
        return 0

    monkeypatch.setattr("subprocess.call", _fake_call)
    return calls


class TestDeployableModes:
    def test_rejects_non_deployable_mode(self):
        from neuroslm.connectors.vast_discover import DiscoverDeployConfig
        with pytest.raises(ValueError, match="not deployable"):
            DiscoverDeployConfig(mode="optimizer")

    def test_accepts_experts_trunk_explore(self):
        from neuroslm.connectors.vast_discover import DiscoverDeployConfig
        for mode in ("experts", "trunk", "explore"):
            DiscoverDeployConfig(mode=mode)   # must not raise


class TestBuildDiscoverOnstart:
    def _build(self, **overrides):
        from neuroslm.connectors.vast_discover import build_discover_onstart
        env = {
            "GH_TOKEN": "ghp_test_token",
            "HF_TOKEN": "hf_test_token",
            "BRANCH": "master",
            "REPO_URL": "https://github.com/269652/BRIAN.git",
            "MODE": "experts",
            "DISCOVER_ARGS": "--models smollm2_360m --rounds 20",
            "PUSH_INTERVAL": "90",
            "LABEL": "neuroslm-discover",
        }
        env.update(overrides)
        return build_discover_onstart(env)

    def test_no_placeholders_remain(self):
        script = self._build()
        remaining = re.findall(r"__[A-Z_]+__", script)
        assert not remaining, f"unsubstituted placeholders: {remaining}"

    def test_github_token_substituted(self):
        script = self._build(GH_TOKEN="ghp_abc123")
        assert "ghp_abc123" in script

    def test_branch_substituted(self):
        script = self._build(BRANCH="feature/discover-test")
        assert "feature/discover-test" in script

    def test_repo_slug_derived_from_url(self):
        script = self._build(REPO_URL="https://github.com/myorg/myrepo.git")
        assert "myorg/myrepo" in script

    def test_mode_and_args_in_the_command_line(self):
        script = self._build(MODE="experts",
                             DISCOVER_ARGS="--models smollm2_360m --rounds 20")
        assert "discover experts --models smollm2_360m --rounds 20" in script

    def test_push_flag_always_appended(self):
        script = self._build()
        assert re.search(r"discover experts .*--push", script)

    def test_push_interval_used_in_sleep(self):
        script = self._build(PUSH_INTERVAL="45")
        assert "sleep '45'" in script or "sleep 45" in script

    def test_background_pusher_scoped_to_discovery_artifacts(self):
        script = self._build()
        assert "modulations" in script
        assert "search_ledger.json" in script
        assert "heatmaps" in script
        # runs in a background subshell (& at end of the loop block)
        assert re.search(r"\)\s*>\s*/workspace/discover_pusher\.log 2>&1 &", script)

    def test_self_destroy_present_and_keyed_to_discover_label(self):
        script = self._build(LABEL="neuroslm-discover")
        assert "vastai destroy instance" in script
        assert "neuroslm-discover" in script
        # must NOT reuse the training label
        assert "neuroslm-full" not in script

    def test_script_starts_with_set_e(self):
        script = self._build()
        assert script.startswith("set -e")

    def test_default_repo_url_fallback(self):
        from neuroslm.connectors.vast_discover import build_discover_onstart
        env = {"GH_TOKEN": "t", "HF_TOKEN": "h", "BRANCH": "master",
               "MODE": "trunk", "DISCOVER_ARGS": "", "PUSH_INTERVAL": "60",
               "LABEL": "neuroslm-discover"}
        script = build_discover_onstart(env)
        assert "269652/BRIAN" in script


class TestConnectorLaunch:
    def test_launch_calls_vast_discover_sh(self, capture_subprocess):
        from neuroslm.connectors.vast_discover import (
            DiscoverDeployConfig, VastDiscoverConnector,
        )
        rc = VastDiscoverConnector().launch(DiscoverDeployConfig(mode="experts"))
        assert rc == 0
        assert len(capture_subprocess) == 1
        call = capture_subprocess[0]
        assert any("vast_discover.sh" in a for a in call["args"])

    def test_onstart_file_set_in_env(self, capture_subprocess):
        from neuroslm.connectors.vast_discover import (
            DiscoverDeployConfig, VastDiscoverConnector,
        )
        VastDiscoverConnector().launch(DiscoverDeployConfig(mode="trunk"))
        assert "ONSTART_FILE" in capture_subprocess[0]["env"]

    def test_stdin_devnull(self, capture_subprocess):
        import subprocess as sp
        from neuroslm.connectors.vast_discover import (
            DiscoverDeployConfig, VastDiscoverConnector,
        )
        VastDiscoverConnector().launch(DiscoverDeployConfig(mode="explore"))
        assert capture_subprocess[0]["stdin"] == sp.DEVNULL

    def test_cwd_is_repo_root(self, capture_subprocess):
        from neuroslm.connectors.vast_discover import (
            DiscoverDeployConfig, VastDiscoverConnector, REPO_ROOT as CONNECTOR_ROOT,
        )
        VastDiscoverConnector().launch(DiscoverDeployConfig(mode="experts"))
        assert Path(capture_subprocess[0]["cwd"]) == CONNECTOR_ROOT

    def test_discover_args_forwarded_shell_quoted(self, monkeypatch):
        from neuroslm.connectors.vast_discover import (
            DiscoverDeployConfig, VastDiscoverConnector,
        )
        captured = {}

        def _fake_call(args, *, cwd=None, env=None, **kwargs):
            # read the onstart file WHILE it still exists — launch()'s
            # finally-block deletes it right after subprocess.call returns
            captured["content"] = Path(env["ONSTART_FILE"]).read_text(encoding="utf-8")
            return 0

        monkeypatch.setattr("subprocess.call", _fake_call)
        cfg = DiscoverDeployConfig(
            mode="experts",
            discover_args=["--models", "smollm2_360m,microsoft/CodeGPT-small-py",
                          "--rounds", "20"])
        VastDiscoverConnector().launch(cfg)
        assert "smollm2_360m,microsoft/CodeGPT-small-py" in captured["content"]
        assert "--rounds 20" in captured["content"]


class TestDiscoverArgsUseTheRentedGpu:
    """A deployed vast.ai instance is a rented GPU — every mode that supports
    --device must actually request it, or the run silently falls back to CPU
    and burns the rental on nothing (seen live: `experts` ran on CPU for 90+
    minutes on a rented A100 because this branch omitted `--device auto`)."""

    def _args(self, mode, **overrides):
        import argparse
        base = dict(
            deploy_discover_mode=mode, models=None, rounds=None,
            batch=None, seq_len=None, pop=None, generations=None,
            length=None, steps=None, seed=None, task=None,
            from_scratch=False, novelty=None, avoid_known=False,
            macros=False, seed_from=None, branch=None, label=None,
            push_interval=None, gpu_query=None,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def _captured_discover_args(self, monkeypatch, mode):
        from neuroslm import cli
        monkeypatch.setattr(cli, "_require_human_confirmation", lambda *a, **kw: None)
        captured = {}

        class _FakeConnector:
            def launch(self, config):
                captured["args"] = config.discover_args
                return 0

        monkeypatch.setattr(
            "neuroslm.connectors.vast_discover.VastDiscoverConnector",
            lambda: _FakeConnector())
        cli.cmd_deploy_discover(self._args(mode))
        return captured["args"]

    def test_experts_requests_auto_device(self, monkeypatch):
        args = self._captured_discover_args(monkeypatch, "experts")
        assert "--device" in args and "auto" in args, (
            f"discover experts must pass --device auto on a vast.ai deploy "
            f"(a rented GPU going unused is a straight cost bug); got {args}")

    def test_trunk_requests_auto_device(self, monkeypatch):
        args = self._captured_discover_args(monkeypatch, "trunk")
        assert "--device" in args and "auto" in args


class TestCliHumanConfirmationGate:
    def test_confirmation_called_before_any_launch(self, monkeypatch):
        from neuroslm import cli
        calls = []
        monkeypatch.setattr(cli, "_require_human_confirmation",
                            lambda *a, **kw: calls.append((a, kw)))

        launched = []

        class _FakeConnector:
            def launch(self, config):
                launched.append(config)
                return 0

        monkeypatch.setattr(
            "neuroslm.connectors.vast_discover.VastDiscoverConnector",
            lambda: _FakeConnector())

        import argparse
        args = argparse.Namespace(
            deploy_discover_mode="experts", models=None, rounds=None,
            batch=None, seq_len=None, pop=None, generations=None,
            length=None, steps=None, seed=None, task=None,
            from_scratch=False, novelty=None, avoid_known=False,
            macros=False, seed_from=None, branch=None, label=None,
            push_interval=None, gpu_query=None,
        )
        cli.cmd_deploy_discover(args)
        assert len(calls) == 1, "human confirmation gate must be called exactly once"
        assert len(launched) == 1, "connector must be invoked after confirmation"

    def test_rejects_non_deployable_mode_before_confirming(self, monkeypatch):
        from neuroslm import cli
        confirm_calls = []
        monkeypatch.setattr(cli, "_require_human_confirmation",
                            lambda *a, **kw: confirm_calls.append((a, kw)))
        import argparse
        args = argparse.Namespace(
            deploy_discover_mode="optimizer", models=None, rounds=None,
            batch=None, seq_len=None, pop=None, generations=None,
            length=None, steps=None, seed=None, task=None,
            from_scratch=False, novelty=None, avoid_known=False,
            macros=False, seed_from=None, branch=None, label=None,
            push_interval=None, gpu_query=None,
        )
        rc = cli.cmd_deploy_discover(args)
        assert rc != 0
        assert confirm_calls == [], (
            "must reject an undeployable mode BEFORE the human-confirmation "
            "gate, not after — no reason to prompt for a mode that can't run")
