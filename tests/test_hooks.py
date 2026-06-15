"""TDD: ``hooks/`` folder + YAML-defined cross-platform hook runner.

User request (verbatim 2026-06-15):

    "Compile new master arch to dna and unfold, or rather make that
    happen automatically when I run `brian deploy`... we need a hooks
    folder with yaml files for each hook which points to a .sh or
    ps1 script... should work cross env..."

Contract:

    A. Hook discovery
       1. `hooks/<event>.yaml` files declare named hook events
       2. The YAML names the OS-specific script(s) to run
       3. `enabled: false` skips silently
       4. Missing folder / missing hook → no-op, return code 0

    B. Hook runner
       1. Picks `scripts.windows` on Windows, `scripts.unix` elsewhere
       2. Streams output to stdout/stderr live
       3. Honours `timeout_seconds` (kills + returns non-zero on TLE)
       4. `fail_on_error: true` propagates non-zero exit upstream
       5. `fail_on_error: false` logs but returns 0

    C. Integration with `brian deploy`
       1. `pre-deploy` runs before any vast.ai call
       2. Failure aborts the deploy (no vast.ai resources consumed)
       3. `post-deploy` runs after the deploy_script call

    D. Shipped `pre-deploy` hook
       1. `hooks/pre-deploy.yaml` exists and is enabled
       2. It points to BOTH `hooks/scripts/pre-deploy.sh` AND
          `hooks/scripts/pre-deploy.ps1`
       3. Both scripts exist
       4. Both scripts compile master arch → DNA (via
          `brian dna compile`) and unfold DNA → current (via
          `brian dna unfold`)
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Section A: Hook discovery + YAML schema ─────────────────────────


class TestHookDiscovery:
    """Hook loading reads ``hooks/<name>.yaml`` files."""

    def test_module_importable(self):
        from neuroslm import hooks
        assert hasattr(hooks, "load_hook")
        assert hasattr(hooks, "run_hook")
        assert hasattr(hooks, "Hook")

    def test_load_hook_returns_none_when_folder_missing(self, tmp_path):
        from neuroslm.hooks import load_hook
        assert load_hook(tmp_path, "pre-deploy") is None

    def test_load_hook_returns_none_when_yaml_missing(self, tmp_path):
        from neuroslm.hooks import load_hook
        (tmp_path / "hooks").mkdir()
        assert load_hook(tmp_path, "pre-deploy") is None

    def test_load_hook_parses_minimal_yaml(self, tmp_path):
        from neuroslm.hooks import load_hook
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "pre-deploy.yaml").write_text(textwrap.dedent("""
            name: pre-deploy
            description: Test hook
            enabled: true
            scripts:
              windows: hooks/scripts/foo.ps1
              unix: hooks/scripts/foo.sh
        """).strip())
        hook = load_hook(tmp_path, "pre-deploy")
        assert hook is not None
        assert hook.name == "pre-deploy"
        assert hook.enabled is True
        assert hook.script_windows.endswith("foo.ps1")
        assert hook.script_unix.endswith("foo.sh")

    def test_load_hook_defaults_enabled_true(self, tmp_path):
        from neuroslm.hooks import load_hook
        (tmp_path / "hooks").mkdir()
        (tmp_path / "hooks" / "x.yaml").write_text("name: x\nscripts: {}\n")
        h = load_hook(tmp_path, "x")
        assert h is not None
        assert h.enabled is True   # default

    def test_load_hook_disabled_explicit(self, tmp_path):
        from neuroslm.hooks import load_hook
        (tmp_path / "hooks").mkdir()
        (tmp_path / "hooks" / "x.yaml").write_text(
            "name: x\nenabled: false\nscripts: {}\n")
        h = load_hook(tmp_path, "x")
        assert h is not None and h.enabled is False


# ── Section B: Hook runner ─────────────────────────────────────────


class TestHookRunner:
    """The runner picks the right script for the OS + respects flags."""

    def _make_simple_hook(self, tmp_path, *, win_cmd="exit 0",
                          unix_cmd="exit 0", enabled=True,
                          fail_on_error=True, timeout=30):
        """Write a hook YAML + two scripts that just exit with given codes."""
        hooks_dir = tmp_path / "hooks"
        scripts_dir = hooks_dir / "scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "x.ps1").write_text(win_cmd)
        (scripts_dir / "x.sh").write_text("#!/usr/bin/env bash\n" + unix_cmd)
        try:
            os.chmod(scripts_dir / "x.sh", 0o755)
        except OSError:
            pass   # Windows
        (hooks_dir / "x.yaml").write_text(textwrap.dedent(f"""
            name: x
            enabled: {str(enabled).lower()}
            fail_on_error: {str(fail_on_error).lower()}
            timeout_seconds: {timeout}
            scripts:
              windows: hooks/scripts/x.ps1
              unix: hooks/scripts/x.sh
        """).strip())

    def test_disabled_hook_returns_zero_without_running(self, tmp_path):
        from neuroslm.hooks import run_hook
        self._make_simple_hook(tmp_path, win_cmd="exit 1",
                               unix_cmd="exit 1", enabled=False)
        # Even though both scripts would fail, disabled hook returns 0
        assert run_hook("x", tmp_path) == 0

    def test_missing_hook_returns_zero(self, tmp_path):
        from neuroslm.hooks import run_hook
        assert run_hook("nonexistent", tmp_path) == 0

    def test_picks_powershell_on_windows(self, tmp_path):
        from neuroslm import hooks as hooks_mod
        self._make_simple_hook(tmp_path)
        with mock.patch.object(hooks_mod, "_is_windows", return_value=True):
            # Use return_value=mock with returncode 0
            with mock.patch.object(hooks_mod, "_run_subprocess",
                                   return_value=0) as run_p:
                rc = hooks_mod.run_hook("x", tmp_path)
        assert rc == 0
        # First positional argv element should be powershell-ish
        call_argv = run_p.call_args[0][0]
        assert any("powershell" in str(a).lower() or "pwsh" in str(a).lower()
                   for a in call_argv), \
            f"expected powershell invocation, got {call_argv!r}"

    def test_picks_bash_on_unix(self, tmp_path):
        from neuroslm import hooks as hooks_mod
        self._make_simple_hook(tmp_path)
        with mock.patch.object(hooks_mod, "_is_windows", return_value=False):
            with mock.patch.object(hooks_mod, "_run_subprocess",
                                   return_value=0) as run_p:
                rc = hooks_mod.run_hook("x", tmp_path)
        assert rc == 0
        call_argv = run_p.call_args[0][0]
        # Either direct script path, bash <script>, or sh <script>
        joined = " ".join(str(a) for a in call_argv)
        assert "x.sh" in joined, f"expected bash script call, got {call_argv!r}"

    def test_fail_on_error_true_propagates_nonzero(self, tmp_path):
        from neuroslm import hooks as hooks_mod
        self._make_simple_hook(tmp_path, fail_on_error=True)
        with mock.patch.object(hooks_mod, "_run_subprocess", return_value=7):
            rc = hooks_mod.run_hook("x", tmp_path)
        assert rc == 7

    def test_fail_on_error_false_swallows_nonzero(self, tmp_path):
        from neuroslm import hooks as hooks_mod
        self._make_simple_hook(tmp_path, fail_on_error=False)
        with mock.patch.object(hooks_mod, "_run_subprocess", return_value=7):
            rc = hooks_mod.run_hook("x", tmp_path)
        assert rc == 0


# ── Section C: brian deploy integration ────────────────────────────


class TestDeployRunsPreHook:
    """``cmd_deploy`` runs ``pre-deploy`` BEFORE any vast.ai call."""

    def _patch_cli(self, monkeypatch, hook_rc=0):
        """Patch out the expensive parts of cmd_deploy + capture order."""
        from neuroslm import cli
        order = []

        def fake_run_hook(name, repo_root=None, env=None):
            order.append(("hook", name))
            return hook_rc

        def fake_deploy_dsl(**kw):
            order.append(("deploy_dsl", kw))
            return 0

        def fake_deploy_dna(**kw):
            order.append(("deploy_dna", kw))
            return 0

        monkeypatch.setattr(cli, "_run_hook", fake_run_hook, raising=False)
        monkeypatch.setattr(cli, "_deploy_dsl", fake_deploy_dsl)
        monkeypatch.setattr(cli, "_deploy_dna", fake_deploy_dna)
        return order

    def test_pre_deploy_hook_runs_before_deploy(self, monkeypatch):
        from neuroslm import cli
        order = self._patch_cli(monkeypatch, hook_rc=0)
        import argparse
        args = argparse.Namespace(
            steps=10, branch=None, dna=None, ood=0, scale=None, label=None,
        )
        rc = cli.cmd_deploy(args)
        assert rc == 0
        # hook came BEFORE the deploy
        assert order[0] == ("hook", "pre-deploy"), \
            f"expected pre-deploy first, got {order}"
        assert any(step[0].startswith("deploy_") for step in order)

    def test_pre_deploy_failure_aborts_deploy(self, monkeypatch):
        from neuroslm import cli
        order = self._patch_cli(monkeypatch, hook_rc=42)
        import argparse
        args = argparse.Namespace(
            steps=10, branch=None, dna=None, ood=0, scale=None, label=None,
        )
        rc = cli.cmd_deploy(args)
        assert rc == 42, "pre-deploy non-zero must propagate"
        # No deploy_* call should have happened
        deploy_calls = [s for s in order if s[0].startswith("deploy_")]
        assert deploy_calls == [], \
            f"pre-deploy failure must abort BEFORE deploy; got {deploy_calls}"


# ── Section D: Shipped pre-deploy hook + scripts ──────────────────


class TestShippedPreDeployHook:
    """The actual ``hooks/pre-deploy.yaml`` + scripts in the repo."""

    HOOKS_DIR = REPO_ROOT / "hooks"
    YAML_PATH = HOOKS_DIR / "pre-deploy.yaml"
    SH_PATH = HOOKS_DIR / "scripts" / "pre-deploy.sh"
    PS1_PATH = HOOKS_DIR / "scripts" / "pre-deploy.ps1"

    def test_yaml_exists(self):
        assert self.YAML_PATH.is_file(), \
            f"missing {self.YAML_PATH} — the pre-deploy hook must ship"

    def test_yaml_parses_with_loader(self):
        from neuroslm.hooks import load_hook
        h = load_hook(REPO_ROOT, "pre-deploy")
        assert h is not None
        assert h.enabled, "shipped pre-deploy must be enabled by default"
        assert h.script_windows, "shipped pre-deploy must set scripts.windows"
        assert h.script_unix, "shipped pre-deploy must set scripts.unix"

    def test_bash_script_exists_and_compiles_master(self):
        assert self.SH_PATH.is_file(), f"missing {self.SH_PATH}"
        body = self.SH_PATH.read_text(encoding="utf-8")
        # Must invoke brian dna compile (or equivalent python module call)
        compile_markers = ("brian dna compile", "neuroslm.cli dna compile",
                           "-m neuroslm.cli dna compile")
        assert any(m in body for m in compile_markers), \
            f"{self.SH_PATH} must call brian dna compile; got: {body[:300]}"
        unfold_markers = ("brian dna unfold", "neuroslm.cli dna unfold",
                          "-m neuroslm.cli dna unfold")
        assert any(m in body for m in unfold_markers), \
            f"{self.SH_PATH} must call brian dna unfold; got: {body[:300]}"

    def test_bash_script_full_pipeline(self):
        """Locks the 5-step pipeline (clean-check → compile → unfold →
        commit → push). Each gate is loadbearing for the user's
        contract: the deploy only proceeds AFTER a successful push of
        the roundtrip artefacts.
        """
        body = self.SH_PATH.read_text(encoding="utf-8")
        # 1. Clean-check
        assert "git status --porcelain" in body, \
            "step 1 missing: must `git status --porcelain` to refuse on dirty tree"
        # 4. Stage + commit with the exact chore message
        assert "git add -A" in body, "step 4 missing: must `git add -A`"
        assert "chore: roundtrip recompile of current architecture" in body, \
            "step 4 missing: must commit with the canonical chore message"
        # 5. Push
        assert "git push" in body, "step 5 missing: must `git push`"

    def test_powershell_script_exists_and_compiles_master(self):
        assert self.PS1_PATH.is_file(), f"missing {self.PS1_PATH}"
        body = self.PS1_PATH.read_text(encoding="utf-8")
        compile_markers = ("brian dna compile", "neuroslm.cli dna compile",
                           "-m neuroslm.cli dna compile")
        assert any(m in body for m in compile_markers), \
            f"{self.PS1_PATH} must call brian dna compile; got: {body[:300]}"
        unfold_markers = ("brian dna unfold", "neuroslm.cli dna unfold",
                          "-m neuroslm.cli dna unfold")
        assert any(m in body for m in unfold_markers), \
            f"{self.PS1_PATH} must call brian dna unfold; got: {body[:300]}"

    def test_powershell_script_full_pipeline(self):
        """PowerShell mirror of test_bash_script_full_pipeline."""
        body = self.PS1_PATH.read_text(encoding="utf-8")
        assert "git status --porcelain" in body, \
            "step 1 missing: must `git status --porcelain` to refuse on dirty tree"
        assert "git add -A" in body, "step 4 missing: must `git add -A`"
        assert "chore: roundtrip recompile of current architecture" in body, \
            "step 4 missing: must commit with the canonical chore message"
        assert "git push" in body, "step 5 missing: must `git push`"
