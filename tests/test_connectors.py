# -*- coding: utf-8 -*-
"""TDD contracts for the connector registry and platform dispatch.

Contracts locked here
─────────────────────
  A. get_connector("vast") → VastConnector instance
  B. get_connector("lightning") → LightningConnector instance
  C. get_connector("unknown") → ValueError with available platforms listed
  D. VastConnector.platform_name() == "vast"
  E. LightningConnector.platform_name() == "lightning"
  F. DeployConfig can be constructed with steps only; all other fields optional
  G. VastConnector._build_env() propagates steps + all optional fields
  H. VastConnector._build_env() skips zero/falsy cadence values (no clutter)
  I. VastConnector.launch() calls bash + vast_train.sh subprocess
  J. ProjectConfig.default_platform == "vast" by default
  K. [deploy].platform in brian.toml → cfg.default_platform
  L. BRIAN_DEFAULT_PLATFORM env override → cfg.default_platform
  M. cmd_deploy --platform vast  → VastConnector.launch() called
  N. cmd_deploy --platform lightning → LightningConnector.launch() called
  O. cmd_deploy without --platform → cfg.default_platform connector used
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def capture_subprocess(monkeypatch):
    """Replace subprocess.call so the test never launches vast_train.sh."""
    calls: List[dict] = []

    def _fake_call(args, *, cwd=None, env=None, **kwargs):
        calls.append({"args": list(args), "cwd": cwd, "env": dict(env or {}), **kwargs})
        return 0

    monkeypatch.setattr("subprocess.call", _fake_call)
    return calls


@pytest.fixture
def minimal_toml(tmp_path: Path, monkeypatch):
    """A minimal brian.toml with only [deploy] platform set."""
    def _make(platform: str) -> Path:
        cfg = tmp_path / "brian.toml"
        cfg.write_text(
            f"[deploy]\nplatform = {platform!r}\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        return cfg
    return _make


# ─────────────────────────────────────────────────────────────────────
# A-C: Registry
# ─────────────────────────────────────────────────────────────────────


def test_A_get_connector_vast():
    from neuroslm.connectors import get_connector
    from neuroslm.connectors.vast import VastConnector
    c = get_connector("vast")
    assert isinstance(c, VastConnector)


def test_B_get_connector_lightning():
    from neuroslm.connectors import get_connector
    from neuroslm.connectors.lightning import LightningConnector
    c = get_connector("lightning")
    assert isinstance(c, LightningConnector)


def test_C_get_connector_unknown_raises():
    from neuroslm.connectors import get_connector
    with pytest.raises(ValueError, match="Unknown platform"):
        get_connector("nonexistent")


def test_C_error_message_lists_available():
    from neuroslm.connectors import get_connector
    with pytest.raises(ValueError) as exc_info:
        get_connector("bogus")
    msg = str(exc_info.value)
    assert "vast" in msg
    assert "lightning" in msg


# ─────────────────────────────────────────────────────────────────────
# D-E: platform_name
# ─────────────────────────────────────────────────────────────────────


def test_D_vast_platform_name():
    from neuroslm.connectors.vast import VastConnector
    assert VastConnector.platform_name() == "vast"


def test_E_lightning_platform_name():
    from neuroslm.connectors.lightning import LightningConnector
    assert LightningConnector.platform_name() == "lightning"


# ─────────────────────────────────────────────────────────────────────
# F: DeployConfig construction
# ─────────────────────────────────────────────────────────────────────


def test_F_deploy_config_minimal():
    from neuroslm.connectors.base import DeployConfig
    cfg = DeployConfig(steps=1000)
    assert cfg.steps == 1000
    assert cfg.branch is None
    assert cfg.arch is None
    assert cfg.ood_every == 0
    assert cfg.extra_env == {}


def test_F_deploy_config_full():
    from neuroslm.connectors.base import DeployConfig
    cfg = DeployConfig(
        steps=5000,
        branch="feature/x",
        arch="architectures/master",
        scale="large",
        label="my-run",
        resume_from="hf://owner/repo/step1000.pt",
        source_dna="dna/evol/arch.dna",
        ood_every=3000,
        log_every=20,
        save_every=500,
        push_every=2500,
        push_backend="hf",
        hf_repo_id="owner/repo",
        push_optimizer=True,
        extra_env={"CUSTOM": "1"},
    )
    assert cfg.steps == 5000
    assert cfg.label == "my-run"
    assert cfg.extra_env == {"CUSTOM": "1"}


# ─────────────────────────────────────────────────────────────────────
# G-H: VastConnector._build_env
# ─────────────────────────────────────────────────────────────────────


def test_G_build_env_propagates_all_fields():
    from neuroslm.connectors.base import DeployConfig
    from neuroslm.connectors.vast import VastConnector

    cfg = DeployConfig(
        steps=7000,
        branch="master",
        arch="architectures/current",
        scale="large",
        label="my-label",
        resume_from="hf://owner/repo/step5000.pt",
        source_dna="dna/evol/arch.dna",
        ood_every=3000,
        log_every=20,
        save_every=500,
        push_every=2500,
        push_backend="hf",
        hf_repo_id="owner/repo",
        extra_env={"CUSTOM_VAR": "42"},
    )
    env = VastConnector()._build_env(cfg)

    assert env["STEPS"] == "7000"
    assert env["BRANCH"] == "master"
    assert env["ARCH"] == "current"  # "architectures/" prefix stripped
    assert env["SCALE"] == "large"
    assert env["LABEL_SUFFIX"] == "my-label"
    assert env["RESUME_FROM"] == "hf://owner/repo/step5000.pt"
    assert env["BRIAN_SOURCE_DNA"] == "dna/evol/arch.dna"
    assert env["OOD_EVERY"] == "3000"
    assert env["LOG_EVERY"] == "20"
    assert env["SAVE_EVERY"] == "500"
    assert env["PUSH_EVERY"] == "2500"
    assert env["CHECKPOINT_PUSH_BACKEND"] == "hf"
    assert env["HF_REPO_ID"] == "owner/repo"
    assert env["CUSTOM_VAR"] == "42"
    assert env["USE_DSL"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_H_build_env_skips_zero_cadence():
    from neuroslm.connectors.base import DeployConfig
    from neuroslm.connectors.vast import VastConnector

    cfg = DeployConfig(steps=1000)  # all cadences default to 0
    env = VastConnector()._build_env(cfg)

    assert "OOD_EVERY" not in env
    assert "LOG_EVERY" not in env
    assert "SAVE_EVERY" not in env
    assert "PUSH_EVERY" not in env


def test_H_build_env_skips_none_fields():
    from neuroslm.connectors.base import DeployConfig
    from neuroslm.connectors.vast import VastConnector

    cfg = DeployConfig(steps=1000)
    env = VastConnector()._build_env(cfg)

    assert "BRANCH" not in env
    assert "ARCH" not in env
    assert "SCALE" not in env
    assert "LABEL_SUFFIX" not in env
    assert "RESUME_FROM" not in env
    assert "BRIAN_SOURCE_DNA" not in env


def test_G2_build_env_strips_architectures_prefix():
    """ARCH must be stripped of the 'architectures/' prefix so that
    vast_train_dsl_loop.sh, which prepends 'architectures/' itself,
    doesn't produce a doubled path like architectures/architectures/SmolLM."""
    from neuroslm.connectors.base import DeployConfig
    from neuroslm.connectors.vast import VastConnector

    # Full path as stored in brian.toml — must be stripped.
    cfg = DeployConfig(steps=100, arch="architectures/SmolLM")
    env = VastConnector()._build_env(cfg)
    assert env["ARCH"] == "SmolLM", (
        "ARCH env var must be 'SmolLM', not 'architectures/SmolLM'. "
        "vast_train_dsl_loop.sh prepends 'architectures/' so passing the full "
        "path produces architectures/architectures/SmolLM which doesn't exist."
    )

    # Bare name (no prefix) — must be preserved as-is.
    cfg2 = DeployConfig(steps=100, arch="SmolLM")
    env2 = VastConnector()._build_env(cfg2)
    assert env2["ARCH"] == "SmolLM"


# ─────────────────────────────────────────────────────────────────────
# I: VastConnector.launch() subprocess call
# ─────────────────────────────────────────────────────────────────────


def test_I_vast_launch_calls_vast_train_sh(capture_subprocess):
    from neuroslm.connectors.base import DeployConfig
    from neuroslm.connectors.vast import VastConnector

    cfg = DeployConfig(steps=500, branch="master")
    rc = VastConnector().launch(cfg)

    assert rc == 0
    assert len(capture_subprocess) == 1
    call = capture_subprocess[0]
    # bash + path-to-vast_train.sh
    assert len(call["args"]) == 2
    assert call["args"][1].endswith("vast_train.sh")
    assert call["env"]["STEPS"] == "500"
    assert call["env"]["USE_DSL"] == "1"


def test_I_vast_launch_cwd_is_repo_root(capture_subprocess):
    from neuroslm.connectors.base import DeployConfig
    from neuroslm.connectors.vast import VastConnector

    VastConnector().launch(DeployConfig(steps=1))
    cwd = Path(capture_subprocess[0]["cwd"])
    assert (cwd / "brian.toml").exists()


def test_I_vast_launch_stdin_devnull(capture_subprocess):
    """launch() must pass stdin=subprocess.DEVNULL to bash.

    On Windows, Python's stdin is a console handle (CONIN$). When bash
    inherits that handle as fd 0, msys2's fork() emulation — used for
    heredoc pipe writers inside Git Bash — behaves incorrectly and the
    ~6 KB ONSTART heredoc deadlocks.  /dev/null (DEVNULL) is a regular
    file descriptor that fork() can duplicate without issue.
    """
    import subprocess as _sp
    from neuroslm.connectors.base import DeployConfig
    from neuroslm.connectors.vast import VastConnector

    VastConnector().launch(DeployConfig(steps=100))
    call = capture_subprocess[0]
    assert call.get("stdin") is _sp.DEVNULL, (
        "VastConnector.launch() must pass stdin=subprocess.DEVNULL to prevent "
        "the Windows Git Bash console-handle bug that deadlocks heredoc pipes."
    )


# ─────────────────────────────────────────────────────────────────────
# J-L: ProjectConfig.default_platform
# ─────────────────────────────────────────────────────────────────────


def test_J_default_platform_is_vast(tmp_path, monkeypatch):
    """No [deploy] section → default_platform == 'vast'."""
    from neuroslm.project_config import load_project_config
    cfg_file = tmp_path / "brian.toml"
    cfg_file.write_text("[current]\narch = 'architectures/master'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cfg = load_project_config(start=tmp_path)
    assert cfg.default_platform == "vast"


def test_K_toml_deploy_section(tmp_path, monkeypatch):
    """[deploy] platform = 'lightning' → cfg.default_platform == 'lightning'."""
    from neuroslm.project_config import load_project_config
    cfg_file = tmp_path / "brian.toml"
    cfg_file.write_text(
        "[current]\narch = 'architectures/master'\n\n[deploy]\nplatform = 'lightning'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    cfg = load_project_config(start=tmp_path)
    assert cfg.default_platform == "lightning"


def test_L_env_override_platform(tmp_path, monkeypatch):
    """BRIAN_DEFAULT_PLATFORM env → cfg.default_platform."""
    from neuroslm.project_config import load_project_config
    cfg_file = tmp_path / "brian.toml"
    cfg_file.write_text(
        "[current]\narch = 'architectures/master'\n\n[deploy]\nplatform = 'vast'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BRIAN_DEFAULT_PLATFORM", "lightning")
    cfg = load_project_config(start=tmp_path)
    assert cfg.default_platform == "lightning"


# ─────────────────────────────────────────────────────────────────────
# M-O: CLI --platform dispatch
# ─────────────────────────────────────────────────────────────────────


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        arch=None, steps=None, branch=None, scale=None, dna=None,
        label=None, ood=None, resume=None, latest=False, hf_repo=None,
        hf_prefix=None, no_verify=True, platform=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.fixture
def patch_connectors(monkeypatch):
    """Patch get_connector to track which platform was requested."""
    launched = []

    class _FakeConnector:
        def __init__(self, name):
            self._name = name

        def launch(self, config):
            launched.append({"platform": self._name, "config": config})
            return 0

    def _fake_get_connector(platform: str):
        return _FakeConnector(platform)

    monkeypatch.setattr("neuroslm.connectors.get_connector", _fake_get_connector)
    return launched


@pytest.fixture
def minimal_project_config(tmp_path, monkeypatch):
    """A minimal project with brian.toml so load_project_config() works."""
    cfg_file = tmp_path / "brian.toml"
    cfg_file.write_text(
        "[current]\narch = 'architectures/master'\n\n[deploy]\nplatform = 'vast'\n",
        encoding="utf-8",
    )
    (tmp_path / "architectures" / "master").mkdir(parents=True)
    (tmp_path / "architectures" / "master" / "arch.neuro").write_text(
        "architecture test { d_sem: 64 }\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_M_platform_flag_vast(patch_connectors, minimal_project_config, monkeypatch):
    """--platform vast → VastConnector.launch() called."""
    from neuroslm.cli import cmd_deploy
    args = _make_args(platform="vast", steps=100)
    rc = cmd_deploy(args)
    assert rc == 0
    assert patch_connectors[0]["platform"] == "vast"


def test_N_platform_flag_lightning(patch_connectors, minimal_project_config, monkeypatch):
    """--platform lightning → LightningConnector.launch() called."""
    from neuroslm.cli import cmd_deploy
    args = _make_args(platform="lightning", steps=100)
    rc = cmd_deploy(args)
    assert rc == 0
    assert patch_connectors[0]["platform"] == "lightning"


def test_P_clone_url_tokenised_in_python_not_shell():
    """_run_setup_and_train passes a pre-tokenised HTTPS URL to _build_setup_command.

    The old approach used sed-in-shell to inject GITHUB_PAT, which breaks when:
    - the PAT has shell-special chars (|, newline, etc.)
    - GITHUB_PAT is empty → sed produces 'x-access-token:@...' which libcurl
      rejects as "Malformed input to a URL function" (CURLE_URL_MALFORMAT)

    Fix: tokenise in Python using urllib.parse.quote; pass via shlex.quote.
    The setup script should NOT contain the sed snippet at all.
    """
    from neuroslm.connectors.lightning import LightningConnector

    PAT = "ghp_abc123XYZ"
    setup = LightningConnector._build_setup_command(
        f"https://x-access-token:{PAT}@github.com/owner/repo.git",
        "master",
        "~/logs/test.log",
    )
    # The token must appear pre-baked in the script, not assembled via sed
    assert PAT in setup, "PAT must be embedded in the setup script before SSH"
    # The fragile sed substitution must be gone
    assert "sed" not in setup or "x-access-token" not in setup.split("sed")[0], (
        "sed should not be used to inject GITHUB_PAT (shell quoting is fragile)"
    )
    # No bare 'x-access-token:@' (empty-PAT sentinel that triggers curl bug)
    assert "x-access-token:@" not in setup


def test_P2_clone_url_no_pat_stays_plain():
    """Without GITHUB_PAT the URL is passed plain — no empty-token injection."""
    from neuroslm.connectors.lightning import LightningConnector

    setup = LightningConnector._build_setup_command(
        "https://github.com/owner/repo.git",
        "master",
        "~/logs/test.log",
    )
    assert "x-access-token" not in setup
    assert "x-access-token:@" not in setup


def test_Q_list_jobs_uses_ssh_not_sdk_when_ssh_target_stored(tmp_path, monkeypatch):
    """list_jobs() must not call Studio() (SDK) when ssh_target is in extra.

    When running in pure-SSH mode (LIGHTNING_SSH_TARGET set), job records
    store ssh_target + ssh_key in extra. list_jobs() should use SSH to check
    whether the training process is still running, not trigger a browser
    login via the SDK.
    """
    import json
    from neuroslm.connectors.lightning import LightningConnector

    # Create a fake SSH key so the key-exists check passes
    fake_key = tmp_path / "fake_rsa"
    fake_key.write_text("FAKE KEY")

    # Write a fake job record with ssh_target in extra
    jobs_dir = tmp_path / ".brian" / "jobs"
    jobs_dir.mkdir(parents=True)
    job = {
        "job_id": "ln-test-001",
        "platform": "lightning",
        "status": "running",
        "studio_name": "brian-train",
        "label": "test-run",
        "machine": "T4",
        "teamspace": "(ssh-target)",
        "host": "(ssh-target)",
        "started_at": 1000000,
        "log_path": "~/brian/logs/ln-test-001.log",
        "extra": {
            "ssh_target": "s_test@ssh.lightning.ai",
            "ssh_key": str(fake_key),
        },
    }
    (jobs_dir / "ln-test-001.json").write_text(json.dumps(job))
    # _jobs_dir() uses _REPO_ROOT, not cwd — override via env var
    monkeypatch.setenv("BRIAN_JOBS_DIR", str(jobs_dir))

    sdk_called = []

    def _fake_import_sdk():
        sdk_called.append(True)
        return None, None, None  # SDK unavailable

    monkeypatch.setattr(
        "neuroslm.connectors.lightning._import_lightning_sdk", _fake_import_sdk
    )

    ssh_calls = []

    def _fake_ssh_run(key_path, ssh_target, script, timeout=30):
        ssh_calls.append({"target": ssh_target, "script": script})
        return "RUNNING", 0

    # _ssh_run is a staticmethod — patch as plain function (pytest monkeypatch
    # does NOT prepend self for non-descriptor attributes set on a class).
    monkeypatch.setattr(LightningConnector, "_ssh_run", staticmethod(_fake_ssh_run))

    connector = LightningConnector()
    jobs = connector.list_jobs()

    # Must have used SSH (not SDK) to determine status
    assert ssh_calls, "list_jobs() must use SSH when ssh_target is in extra"
    # SDK must not have been called (would trigger browser login)
    assert not sdk_called, (
        "list_jobs() must NOT call _import_lightning_sdk() when ssh_target is stored"
    )
    assert len(jobs) == 1
    assert jobs[0].job_id == "ln-test-001"
    assert jobs[0].status == "running"


def test_O_no_platform_uses_toml(patch_connectors, tmp_path, monkeypatch):
    """No --platform flag → reads [deploy].platform from brian.toml."""
    cfg_file = tmp_path / "brian.toml"
    cfg_file.write_text(
        "[current]\narch = 'architectures/master'\n\n[deploy]\nplatform = 'lightning'\n",
        encoding="utf-8",
    )
    (tmp_path / "architectures" / "master").mkdir(parents=True)
    (tmp_path / "architectures" / "master" / "arch.neuro").write_text(
        "architecture test { d_sem: 64 }\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    from neuroslm.cli import cmd_deploy
    args = _make_args(platform=None, steps=100)
    rc = cmd_deploy(args)
    assert rc == 0
    assert patch_connectors[0]["platform"] == "lightning"


# ─────────────────────────────────────────────────────────────────────
# R: vast_train.sh source contracts — no-heredoc onstart + deploy chain
# ─────────────────────────────────────────────────────────────────────
#
# Root issue (Windows Git Bash, 2026-06-23): bash's internal heredoc pipe
# writer fills synchronously on Windows.  The ~6 KB ONSTART content exceeds
# the ~4 KB pipe buffer, so the writer blocks and the whole process deadlocks.
# Fix: Python builds and writes the onstart script to a NamedTemporaryFile
# (VastConnector._build_onstart), passes the path as ONSTART_FILE, and
# vast_train.sh reads it line-by-line with `while IFS= read -r` (no pipe).

@pytest.fixture(scope="module")
def vast_train_sh_src() -> str:
    return (REPO_ROOT / "scripts" / "vast_train.sh").read_text(encoding="utf-8")


class TestVastTrainShPipeBufferFix:
    """vast_train.sh must not use bash heredocs to build the onstart script."""

    def test_R1_no_old_pipe_capture_pattern(self, vast_train_sh_src: str):
        """The old ONSTART=\"$(cat <<ONSTART\" pattern must not appear."""
        assert 'ONSTART="$(cat <<ONSTART' not in vast_train_sh_src, (
            "vast_train.sh still uses the pipe-capture heredoc pattern "
            "ONSTART=\"$(cat <<ONSTART…)\". This deadlocks on Windows Git Bash "
            "when the ~6 KB onstart script exceeds the pipe buffer."
        )

    def test_R2_reads_via_while_loop(self, vast_train_sh_src: str):
        """Must read ONSTART_FILE with a while-read loop (no pipe, no heredoc)."""
        assert "while IFS= read -r _onstart_line" in vast_train_sh_src, (
            "vast_train.sh must read the onstart file with "
            "`while IFS= read -r _onstart_line … done < \"$ONSTART_FILE\"` "
            "to avoid bash heredoc / pipe-buffer deadlock on Windows Git Bash."
        )

    def test_R3_uses_onstart_file_env_var(self, vast_train_sh_src: str):
        """Must reference ONSTART_FILE (set by the Python connector)."""
        assert "ONSTART_FILE" in vast_train_sh_src, (
            "vast_train.sh must reference ONSTART_FILE, the path written by "
            "VastConnector._build_onstart() before launching bash."
        )

    def test_R4_unsets_loop_variable(self, vast_train_sh_src: str):
        """Loop variable _onstart_line must be unset after the while loop."""
        assert "unset _onstart_line" in vast_train_sh_src, (
            "vast_train.sh must `unset _onstart_line` after the while-read "
            "loop to avoid leaking the loop variable into the rest of the script."
        )

    def test_R5_trace_sequence_complete(self, vast_train_sh_src: str):
        """All four [stage] trace markers must be present in order."""
        markers = [
            "offer selected",
            "onstart script loaded",
            "calling: vastai create instance",
            "create call starting",
        ]
        positions = []
        for m in markers:
            idx = vast_train_sh_src.find(m)
            assert idx >= 0, f"missing trace marker: {m!r}"
            positions.append(idx)
        assert positions == sorted(positions), (
            "trace markers must appear in order: offer-selected → onstart-loaded "
            "→ calling-create → create-starting"
        )

    def test_R6_bash_syntax_valid(self):
        """vast_train.sh must pass `bash -n` (no syntax errors).

        Uses VastConnector._find_bash() to get the same bash binary that
        the connector uses, so the syntax check matches the deploy environment.
        Skipped when bash is not available on the current platform.
        """
        import subprocess
        from neuroslm.connectors.vast import VastConnector
        bash = VastConnector._find_bash()
        if not bash:
            pytest.skip("bash not found on this platform")
        result = subprocess.run(
            [bash, "-n", str(REPO_ROOT / "scripts" / "vast_train.sh")],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"bash -n vast_train.sh reported syntax errors:\n{result.stderr}"
        )


@pytest.fixture(scope="module")
def onstart_template() -> str:
    """The raw _ONSTART_TEMPLATE string from vast.py (before placeholder sub)."""
    from neuroslm.connectors.vast import _ONSTART_TEMPLATE
    return _ONSTART_TEMPLATE


class TestVastTrainShOnstartContent:
    """The ONSTART content (container-side script) must include all the
    critical sections that make training work end-to-end.

    These tests check _ONSTART_TEMPLATE in vast.py — the content was moved
    from bash <<ONSTART heredoc to Python to fix the pipe-buffer deadlock.
    """

    def test_R7_onstart_clones_repo(self, onstart_template: str):
        assert "git clone" in onstart_template, (
            "ONSTART must clone the repo inside the container."
        )

    def test_R8_onstart_runs_bootstrap(self, onstart_template: str):
        assert "vast_bootstrap.sh" in onstart_template, (
            "ONSTART must run scripts/vast_bootstrap.sh to install deps."
        )

    def test_R9_onstart_runs_training(self, onstart_template: str):
        assert "vast_train_dsl_loop.sh" in onstart_template, (
            "ONSTART must invoke vast_train_dsl_loop.sh for DSL training."
        )

    def test_R10_onstart_has_log_pusher(self, onstart_template: str):
        assert "log_pusher.sh" in onstart_template, (
            "ONSTART must start log_pusher.sh so training progress is "
            "visible from git without SSH-ing into the instance."
        )

    def test_R11_no_ssh_flag_in_create(self, vast_train_sh_src: str):
        """--ssh must NOT appear in the vastai create instance call.

        vast.ai's /.launch spawns an ssh keepalive when --ssh is set.
        The pytorch/pytorch image has no openssh-client, so /.launch
        spins on 'ssh: command not found' forever and onstart-cmd never
        runs (container idle, billed indefinitely).
        """
        import re
        # Find the create instance block and check --ssh is absent
        create_idx = vast_train_sh_src.find("create instance")
        assert create_idx >= 0
        create_block = vast_train_sh_src[create_idx: create_idx + 600]
        assert "--ssh" not in create_block, (
            "`vastai create instance` must NOT use --ssh: the pytorch image "
            "lacks openssh-client, causing /.launch to spin forever and "
            "preventing onstart-cmd from ever running."
        )

    def test_R12_create_has_timeout(self, vast_train_sh_src: str):
        """The vastai create instance call must be wrapped with timeout."""
        assert "timeout 120" in vast_train_sh_src, (
            "vastai create instance must be wrapped with `timeout 120` "
            "so a hung API call exits visibly instead of blocking forever."
        )

    def test_R13_vast_api_key_forwarded_to_container(self, vast_train_sh_src: str):
        """VAST_API_KEY must be forwarded as an env var to the container."""
        assert "VAST_API_KEY=$VAST_API_KEY" in vast_train_sh_src or \
               "VAST_API_KEY=${VAST_API_KEY" in vast_train_sh_src, (
            "The container env (-e VAST_API_KEY=...) must forward VAST_API_KEY "
            "so the container can self-destroy after training."
        )

    def test_R15_onstart_log_pusher_launch_receives_arch(self, onstart_template: str):
        """The log_pusher.sh nohup launch must receive ARCH= so ARCH_NAME in
        log_pusher.sh resolves to the real arch name rather than 'current'."""
        idx = onstart_template.find("log_pusher.sh")
        assert idx >= 0, "onstart template must contain log_pusher.sh launch"
        # Look at the ~400 chars before the nohup line (env var prefix region)
        context = onstart_template[max(0, idx - 400): idx + 100]
        assert "ARCH=" in context, (
            "The nohup bash scripts/log_pusher.sh launch must include ARCH= "
            "so log filenames use the arch name (e.g. SmolLM) not 'current'. "
            "Without ARCH, log_pusher.sh falls back to ARCH_NAME='current'."
        )


class TestBrianTomlPlatform:
    """brian.toml must have platform = 'vast' so `brian deploy` targets
    vast.ai by default."""

    def test_R14_brian_toml_platform_is_vast(self):
        import tomllib
        toml_path = REPO_ROOT / "brian.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        platform = data.get("deploy", {}).get("platform", "")
        assert platform == "vast", (
            f"brian.toml [deploy].platform must be 'vast', got {platform!r}. "
            "Run `brian deploy` to target vast.ai by default."
        )


# ─────────────────────────────────────────────────────────────────────
# S: VastConnector._build_onstart() contracts
#
# The Python-side onstart builder replaces the bash <<ONSTART heredoc
# (which deadlocked on Windows Git Bash due to pipe-buffer overflow).
# These tests pin the substitution logic and launch integration.
# ─────────────────────────────────────────────────────────────────────


class TestBuildOnstart:
    """_build_onstart() substitutes all __PLACEHOLDER__ markers correctly."""

    def _build(self, **env_overrides) -> str:
        from neuroslm.connectors.vast import VastConnector
        env = {
            "GH_TOKEN": "ghp_test_token",
            "HF_TOKEN": "hf_test_token",
            "BRANCH": "master",
            "REPO_URL": "https://github.com/testuser/BRIAN.git",
            "FRESH": "1",
            "ARCH": "architectures/current",
            "SCALE": "large",
            "STEPS": "10000",
            "BATCH": "32",
            "GRAD_ACCUM": "4",
            "SEQ_LEN": "512",
            "D_SEM": "256",
            "OOD_EVERY": "500",
            "SAVE_EVERY": "2000",
            "LOG_EVERY": "20",
            "PRESET": "",
            "USE_DSL": "1",
            "LOG_PUSH_INTERVAL": "300",
        }
        env.update(env_overrides)
        return VastConnector._build_onstart(env)

    def test_S1_github_token_substituted(self):
        script = self._build(GH_TOKEN="ghp_abc123")
        assert "ghp_abc123" in script
        assert "__GH_TOKEN__" not in script

    def test_S2_hf_token_substituted(self):
        script = self._build(HF_TOKEN="hf_xyz789")
        assert "hf_xyz789" in script
        assert "__HF_TOKEN__" not in script

    def test_S3_branch_substituted(self):
        script = self._build(BRANCH="feature/test-branch")
        assert "feature/test-branch" in script
        assert "__BRANCH__" not in script

    def test_S4_repo_slug_derived_from_url(self):
        script = self._build(REPO_URL="https://github.com/myorg/myrepo.git")
        assert "myorg/myrepo" in script
        assert "__REPO_SLUG__" not in script

    def test_S5_steps_substituted(self):
        script = self._build(STEPS="7777")
        assert "7777" in script
        assert "__STEPS__" not in script

    def test_S6_arch_substituted(self):
        script = self._build(ARCH="architectures/SmolLM")
        assert "architectures/SmolLM" in script
        assert "__ARCH__" not in script

    def test_S7_no_placeholders_remain(self):
        """No __PLACEHOLDER__ markers must survive substitution."""
        script = self._build()
        import re
        remaining = re.findall(r"__[A-Z_]+__", script)
        assert not remaining, (
            f"Unsubstituted placeholders remain in onstart script: {remaining}"
        )

    def test_S8_script_starts_with_set_e(self):
        """Container script must start with `set -e` for fail-fast behaviour."""
        script = self._build()
        assert script.startswith("set -e"), (
            "Onstart script must begin with `set -e` so any failing command "
            "aborts the container run visibly."
        )

    def test_S9_fresh_flag_substituted(self):
        script = self._build(FRESH="0")
        assert "__FRESH__" not in script

    def test_S10_default_repo_url_fallback(self):
        """When REPO_URL is absent, falls back to the hardcoded BRIAN.git URL."""
        from neuroslm.connectors.vast import VastConnector
        env = {"GH_TOKEN": "tok", "HF_TOKEN": "hf", "BRANCH": "master"}
        script = VastConnector._build_onstart(env)
        assert "269652/BRIAN" in script
        assert "__REPO_SLUG__" not in script


class TestOnstartFileInLaunchEnv:
    """VastConnector.launch() must set ONSTART_FILE in the subprocess env."""

    def test_S11_onstart_file_in_env(self, capture_subprocess):
        """launch() must pass ONSTART_FILE to the bash subprocess."""
        from neuroslm.connectors.base import DeployConfig
        from neuroslm.connectors.vast import VastConnector

        VastConnector().launch(DeployConfig(steps=100))
        call = capture_subprocess[0]
        assert "ONSTART_FILE" in call["env"], (
            "VastConnector.launch() must set ONSTART_FILE in the env passed to "
            "bash so vast_train.sh can read the onstart script without a heredoc."
        )

    def test_S12_onstart_file_is_path_string(self, capture_subprocess):
        """ONSTART_FILE must be a non-empty string (a file path)."""
        from neuroslm.connectors.base import DeployConfig
        from neuroslm.connectors.vast import VastConnector

        VastConnector().launch(DeployConfig(steps=100))
        onstart_file = capture_subprocess[0]["env"].get("ONSTART_FILE", "")
        assert isinstance(onstart_file, str) and len(onstart_file) > 0, (
            "ONSTART_FILE must be a non-empty file path string."
        )
