# -*- coding: utf-8 -*-
"""TDD contracts for the unified job registry + polling surface.

Contracts locked here
─────────────────────
  R1.  JobInfo round-trips through JSON (to_json/from_json).
  R2.  register_job writes ``.brian/jobs/<job_id>.json`` and sets
       ``started_at`` if missing.
  R3.  load_jobs returns ALL persisted records sorted newest-first.
  R4.  load_jobs(platform=X) filters to platform X only.
  R5.  load_job(job_id) returns the matching record or ``None``.
  R6.  remove_job deletes the on-disk file and returns True; False
       when the file is missing.
  R7.  BaseConnector.list_jobs() default returns this platform's
       registered jobs.
  R8.  BaseConnector.tail_logs() default raises NotImplementedError.
  R9.  VastConnector.list_jobs() returns vast-platform jobs only.
  R10. LightningConnector.list_jobs() returns lightning jobs only
       (gracefully handles SDK / network unreachable).
  R11. all_connectors() returns one instance per registered platform.
  R12. BRIAN_JOBS_DIR env redirects the registry path (tests use it
       to isolate fixture writes from the real workspace).

The Lightning SDK is NOT exercised in any test that touches the
network. Auth / Studio.run* are mocked.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    """Redirect .brian/jobs/ to a tmp dir via BRIAN_JOBS_DIR env."""
    target = tmp_path / "jobs"
    monkeypatch.setenv("BRIAN_JOBS_DIR", str(target))
    return target


def _make_info(**overrides):
    """Construct a JobInfo with sensible defaults for tests."""
    from neuroslm.connectors import JobInfo, JobStatus
    base = dict(
        job_id="ln-20260618-001",
        platform="lightning",
        label="smoke",
        status=JobStatus.RUNNING.value,
        machine="T4",
        branch="master",
        arch="architectures/SmolLM",
        steps=10000,
        studio_name="brian-smoke",
        teamspace="model-experimentation-project",
        host="moritzroessler",
        log_path="~/brian/logs/ln-20260618-001.log",
    )
    base.update(overrides)
    return JobInfo(**base)


# ─────────────────────────────────────────────────────────────────────
# R1: JSON round-trip
# ─────────────────────────────────────────────────────────────────────


def test_R1_jobinfo_roundtrip():
    """to_json + from_json preserves every field."""
    from neuroslm.connectors import JobInfo
    info = _make_info(extra={"sdk": "lightning_sdk", "repo_url": "https://x"})
    data = info.to_json()
    assert isinstance(data, dict)
    again = JobInfo.from_json(data)
    assert again.job_id == info.job_id
    assert again.platform == info.platform
    assert again.label == info.label
    assert again.status == info.status
    assert again.machine == info.machine
    assert again.steps == info.steps
    assert again.studio_name == info.studio_name
    assert again.teamspace == info.teamspace
    assert again.host == info.host
    assert again.log_path == info.log_path
    assert again.extra == {"sdk": "lightning_sdk", "repo_url": "https://x"}


def test_R1_jobinfo_from_json_tolerates_unknown_fields():
    """Future schema fields don't crash older readers."""
    from neuroslm.connectors import JobInfo
    data = {
        "job_id": "x", "platform": "vast", "label": "y",
        "made_up_field": 42, "another_new_one": "ok",
    }
    info = JobInfo.from_json(data)
    assert info.job_id == "x"
    assert info.platform == "vast"
    assert info.label == "y"


# ─────────────────────────────────────────────────────────────────────
# R2: register_job
# ─────────────────────────────────────────────────────────────────────


def test_R2_register_job_writes_file(jobs_dir):
    from neuroslm.connectors import register_job
    info = _make_info(started_at=0)  # force the auto-fill path
    path = register_job(info)
    assert path.is_file()
    assert path.name == "ln-20260618-001.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["job_id"] == "ln-20260618-001"
    assert data["started_at"] > 0  # auto-filled to now


def test_R2_register_job_keeps_explicit_started_at(jobs_dir):
    from neuroslm.connectors import register_job
    info = _make_info(started_at=1234567890)
    register_job(info)
    data = json.loads((jobs_dir / "ln-20260618-001.json").read_text(
        encoding="utf-8"))
    assert data["started_at"] == 1234567890


# ─────────────────────────────────────────────────────────────────────
# R3-R4: load_jobs
# ─────────────────────────────────────────────────────────────────────


def test_R3_load_jobs_empty_dir_returns_empty_list(jobs_dir):
    from neuroslm.connectors import load_jobs
    assert load_jobs() == []


def test_R3_load_jobs_sorts_newest_first(jobs_dir):
    from neuroslm.connectors import load_jobs, register_job
    register_job(_make_info(job_id="ln-old", started_at=1000))
    register_job(_make_info(job_id="ln-new", started_at=2000))
    register_job(_make_info(job_id="ln-mid", started_at=1500))
    jobs = load_jobs()
    assert [j.job_id for j in jobs] == ["ln-new", "ln-mid", "ln-old"]


def test_R4_load_jobs_filters_by_platform(jobs_dir):
    from neuroslm.connectors import load_jobs, register_job
    register_job(_make_info(job_id="ln-a", platform="lightning",
                            started_at=1000))
    register_job(_make_info(job_id="vt-a", platform="vast",
                            started_at=2000))
    register_job(_make_info(job_id="ln-b", platform="lightning",
                            started_at=3000))
    lightning_jobs = load_jobs(platform="lightning")
    vast_jobs = load_jobs(platform="vast")
    assert [j.job_id for j in lightning_jobs] == ["ln-b", "ln-a"]
    assert [j.job_id for j in vast_jobs] == ["vt-a"]


def test_R3_load_jobs_skips_malformed_files(jobs_dir):
    """A corrupt .json shouldn't tank the whole listing."""
    from neuroslm.connectors import load_jobs, register_job
    register_job(_make_info(job_id="good"))
    (jobs_dir / "broken.json").write_text("{ not valid json",
                                          encoding="utf-8")
    jobs = load_jobs()
    assert len(jobs) == 1
    assert jobs[0].job_id == "good"


# ─────────────────────────────────────────────────────────────────────
# R5-R6: load_job / remove_job
# ─────────────────────────────────────────────────────────────────────


def test_R5_load_job_returns_match(jobs_dir):
    from neuroslm.connectors import load_job, register_job
    register_job(_make_info(job_id="ln-find-me"))
    found = load_job("ln-find-me")
    assert found is not None
    assert found.job_id == "ln-find-me"


def test_R5_load_job_returns_none_when_missing(jobs_dir):
    from neuroslm.connectors import load_job
    assert load_job("nonexistent") is None


def test_R6_remove_job_deletes_file(jobs_dir):
    from neuroslm.connectors import load_job, register_job, remove_job
    register_job(_make_info(job_id="ln-doomed"))
    assert load_job("ln-doomed") is not None
    assert remove_job("ln-doomed") is True
    assert load_job("ln-doomed") is None


def test_R6_remove_job_false_when_missing(jobs_dir):
    from neuroslm.connectors import remove_job
    assert remove_job("never-existed") is False


# ─────────────────────────────────────────────────────────────────────
# R7-R8: BaseConnector defaults
# ─────────────────────────────────────────────────────────────────────


def test_R7_base_list_jobs_default_reads_registry(jobs_dir):
    """A subclass that doesn't override list_jobs gets registry-based listing."""
    from neuroslm.connectors import register_job
    from neuroslm.connectors.base import BaseConnector, DeployConfig

    class _FakeConn(BaseConnector):
        @classmethod
        def platform_name(cls):
            return "fake"

        def launch(self, config):
            return 0

    register_job(_make_info(job_id="fake-1", platform="fake"))
    register_job(_make_info(job_id="lightning-1", platform="lightning"))
    jobs = _FakeConn().list_jobs()
    assert [j.job_id for j in jobs] == ["fake-1"]


def test_R8_base_tail_logs_default_raises():
    from neuroslm.connectors.base import BaseConnector

    class _FakeConn(BaseConnector):
        @classmethod
        def platform_name(cls):
            return "fake"

        def launch(self, config):
            return 0

    with pytest.raises(NotImplementedError, match="does not support log tailing"):
        _FakeConn().tail_logs("any-id")


def test_R8_base_stop_default_raises():
    from neuroslm.connectors.base import BaseConnector

    class _FakeConn(BaseConnector):
        @classmethod
        def platform_name(cls):
            return "fake"

        def launch(self, config):
            return 0

    with pytest.raises(NotImplementedError, match="does not support stop"):
        _FakeConn().stop("any-id")


# ─────────────────────────────────────────────────────────────────────
# R9-R10: Vast + Lightning list_jobs
# ─────────────────────────────────────────────────────────────────────


def test_R9_vast_list_jobs_filters(jobs_dir):
    from neuroslm.connectors import register_job
    from neuroslm.connectors.vast import VastConnector
    register_job(_make_info(job_id="ln-a", platform="lightning"))
    register_job(_make_info(job_id="vt-a", platform="vast"))
    jobs = VastConnector().list_jobs()
    assert len(jobs) == 1
    assert jobs[0].job_id == "vt-a"


def test_R10_lightning_list_jobs_returns_registry_when_sdk_missing(
        jobs_dir, monkeypatch):
    """If SDK import fails, the connector falls back to disk-only records."""
    from neuroslm.connectors import register_job
    from neuroslm.connectors.lightning import LightningConnector
    register_job(_make_info(job_id="ln-only", platform="lightning"))

    # Force the SDK import to return None,None,None as if uninstalled.
    monkeypatch.setattr(
        "neuroslm.connectors.lightning._import_lightning_sdk",
        lambda: (None, None, None),
    )
    jobs = LightningConnector().list_jobs()
    assert len(jobs) == 1
    assert jobs[0].job_id == "ln-only"


def test_R10_lightning_list_jobs_swallows_auth_failure(jobs_dir, monkeypatch):
    """When the SDK is present but auth fails, we still return the registry."""
    from neuroslm.connectors import register_job
    from neuroslm.connectors.lightning import LightningConnector
    register_job(_make_info(job_id="ln-stale", platform="lightning",
                            status="running"))

    monkeypatch.setattr(
        "neuroslm.connectors.lightning._import_lightning_sdk",
        lambda: (MagicMock(), MagicMock(), "lightning_sdk"),
    )

    def _explode(*_args, **_kwargs):
        raise RuntimeError("no token in env")

    monkeypatch.setattr(
        "neuroslm.connectors.lightning._get_authed_user_safe",
        _explode,
    )
    jobs = LightningConnector().list_jobs()
    assert len(jobs) == 1
    assert jobs[0].job_id == "ln-stale"
    # status stayed at the on-disk value (no auth → can't refresh)
    assert jobs[0].status == "running"


# ─────────────────────────────────────────────────────────────────────
# R11: all_connectors() returns every registered platform
# ─────────────────────────────────────────────────────────────────────


def test_R11_all_connectors_returns_every_platform():
    from neuroslm.connectors import all_connectors
    instances = all_connectors()
    platforms = sorted(c.platform_name() for c in instances)
    assert platforms == ["lightning", "vast"]


# ─────────────────────────────────────────────────────────────────────
# R12: BRIAN_JOBS_DIR env redirect
# ─────────────────────────────────────────────────────────────────────


def test_R12_jobs_dir_env_redirects(tmp_path, monkeypatch):
    """Setting BRIAN_JOBS_DIR changes where jobs are written."""
    from neuroslm.connectors import register_job, load_jobs
    custom = tmp_path / "elsewhere"
    monkeypatch.setenv("BRIAN_JOBS_DIR", str(custom))
    register_job(_make_info(job_id="ln-elsewhere"))
    assert (custom / "ln-elsewhere.json").is_file()
    # And load_jobs reads from the same redirected path
    jobs = load_jobs()
    assert [j.job_id for j in jobs] == ["ln-elsewhere"]


# ─────────────────────────────────────────────────────────────────────
# Lightning helper smoke tests (no network)
# ─────────────────────────────────────────────────────────────────────


def test_lightning_short_id_format():
    """_short_id returns a deterministic-ish prefix + timestamp + suffix."""
    from neuroslm.connectors.lightning import _short_id
    a = _short_id()
    b = _short_id()
    assert a.startswith("ln-")
    assert b.startswith("ln-")
    # Format ln-YYYYMMDD-HHMMSS-XXXX (XXXX = 2 random bytes hex = 4 chars)
    parts = a.split("-")
    assert len(parts) == 4
    assert len(parts[1]) == 8  # YYYYMMDD
    assert len(parts[2]) == 6  # HHMMSS
    assert len(parts[3]) == 4  # 2 random bytes hex
    # Random suffix means two calls differ
    assert a != b


def test_lightning_build_setup_command_includes_branch_and_pip():
    """Setup script clones the right branch and installs deps."""
    from neuroslm.connectors.lightning import LightningConnector
    cmd = LightningConnector._build_setup_command(
        "https://github.com/owner/repo.git", "feature/x", "~/brian/logs/x.log"
    )
    assert "git clone" in cmd
    assert "feature/x" in cmd
    # Critical: training needs the [ml] extras (torch / transformers /
    # tiktoken / einops). Base ``pip install -e .`` skips them by
    # design (CLI-only install), so the connector MUST install the
    # heavy extras or training crashes with ModuleNotFoundError.
    assert "pip install -e '.[ml]'" in cmd
    assert "requirements.txt" in cmd
    # Verify-imports step aborts with exit 2 when deps are missing
    assert "import torch" in cmd
    assert "transformers" in cmd
    assert "tiktoken" in cmd
    # Token injection branch present
    assert "GITHUB_PAT" in cmd
    assert "x-access-token" in cmd
    # Sets up base + logs dirs
    assert "mkdir" in cmd
    assert "set -e" in cmd


def test_lightning_build_train_command_detaches_and_redirects():
    """Train command is detached and writes to the per-job log file."""
    from neuroslm.connectors.base import DeployConfig
    from neuroslm.connectors.lightning import LightningConnector
    cfg = DeployConfig(steps=5000, log_every=20, save_every=500,
                       push_every=2500, push_backend="hf",
                       arch="architectures/SmolLM")
    log_path = "~/brian/logs/test.log"
    cmd = LightningConnector._build_train_command(cfg, log_path)
    assert "nohup" in cmd
    assert "disown" in cmd
    assert "&" in cmd
    assert "neuroslm.train_dsl" in cmd
    assert "--steps 5000" in cmd
    assert "--log_every 20" in cmd
    assert "--save_every 500" in cmd
    assert "--push_every 2500" in cmd
    assert "--push_backend hf" in cmd
    assert log_path in cmd
    assert "[launch] pid=" in cmd


def test_lightning_build_remote_env_includes_secrets(monkeypatch):
    """Token env vars are forwarded; Lightning-internal vars are stripped."""
    from neuroslm.connectors.base import DeployConfig
    from neuroslm.connectors.lightning import LightningConnector
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.setenv("GITHUB_PAT", "ghp_secret_value")
    cfg = DeployConfig(
        steps=100,
        extra_env={
            "LIGHTNING_MACHINE": "T4",          # should be stripped
            "LIGHTNING_TEAMSPACE": "ts",        # should be stripped
            "LIGHTNING_API_KEY": "should-skip", # should be stripped
            "CUSTOM_TRAIN_FLAG": "1",           # should pass through
        },
    )
    env = LightningConnector._build_remote_env(
        cfg, "https://github.com/x/y.git", "master", "ln-abc"
    )
    # Secrets forwarded
    assert env["HF_TOKEN"] == "hf_secret_value"
    assert env["GITHUB_PAT"] == "ghp_secret_value"
    # Job coordinates
    assert env["BRIAN_JOB_ID"] == "ln-abc"
    assert env["BRIAN_REPO_URL"] == "https://github.com/x/y.git"
    assert env["BRIAN_BRANCH"] == "master"
    # Lightning internals stripped
    assert "LIGHTNING_MACHINE" not in env
    assert "LIGHTNING_TEAMSPACE" not in env
    assert "LIGHTNING_API_KEY" not in env
    # Caller-supplied extras passed through
    assert env["CUSTOM_TRAIN_FLAG"] == "1"


def test_lightning_status_map_covers_sdk_states():
    """Every Lightning SDK status maps to a known JobStatus."""
    from neuroslm.connectors import JobStatus
    from neuroslm.connectors.lightning import _STATUS_MAP
    # The SDK enum (per probe): NotCreated, Pending, Running, Stopping,
    # Stopped, Completed, Failed — every one must round-trip.
    for native in ("NotCreated", "Pending", "Running", "Stopping",
                   "Stopped", "Completed", "Failed"):
        assert native in _STATUS_MAP, f"unmapped SDK status: {native}"
        assert isinstance(_STATUS_MAP[native], JobStatus)


def test_lightning_resolve_teamspace_picks_solo():
    """Single teamspace → auto-picked without ambiguity."""
    from neuroslm.connectors.lightning import _resolve_teamspace_handle
    fake_ts = MagicMock(name="solo-ts")
    fake_ts.name = "model-experimentation-project"
    fake_user = MagicMock()
    fake_user.teamspaces = [fake_ts]
    out = _resolve_teamspace_handle(fake_user, "")
    assert out is fake_ts


def test_lightning_resolve_teamspace_explicit_name_wins():
    from neuroslm.connectors.lightning import _resolve_teamspace_handle
    ts_a = MagicMock(); ts_a.name = "ts-a"
    ts_b = MagicMock(); ts_b.name = "ts-b"
    fake_user = MagicMock()
    fake_user.teamspaces = [ts_a, ts_b]
    out = _resolve_teamspace_handle(fake_user, "ts-b")
    assert out is ts_b


def test_lightning_resolve_teamspace_multi_no_choice_errors():
    """Multiple teamspaces + no name → helpful error with chooser hint."""
    from neuroslm.connectors.lightning import _resolve_teamspace_handle
    ts_a = MagicMock(); ts_a.name = "ts-a"
    ts_b = MagicMock(); ts_b.name = "ts-b"
    fake_user = MagicMock()
    fake_user.name = "alice"
    fake_user.teamspaces = [ts_a, ts_b]
    with pytest.raises(RuntimeError, match="multiple teamspaces"):
        _resolve_teamspace_handle(fake_user, "")


def test_lightning_resolve_teamspace_unknown_name_errors():
    from neuroslm.connectors.lightning import _resolve_teamspace_handle
    ts_a = MagicMock(); ts_a.name = "ts-a"
    fake_user = MagicMock()
    fake_user.name = "alice"
    fake_user.teamspaces = [ts_a]
    with pytest.raises(RuntimeError, match="not found"):
        _resolve_teamspace_handle(fake_user, "wrong-name")
