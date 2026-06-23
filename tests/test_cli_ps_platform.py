# -*- coding: utf-8 -*-
"""TDD contracts for the new `brian ps --platform/--logs` + `brian stop` flow.

Contracts locked here
─────────────────────
  P1. `brian ps --platform lightning` against an empty registry shows fallback
      message and exits 0 (no section header, no table).
  P2. `brian ps --platform lightning` lists registered running Lightning jobs
      in the unified table with "l" platform code.
  P3. `brian ps --platform vast` skips the Lightning section entirely.
  P4. `brian ps --logs <job_id>` dispatches to the right connector's
      tail_logs() and returns 0.
  P5. `brian ps --logs <unknown_id>` returns 1 with an error message.
  P6. `brian stop <job_id>` dispatches to the connector's stop().
  P7. `brian stop <unknown_id>` returns 1 with an error message.

  Q1. No active instances across all platforms → single fallback message, no
      table and no per-platform section headers.
  Q2. Active instances present → unified table with a "P" column (platform
      code: v/l/c).
  Q3. `--all` flag shows stopped/completed instances that would otherwise
      be filtered from the table.
  Q4. Stopped Lightning job hidden by default, shown with --all.
  Q5. Legend line "v=vast.ai  l=lightning" appears after the table when rows
      are rendered.
  Q6. Fallback message when no active instances includes "--all" hint.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_ps_args(**overrides):
    """Construct a Namespace shaped like `brian ps` produces."""
    defaults = dict(
        all=False, it=False, interval=1.0, colab=None,
        platform="all", logs=None, tail=200,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_info(**overrides):
    from neuroslm.connectors import JobInfo, JobStatus
    base = dict(
        job_id="ln-test-001",
        platform="lightning",
        label="ps-test",
        status=JobStatus.RUNNING.value,
        machine="T4",
        branch="master",
        arch="architectures/SmolLM",
        steps=10000,
        studio_name="brian-ps-test",
        teamspace="model-experimentation-project",
        host="moritzroessler",
        log_path="~/brian/logs/ln-test-001.log",
    )
    base.update(overrides)
    return JobInfo(**base)


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    target = tmp_path / "jobs"
    monkeypatch.setenv("BRIAN_JOBS_DIR", str(target))
    return target


@pytest.fixture
def stub_vast(monkeypatch):
    """Make _vastai_exe + _run_capture inert so the test doesn't hit vastai."""
    monkeypatch.setattr("neuroslm.cli._vastai_exe", lambda: "vastai")
    monkeypatch.setattr(
        "neuroslm.cli._run_capture",
        lambda cmd: ("(offline)", 1),
    )
    return None


# ─────────────────────────────────────────────────────────────────────
# P1-P3: --platform filter
# ─────────────────────────────────────────────────────────────────────


def test_P1_ps_platform_lightning_empty_registry(jobs_dir, stub_vast):
    from neuroslm.cli import cmd_ps
    args = _make_ps_args(platform="lightning")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_ps(args)
    out = buf.getvalue()
    assert rc == 0
    # Unified format: no per-platform section headers
    assert "── Lightning AI ──" not in out
    assert "── vast.ai ──" not in out
    # No active instances → fallback message shown
    assert "no active" in out.lower() or "No active" in out


def test_P2_ps_platform_lightning_lists_registered(jobs_dir, stub_vast,
                                                    monkeypatch):
    from neuroslm.connectors import register_job
    from neuroslm.cli import cmd_ps

    register_job(_make_info(job_id="ln-alpha"))
    register_job(_make_info(job_id="ln-beta", studio_name="brian-beta"))

    # Force the SDK to look uninstalled so list_jobs returns disk records
    # untouched (avoids any network call during the unit test).
    monkeypatch.setattr(
        "neuroslm.connectors.lightning._import_lightning_sdk",
        lambda: (None, None, None),
    )

    args = _make_ps_args(platform="lightning")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_ps(args)
    out = buf.getvalue()
    assert rc == 0
    assert "ln-alpha" in out
    assert "ln-beta" in out
    assert "T4" in out
    assert "ps-test" in out  # label


def test_P3_ps_platform_vast_skips_lightning_section(jobs_dir, stub_vast,
                                                      monkeypatch):
    from neuroslm.connectors import register_job
    from neuroslm.cli import cmd_ps

    # Even if a Lightning job is registered, --platform vast must NOT render it
    register_job(_make_info(job_id="ln-hidden"))

    args = _make_ps_args(platform="vast")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_ps(args)
    out = buf.getvalue()
    assert rc == 0
    # Unified format: no per-platform section headers
    assert "── vast.ai ──" not in out
    assert "── Lightning AI ──" not in out
    # Lightning job must never appear in vast-only view
    assert "ln-hidden" not in out


# ─────────────────────────────────────────────────────────────────────
# P4-P5: --logs <job_id>
# ─────────────────────────────────────────────────────────────────────


def test_P4_ps_logs_dispatches_to_connector(jobs_dir, monkeypatch):
    from neuroslm.connectors import register_job
    from neuroslm.cli import cmd_ps

    register_job(_make_info(job_id="ln-with-logs"))

    fake_connector = MagicMock()
    fake_connector.tail_logs.return_value = (
        "[train] step=100 ppl=42.5\n[train] step=200 ppl=41.2\n"
    )

    def _fake_get(platform):
        return fake_connector

    monkeypatch.setattr("neuroslm.connectors.get_connector", _fake_get)

    args = _make_ps_args(logs="ln-with-logs", tail=50)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_ps(args)
    out = buf.getvalue()
    assert rc == 0
    # tail_logs called with the right job_id and tail count
    fake_connector.tail_logs.assert_called_once_with("ln-with-logs", n=50)
    assert "step=200 ppl=41.2" in out
    # Header line includes the platform + studio name
    assert "lightning/ln-with-logs" in out


def test_P5_ps_logs_unknown_job_returns_1(jobs_dir):
    from neuroslm.cli import cmd_ps

    args = _make_ps_args(logs="never-existed")
    err = io.StringIO()
    with redirect_stderr(err):
        rc = cmd_ps(args)
    assert rc == 1
    assert "no job 'never-existed'" in err.getvalue()


def test_P5_ps_logs_connector_error_returns_1(jobs_dir, monkeypatch):
    from neuroslm.connectors import register_job
    from neuroslm.cli import cmd_ps

    register_job(_make_info(job_id="ln-error"))

    fake_connector = MagicMock()
    fake_connector.tail_logs.side_effect = RuntimeError("ssh tunnel dead")

    monkeypatch.setattr("neuroslm.connectors.get_connector",
                        lambda _p: fake_connector)

    args = _make_ps_args(logs="ln-error")
    err = io.StringIO()
    with redirect_stderr(err):
        rc = cmd_ps(args)
    assert rc == 1
    assert "ssh tunnel dead" in err.getvalue()


# ─────────────────────────────────────────────────────────────────────
# P6-P7: `brian stop`
# ─────────────────────────────────────────────────────────────────────


def test_P6_stop_dispatches_to_connector(jobs_dir, monkeypatch):
    from neuroslm.connectors import register_job
    from neuroslm.cli import cmd_stop

    register_job(_make_info(job_id="ln-stoppable"))

    fake_connector = MagicMock()
    fake_connector.stop.return_value = 0

    monkeypatch.setattr("neuroslm.connectors.get_connector",
                        lambda _p: fake_connector)

    args = argparse.Namespace(job_id="ln-stoppable")
    rc = cmd_stop(args)
    assert rc == 0
    fake_connector.stop.assert_called_once_with("ln-stoppable")


def test_P7_stop_unknown_job_returns_1(jobs_dir):
    from neuroslm.cli import cmd_stop
    args = argparse.Namespace(job_id="never-existed")
    err = io.StringIO()
    with redirect_stderr(err):
        rc = cmd_stop(args)
    assert rc == 1
    assert "no job 'never-existed'" in err.getvalue()


def test_P7_stop_notimplemented_connector_returns_1(jobs_dir, monkeypatch):
    from neuroslm.connectors import register_job
    from neuroslm.cli import cmd_stop

    register_job(_make_info(job_id="vt-vast", platform="vast"))

    fake_connector = MagicMock()
    fake_connector.stop.side_effect = NotImplementedError

    monkeypatch.setattr("neuroslm.connectors.get_connector",
                        lambda _p: fake_connector)

    args = argparse.Namespace(job_id="vt-vast")
    err = io.StringIO()
    with redirect_stderr(err):
        rc = cmd_stop(args)
    assert rc == 1
    assert "does not support stop" in err.getvalue()
    assert "brian destroy" in err.getvalue()


# ─────────────────────────────────────────────────────────────────────
# Q: Unified table + active-only filtering contracts
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def stub_lightning_no_jobs(monkeypatch):
    """Lightning connector returns no jobs (empty registry)."""
    monkeypatch.setattr(
        "neuroslm.cli._collect_lightning_rows",
        lambda args: [],
    )


@pytest.fixture
def stub_lightning_stopped(monkeypatch):
    """Lightning connector returns one stopped job."""
    from neuroslm.connectors import JobStatus
    monkeypatch.setattr(
        "neuroslm.cli._collect_lightning_rows",
        lambda args: [{
            "plat": "l", "id": "ln-stopped", "label": "old-run",
            "gpu": "T4", "cost": None, "uptime_mins": 100,
            "phase": JobStatus.STOPPED.value, "is_active": False,
            "_step_s": "-", "_ppl_s": "-", "_ood_s": "-", "_tps_s": "-",
        }],
    )


@pytest.fixture
def stub_lightning_running(monkeypatch):
    """Lightning connector returns one running job."""
    monkeypatch.setattr(
        "neuroslm.cli._collect_lightning_rows",
        lambda args: [{
            "plat": "l", "id": "ln-running", "label": "active-run",
            "gpu": "T4", "cost": None, "uptime_mins": 30,
            "phase": "running", "is_active": True,
            "_step_s": "5000", "_ppl_s": "42.1", "_ood_s": "-", "_tps_s": "12k",
        }],
    )


def test_Q1_no_active_instances_shows_fallback(
        jobs_dir, stub_vast, stub_lightning_no_jobs):
    """No active instances on any platform → fallback message, no table."""
    from neuroslm.cli import cmd_ps
    args = _make_ps_args(platform="all")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_ps(args)
    out = buf.getvalue()
    assert rc == 0
    # No table header
    assert "─────" not in out or "No active" in out
    # Fallback message
    assert "no active" in out.lower()
    # No per-platform section headers
    assert "── Lightning AI ──" not in out
    assert "── vast.ai ──" not in out


def test_Q2_active_rows_render_unified_table_with_plat_column(
        jobs_dir, stub_vast, stub_lightning_running):
    """Active instances → one table with 'P' header column, no section headers."""
    from neuroslm.cli import cmd_ps
    args = _make_ps_args(platform="all")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_ps(args)
    out = buf.getvalue()
    assert rc == 0
    # Table header has a platform column
    assert "P " in out or "  P  " in out or "P\n" in out or "P " in out
    # Job appears
    assert "ln-running" in out
    assert "active-run" in out
    # No section headers
    assert "── Lightning AI ──" not in out
    assert "── vast.ai ──" not in out


def test_Q3_stopped_lightning_hidden_by_default(
        jobs_dir, stub_vast, stub_lightning_stopped):
    """Stopped lightning job must not appear in the table unless --all."""
    from neuroslm.cli import cmd_ps
    args = _make_ps_args(platform="all", all=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_ps(args)
    out = buf.getvalue()
    assert "ln-stopped" not in out
    assert "old-run" not in out


def test_Q4_all_flag_shows_stopped_instances(
        jobs_dir, stub_vast, stub_lightning_stopped):
    """--all reveals stopped lightning instances that would otherwise be hidden."""
    from neuroslm.cli import cmd_ps
    args = _make_ps_args(platform="all", all=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_ps(args)
    out = buf.getvalue()
    assert "ln-stopped" in out
    assert "old-run" in out


def test_Q5_legend_appears_after_table(
        jobs_dir, stub_vast, stub_lightning_running):
    """Legend line appears at the bottom of the table when rows are shown."""
    from neuroslm.cli import cmd_ps
    args = _make_ps_args(platform="all")
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_ps(args)
    out = buf.getvalue()
    # Legend present
    assert "v=vast" in out
    assert "l=lightning" in out
    # Legend after the table rows
    table_pos = out.find("ln-running")
    legend_pos = out.find("v=vast")
    assert table_pos >= 0
    assert legend_pos > table_pos


def test_Q6_fallback_includes_all_hint(
        jobs_dir, stub_vast, stub_lightning_no_jobs):
    """Fallback message when no active instances includes the --all hint."""
    from neuroslm.cli import cmd_ps
    args = _make_ps_args(platform="all", all=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_ps(args)
    out = buf.getvalue()
    assert "no active" in out.lower()
    assert "--all" in out
