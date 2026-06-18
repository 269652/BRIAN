# -*- coding: utf-8 -*-
"""TDD contracts for the new `brian ps --platform/--logs` + `brian stop` flow.

Contracts locked here
─────────────────────
  P1. `brian ps --platform lightning` against an empty registry prints a
      friendly "no Lightning jobs registered" message and exits 0.
  P2. `brian ps --platform lightning` lists registered Lightning jobs.
  P3. `brian ps --platform vast` skips the Lightning section entirely.
  P4. `brian ps --logs <job_id>` dispatches to the right connector's
      tail_logs() and returns 0.
  P5. `brian ps --logs <unknown_id>` returns 1 with an error message.
  P6. `brian stop <job_id>` dispatches to the connector's stop().
  P7. `brian stop <unknown_id>` returns 1 with an error message.
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
    assert "── Lightning AI ──" in out
    assert "no Lightning jobs registered" in out
    # Lightning-only run skips the vast section header
    assert "── vast.ai ──" not in out


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
    assert "── vast.ai ──" in out
    assert "── Lightning AI ──" not in out
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
