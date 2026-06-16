"""Tests for the 0002_log_name_refactor migration.

Validates that train.log files are renamed to <HHMMSS>_<start>_<end>.log
format with step range extracted from log content.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from neuroslm.migrations import _framework as fw
from neuroslm.references import build_reference_index

# Import migration module (can't use 'from' syntax due to leading digit)
m002 = importlib.import_module("neuroslm.migrations.0002_log_name_refactor")


def _make_log(folder: Path, name: str, content: str = "step 0 | loss 1.0\n") -> None:
    """Write a log file with optional content."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(content, encoding="utf-8")


def test_parse_folder_extracts_boot_time():
    """Boot time is extracted from folder names like 175931_c19bf629."""
    assert m002._parse_folder_name(Path("175931_c19bf629")) == "175931"
    assert m002._parse_folder_name(Path("092625_7fdc3ccd")) == "092625"
    assert m002._parse_folder_name(Path("unparseable")) is None


def test_parse_log_content_finds_step_range(tmp_path: Path):
    """Step range is extracted from log content."""
    log = tmp_path / "test.log"
    log.write_text(
        "step 0 | loss 5.0\nstep 20 | loss 4.5\nstep 100 | loss 3.2\n",
        encoding="utf-8",
    )
    result = m002._parse_log_content(log)
    assert result == (0, 100)


def test_parse_log_content_handles_single_step(tmp_path: Path):
    """Logs with only one step work correctly."""
    log = tmp_path / "test.log"
    log.write_text("step 5000 | loss 2.5\n", encoding="utf-8")
    result = m002._parse_log_content(log)
    assert result == (5000, 5000)


def test_parse_log_content_returns_none_for_empty(tmp_path: Path):
    """Empty or unparseable logs return None."""
    log = tmp_path / "test.log"
    log.write_text("no steps here\n", encoding="utf-8")
    assert m002._parse_log_content(log) is None


def test_new_log_name_format():
    """New log name includes boot time and step range."""
    parsed = m002.ParsedLog(
        boot_time="175931",
        start_step=0,
        end_step=10000,
        arch_name="neuroslm-full-dna-arch",
        date="20260615",
    )
    assert m002._new_log_name(parsed) == "175931_0_10000.log"


def test_plan_renames_3level_train_log(tmp_path: Path):
    """3-level layout logs/<date>/<arch>/<folder>/train.log are renamed."""
    # Create a log in 3-level layout
    folder = tmp_path / "logs" / "20260615" / "neuroslm-full-dna-arch" / "175931_c19bf629"
    _make_log(
        folder, "train.log",
        "step 0 | loss 5.0\nstep 5000 | loss 3.0\nstep 10000 | loss 2.5\n"
    )
    
    ctx = fw.Context(root=tmp_path, refs=build_reference_index(tmp_path))
    ops = m002.plan(ctx)
    
    assert len(ops) == 1
    assert ops[0].src == folder / "train.log"
    assert ops[0].dst == tmp_path / "logs" / "20260615" / "neuroslm-full-dna-arch" / "175931_0_10000.log"


def test_plan_skips_already_renamed(tmp_path: Path):
    """Logs already in new format are not planned for renaming."""
    folder = tmp_path / "logs" / "20260615" / "neuroslm-full-dna-arch"
    _make_log(folder, "175931_0_10000.log", "step 0 | loss 1.0\n")
    
    ctx = fw.Context(root=tmp_path, refs=build_reference_index(tmp_path))
    ops = m002.plan(ctx)
    
    assert len(ops) == 0  # Already in new format


def test_plan_handles_unparseable_logs(tmp_path: Path):
    """Unparseable logs go to _unsorted_legacy."""
    folder = tmp_path / "logs" / "20260615" / "neuroslm-full-dna-arch" / "badname"
    _make_log(folder, "train.log", "no steps\n")
    
    ctx = fw.Context(root=tmp_path, refs=build_reference_index(tmp_path))
    ops = m002.plan(ctx)
    
    assert len(ops) == 1
    assert ops[0].dst.parent.name == "badname"
    assert "_unsorted_legacy" in str(ops[0].dst)


def test_plan_handles_resumed_runs(tmp_path: Path):
    """Resumed runs (start_step > 0) are named correctly."""
    folder = tmp_path / "logs" / "20260615" / "neuroslm-full-dna-arch" / "184943_c19bf629"
    _make_log(
        folder, "train.log",
        "step 10000 | loss 2.0\nstep 12000 | loss 1.8\nstep 15000 | loss 1.5\n"
    )
    
    ctx = fw.Context(root=tmp_path, refs=build_reference_index(tmp_path))
    ops = m002.plan(ctx)
    
    assert len(ops) == 1
    assert ops[0].dst.name == "184943_10000_15000.log"


def test_apply_copies_and_removes_folder(tmp_path: Path):
    """apply() copies log and removes empty folder."""
    folder = tmp_path / "logs" / "20260615" / "neuroslm-full-dna-arch" / "175931_c19bf629"
    _make_log(folder, "train.log", "step 0 | loss 1.0\nstep 100 | loss 0.5\n")
    
    ctx = fw.Context(root=tmp_path, refs=build_reference_index(tmp_path))
    ops = m002.plan(ctx)
    applied = m002.apply(ctx, ops)
    
    assert applied == 1
    new_log = tmp_path / "logs" / "20260615" / "neuroslm-full-dna-arch" / "175931_0_100.log"
    assert new_log.exists()
    assert "step 0 | loss 1.0" in new_log.read_text()
    # Old folder should be removed if empty
    assert not folder.exists() or not (folder / "train.log").exists()


def test_idempotent_rerun(tmp_path: Path):
    """Re-running migration after apply yields no new ops."""
    folder = tmp_path / "logs" / "20260615" / "neuroslm-full-dna-arch" / "175931_c19bf629"
    _make_log(folder, "train.log", "step 0 | loss 1.0\nstep 100 | loss 0.5\n")
    
    ctx = fw.Context(root=tmp_path, refs=build_reference_index(tmp_path))
    
    # First run
    ops1 = m002.plan(ctx)
    assert len(ops1) == 1
    m002.apply(ctx, ops1)
    
    # Second run
    ops2 = m002.plan(ctx)
    assert len(ops2) == 0  # Idempotent


def test_multiple_logs_same_arch(tmp_path: Path):
    """Multiple runs in the same arch folder are handled correctly."""
    arch_base = tmp_path / "logs" / "20260615" / "neuroslm-full-dna-arch"
    
    _make_log(arch_base / "175931_c19bf629", "train.log", "step 0 | loss 1.0\nstep 10000 | loss 0.5\n")
    _make_log(arch_base / "184943_c19bf629", "train.log", "step 10000 | loss 0.5\nstep 20000 | loss 0.3\n")
    
    ctx = fw.Context(root=tmp_path, refs=build_reference_index(tmp_path))
    ops = m002.plan(ctx)
    
    assert len(ops) == 2
    applied = m002.apply(ctx, ops)
    assert applied == 2
    
    # Check both logs were created
    assert (arch_base / "175931_0_10000.log").exists()
    assert (arch_base / "184943_10000_20000.log").exists()
