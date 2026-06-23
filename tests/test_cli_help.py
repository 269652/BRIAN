# -*- coding: utf-8 -*-
"""Tests for the brian help / tease / cite CLI commands (neuroslm.cli_help)."""
from __future__ import annotations

import pytest
from pathlib import Path
from neuroslm.cli_help import (
    RunEntry,
    parse_runs_ledger,
    tease_runs,
    tease_log_tail,
    format_runs_terminal,
    format_runs_table_md,
    cite_entry,
)

# ── Sample data ───────────────────────────────────────────────────────────────

SAMPLE_LEDGER = """\
# Run Ledger

Preamble text.

---

## Run: 20260615-175931 · H22 SmolLM2 Upgrade

**Date:** 2026-06-15
**Log:** `logs/20260615/neuroslm-full-dna-arch/175931_0_10000.log`
**Checkpoint:** `hf://moritzroessler/BRIAN/checkpoints/20260615-175931/step10000.pt`
**Metrics:** train_ppl=23.6 · ood_ppl=155.0 · gap_ratio=6.55 · steps=10000

First complete 10k run with new cortex fusion stack. Establishes SmolLM2 baseline.

## Run: 20260616-140627 · H22 Best Combined

**Date:** 2026-06-16
**Log:** `logs/20260616/gpt2/140627_500_7500.log`
**Checkpoint:** `hf://moritzroessler/BRIAN/checkpoints/20260616-140629/step7500.pt`
**Metrics:** train_ppl=22.1 · ood_ppl=148.3 · gap_ratio=6.70 · steps=7500

Best combined-score run. Lower train PPL than the 10k run.
"""


# ── parse_runs_ledger ─────────────────────────────────────────────────────────

def test_parse_empty_string():
    assert parse_runs_ledger("") == []


def test_parse_preamble_only():
    assert parse_runs_ledger("# Run Ledger\n\nSome text without any runs.\n") == []


def test_parse_two_entries():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    assert len(entries) == 2


def test_parse_first_entry_id():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    assert entries[0].run_id == "20260615-175931"


def test_parse_first_entry_title():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    assert "H22 SmolLM2 Upgrade" in entries[0].title


def test_parse_first_entry_date():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    assert entries[0].date == "2026-06-15"


def test_parse_first_entry_metrics():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    e = entries[0]
    assert e.train_ppl == pytest.approx(23.6)
    assert e.ood_ppl == pytest.approx(155.0)
    assert e.gap_ratio == pytest.approx(6.55)
    assert e.steps == 10000


def test_parse_first_entry_log_path():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    assert "175931_0_10000.log" in entries[0].log_path


def test_parse_first_entry_checkpoint():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    assert entries[0].checkpoint.startswith("hf://")


def test_parse_first_entry_significance():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    assert "First complete 10k" in entries[0].significance


def test_parse_second_entry_id():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    assert entries[1].run_id == "20260616-140627"


def test_parse_second_entry_steps():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    assert entries[1].steps == 7500


def test_parse_preserves_order():
    """Entries appear in document order, oldest first."""
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    assert entries[0].date < entries[1].date


# ── tease_runs ────────────────────────────────────────────────────────────────

def _make_entries(n: int) -> list[RunEntry]:
    return [
        RunEntry(
            run_id=f"run-{i:03}",
            title=f"Run {i}",
            date=f"2026-06-{i+1:02}",
            log_path="",
            checkpoint="",
            significance="sig",
            gap_ratio=float(i),
            ood_ppl=100.0,
            train_ppl=20.0,
            steps=1000,
        )
        for i in range(n)
    ]


def test_tease_runs_limit():
    entries = _make_entries(10)
    teased = tease_runs(entries, n=3)
    assert len(teased) == 3


def test_tease_runs_returns_last_n():
    """Most recent = last in document = last in list."""
    entries = _make_entries(10)
    teased = tease_runs(entries, n=3)
    assert teased[-1].run_id == "run-009"
    assert teased[0].run_id == "run-007"


def test_tease_runs_fewer_than_n():
    entries = _make_entries(2)
    assert len(tease_runs(entries, n=5)) == 2


def test_tease_runs_empty():
    assert tease_runs([], n=5) == []


# ── tease_log_tail ────────────────────────────────────────────────────────────

def test_tease_log_tail_basic(tmp_path: Path):
    log = tmp_path / "train.log"
    lines = [f"step {i}: loss=0.{i:03}\n" for i in range(50)]
    log.write_text("".join(lines))
    tail = tease_log_tail(log, n=5)
    assert "step 49" in tail
    assert "step 0" not in tail


def test_tease_log_tail_line_count(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text("".join(f"line {i}\n" for i in range(20)))
    tail = tease_log_tail(log, n=5)
    assert len(tail.strip().splitlines()) <= 5


def test_tease_log_tail_missing_file(tmp_path: Path):
    out = tease_log_tail(tmp_path / "nonexistent.log", n=5)
    assert out == "" or "not found" in out.lower()


def test_tease_log_tail_short_file(tmp_path: Path):
    log = tmp_path / "short.log"
    log.write_text("only two lines\nsecond line\n")
    tail = tease_log_tail(log, n=10)
    assert "only two lines" in tail
    assert "second line" in tail


# ── format_runs_terminal ──────────────────────────────────────────────────────

def test_format_runs_terminal_contains_id():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    out = format_runs_terminal(entries[:1])
    assert "20260615-175931" in out


def test_format_runs_terminal_contains_metrics():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    out = format_runs_terminal(entries)
    # gap_ratio should appear
    assert "6.55" in out or "6.70" in out


def test_format_runs_terminal_empty():
    out = format_runs_terminal([])
    assert isinstance(out, str)


# ── format_runs_table_md ──────────────────────────────────────────────────────

def test_format_runs_table_md_has_table_syntax():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    out = format_runs_table_md(entries)
    assert "|" in out


def test_format_runs_table_md_contains_run_id():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    out = format_runs_table_md(entries)
    assert "20260615-175931" in out


def test_format_runs_table_md_empty():
    out = format_runs_table_md([])
    assert isinstance(out, str)


# ── cite_entry ────────────────────────────────────────────────────────────────

def test_cite_entry_contains_id():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    citation = cite_entry(entries[0])
    assert "20260615-175931" in citation


def test_cite_entry_contains_metrics():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    citation = cite_entry(entries[0])
    # At least one metric should appear in the citation
    assert "23.6" in citation or "155.0" in citation or "6.55" in citation


def test_cite_entry_contains_checkpoint():
    entries = parse_runs_ledger(SAMPLE_LEDGER)
    citation = cite_entry(entries[0])
    assert "hf://" in citation
