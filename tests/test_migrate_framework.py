"""Tests for the `brian migrate` framework.

Contract under test:

  1. Discovery: `discover_migrations()` finds every `NNNN_<slug>.py`
     module under `neuroslm/migrations/` (excludes `_framework.py`,
     `__init__.py`, anything starting with `_`), sorted by ID.
  2. Protocol: each migration module exposes `ID: str`,
     `DESCRIPTION: str`, `plan(ctx) -> list[Op]`, `apply(ctx, ops) -> int`.
  3. Ledger: `Ledger.applied()` returns a dict keyed by migration ID;
     `Ledger.record(id, commit, ops_applied)` persists to
     `.brian/migrations.json`.
  4. plan() is ALWAYS run, even for applied migrations, so drift
     (applied + non-empty plan) is detectable.
  5. Status taxonomy:
       APPLIED        — in ledger AND plan() returns [] ops
       PENDING        — not in ledger
       DRIFT          — in ledger AND plan() returns ops
       NOOP_PENDING   — not in ledger AND plan() returns [] ops
  6. Dry-run (default) writes nothing — no ledger, no filesystem ops.
  7. `--force` executes apply() and records to the ledger.
  8. `--rerun` re-applies even if in ledger (escape hatch).
  9. `--list` shows every discovered migration with its status.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Minimal repo root with a .git/ marker so REPO_ROOT logic works."""
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def fake_migrations_dir(tmp_path: Path) -> Path:
    """Isolated migrations directory we can populate per-test."""
    d = tmp_path / "migrations_fake"
    d.mkdir()
    (d / "__init__.py").write_text("", encoding="utf-8")
    return d


# ── 1. Op / Context dataclasses ────────────────────────────────────────


def test_op_is_frozen_dataclass() -> None:
    from neuroslm.migrations._framework import Op
    op = Op(kind="copy", src=Path("a"), dst=Path("b"), note="hi")
    with pytest.raises(Exception):
        op.kind = "delete"  # frozen ⇒ FrozenInstanceError


def test_context_holds_root_refs_dry_run_force(fake_repo: Path) -> None:
    from neuroslm.migrations._framework import Context
    from neuroslm.references import ReferenceIndex
    ctx = Context(root=fake_repo, refs=ReferenceIndex(),
                  dry_run=True, force=False)
    assert ctx.root == fake_repo
    assert ctx.dry_run is True
    assert ctx.force is False


# ── 2. Discovery ───────────────────────────────────────────────────────


def test_discover_migrations_sorted_by_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discover walks the migrations dir, ignores underscore files."""
    from neuroslm.migrations import _framework as fw

    # Build a fake migrations package on disk
    pkg = tmp_path / "mig_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "_framework.py").write_text("# skip me\n", encoding="utf-8")
    (pkg / "0002_b.py").write_text(
        'ID = "0002_b"\nDESCRIPTION = "b"\n'
        'def plan(ctx): return []\n'
        'def apply(ctx, ops): return 0\n', encoding="utf-8")
    (pkg / "0001_a.py").write_text(
        'ID = "0001_a"\nDESCRIPTION = "a"\n'
        'def plan(ctx): return []\n'
        'def apply(ctx, ops): return 0\n', encoding="utf-8")
    (pkg / "_helper.py").write_text("# skip me too\n", encoding="utf-8")

    # Make the fake package importable as `mig_pkg`
    monkeypatch.syspath_prepend(str(tmp_path))

    mods = fw.discover_migrations(pkg)
    ids = [m.ID for m in mods]
    assert ids == ["0001_a", "0002_b"], (
        "discover should sort by ID and exclude underscore-prefixed files"
    )


def test_discover_skips_non_python_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neuroslm.migrations import _framework as fw
    pkg = tmp_path / "mig_pkg2"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "0001_real.py").write_text(
        'ID = "0001_real"\nDESCRIPTION = ""\n'
        'def plan(ctx): return []\n'
        'def apply(ctx, ops): return 0\n', encoding="utf-8")
    (pkg / "0002_nope.md").write_text("not a migration", encoding="utf-8")
    (pkg / "0003_nope.txt").write_text("not a migration", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    mods = fw.discover_migrations(pkg)
    assert [m.ID for m in mods] == ["0001_real"]


# ── 3. Ledger ──────────────────────────────────────────────────────────


def test_ledger_empty_when_file_missing(fake_repo: Path) -> None:
    from neuroslm.migrations._framework import Ledger
    led = Ledger(fake_repo)
    assert led.applied() == {}


def test_ledger_records_and_reads_back(fake_repo: Path) -> None:
    from neuroslm.migrations._framework import Ledger
    led = Ledger(fake_repo)
    led.record(mig_id="0001_test", commit="abc123", ops_applied=5)
    led2 = Ledger(fake_repo)  # fresh instance must see persisted data
    applied = led2.applied()
    assert "0001_test" in applied
    assert applied["0001_test"]["commit"] == "abc123"
    assert applied["0001_test"]["ops_applied"] == 5
    assert "applied_at_utc" in applied["0001_test"]
    assert "host" in applied["0001_test"]


def test_ledger_file_path_is_dot_brian(fake_repo: Path) -> None:
    from neuroslm.migrations._framework import Ledger
    led = Ledger(fake_repo)
    led.record(mig_id="0001_test", commit="abc", ops_applied=0)
    assert (fake_repo / ".brian" / "migrations.json").exists()


def test_ledger_round_trip_preserves_multiple_entries(fake_repo: Path) -> None:
    from neuroslm.migrations._framework import Ledger
    led = Ledger(fake_repo)
    led.record(mig_id="0001_a", commit="aaa", ops_applied=1)
    led.record(mig_id="0002_b", commit="bbb", ops_applied=2)
    led.record(mig_id="0003_c", commit="ccc", ops_applied=3)
    applied = Ledger(fake_repo).applied()
    assert set(applied.keys()) == {"0001_a", "0002_b", "0003_c"}
    assert applied["0002_b"]["commit"] == "bbb"


# ── 4. Status taxonomy ─────────────────────────────────────────────────


def _make_migration(
    pkg: Path, mig_id: str, ops_count: int = 0, description: str = ""
) -> None:
    """Helper: write a tiny migration file with plan() returning N noop Ops."""
    src = (
        f'from pathlib import Path\n'
        f'from neuroslm.migrations._framework import Op\n'
        f'ID = "{mig_id}"\n'
        f'DESCRIPTION = "{description}"\n'
        f'def plan(ctx):\n'
        f'    return [Op(kind="noop", src=None, dst=None, note=f"op{{i}}") '
        f'for i in range({ops_count})]\n'
        f'def apply(ctx, ops):\n'
        f'    return len(ops)\n'
    )
    (pkg / f"{mig_id}.py").write_text(src, encoding="utf-8")


def test_status_pending_when_not_in_ledger(
    fake_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neuroslm.migrations import _framework as fw
    pkg = tmp_path / "mig_pkg_a"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    _make_migration(pkg, "0001_pending", ops_count=3)
    monkeypatch.syspath_prepend(str(tmp_path))

    ctx = fw.Context(
        root=fake_repo,
        refs=__import__("neuroslm.references", fromlist=["ReferenceIndex"]).ReferenceIndex(),
        dry_run=True, force=False,
    )
    statuses = fw.status_all(pkg, ctx)
    assert statuses["0001_pending"].kind == "PENDING"
    assert statuses["0001_pending"].planned_ops == 3


def test_status_applied_when_in_ledger_and_no_ops(
    fake_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neuroslm.migrations import _framework as fw
    from neuroslm.references import ReferenceIndex
    pkg = tmp_path / "mig_pkg_b"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    _make_migration(pkg, "0001_done", ops_count=0)
    monkeypatch.syspath_prepend(str(tmp_path))

    fw.Ledger(fake_repo).record(mig_id="0001_done", commit="x", ops_applied=0)
    ctx = fw.Context(root=fake_repo, refs=ReferenceIndex(),
                     dry_run=True, force=False)
    statuses = fw.status_all(pkg, ctx)
    assert statuses["0001_done"].kind == "APPLIED"


def test_status_drift_when_applied_but_plan_returns_ops(
    fake_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neuroslm.migrations import _framework as fw
    from neuroslm.references import ReferenceIndex
    pkg = tmp_path / "mig_pkg_c"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    _make_migration(pkg, "0001_drift", ops_count=2)
    monkeypatch.syspath_prepend(str(tmp_path))

    fw.Ledger(fake_repo).record(mig_id="0001_drift", commit="y", ops_applied=5)
    ctx = fw.Context(root=fake_repo, refs=ReferenceIndex(),
                     dry_run=True, force=False)
    statuses = fw.status_all(pkg, ctx)
    assert statuses["0001_drift"].kind == "DRIFT"
    assert statuses["0001_drift"].planned_ops == 2


def test_status_noop_pending_when_not_in_ledger_and_zero_ops(
    fake_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neuroslm.migrations import _framework as fw
    from neuroslm.references import ReferenceIndex
    pkg = tmp_path / "mig_pkg_d"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    _make_migration(pkg, "0001_empty", ops_count=0)
    monkeypatch.syspath_prepend(str(tmp_path))

    ctx = fw.Context(root=fake_repo, refs=ReferenceIndex(),
                     dry_run=True, force=False)
    statuses = fw.status_all(pkg, ctx)
    assert statuses["0001_empty"].kind == "NOOP_PENDING"
    assert statuses["0001_empty"].planned_ops == 0


# ── 5. Apply / dry-run semantics ───────────────────────────────────────


def test_dry_run_does_not_record_to_ledger(
    fake_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neuroslm.migrations import _framework as fw
    from neuroslm.references import ReferenceIndex
    pkg = tmp_path / "mig_pkg_e"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    _make_migration(pkg, "0001_x", ops_count=2)
    monkeypatch.syspath_prepend(str(tmp_path))

    ctx = fw.Context(root=fake_repo, refs=ReferenceIndex(),
                     dry_run=True, force=False)
    rc = fw.run_one(pkg, "0001_x", ctx)
    assert rc == 0, "dry-run should exit 0"
    assert fw.Ledger(fake_repo).applied() == {}, (
        "dry-run must not touch the ledger"
    )


def test_force_records_to_ledger(
    fake_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neuroslm.migrations import _framework as fw
    from neuroslm.references import ReferenceIndex
    pkg = tmp_path / "mig_pkg_f"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    _make_migration(pkg, "0001_y", ops_count=3)
    monkeypatch.syspath_prepend(str(tmp_path))

    ctx = fw.Context(root=fake_repo, refs=ReferenceIndex(),
                     dry_run=False, force=True)
    rc = fw.run_one(pkg, "0001_y", ctx)
    assert rc == 0
    applied = fw.Ledger(fake_repo).applied()
    assert "0001_y" in applied
    assert applied["0001_y"]["ops_applied"] == 3


def test_already_applied_returns_0_without_rerun(
    fake_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling run_one --force on an applied migration is a no-op."""
    from neuroslm.migrations import _framework as fw
    from neuroslm.references import ReferenceIndex
    pkg = tmp_path / "mig_pkg_g"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    _make_migration(pkg, "0001_z", ops_count=0)
    monkeypatch.syspath_prepend(str(tmp_path))

    fw.Ledger(fake_repo).record(mig_id="0001_z", commit="q", ops_applied=0)

    ctx = fw.Context(root=fake_repo, refs=ReferenceIndex(),
                     dry_run=False, force=True)
    rc = fw.run_one(pkg, "0001_z", ctx)
    assert rc == 0
    # Ledger entry should be unchanged (still commit="q")
    assert fw.Ledger(fake_repo).applied()["0001_z"]["commit"] == "q"


def test_rerun_overwrites_ledger_entry(
    fake_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--rerun forces re-application even if in ledger."""
    from neuroslm.migrations import _framework as fw
    from neuroslm.references import ReferenceIndex
    pkg = tmp_path / "mig_pkg_h"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    _make_migration(pkg, "0001_w", ops_count=4)
    monkeypatch.syspath_prepend(str(tmp_path))

    fw.Ledger(fake_repo).record(mig_id="0001_w", commit="old", ops_applied=0)
    ctx = fw.Context(root=fake_repo, refs=ReferenceIndex(),
                     dry_run=False, force=True)
    rc = fw.run_one(pkg, "0001_w", ctx, rerun=True)
    assert rc == 0
    applied = fw.Ledger(fake_repo).applied()
    assert applied["0001_w"]["ops_applied"] == 4, (
        "rerun should overwrite the ledger entry with the latest run"
    )
    assert applied["0001_w"]["commit"] != "old"


def test_unknown_migration_id_returns_nonzero(
    fake_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neuroslm.migrations import _framework as fw
    from neuroslm.references import ReferenceIndex
    pkg = tmp_path / "mig_pkg_i"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    ctx = fw.Context(root=fake_repo, refs=ReferenceIndex(),
                     dry_run=True, force=False)
    rc = fw.run_one(pkg, "9999_nope", ctx)
    assert rc != 0


# ── 6. CLI entry point ─────────────────────────────────────────────────


def test_cli_list_shows_pending_and_applied(
    fake_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from neuroslm.migrations import _framework as fw
    from neuroslm.references import ReferenceIndex
    pkg = tmp_path / "mig_pkg_j"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    _make_migration(pkg, "0001_done", ops_count=0)
    _make_migration(pkg, "0002_pend", ops_count=2)
    monkeypatch.syspath_prepend(str(tmp_path))

    fw.Ledger(fake_repo).record(mig_id="0001_done", commit="x", ops_applied=0)

    ctx = fw.Context(root=fake_repo, refs=ReferenceIndex(),
                     dry_run=True, force=False)
    rc = fw.cli_list(pkg, ctx)
    assert rc == 0
    out = capsys.readouterr().out
    assert "0001_done" in out
    assert "0002_pend" in out
    assert "APPLIED" in out
    assert "PENDING" in out
