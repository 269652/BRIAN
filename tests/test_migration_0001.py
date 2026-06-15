"""Tests for migration 0001_logs_to_run_folders.

Contract under test:

  1. Parses the "new" filename format
     `<YYYYMMDD>T<HHMMSS>Z_<sha>_<arch>_step<X>of<Y>.log` -> all fields.
  2. Parses the "legacy" double-underscore format
     `<sha>__<arch>.log` using file mtime for the date.
  3. Filenames matching neither pattern are routed to
     `logs/_unsorted_legacy/<original>`.
  4. Only logs whose basename is referenced anywhere in the repo
     (per `ReferenceIndex.references`) are included in plan().
  5. Output folder format (3-level hierarchy):
       `logs/<YYYYMMDD>/<arch>/<HHMMSS>_<sha>/train.log`
  6. `plan()` is idempotent — if the destination file already exists,
     no Op is emitted.
  7. `apply()` actually copies (not moves) and creates parent dirs.
  8. After apply(), re-running plan() returns an empty list (drift =
     0; status APPLIED).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import time
from pathlib import Path

import pytest


# ── Helpers ────────────────────────────────────────────────────────────


def _load_migration() -> object:
    """Load the migration module by path so we don't depend on package
    import side effects."""
    import sys
    here = Path(__file__).resolve().parent.parent
    mig_path = here / "neuroslm" / "migrations" / "0001_logs_to_run_folders.py"
    assert mig_path.exists(), f"missing migration: {mig_path}"
    spec = importlib.util.spec_from_file_location("mig0001", mig_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # MUST register before exec for @dataclass to work under PY 3.13.
    sys.modules["mig0001"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_log(dirpath: Path, name: str, *, mtime: float | None = None,
              content: str = "log content\n") -> Path:
    """Write a log file, optionally with a specific mtime."""
    dirpath.mkdir(parents=True, exist_ok=True)
    fp = dirpath / name
    fp.write_text(content, encoding="utf-8")
    if mtime is not None:
        os.utime(fp, (mtime, mtime))
    return fp


def _make_ctx(root: Path, *, dry_run: bool = True, force: bool = False):
    """Build a Context against `root` with an index built from `root`."""
    from neuroslm.migrations._framework import Context
    from neuroslm.references import build_reference_index
    refs = build_reference_index(root)
    return Context(root=root, refs=refs, dry_run=dry_run, force=force)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo skeleton with logs/vast/ and a docs/FINDINGS.md."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "logs" / "vast").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    return tmp_path


# ── 1-3. Filename parsing ──────────────────────────────────────────────


def test_parse_new_format_extracts_all_fields() -> None:
    m = _load_migration()
    p = m.parse_log_basename(
        "20260614T182653Z_07aba24be2bf_rcc_bowtie_889M_run_step920of10k.log"
    )
    assert p is not None
    assert p.date_token == "20260614-182653"
    assert p.sha == "07aba24be2bf"
    assert "rcc_bowtie_889M_run" in p.arch
    assert p.source == "new"


def test_parse_new_format_without_step_suffix() -> None:
    m = _load_migration()
    p = m.parse_log_basename("20260614T184807Z_31cf84a0b3c6_arch_1127M.log")
    assert p is not None
    assert p.date_token == "20260614-184807"
    assert p.sha == "31cf84a0b3c6"
    assert p.arch == "arch_1127M"


def test_parse_legacy_format_requires_mtime() -> None:
    m = _load_migration()
    # No mtime -> None (caller routes to _unsorted_legacy)
    assert m.parse_log_basename("101ceb95a960__neuroslm-full.log") is None


def test_parse_legacy_format_with_mtime() -> None:
    m = _load_migration()
    # 2026-01-15 12:34:56 UTC
    import datetime as dt
    ts = dt.datetime(2026, 1, 15, 12, 34, 56, tzinfo=dt.timezone.utc).timestamp()
    p = m.parse_log_basename("101ceb95a960__neuroslm-full.log", mtime=ts)
    assert p is not None
    assert p.date_token == "20260115-123456"
    assert p.sha == "101ceb95a960"
    assert p.arch == "neuroslm-full"
    assert p.source == "legacy"


def test_parse_returns_none_for_unsortable() -> None:
    m = _load_migration()
    assert m.parse_log_basename("inspect.log") is None
    assert m.parse_log_basename("temp.log") is None
    assert m.parse_log_basename("not_a_log.txt") is None


def test_new_folder_name_format() -> None:
    m = _load_migration()
    from dataclasses import asdict
    p = m._ParsedName(
        date_token="20260614-182653",
        arch="rcc_bowtie_889M_run",
        sha="07aba24be2bf",
        source="new",
    )
    folder = m._new_folder_name(p)
    # 3-level hierarchy: <YYYYMMDD>/<arch>/<HHMMSS>_<sha>
    assert folder == "20260614/rcc_bowtie_889M_run/182653_07aba24be2bf"


# ── 4. Reference gating ────────────────────────────────────────────────


def test_plan_only_emits_ops_for_referenced_logs(repo: Path) -> None:
    """A log NOT mentioned in any doc/code is skipped entirely."""
    m = _load_migration()
    _make_log(repo / "logs" / "vast",
              "20260614T182653Z_07aba24be2bf_rcc_889M_step100of100.log")
    _make_log(repo / "logs" / "vast",
              "20260615T100000Z_aaaaaaaaaaaa_unref_42M_step200of200.log")
    # Reference only the FIRST file via FINDINGS.md
    (repo / "docs" / "FINDINGS.md").write_text(
        "## H1 — example\n**Status.** see "
        "`20260614T182653Z_07aba24be2bf_rcc_889M_step100of100.log`\n",
        encoding="utf-8",
    )
    ctx = _make_ctx(repo)
    ops = m.plan(ctx)
    srcs = {op.src.name for op in ops}
    assert "20260614T182653Z_07aba24be2bf_rcc_889M_step100of100.log" in srcs
    assert "20260615T100000Z_aaaaaaaaaaaa_unref_42M_step200of200.log" not in srcs


def test_plan_skips_logs_in_already_migrated_folders(repo: Path) -> None:
    """Logs already inside `logs/<day>/<arch>/<run>/` (not `logs/vast/`)
    are NEVER candidates — they've been migrated."""
    m = _load_migration()
    # v4 3-level hierarchy: logs/<day>/<arch>/<time>_<sha>/train.log
    _make_log(
        repo / "logs" / "20260614" / "rcc_889M" / "182653_07aba24be2bf",
        "train.log",
    )
    ctx = _make_ctx(repo)
    ops = m.plan(ctx)
    assert all("vast" in str(op.src) for op in ops)


# ── 5. Output folder format ────────────────────────────────────────────


def test_plan_destination_matches_format(repo: Path) -> None:
    m = _load_migration()
    _make_log(repo / "logs" / "vast",
              "20260614T182653Z_07aba24be2bf_arch_889M_step100of100.log")
    (repo / "docs" / "FINDINGS.md").write_text(
        "see `20260614T182653Z_07aba24be2bf_arch_889M_step100of100.log`",
        encoding="utf-8",
    )
    ctx = _make_ctx(repo)
    ops = m.plan(ctx)
    assert len(ops) == 1
    op = ops[0]
    dst = op.dst
    # 3-level hierarchy: logs/<day>/<arch>/<time>_<sha>/train.log
    # Walking parents from train.log upward:
    #   train.log -> 182653_07aba24be2bf -> arch_889M -> 20260614 -> logs
    assert dst.name == "train.log"
    assert dst.parent.name == "182653_07aba24be2bf"
    assert dst.parent.parent.name == "arch_889M"
    assert dst.parent.parent.parent.name == "20260614"
    assert dst.parent.parent.parent.parent.name == "logs"


def test_plan_unsortable_goes_to_unsorted_legacy(repo: Path) -> None:
    m = _load_migration()
    _make_log(repo / "logs" / "vast", "inspect.log")
    (repo / "docs" / "FINDINGS.md").write_text(
        "see `inspect.log`", encoding="utf-8"
    )
    ctx = _make_ctx(repo)
    ops = m.plan(ctx)
    assert len(ops) == 1
    op = ops[0]
    assert op.dst.parent.name == "_unsorted_legacy"
    assert op.dst.name == "inspect.log"


# ── 6. Idempotency ─────────────────────────────────────────────────────


def test_plan_skips_when_destination_already_exists(repo: Path) -> None:
    """If the dst file already exists, plan() returns no Op for it."""
    m = _load_migration()
    name = "20260614T182653Z_07aba24be2bf_arch_889M_step100of100.log"
    _make_log(repo / "logs" / "vast", name)
    (repo / "docs" / "FINDINGS.md").write_text(
        f"see `{name}`", encoding="utf-8"
    )
    # Pre-create the destination in the v4 3-level hierarchy
    dst_folder = repo / "logs" / "20260614" / "arch_889M" / "182653_07aba24be2bf"
    _make_log(dst_folder, "train.log")
    ctx = _make_ctx(repo)
    ops = m.plan(ctx)
    assert ops == []


# ── 7-8. apply() copies + idempotency ──────────────────────────────────


def test_apply_copies_files_and_creates_parents(repo: Path) -> None:
    m = _load_migration()
    name = "20260614T182653Z_07aba24be2bf_arch_889M_step100of100.log"
    src = _make_log(repo / "logs" / "vast", name,
                    content="real training output\n")
    (repo / "docs" / "FINDINGS.md").write_text(
        f"see `{name}`", encoding="utf-8"
    )
    ctx = _make_ctx(repo, dry_run=False, force=True)
    ops = m.plan(ctx)
    assert len(ops) == 1
    n = m.apply(ctx, ops)
    assert n == 1
    # v4 3-level hierarchy: logs/<day>/<arch>/<time>_<sha>/train.log
    dst = (
        repo / "logs" / "20260614" / "arch_889M"
        / "182653_07aba24be2bf" / "train.log"
    )
    assert dst.exists()
    assert dst.read_text(encoding="utf-8") == "real training output\n"
    # Source must STILL exist (copy, not move)
    assert src.exists()


def test_plan_returns_empty_after_apply(repo: Path) -> None:
    """The idempotency contract: after apply(), plan() == []."""
    m = _load_migration()
    name = "20260614T182653Z_07aba24be2bf_arch_889M_step100of100.log"
    _make_log(repo / "logs" / "vast", name)
    (repo / "docs" / "FINDINGS.md").write_text(
        f"see `{name}`", encoding="utf-8"
    )
    ctx = _make_ctx(repo, dry_run=False, force=True)
    ops = m.plan(ctx)
    m.apply(ctx, ops)
    # Rebuild context so refs is fresh (apply may have written files
    # that change the index — though in practice copies of logs don't).
    ctx2 = _make_ctx(repo, dry_run=False, force=True)
    assert m.plan(ctx2) == []


def test_migration_has_required_protocol_attrs() -> None:
    m = _load_migration()
    assert hasattr(m, "ID") and m.ID == "0001_logs_to_run_folders"
    assert hasattr(m, "DESCRIPTION") and isinstance(m.DESCRIPTION, str)
    assert callable(m.plan)
    assert callable(m.apply)
