"""Tests for `neuroslm.tools.clean` — the reference-aware repo janitor.

The contract under test:

  1. A file whose basename is mentioned in any scanned text file
     (especially `docs/FINDINGS.md`) is NEVER on the delete list.
  2. A file whose basename matches a glob token in a scanned text file
     is NEVER on the delete list.
  3. `*_best.*` checkpoints + their sidecars are never deleted.
  4. Anchor files (`README.md`, `.gitkeep`, FINDINGS.md…) are never
     deleted, regardless of bucket.
  5. Markdown files containing finding markers (✅ CONFIRMED,
     **Hypothesis.**, ## H1 — …) are protected in the docs bucket.
  6. The N most-recent files per bucket are kept (default 3).
  7. Files in the git porcelain (staged or modified) are never deleted.
  8. Dry-run (default) makes no filesystem mutations.
  9. `--force` deletes the planned files and returns 0 on success.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from neuroslm.tools import clean as cl


# ── fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A minimal fake repo with logs, checkpoints, docs, and a FINDINGS.md."""
    root = tmp_path

    # docs/
    (root / "docs").mkdir()
    (root / "docs" / "FINDINGS.md").write_text(
        "# FINDINGS\n\n"
        "## H1 — example\n\n"
        "**Hypothesis.** Something.\n"
        "**Status.** ✅ CONFIRMED — see "
        "`lfs_checkpoints/neuroslm_keep_me_42M_step5000.pt` and "
        "`logs/vast/20260614*_keep_me_glob_step2kof2k.log`.\n",
        encoding="utf-8",
    )
    (root / "docs" / "architecture.md").write_text("# Architecture\n", encoding="utf-8")
    (root / "docs" / "README.md").write_text("# Docs\n", encoding="utf-8")
    (root / "docs" / "archive").mkdir()
    (root / "docs" / "archive" / "2025-01-01_old_note.md").write_text(
        "Random orphan archive note.\n", encoding="utf-8",
    )
    (root / "docs" / "archive" / "2025-01-02_finding_archive.md").write_text(
        "# An archived finding\n\n**Hypothesis.** X.\n**Status.** ✅ CONFIRMED\n",
        encoding="utf-8",
    )

    # logs/
    (root / "logs" / "vast").mkdir(parents=True)
    (root / "logs" / "vast" / "orphan_run_step1k.log").write_text("log\n", encoding="utf-8")
    (root / "logs" / "vast" / "af758c381388_keep_me_glob_step2kof2k.log").write_text(
        "log\n", encoding="utf-8",
    )

    # lfs_checkpoints/
    (root / "lfs_checkpoints").mkdir()
    (root / "lfs_checkpoints" / ".gitkeep").write_text("", encoding="utf-8")
    (root / "lfs_checkpoints" / "neuroslm_keep_me_42M_step5000.pt").write_bytes(b"x" * 16)
    (root / "lfs_checkpoints" / "neuroslm_orphan_42M_step1000.pt").write_bytes(b"x" * 16)
    (root / "lfs_checkpoints" / "neuroslm_orphan_42M_best.pt").write_bytes(b"x" * 16)
    (root / "lfs_checkpoints" / "neuroslm_orphan_42M_best.mem.json").write_text(
        "{}", encoding="utf-8",
    )

    # brian.toml with extra_keep
    (root / "brian.toml").write_text(
        '[clean]\nextra_keep = ["lfs_checkpoints/neuroslm_orphan_42M_step1000.pt"]\n',
        encoding="utf-8",
    )

    return root


# ── reference index ───────────────────────────────────────────────────


def test_index_picks_up_basename_and_glob(fake_repo: Path) -> None:
    idx = cl.build_reference_index(
        fake_repo,
        skip_dirs=("logs", "lfs_checkpoints", "docs/archive"),
    )
    # Basename from FINDINGS.md should be in `exact`.
    assert "neuroslm_keep_me_42M_step5000.pt" in idx.exact
    # Glob token should be captured.
    assert any("*" in g and g.endswith(".log") for g in idx.globs)
    # Glob protects matching basenames.
    assert idx.references("af758c381388_keep_me_glob_step2kof2k.log")
    # Orphan must NOT register as referenced.
    assert not idx.references("orphan_run_step1k.log")


def test_finding_markers_detect_findings_md(fake_repo: Path) -> None:
    idx = cl.build_reference_index(fake_repo)
    findings = (fake_repo / "docs" / "FINDINGS.md").resolve()
    assert findings in idx.finding_files
    archived = (fake_repo / "docs" / "archive" / "2025-01-02_finding_archive.md").resolve()
    assert archived in idx.finding_files


# ── plan: logs bucket ─────────────────────────────────────────────────


def test_logs_plan_keeps_referenced_and_deletes_orphans(fake_repo: Path) -> None:
    idx = cl.build_reference_index(
        fake_repo,
        skip_dirs=("logs", "lfs_checkpoints", "docs/archive"),
    )
    plan = cl.plan_for_bucket(
        "logs", idx, root=fake_repo, keep_recent=0,
        extra_keep=set(), git_protected=set(),
    )
    delete_names = {p.name for p in plan.delete}
    kept_names = {p.name for p in plan.keep_referenced}

    assert "af758c381388_keep_me_glob_step2kof2k.log" in kept_names
    assert "orphan_run_step1k.log" in delete_names


def test_keep_recent_protects_newest(fake_repo: Path) -> None:
    # Make the orphan the newest file.
    orphan = fake_repo / "logs" / "vast" / "orphan_run_step1k.log"
    future = time.time() + 1000
    os.utime(orphan, (future, future))

    idx = cl.build_reference_index(
        fake_repo,
        skip_dirs=("logs", "lfs_checkpoints", "docs/archive"),
    )
    plan = cl.plan_for_bucket(
        "logs", idx, root=fake_repo, keep_recent=1,
        extra_keep=set(), git_protected=set(),
    )
    # With keep_recent=1 the most-recent (orphan) is in keep_recent,
    # not delete.
    assert orphan.resolve() not in {p.resolve() for p in plan.delete}
    assert orphan.resolve() in {p.resolve() for p in plan.keep_recent}


# ── plan: checkpoints bucket ──────────────────────────────────────────


def test_checkpoints_plan_protects_referenced_best_and_extra_keep(
    fake_repo: Path,
) -> None:
    idx = cl.build_reference_index(
        fake_repo,
        skip_dirs=("logs", "lfs_checkpoints", "docs/archive"),
    )
    extra = cl._load_extra_keep(fake_repo)
    plan = cl.plan_for_bucket(
        "checkpoints", idx, root=fake_repo, keep_recent=0,
        extra_keep=extra, git_protected=set(),
    )
    by_name = {p.name: p for p in plan.delete}

    # The FINDINGS-referenced checkpoint is protected.
    assert "neuroslm_keep_me_42M_step5000.pt" not in by_name
    # `_best.pt` and its sidecar are protected (anchor predicate).
    assert "neuroslm_orphan_42M_best.pt" not in by_name
    assert "neuroslm_orphan_42M_best.mem.json" not in by_name
    # extra_keep from brian.toml protects an otherwise-orphaned file.
    assert "neuroslm_orphan_42M_step1000.pt" not in by_name


def test_checkpoints_orphan_with_no_protection_is_deleted(
    fake_repo: Path,
) -> None:
    # Add a checkpoint that is unreferenced, not _best, and not in extra_keep.
    orphan = fake_repo / "lfs_checkpoints" / "neuroslm_truly_orphan_42M_step9000.pt"
    orphan.write_bytes(b"x" * 16)

    idx = cl.build_reference_index(
        fake_repo,
        skip_dirs=("logs", "lfs_checkpoints", "docs/archive"),
    )
    extra = cl._load_extra_keep(fake_repo)
    plan = cl.plan_for_bucket(
        "checkpoints", idx, root=fake_repo, keep_recent=0,
        extra_keep=extra, git_protected=set(),
    )
    assert orphan.resolve() in {p.resolve() for p in plan.delete}


# ── plan: docs bucket ─────────────────────────────────────────────────


def test_docs_plan_protects_finding_archive(fake_repo: Path) -> None:
    idx = cl.build_reference_index(fake_repo)
    plan = cl.plan_for_bucket(
        "docs", idx, root=fake_repo, keep_recent=0,
        extra_keep=set(), git_protected=set(),
    )
    finding_path = (fake_repo / "docs" / "archive" / "2025-01-02_finding_archive.md").resolve()
    orphan_path = (fake_repo / "docs" / "archive" / "2025-01-01_old_note.md").resolve()
    assert finding_path in {p.resolve() for p in plan.keep_finding}
    assert orphan_path in {p.resolve() for p in plan.delete}


# ── git porcelain protection ──────────────────────────────────────────


def test_git_protected_paths_never_deleted(fake_repo: Path) -> None:
    orphan = fake_repo / "logs" / "vast" / "orphan_run_step1k.log"
    idx = cl.build_reference_index(
        fake_repo,
        skip_dirs=("logs", "lfs_checkpoints", "docs/archive"),
    )
    plan = cl.plan_for_bucket(
        "logs", idx, root=fake_repo, keep_recent=0,
        extra_keep=set(),
        git_protected={orphan.resolve()},
    )
    assert orphan.resolve() in {p.resolve() for p in plan.keep_git}
    assert orphan.resolve() not in {p.resolve() for p in plan.delete}


# ── dry-run vs force ──────────────────────────────────────────────────


def test_dry_run_makes_no_filesystem_changes(fake_repo: Path) -> None:
    orphan = fake_repo / "logs" / "vast" / "orphan_run_step1k.log"
    assert orphan.exists()
    rc = cl.run(["logs"], force=False, root=fake_repo, keep_recent=0,
                use_git=False)
    assert rc == 0
    assert orphan.exists(), "dry-run must not delete anything"


def test_force_deletes_orphan(fake_repo: Path) -> None:
    orphan = fake_repo / "logs" / "vast" / "orphan_run_step1k.log"
    referenced = fake_repo / "logs" / "vast" / "af758c381388_keep_me_glob_step2kof2k.log"
    assert orphan.exists() and referenced.exists()

    rc = cl.run(["logs"], force=True, root=fake_repo, keep_recent=0,
                use_git=False)
    assert rc == 0
    assert not orphan.exists(), "orphan should be gone after --force"
    assert referenced.exists(), "referenced log must survive --force"


def test_all_bucket_expands_to_three(fake_repo: Path) -> None:
    rc = cl.run(["all"], force=False, root=fake_repo, keep_recent=0,
                use_git=False)
    assert rc == 0
