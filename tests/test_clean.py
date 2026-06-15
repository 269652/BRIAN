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


def test_index_picks_up_basename_only(fake_repo: Path) -> None:
    """Under the EXACT-ONLY contract (see
    ``tests/test_references_exact_only.py``), only verbatim basename
    citations register. Glob tokens are dropped on the floor."""
    idx = cl.build_reference_index(
        fake_repo,
        skip_dirs=("logs", "lfs_checkpoints", "docs/archive"),
    )
    # Basename from FINDINGS.md should be in `exact`.
    assert "neuroslm_keep_me_42M_step5000.pt" in idx.exact
    # Glob tokens are no longer stored ANYWHERE.
    assert idx.globs == set(), \
        f"globs should remain empty under exact-only matching, got {idx.globs!r}"
    # Glob does NOT protect any matching basename — the doc author
    # must write the exact basename to protect a file.
    assert not idx.references("af758c381388_keep_me_glob_step2kof2k.log")
    # Orphan likewise unreferenced (was unreferenced under the old
    # contract too; preserved).
    assert not idx.references("orphan_run_step1k.log")


def test_finding_markers_detect_findings_md(fake_repo: Path) -> None:
    idx = cl.build_reference_index(fake_repo)
    findings = (fake_repo / "docs" / "FINDINGS.md").resolve()
    assert findings in idx.finding_files
    archived = (fake_repo / "docs" / "archive" / "2025-01-02_finding_archive.md").resolve()
    assert archived in idx.finding_files


# ── plan: logs bucket ─────────────────────────────────────────────────


def test_logs_plan_deletes_both_orphan_and_glob_only_log(
    fake_repo: Path,
) -> None:
    """Under exact-only matching, BOTH logs in ``fake_repo`` end up on
    the delete list — neither has its basename cited verbatim. The
    glob ``20260614*_keep_me_glob_step2kof2k.log`` in FINDINGS.md no
    longer protects the matching log file."""
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

    # Glob-only protection is gone: this log is now prunable.
    assert "af758c381388_keep_me_glob_step2kof2k.log" in delete_names
    # The orphan (always was prunable) still is.
    assert "orphan_run_step1k.log" in delete_names
    # Nothing should land in keep_referenced — both logs were only
    # ever referenced (if at all) via globs, which no longer count.
    assert kept_names == set(), \
        f"no logs should be kept under exact-only matching, got {kept_names!r}"


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
    """Under exact-only matching, only the EXACT basename
    ``neuroslm_keep_me_42M_step5000.pt`` (cited verbatim in
    ``fake_repo``'s FINDINGS.md) is reference-protected. To pin the
    "referenced files survive --force" behaviour with a log, we add
    an explicit verbatim citation to FINDINGS.md first."""
    orphan = fake_repo / "logs" / "vast" / "orphan_run_step1k.log"
    exact_ref_log = fake_repo / "logs" / "vast" / "exact_protected_log.log"
    exact_ref_log.write_text("loss=0.5\n", encoding="utf-8")
    # Add an exact-basename citation so the new contract protects it.
    findings = fake_repo / "docs" / "FINDINGS.md"
    findings.write_text(
        findings.read_text(encoding="utf-8")
        + "\nAlso see `exact_protected_log.log` for the H42 evidence.\n",
        encoding="utf-8",
    )
    assert orphan.exists() and exact_ref_log.exists()

    rc = cl.run(["logs"], force=True, root=fake_repo, keep_recent=0,
                use_git=False)
    assert rc == 0
    assert not orphan.exists(), "orphan should be gone after --force"
    assert exact_ref_log.exists(), \
        "exact-basename-cited log must survive --force"


def test_all_bucket_expands_to_three(fake_repo: Path) -> None:
    rc = cl.run(["all"], force=False, root=fake_repo, keep_recent=0,
                use_git=False)
    assert rc == 0


# ── _is_informative_glob is a deprecation stub under exact-only ──────


@pytest.mark.parametrize("any_glob", [
    # The old "uninformative" set — all still rejected.
    "*.pt", "*.log", "*.json", "*.mem", "*.mem.json", "*", "*.md",
    # The old "informative" set — now ALSO rejected, because globs
    # are not references at all under the new contract.
    "20260614*_step2kof2k.log",
    "neuroslm_large_*_step5000.pt",
    "*_keep_me_glob_*.log",
    "dsl_arch_20260531-*_step3000.pt",
])
def test_is_informative_glob_always_false_under_exact_only(
    any_glob: str,
) -> None:
    """Pinned by ``tests/test_references_exact_only.py``. The stub
    keeps the symbol importable for any out-of-tree diagnostic that
    used to call it, but always returns False — there is no concept
    of an "informative" glob under exact-only matching."""
    assert cl._is_informative_glob(any_glob) is False, (
        f"_is_informative_glob({any_glob!r}) must be False under "
        "exact-only matching; it is a deprecation stub now"
    )


def test_gitignore_style_glob_does_not_protect_checkpoint(tmp_path: Path) -> None:
    """Full integration: a `.gitignore` containing `*.pt` must NOT cause
    an unreferenced checkpoint to be marked referenced."""
    root = tmp_path
    (root / "lfs_checkpoints").mkdir()
    (root / "lfs_checkpoints" / ".gitkeep").write_text("", encoding="utf-8")
    orphan = root / "lfs_checkpoints" / "completely_orphan_step9000.pt"
    orphan.write_bytes(b"x" * 16)
    # The kind of file that introduced the bug:
    (root / ".gitignore").write_text("*.pt\n*.log\n*.mem\n", encoding="utf-8")
    # Plus a docs README that doesn't reference the orphan:
    (root / "docs").mkdir()
    (root / "docs" / "README.md").write_text("# Docs\n", encoding="utf-8")

    idx = cl.build_reference_index(
        root, skip_dirs=("logs", "lfs_checkpoints", "checkpoints", "docs/archive"),
    )
    assert not idx.references("completely_orphan_step9000.pt"), (
        "`.gitignore *.pt` must not silently protect every checkpoint"
    )
    plan = cl.plan_for_bucket(
        "checkpoints", idx, root=root, keep_recent=0,
        extra_keep=set(), git_protected=set(),
    )
    assert orphan.resolve() in {p.resolve() for p in plan.delete}


# ── per-run log folders (0001 migration layout) ───────────────────────


@pytest.fixture
def fake_repo_with_run_folders(fake_repo: Path) -> Path:
    """Extend ``fake_repo`` with the per-run folder layout produced by
    the 0001 logs-layout migration:

        logs/<YYYYMMDD-HHMMSS>_<arch>_<params>[_<label>]_<sha>/train.log

    Adds one *orphan* folder (no citation anywhere) and one *cited*
    folder (its full date-prefixed name is appended to ``FINDINGS.md``).
    """
    root = fake_repo

    orphan = root / "logs" / "20260601-100000_arch_42M_orphan_abc123"
    orphan.mkdir(parents=True)
    (orphan / "train.log").write_text(
        "step 1 loss=4.2\nstep 2 loss=4.1\n", encoding="utf-8",
    )

    cited = root / "logs" / "20260602-110000_arch_42M_cited_def456"
    cited.mkdir(parents=True)
    (cited / "train.log").write_text(
        "step 1 loss=4.0\nstep 2 loss=3.9\n", encoding="utf-8",
    )

    # Cite the second folder verbatim in FINDINGS.md (path-form, so
    # the doc author doesn't have to know about the bare-basename rule).
    findings = root / "docs" / "FINDINGS.md"
    findings.write_text(
        findings.read_text(encoding="utf-8")
        + "\nThe H99 evidence run lives in "
        + "`logs/20260602-110000_arch_42M_cited_def456/train.log`.\n",
        encoding="utf-8",
    )
    return root


def test_folder_token_regex_indexes_run_folder_names(
    fake_repo_with_run_folders: Path,
) -> None:
    """``_FOLDER_TOKEN_RE`` must pick the date-prefixed folder name out
    of a path-form citation like
    ``logs/20260602-110000_arch_42M_cited_def456/train.log`` and seed
    ``idx.exact`` with the bare folder name. Without this, per-run
    folders could never be protected (the folder has no suffix, so the
    suffix-anchored ``_TOKEN_RE`` can't see it)."""
    root = fake_repo_with_run_folders
    idx = cl.build_reference_index(
        root, skip_dirs=("logs", "lfs_checkpoints", "docs/archive"),
    )
    assert "20260602-110000_arch_42M_cited_def456" in idx.exact, (
        "_FOLDER_TOKEN_RE must seed cited folder names into idx.exact; "
        f"got exact = {sorted(idx.exact)!r}"
    )
    # The orphan folder is NOT cited anywhere, so it must NOT be in
    # the index (this is the whole point of the protect-by-citation
    # contract).
    assert "20260601-100000_arch_42M_orphan_abc123" not in idx.exact


def test_unreferenced_run_folder_is_on_delete_list(
    fake_repo_with_run_folders: Path,
) -> None:
    """Per-run folders whose date-prefixed name isn't cited anywhere
    end up in ``plan.delete`` as a directory ``Path``."""
    root = fake_repo_with_run_folders
    idx = cl.build_reference_index(
        root, skip_dirs=("logs", "lfs_checkpoints", "docs/archive"),
    )
    plan = cl.plan_for_bucket(
        "logs", idx, root=root, keep_recent=0,
        extra_keep=set(), git_protected=set(),
    )
    orphan = root / "logs" / "20260601-100000_arch_42M_orphan_abc123"
    delete_set = {p.resolve() for p in plan.delete}
    assert orphan.resolve() in delete_set, (
        f"orphan run folder should be marked for deletion; "
        f"got delete = {[p.name for p in plan.delete]!r}"
    )
    # And it must be a directory candidate, not the contained train.log.
    matching = [p for p in plan.delete if p.resolve() == orphan.resolve()]
    assert matching and matching[0].is_dir(), (
        "candidate must be the folder itself (so the whole dir gets "
        "rmtree'd), not the contained train.log file"
    )


def test_referenced_run_folder_is_protected(
    fake_repo_with_run_folders: Path,
) -> None:
    """When a per-run folder's full date-prefixed name is cited in a
    scientific record (FINDINGS.md), the folder lands in
    ``plan.keep_referenced`` and never in ``plan.delete``."""
    root = fake_repo_with_run_folders
    idx = cl.build_reference_index(
        root, skip_dirs=("logs", "lfs_checkpoints", "docs/archive"),
    )
    plan = cl.plan_for_bucket(
        "logs", idx, root=root, keep_recent=0,
        extra_keep=set(), git_protected=set(),
    )
    cited = root / "logs" / "20260602-110000_arch_42M_cited_def456"
    keep_set = {p.resolve() for p in plan.keep_referenced}
    delete_set = {p.resolve() for p in plan.delete}
    assert cited.resolve() in keep_set, (
        "cited folder must land in keep_referenced; "
        f"got keep_referenced = {[p.name for p in plan.keep_referenced]!r}"
    )
    assert cited.resolve() not in delete_set


def test_train_log_basename_alone_does_not_protect_folder(
    fake_repo_with_run_folders: Path,
) -> None:
    """A doc that writes the bare basename ``train.log`` somewhere must
    NOT silently protect every per-run folder. The discriminator is the
    folder's full date-prefixed name, not the inner filename (which is
    identical across every run)."""
    root = fake_repo_with_run_folders
    # Add a stray ``train.log`` mention in a side doc.
    side = root / "docs" / "rambling_note.md"
    side.write_text(
        "Random aside: the per-run output goes to `train.log`.\n",
        encoding="utf-8",
    )
    idx = cl.build_reference_index(
        root, skip_dirs=("logs", "lfs_checkpoints", "docs/archive"),
    )
    plan = cl.plan_for_bucket(
        "logs", idx, root=root, keep_recent=0,
        extra_keep=set(), git_protected=set(),
    )
    orphan = root / "logs" / "20260601-100000_arch_42M_orphan_abc123"
    # The orphan must STILL be on the delete list — its full folder
    # name is not in idx.exact, regardless of train.log mentions.
    assert orphan.resolve() in {p.resolve() for p in plan.delete}


def test_recent_run_folder_is_protected_by_mtime(
    fake_repo_with_run_folders: Path,
) -> None:
    """``keep_recent=N`` ranks per-run folders by folder mtime (not
    contained-file mtime), so an in-flight run is never surprise-pruned."""
    root = fake_repo_with_run_folders
    orphan = root / "logs" / "20260601-100000_arch_42M_orphan_abc123"
    future = time.time() + 1000
    os.utime(orphan, (future, future))

    idx = cl.build_reference_index(
        root, skip_dirs=("logs", "lfs_checkpoints", "docs/archive"),
    )
    plan = cl.plan_for_bucket(
        "logs", idx, root=root, keep_recent=1,
        extra_keep=set(), git_protected=set(),
    )
    assert orphan.resolve() in {p.resolve() for p in plan.keep_recent}
    assert orphan.resolve() not in {p.resolve() for p in plan.delete}


def test_force_recursively_deletes_orphan_run_folder(
    fake_repo_with_run_folders: Path,
) -> None:
    """``brian clean logs --force`` rmtree's an unreferenced per-run
    folder — both the folder and the contained ``train.log`` must be
    gone after the run."""
    root = fake_repo_with_run_folders
    orphan = root / "logs" / "20260601-100000_arch_42M_orphan_abc123"
    train_log = orphan / "train.log"
    cited = root / "logs" / "20260602-110000_arch_42M_cited_def456"
    cited_train = cited / "train.log"
    assert train_log.is_file() and cited_train.is_file()

    rc = cl.run(["logs"], force=True, root=root, keep_recent=0,
                use_git=False)
    assert rc == 0
    assert not orphan.exists(), (
        f"orphan run folder must be rmtree'd, still exists at {orphan}"
    )
    # The cited folder must survive intact (folder + contained file).
    assert cited.is_dir() and cited_train.is_file(), (
        "cited run folder must survive --force"
    )


def test_plan_bytes_to_free_includes_folder_contents(
    fake_repo_with_run_folders: Path,
) -> None:
    """``CleanPlan.bytes_to_free`` must recurse into per-run folder
    candidates — otherwise the dry-run summary would report 0 B
    reclaimable for the dominant disk-hog (multi-GB train.log)."""
    root = fake_repo_with_run_folders
    orphan = root / "logs" / "20260601-100000_arch_42M_orphan_abc123"
    train_log = orphan / "train.log"
    expected_min = train_log.stat().st_size  # the only file in there

    idx = cl.build_reference_index(
        root, skip_dirs=("logs", "lfs_checkpoints", "docs/archive"),
    )
    plan = cl.plan_for_bucket(
        "logs", idx, root=root, keep_recent=0,
        extra_keep=set(), git_protected=set(),
    )
    assert plan.bytes_to_free >= expected_min, (
        f"bytes_to_free ({plan.bytes_to_free}) should include the "
        f"orphan folder's train.log size ({expected_min})"
    )
