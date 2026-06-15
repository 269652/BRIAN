"""Tests for `neuroslm.tools.clean_lfs` — per-run LFS checkpoint pruner.

CONTRACT under test
-------------------

A checkpoint file under ``lfs_checkpoints/`` is KEPT iff at least one of:

  R1. Its **basename** is referenced anywhere in the repo
      (``ReferenceIndex.references()`` — markdown / py / json / yaml …).
  R2. It is one of the **N most-recent steps within its parent folder**
      (default ``keep_recent=3``; folder = checkpoint's parent dir).
  R3. Its parent folder contains a ``manifest.json`` whose ``commit``
      equals the current git ``HEAD``.
  R4. It is a ``*_best.*`` checkpoint AND **its run's log file is
      referenced** (i.e. the existing reference rules already protect
      the log). Two layouts supported:
        (a) run-folder layout: ``logs/<same-folder-name>/*.log``
        (b) flat layout: any ``logs/**/*.log`` whose basename shares a
            distinctive run-id token (≥8 chars) with the checkpoint's
            stem (after stripping ``_best`` / ``_step<N>``) AND whose
            basename itself is referenced.

Anything else is PRUNABLE.

Default mode = dry-run (no filesystem mutation, exit 0). ``--force``
deletes (a) the in-tree LFS pointer file and (b) any blob in
``.git/lfs/objects/<oid>`` whose OID matches the pointer.

With ``use_git=True`` (the default), ``--force`` additionally:
  G1. Stages deletions via ``git rm --cached`` (removes from git index).
  G2. Commits the staged removals.
  G3. Pushes to origin so the remote LFS server can reclaim storage.
  G4. Runs ``git lfs prune`` to clean the local blob cache.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from neuroslm.tools import clean_lfs as cl
from neuroslm.tools.clean import ReferenceIndex, build_reference_index


# ── parsing helpers ────────────────────────────────────────────────────


@pytest.mark.parametrize("name, expected", [
    ("step01000.pt",                                   1000),
    ("step01000_best.pt",                              1000),
    ("dsl_arch_20260531-174107_step5000.pt",           5000),
    ("neuroslm_large_107M_adamw_mix_3000.pt",          3000),
    ("dsl_arch.pt",                                       0),  # no step → 0
    ("neuroslm_large_107M_adamw_mix_best.pt",             0),  # no step → 0
])
def test_extract_step_number(name: str, expected: int) -> None:
    assert cl._extract_step_number(name) == expected


@pytest.mark.parametrize("name, expected", [
    ("step01000_best.pt",                                True),
    ("step04000_best.pt",                                True),
    ("neuroslm_large_107M_adamw_mix_best.pt",            True),
    ("neuroslm_large_107M_adamw_mix_best.mem.json",      True),
    ("neuroslm_large_107M_adamw_mix_best.mem",           True),
    ("step01000.pt",                                    False),
    ("dsl_arch.pt",                                     False),
    ("neuroslm_large_107M_adamw_mix_3000.pt",           False),
])
def test_is_best_filename(name: str, expected: bool) -> None:
    assert cl._is_best_filename(name) is expected


# ── grouping ───────────────────────────────────────────────────────────


def _make_lfs_pointer(path: Path, oid_hex: str | None = None) -> None:
    """Write a minimal valid LFS pointer file at `path`."""
    oid = oid_hex or ("a" * 64)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{oid}\n"
        "size 12345\n",
        encoding="utf-8",
    )


def _make_native_pt(path: Path, n_bytes: int = 1024) -> None:
    """Write a non-LFS-pointer .pt (real binary). Should be ignored."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * n_bytes)


def test_is_lfs_pointer_detects_real_pointer(tmp_path: Path) -> None:
    p = tmp_path / "ptr.pt"
    _make_lfs_pointer(p)
    assert cl._is_lfs_pointer(p) is True


def test_is_lfs_pointer_rejects_native_binary(tmp_path: Path) -> None:
    p = tmp_path / "real.pt"
    _make_native_pt(p)
    assert cl._is_lfs_pointer(p) is False


def test_collect_groups_checkpoints_by_parent_folder(tmp_path: Path) -> None:
    ck = tmp_path / "lfs_checkpoints"
    folder_a = ck / "20260514-130749_arch_3aaff15"
    folder_b = ck / "20260514-191117_arch_3aaff15"

    _make_lfs_pointer(folder_a / "step01000.pt")
    _make_lfs_pointer(folder_a / "step02000.pt")
    _make_lfs_pointer(folder_b / "step00100.pt")
    # native binary in folder_a — must be ignored
    _make_native_pt(folder_a / "stray.pt")
    # non-.pt file — must be ignored
    (folder_a / "step01000.mem.json").write_text("{}", encoding="utf-8")

    folders = cl._collect_checkpoints(tmp_path)

    assert set(folders.keys()) == {folder_a, folder_b}
    assert {c.path.name for c in folders[folder_a]} == {"step01000.pt", "step02000.pt"}
    assert {c.path.name for c in folders[folder_b]} == {"step00100.pt"}


# ── selection (no _best involved) ─────────────────────────────────────


def test_keeps_n_most_recent_per_folder(tmp_path: Path) -> None:
    ck = tmp_path / "lfs_checkpoints" / "run_a"
    for step in (1000, 2000, 3000, 4000, 5000):
        _make_lfs_pointer(ck / f"step{step:05d}.pt")

    folders = cl._collect_checkpoints(tmp_path)
    prunable, kept = cl._select_prunable(
        folders, keep_recent=3, reference_index=None, root=tmp_path,
    )
    pruned_names = {p.name for p in prunable}
    assert pruned_names == {"step01000.pt", "step02000.pt"}
    assert any("top-3" in r for r in kept.values())


def test_keeps_referenced_basename(tmp_path: Path) -> None:
    ck = tmp_path / "lfs_checkpoints" / "run_a"
    _make_lfs_pointer(ck / "step01000.pt")
    _make_lfs_pointer(ck / "step02000.pt")
    _make_lfs_pointer(ck / "step03000.pt")
    _make_lfs_pointer(ck / "step04000.pt")
    _make_lfs_pointer(ck / "step05000.pt")

    # Reference one of the would-be-pruned files in a docs page
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "FINDINGS.md").write_text(
        "# Findings\n\nSee `step01000.pt` — it's the seed for H42.\n",
        encoding="utf-8",
    )

    idx = build_reference_index(tmp_path, skip_dirs=("lfs_checkpoints",))
    folders = cl._collect_checkpoints(tmp_path)
    prunable, kept = cl._select_prunable(
        folders, keep_recent=3, reference_index=idx, root=tmp_path,
    )
    pruned_names = {p.name for p in prunable}
    # step01000 is referenced → kept; step02000 is the oldest unreferenced → pruned
    assert "step01000.pt" not in pruned_names
    assert "step02000.pt" in pruned_names


# ── NEW _best rule: best is kept iff log is referenced ────────────────


def test_best_kept_when_run_folder_log_is_referenced(tmp_path: Path) -> None:
    """Run-folder layout: lfs_checkpoints/<rf>/best.pt + logs/<rf>/train.log."""
    rf = "20260514-130749_arch_3aaff15"
    _make_lfs_pointer(tmp_path / "lfs_checkpoints" / rf / "step10000_best.pt")
    # An ordinary log file in the matching log folder
    (tmp_path / "logs" / rf).mkdir(parents=True)
    (tmp_path / "logs" / rf / "train.log").write_text("loss=0.5\n", encoding="utf-8")
    # FINDINGS references the log file by basename
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "FINDINGS.md").write_text(
        "See `logs/20260514-130749_arch_3aaff15/train.log` for H42 evidence.\n",
        encoding="utf-8",
    )

    idx = build_reference_index(tmp_path, skip_dirs=("lfs_checkpoints", "logs"))
    folders = cl._collect_checkpoints(tmp_path)
    prunable, kept = cl._select_prunable(
        folders, keep_recent=3, reference_index=idx, root=tmp_path,
    )
    best_path = tmp_path / "lfs_checkpoints" / rf / "step10000_best.pt"
    # It's the only checkpoint so top-3 also keeps it; remove that ambiguity
    # by adding 5 newer steps:
    for step in (10100, 10200, 10300, 10400, 10500):
        _make_lfs_pointer(tmp_path / "lfs_checkpoints" / rf / f"step{step:05d}.pt")
    folders = cl._collect_checkpoints(tmp_path)
    prunable, kept = cl._select_prunable(
        folders, keep_recent=3, reference_index=idx, root=tmp_path,
    )
    assert best_path not in prunable, \
        "best.pt should be kept because its log is referenced"
    assert "best" in kept[best_path] and "log" in kept[best_path]


def test_best_pruned_when_no_log_protects_it(tmp_path: Path) -> None:
    """Best is just another old checkpoint when nothing references its run."""
    rf = "20260514-130749_arch_3aaff15"
    ck_dir = tmp_path / "lfs_checkpoints" / rf
    _make_lfs_pointer(ck_dir / "step10000_best.pt")
    # 5 newer steps so step10000_best is out of the top-3 by step
    for step in (10100, 10200, 10300, 10400, 10500):
        _make_lfs_pointer(ck_dir / f"step{step:05d}.pt")

    # logs/<rf>/train.log exists but NOTHING references it
    (tmp_path / "logs" / rf).mkdir(parents=True)
    (tmp_path / "logs" / rf / "train.log").write_text("loss=0.5\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "FINDINGS.md").write_text(
        "Some other finding — nothing about this run.\n", encoding="utf-8",
    )

    idx = build_reference_index(tmp_path, skip_dirs=("lfs_checkpoints", "logs"))
    folders = cl._collect_checkpoints(tmp_path)
    prunable, _ = cl._select_prunable(
        folders, keep_recent=3, reference_index=idx, root=tmp_path,
    )
    best_path = ck_dir / "step10000_best.pt"
    assert best_path in prunable, \
        "best.pt should be PRUNED because its log isn't referenced"


def test_best_kept_when_flat_layout_log_shares_token(tmp_path: Path) -> None:
    """Flat layout: ckpt and log live at top of their dirs; correlation
    by distinctive run-id token (timestamp / experiment name)."""
    ck = tmp_path / "lfs_checkpoints"
    # Real flat-layout filename from the actual repo
    _make_lfs_pointer(ck / "dsl_arch_20260531-174107_step5000_best.pt")
    # 5 newer flat checkpoints so the best is out of top-3 by step
    for step in (5100, 5200, 5300, 5400, 5500):
        _make_lfs_pointer(ck / f"dsl_arch_20260531-174107_step{step}.pt")

    # The matching log file — same timestamp token
    (tmp_path / "logs" / "vast").mkdir(parents=True)
    log_path = tmp_path / "logs" / "vast" / "abc123__dsl_arch_20260531-174107.log"
    log_path.write_text("loss=0.5\n", encoding="utf-8")

    # FINDINGS references the log by basename
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "FINDINGS.md").write_text(
        f"See `{log_path.name}` for H42 evidence.\n",
        encoding="utf-8",
    )

    idx = build_reference_index(tmp_path, skip_dirs=("lfs_checkpoints", "logs"))
    folders = cl._collect_checkpoints(tmp_path)
    prunable, kept = cl._select_prunable(
        folders, keep_recent=3, reference_index=idx, root=tmp_path,
    )
    best_path = ck / "dsl_arch_20260531-174107_step5000_best.pt"
    assert best_path not in prunable, \
        "best.pt should be kept (flat layout: token-matched log is referenced)"


def test_best_pruned_when_flat_log_exists_but_unreferenced(tmp_path: Path) -> None:
    ck = tmp_path / "lfs_checkpoints"
    _make_lfs_pointer(ck / "dsl_arch_20260531-174107_step5000_best.pt")
    for step in (5100, 5200, 5300, 5400, 5500):
        _make_lfs_pointer(ck / f"dsl_arch_20260531-174107_step{step}.pt")

    (tmp_path / "logs" / "vast").mkdir(parents=True)
    (tmp_path / "logs" / "vast" / "abc123__dsl_arch_20260531-174107.log") \
        .write_text("loss=0.5\n", encoding="utf-8")

    # Doc references a DIFFERENT log
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "FINDINGS.md").write_text(
        "See `xyz__some_other_run.log` — different run.\n", encoding="utf-8",
    )

    idx = build_reference_index(tmp_path, skip_dirs=("lfs_checkpoints", "logs"))
    folders = cl._collect_checkpoints(tmp_path)
    prunable, _ = cl._select_prunable(
        folders, keep_recent=3, reference_index=idx, root=tmp_path,
    )
    best_path = ck / "dsl_arch_20260531-174107_step5000_best.pt"
    assert best_path in prunable


# ── manifest commit rule ──────────────────────────────────────────────


def test_kept_when_manifest_commit_matches_head(tmp_path: Path) -> None:
    rf = "20260514-130749_arch_abcdef0"
    ck = tmp_path / "lfs_checkpoints" / rf
    for step in (1000, 2000, 3000, 4000, 5000):
        _make_lfs_pointer(ck / f"step{step:05d}.pt")
    (ck / "manifest.json").write_text(
        '{"commit": "deadbeef1234567890", "arch": "arch", "preset": "x"}',
        encoding="utf-8",
    )

    folders = cl._collect_checkpoints(tmp_path)
    prunable, kept = cl._select_prunable(
        folders, keep_recent=3, reference_index=None,
        head_commit="deadbeef1234567890", root=tmp_path,
    )
    assert prunable == {}, "all checkpoints kept because manifest commit == HEAD"
    assert all("manifest" in r or "top-" in r for r in kept.values())


# ── dry-run / force execution ─────────────────────────────────────────


def test_dry_run_does_not_delete(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    ck = tmp_path / "lfs_checkpoints" / "run_a"
    pointer = ck / "step01000.pt"
    _make_lfs_pointer(pointer)
    for step in (2000, 3000, 4000, 5000):
        _make_lfs_pointer(ck / f"step{step:05d}.pt")

    rc = cl.run(root=tmp_path, force=False, keep_recent=3, use_git=False)
    assert rc == 0
    assert pointer.exists(), "dry-run must not delete files"


def test_force_deletes_pointer_and_lfs_cache(tmp_path: Path) -> None:
    ck = tmp_path / "lfs_checkpoints" / "run_a"
    oid_hex = "b" * 64
    pointer = ck / "step01000.pt"
    _make_lfs_pointer(pointer, oid_hex=oid_hex)
    for step in (2000, 3000, 4000, 5000):
        _make_lfs_pointer(ck / f"step{step:05d}.pt")

    # Simulate the local LFS cache layout
    cache = tmp_path / ".git" / "lfs" / "objects" / oid_hex[:2] / oid_hex[2:4]
    cache.mkdir(parents=True)
    cached_blob = cache / oid_hex
    cached_blob.write_bytes(b"x" * 100)

    rc = cl.run(root=tmp_path, force=True, keep_recent=3, use_git=False)
    assert rc == 0
    assert not pointer.exists(), "force must delete the pointer"
    assert not cached_blob.exists(), "force must delete the cached blob"


# ── regression: is_best must be populated in CheckpointInfo ───────────


def test_checkpoint_info_records_is_best(tmp_path: Path) -> None:
    """Regression: previously `_collect_checkpoints` hardcoded
    `is_best=False`, so the best-rule was effectively dead."""
    ck = tmp_path / "lfs_checkpoints" / "run_a"
    _make_lfs_pointer(ck / "step01000_best.pt")
    _make_lfs_pointer(ck / "step01000.pt")

    folders = cl._collect_checkpoints(tmp_path)
    infos = {c.path.name: c for c in folders[ck]}
    assert infos["step01000_best.pt"].is_best is True
    assert infos["step01000.pt"].is_best is False


# ── git staging / push contracts (G1-G4) ─────────────────────────────


def _fake_subprocess_factory(git_calls: List[List[str]]):
    """Return a subprocess.run replacement that:
    - Records every git call in `git_calls`.
    - Returns non-zero for `git lfs ls-files` and `git rev-parse HEAD`
      (forces filesystem fallback, no real git repo needed).
    - Returns zero for all other calls (git rm, commit, push, lfs prune).
    """
    def _fake(cmd, **kwargs):
        cmd_list = list(cmd)
        git_calls.append(cmd_list)
        # Force fallback to filesystem walk (no git LFS in tmp_path).
        if "lfs" in cmd_list and "ls-files" in cmd_list:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        # No HEAD in tmp_path.
        if cmd_list[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return _fake


def test_force_with_use_git_calls_git_rm_cached(tmp_path: Path) -> None:
    """G1: --force with use_git=True stages deletions via git rm --cached."""
    ck = tmp_path / "lfs_checkpoints" / "run1"
    ck.mkdir(parents=True)
    old_ptr = ck / "step01000.pt"
    _make_lfs_pointer(old_ptr)
    for step in (3000, 4000, 5000):
        _make_lfs_pointer(ck / f"step{step:05d}.pt")

    git_calls: List[List[str]] = []
    with patch("subprocess.run", side_effect=_fake_subprocess_factory(git_calls)):
        rc = cl.run(root=tmp_path, force=True, keep_recent=3, use_git=True)

    assert rc == 0
    assert not old_ptr.exists(), "pointer must be deleted from disk"
    rm_calls = [c for c in git_calls if len(c) >= 3 and c[:3] == ["git", "rm", "--cached"]]
    assert rm_calls, "git rm --cached must be called"
    all_rm_args = " ".join(str(a) for a in rm_calls[0])
    assert "step01000.pt" in all_rm_args, \
        f"deleted pointer must appear in git rm args, got: {rm_calls[0]}"


def test_force_with_use_git_commits_removals(tmp_path: Path) -> None:
    """G2: --force with use_git=True commits the staged deletions."""
    ck = tmp_path / "lfs_checkpoints" / "run1"
    ck.mkdir(parents=True)
    _make_lfs_pointer(ck / "step01000.pt")
    for step in (3000, 4000, 5000):
        _make_lfs_pointer(ck / f"step{step:05d}.pt")

    git_calls: List[List[str]] = []
    with patch("subprocess.run", side_effect=_fake_subprocess_factory(git_calls)):
        cl.run(root=tmp_path, force=True, keep_recent=3, use_git=True)

    commit_calls = [c for c in git_calls if c[:2] == ["git", "commit"]]
    assert commit_calls, "git commit must be called after staging deletions"


def test_force_with_use_git_pushes_to_origin(tmp_path: Path) -> None:
    """G3: --force with use_git=True pushes so remote LFS can reclaim storage."""
    ck = tmp_path / "lfs_checkpoints" / "run1"
    ck.mkdir(parents=True)
    _make_lfs_pointer(ck / "step01000.pt")
    for step in (3000, 4000, 5000):
        _make_lfs_pointer(ck / f"step{step:05d}.pt")

    git_calls: List[List[str]] = []
    with patch("subprocess.run", side_effect=_fake_subprocess_factory(git_calls)):
        cl.run(root=tmp_path, force=True, keep_recent=3, use_git=True)

    push_calls = [c for c in git_calls if c[:2] == ["git", "push"]]
    assert push_calls, "git push must be called to free remote LFS quota"


def test_force_with_use_git_runs_lfs_prune(tmp_path: Path) -> None:
    """G4: --force with use_git=True runs git lfs prune to clean local cache."""
    ck = tmp_path / "lfs_checkpoints" / "run1"
    ck.mkdir(parents=True)
    _make_lfs_pointer(ck / "step01000.pt")
    for step in (3000, 4000, 5000):
        _make_lfs_pointer(ck / f"step{step:05d}.pt")

    git_calls: List[List[str]] = []
    with patch("subprocess.run", side_effect=_fake_subprocess_factory(git_calls)):
        cl.run(root=tmp_path, force=True, keep_recent=3, use_git=True)

    lfs_prune_calls = [
        c for c in git_calls if "lfs" in c and "prune" in c
    ]
    assert lfs_prune_calls, "git lfs prune must be called to clean local cache"


def test_force_without_use_git_skips_git_ops(tmp_path: Path) -> None:
    """use_git=False must not call git rm / commit / push / lfs prune."""
    ck = tmp_path / "lfs_checkpoints" / "run1"
    ck.mkdir(parents=True)
    _make_lfs_pointer(ck / "step01000.pt")
    for step in (3000, 4000, 5000):
        _make_lfs_pointer(ck / f"step{step:05d}.pt")

    git_calls: List[List[str]] = []
    with patch("subprocess.run", side_effect=_fake_subprocess_factory(git_calls)):
        rc = cl.run(root=tmp_path, force=True, keep_recent=3, use_git=False)

    assert rc == 0
    # No git calls at all when use_git=False.
    assert not git_calls, f"git must not be called with use_git=False, got: {git_calls}"


def test_force_git_rm_failure_does_not_commit_or_push(tmp_path: Path) -> None:
    """If git rm --cached fails, commit and push must NOT be attempted."""
    ck = tmp_path / "lfs_checkpoints" / "run1"
    ck.mkdir(parents=True)
    _make_lfs_pointer(ck / "step01000.pt")
    for step in (3000, 4000, 5000):
        _make_lfs_pointer(ck / f"step{step:05d}.pt")

    git_calls: List[List[str]] = []

    def _fail_on_rm(cmd, **kwargs):
        cmd_list = list(cmd)
        git_calls.append(cmd_list)
        if "lfs" in cmd_list and "ls-files" in cmd_list:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if cmd_list[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if len(cmd_list) >= 3 and cmd_list[:3] == ["git", "rm", "--cached"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error: rm failed")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fail_on_rm):
        rc = cl.run(root=tmp_path, force=True, keep_recent=3, use_git=True)

    assert rc == 0  # deletion of files still succeeded
    commit_calls = [c for c in git_calls if c[:2] == ["git", "commit"]]
    push_calls = [c for c in git_calls if c[:2] == ["git", "push"]]
    assert not commit_calls, "must not commit when git rm failed"
    assert not push_calls, "must not push when git rm failed"


# ── git lfs ls-files based enumeration ────────────────────────────────


def test_collect_via_git_lfs_parses_long_format(tmp_path, monkeypatch) -> None:
    """`git lfs ls-files --long` output is the authoritative source."""
    ck = tmp_path / "lfs_checkpoints" / "run_a"
    ck.mkdir(parents=True)
    # Real files don't need pointer contents — collector trusts git.
    (ck / "step01000.pt").write_bytes(b"x")
    (ck / "step02000_best.pt").write_bytes(b"y")
    # A non-.pt LFS file should be ignored by the collector
    (ck / "step01000.mem.json").write_bytes(b"z")

    fake_oid_a = "a" * 64
    fake_oid_b = "b" * 64
    fake_oid_c = "c" * 64
    fake_stdout = (
        f"{fake_oid_a} * lfs_checkpoints/run_a/step01000.pt\n"
        f"{fake_oid_b} - lfs_checkpoints/run_a/step02000_best.pt\n"
        f"{fake_oid_c} * lfs_checkpoints/run_a/step01000.mem.json\n"
    )

    class FakeCompleted:
        returncode = 0
        stdout = fake_stdout
        stderr = ""

    def fake_run(*args, **kwargs):
        return FakeCompleted()

    monkeypatch.setattr(cl.subprocess, "run", fake_run)

    folders = cl._collect_via_git_lfs(tmp_path)
    assert folders is not None
    only_folder = list(folders.keys())
    assert len(only_folder) == 1
    infos = {c.path.name: c for c in folders[only_folder[0]]}
    # Only the .pt files
    assert set(infos) == {"step01000.pt", "step02000_best.pt"}
    assert infos["step02000_best.pt"].is_best is True
    # OID cache populated
    best_path = only_folder[0] / "step02000_best.pt"
    assert cl._OID_CACHE[best_path.resolve()] == fake_oid_b


def test_collect_via_git_lfs_returns_none_when_git_missing(
    tmp_path, monkeypatch,
) -> None:
    def boom(*args, **kwargs):
        raise FileNotFoundError("no git")
    monkeypatch.setattr(cl.subprocess, "run", boom)
    assert cl._collect_via_git_lfs(tmp_path) is None


def test_collect_via_git_lfs_returns_none_on_nonzero_exit(
    tmp_path, monkeypatch,
) -> None:
    class FakeFailed:
        returncode = 128  # "not a git repo"
        stdout = ""
        stderr = "fatal: not a git repository"
    monkeypatch.setattr(cl.subprocess, "run", lambda *a, **k: FakeFailed())
    assert cl._collect_via_git_lfs(tmp_path) is None

