"""brian clean lfs — per-run LFS checkpoint pruner.

Deletes ``lfs_checkpoints/**/*.pt`` files (and their cached blobs at
``.git/lfs/objects/<oid>``) that are no longer worth keeping. Unlike
``brian clean checkpoints`` (which only looks at flat reference rules),
this command groups by run-folder and enforces a per-folder retention
window — so a noisy run that produced 40 checkpoints leaves a manageable
tail behind.

A checkpoint is KEPT iff at least one of:

  R1.  Its basename is referenced anywhere in the repo
       (``ReferenceIndex.references()``).
  R2.  It is one of the N most-recent steps within its parent folder
       (default ``keep_recent=3``).
  R3.  Its parent folder contains a ``manifest.json`` whose ``commit``
       matches the current git ``HEAD``.
  R4.  It is a ``*_best.*`` checkpoint AND its run's log file is
       referenced (the existing reference rules already protect the
       log). Two layouts:
         (a) run-folder: ``logs/<same-folder-name>/*.log``.
         (b) flat: any ``logs/**/*.log`` whose basename shares a
             distinctive token (>= 8 chars) with the checkpoint's stem
             (after stripping ``_best`` / ``_step<N>``) AND whose
             basename itself is referenced.

Anything else is PRUNABLE.

This module deletes the LFS *pointer* file and the local cache blob.
Reclaiming server-side LFS quota needs a separate ``git filter-repo`` /
``git lfs migrate`` pass.

Default mode is dry-run; ``--force`` actually unlinks.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from neuroslm.tools.clean import (
    REPO_ROOT,
    ReferenceIndex,
    build_reference_index,
)


# ── filename parsers ─────────────────────────────────────────────────────


_STEP_RX = re.compile(r"_?step(\d+)", re.IGNORECASE)
# Legacy flat layout uses `..._3000.pt` (no "step" prefix). Match a
# trailing _<digits> just before the extension.
_TRAILING_NUM_RX = re.compile(r"_(\d+)\.[A-Za-z]+$")
_BEST_RX = re.compile(r"_best(\.|$)", re.IGNORECASE)
# Strip these suffixes when extracting the "run-id token" from a
# checkpoint stem so the leftover is the unique-per-run substring.
_RUN_TOKEN_TRIM = re.compile(
    r"_(best|step\d+|\d+)(?=\.|$)", re.IGNORECASE,
)
_MIN_TOKEN_LEN = 8


def _extract_step_number(name: str) -> int:
    """Return the step number embedded in the filename, or 0 if absent.

    Handles two filename conventions:
      * ``step1000`` / ``_step1000`` (new layout)
      * trailing ``_1000.pt``        (legacy flat layout)
    """
    m = _STEP_RX.search(name)
    if m:
        return int(m.group(1))
    m = _TRAILING_NUM_RX.search(name)
    return int(m.group(1)) if m else 0


def _is_best_filename(name: str) -> bool:
    """Return True if `name` is a ``*_best.*`` checkpoint or sidecar."""
    return bool(_BEST_RX.search(name))


def _run_token(stem: str) -> str:
    """Reduce a checkpoint stem to its distinctive run-id substring.

    ``dsl_arch_20260531-174107_step5000_best`` → ``dsl_arch_20260531-174107``
    ``neuroslm_large_107M_adamw_mix_best``      → ``neuroslm_large_107M_adamw_mix``
    ``step01000_best``                          → ``step01000`` (short — handled by caller)
    """
    out = stem
    # Strip suffixes iteratively so combined ones (`_step5000_best`) collapse.
    while True:
        nxt = _RUN_TOKEN_TRIM.sub("", out)
        if nxt == out:
            break
        out = nxt
    return out


# ── LFS pointer detection ───────────────────────────────────────────────


def _is_lfs_pointer(path: Path) -> bool:
    """A git-lfs pointer file is small UTF-8 text that begins with
    ``version https://git-lfs.github.com/spec/v1``. We read just the
    first 200 bytes to avoid loading native checkpoint binaries."""
    try:
        with path.open("rb") as fh:
            head = fh.read(200)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def _read_lfs_oid(path: Path) -> Optional[str]:
    """Return the bare hex OID from an LFS pointer file, or None."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"^oid sha256:([0-9a-f]{64})\b", text, re.MULTILINE)
    return m.group(1) if m else None


# ── data model ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    step: int
    is_best: bool

    @property
    def folder(self) -> Path:
        return self.path.parent


def _collect_via_git_lfs(root: Path) -> Optional[Dict[Path, List[CheckpointInfo]]]:
    """Enumerate LFS-tracked files via ``git lfs ls-files --long``.

    Returns the per-folder grouping if git+lfs are available and the
    command succeeds; ``None`` if git/lfs is missing or the repo isn't
    a git repo (caller should fall back to filesystem walk).

    Output format we parse::

        <oid64> - path/to/file.pt        (- = not downloaded)
        <oid64> * path/to/file.pt        (* = present locally)

    Only ``.pt`` files are kept (checkpoints; sidecars are handled by
    their own naming, e.g. ``foo.mem.json`` references foo's ``.pt``).
    """
    try:
        r = subprocess.run(
            ["git", "lfs", "ls-files", "--long"],
            cwd=root, capture_output=True, text=True, check=False,
            timeout=30,
        )
    except (OSError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None

    out: Dict[Path, List[CheckpointInfo]] = {}
    # Cache the OID per path on the CheckpointInfo via a side-table.
    # We stash it in a module-level dict keyed by path because
    # CheckpointInfo is frozen; the deleter reads OIDs from here first
    # before falling back to re-reading the pointer file.
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: "<oid> {- | *} <relpath>"
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        oid, _flag, relpath = parts
        if not re.fullmatch(r"[0-9a-f]{64}", oid):
            continue
        if not relpath.endswith(".pt"):
            continue
        full = (root / relpath).resolve()
        info = CheckpointInfo(
            path=full,
            step=_extract_step_number(full.name),
            is_best=_is_best_filename(full.name),
        )
        _OID_CACHE[full] = oid
        out.setdefault(full.parent, []).append(info)
    return out


# OID cache populated by the git-lfs collector; used by the deleter to
# avoid re-reading the pointer file. Cleared per `run()` invocation.
_OID_CACHE: Dict[Path, str] = {}


def _collect_checkpoints(root: Path) -> Dict[Path, List[CheckpointInfo]]:
    """Filesystem fallback: walk ``<root>/lfs_checkpoints/`` and group
    pointer .pt files by parent folder. Used when git/lfs is missing
    (and by tests that don't set up a real git repo)."""
    out: Dict[Path, List[CheckpointInfo]] = {}
    base = root / "lfs_checkpoints"
    if not base.is_dir():
        return out
    for p in base.rglob("*.pt"):
        if not p.is_file():
            continue
        if not _is_lfs_pointer(p):
            continue
        info = CheckpointInfo(
            path=p,
            step=_extract_step_number(p.name),
            is_best=_is_best_filename(p.name),
        )
        out.setdefault(p.parent, []).append(info)
    return out


# ── log-protection helper (the NEW rule R4) ────────────────────────────


def _candidate_logs_for(ckpt: CheckpointInfo, root: Path) -> List[Path]:
    """Return log files that could plausibly belong to the same run as
    `ckpt`. Both layouts handled."""
    logs_root = root / "logs"
    if not logs_root.is_dir():
        return []

    out: List[Path] = []

    # Layout (a): run-folder. logs/<same-folder-name>/*.log
    rf_name = ckpt.folder.name
    sibling = logs_root / rf_name
    if sibling.is_dir():
        out.extend(p for p in sibling.rglob("*.log") if p.is_file())

    # Layout (b): flat. Match by distinctive run-token.
    stem = ckpt.path.stem
    token = _run_token(stem)
    if len(token) >= _MIN_TOKEN_LEN:
        # The token must remain distinctive — strip common prefixes that
        # would otherwise match every checkpoint of the same family.
        for log_path in logs_root.rglob("*.log"):
            if not log_path.is_file():
                continue
            if token in log_path.name:
                out.append(log_path)

    return out


def _best_protected_via_log(
    ckpt: CheckpointInfo, root: Path, ref_idx: ReferenceIndex,
) -> bool:
    """R4: is `ckpt`'s run's log file referenced anywhere?"""
    for log in _candidate_logs_for(ckpt, root):
        if ref_idx.references(log.name):
            return True
    return False


# ── manifest commit helper (rule R3) ───────────────────────────────────


def _manifest_commit(folder: Path) -> Optional[str]:
    """Read ``<folder>/manifest.json`` and return its ``commit`` field."""
    m = folder / "manifest.json"
    if not m.is_file():
        return None
    try:
        data = json.loads(m.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    val = data.get("commit")
    return val if isinstance(val, str) else None


def _git_head(root: Path) -> Optional[str]:
    """Return current HEAD commit SHA, or None if not a git repo."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, check=False,
        )
    except (OSError, FileNotFoundError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


# ── selection ───────────────────────────────────────────────────────────


def _select_prunable(
    folders: Dict[Path, List[CheckpointInfo]],
    *,
    keep_recent: int = 3,
    reference_index: Optional[ReferenceIndex] = None,
    head_commit: Optional[str] = None,
    root: Path = REPO_ROOT,
) -> Tuple[Dict[Path, CheckpointInfo], Dict[Path, str]]:
    """Decide per-file which checkpoints survive.

    Returns ``(prunable, kept_reasons)`` where keys are paths and:
      * ``prunable[path] = CheckpointInfo``
      * ``kept_reasons[path] = human-readable rule that saved it``
    """
    prunable: Dict[Path, CheckpointInfo] = {}
    kept: Dict[Path, str] = {}

    for folder, ckpts in folders.items():
        # R3 short-circuit: manifest commit == HEAD → keep everything here.
        if head_commit is not None:
            mc = _manifest_commit(folder)
            if mc and mc == head_commit:
                for info in ckpts:
                    kept[info.path] = "manifest commit matches HEAD"
                continue

        # Rank by step desc; pick top-N for R2.
        sorted_by_step = sorted(ckpts, key=lambda c: c.step, reverse=True)
        top_n = {c.path for c in sorted_by_step[:keep_recent]}

        for info in ckpts:
            # R1: basename referenced
            if reference_index is not None and \
                    reference_index.references(info.path.name):
                kept[info.path] = "referenced by name"
                continue
            # R4: best + log protected
            if info.is_best and reference_index is not None and \
                    _best_protected_via_log(info, root, reference_index):
                kept[info.path] = "best (run log referenced)"
                continue
            # R2: top-N by step
            if info.path in top_n:
                kept[info.path] = f"top-{keep_recent} by step"
                continue
            # nothing saved it
            prunable[info.path] = info

    return prunable, kept


# ── deletion ────────────────────────────────────────────────────────────


def _delete_pointer_and_cache(path: Path, root: Path) -> Tuple[bool, bool]:
    """Delete the LFS pointer file at `path` and the cached blob (if
    present). Returns (pointer_deleted, blob_deleted).

    Prefers the OID cached by ``_collect_via_git_lfs`` to avoid a
    re-read of the pointer file; falls back to parsing the pointer if
    it isn't cached (e.g. filesystem-walk path)."""
    oid = _OID_CACHE.get(path) or _read_lfs_oid(path)
    pointer_deleted = False
    blob_deleted = False
    try:
        path.unlink()
        pointer_deleted = True
    except OSError:
        pass
    if oid:
        # Standard git-lfs cache layout: .git/lfs/objects/aa/bb/aabbcc...
        blob = root / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
        try:
            if blob.is_file():
                blob.unlink()
                blob_deleted = True
        except OSError:
            pass
    return pointer_deleted, blob_deleted


# ── pretty printing ────────────────────────────────────────────────────


def _ascii_safe() -> bool:
    import sys
    enc = (getattr(sys.stdout, "encoding", None) or "ascii").lower()
    try:
        "\u2500".encode(enc)
        return False
    except (UnicodeEncodeError, LookupError):
        return True


_SEP = "-" if _ascii_safe() else "\u2500"


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        n_f = n / 1024
        if n_f < 1024:
            return f"{n_f:.1f} {unit}"
        n //= 1024
    return f"{n} PiB"


# ── public entry point ─────────────────────────────────────────────────


def run(
    *,
    root: Path = REPO_ROOT,
    force: bool = False,
    keep_recent: int = 3,
    verbose: bool = False,
    use_git: bool = True,
) -> int:
    """Plan and (optionally) execute LFS checkpoint pruning."""
    # Fresh OID cache per invocation.
    _OID_CACHE.clear()

    # Prefer git-lfs ls-files: authoritative, fast, doesn't depend on
    # working-tree pointer format. Falls back to filesystem walk only
    # if git/lfs is unavailable (or this isn't a git checkout).
    folders: Optional[Dict[Path, List[CheckpointInfo]]] = None
    via = "filesystem"
    if use_git:
        print("[lfs prune] enumerating LFS files via `git lfs ls-files`...",
              flush=True)
        folders = _collect_via_git_lfs(root)
        if folders is not None:
            via = "git lfs ls-files"

    if folders is None:
        print("[lfs prune] git lfs unavailable; falling back to filesystem walk",
              flush=True)
        folders = _collect_checkpoints(root)

    if not folders:
        print(f"[lfs prune] no LFS .pt files found ({via})")
        return 0

    total_files = sum(len(v) for v in folders.values())
    print(f"[lfs prune] {total_files} LFS checkpoint(s) in "
          f"{len(folders)} folder(s) (via {via})",
          flush=True)

    # Build reference index. Skip our own bucket + the LFS tree so
    # checkpoint filenames mentioned inside sidecars don't self-protect.
    # Skip logs/ too because raw training logs name every checkpoint
    # they wrote — counting that as a "reference" defeats the point.
    print("[lfs prune] scanning repo for references "
          "(markdown / py / json / yaml ...)...", flush=True)
    idx = build_reference_index(
        root,
        skip_dirs=("lfs_checkpoints", "logs"),
        progress=True,
    )

    head = _git_head(root) if use_git else None
    if head:
        print(f"[lfs prune] HEAD = {head[:12]} (used for manifest-commit rule)",
              flush=True)

    prunable, kept = _select_prunable(
        folders, keep_recent=keep_recent, reference_index=idx,
        head_commit=head, root=root,
    )

    bar = _SEP * 60
    print(bar)
    print(f"[lfs prune] kept {len(kept)}   prunable {len(prunable)}")

    if verbose:
        for p, reason in sorted(kept.items()):
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                rel = str(p)
            print(f"  + {rel}  [{reason}]")

    if not prunable:
        print(bar)
        return 0

    # Show what would be pruned.
    print(f"\n[lfs prune] would prune {len(prunable)} file(s):")
    total_bytes = 0
    for p, info in sorted(prunable.items()):
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            rel = str(p)
        try:
            sz = p.stat().st_size
        except OSError:
            sz = 0
        total_bytes += sz
        print(f"  - {rel} (step={info.step}, {_fmt_size(sz)})")

    print(f"\n[lfs prune] reclaimable (pointers only): {_fmt_size(total_bytes)}")

    if not force:
        print(bar)
        print("[lfs prune] dry-run -- re-run with --force to actually delete")
        print(bar)
        return 0

    # FORCE.
    print(bar)
    print("[lfs prune] --force given -- DELETING NOW ...")
    print(bar)
    n_ptr = n_blob = n_err = 0
    for p in prunable:
        ok_ptr, ok_blob = _delete_pointer_and_cache(p, root)
        if ok_ptr:
            n_ptr += 1
        else:
            n_err += 1
        if ok_blob:
            n_blob += 1
    print(f"[lfs prune] deleted {n_ptr} pointer(s), {n_blob} cached blob(s), "
          f"{n_err} error(s)")
    return 1 if n_err else 0


# ── CLI dispatcher (used by neuroslm.cli) ─────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="brian clean lfs",
        description="Prune unreferenced LFS checkpoints (default: dry-run).",
    )
    p.add_argument("--force", action="store_true",
                   help="actually delete (default: dry-run)")
    p.add_argument("--keep-recent", type=int, default=3, metavar="N",
                   help="keep the N most-recent steps per folder (default: 3)")
    p.add_argument("--verbose", action="store_true",
                   help="print every kept file with its protecting rule")
    p.add_argument("--no-git", action="store_true",
                   help="skip the git HEAD check (R3 manifest rule)")
    args = p.parse_args(argv)
    return run(
        force=args.force,
        keep_recent=args.keep_recent,
        verbose=args.verbose,
        use_git=not args.no_git,
    )


if __name__ == "__main__":
    raise SystemExit(main())
