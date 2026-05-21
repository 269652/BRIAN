"""Checkpoint rotation — keep only the N most-recent checkpoints per stream.

Used by `train.py` after each save so the repo doesn't grow unbounded as
training progresses.  Also exposed as a standalone CLI so the rotation
can be applied manually:

    python -m neuroslm.tools.prune_ckpts \
        --dirs lfs_checkpoints checkpoints \
        --keep 3 \
        --git

Behaviour:

  • Files are grouped by stream (preset, optimizer, baseline-or-not,
    train mode).  Each stream is rotated independently so an Adafactor
    run cannot prune an AdamW run's checkpoints.
  • Each group is sorted by the step number embedded in the filename
    (the trailing `_<step>.pt`).  `_latest.pt` files are ignored —
    those are deliberately overwritten in-place during training.
  • The oldest (group_size − keep) checkpoints are deleted from disk.
    Companion files (`.mem`, `.mem.json`, `.dna.json`) sharing the same
    stem are deleted alongside.
  • If --git is passed AND the directory is inside a git work tree,
    the deletions are staged via `git rm` and committed locally with a
    rotation message.  Push is intentionally left to the caller (a
    crashed `git push` during training should never abort the run).

Failure modes are non-fatal: any error logs a single warning line and
returns control to the caller. Training is more important than perfect
rotation.
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Regex captures the step number from a checkpoint filename, e.g.
#   neuroslm_xl_240M_adamw_mix_1000.pt → step=1000
# `_latest.pt` files explicitly do not match (no trailing digits).
_STEP_RE = re.compile(r"^(?P<stream>.+)_(?P<step>\d+)\.pt$")

# Companion suffixes that share the .pt stem and must be deleted with it.
_COMPANION_SUFFIXES: Tuple[str, ...] = (".pt", ".mem", ".mem.json", ".dna.json")


# ────────────────────────────────────────────────────────────────────────
# Grouping + selection
# ────────────────────────────────────────────────────────────────────────

def _group_checkpoints(pt_paths: Iterable[Path]) -> dict[str, List[Tuple[int, Path]]]:
    """Group .pt files by stream prefix → sorted [(step, path), ...]."""
    groups: dict[str, List[Tuple[int, Path]]] = {}
    for p in pt_paths:
        m = _STEP_RE.match(p.name)
        if not m:
            continue                          # skip _latest.pt and similar
        stream = m.group("stream")
        step   = int(m.group("step"))
        groups.setdefault(stream, []).append((step, p))
    for stream in groups:
        groups[stream].sort(key=lambda x: x[0])     # ascending by step
    return groups


def _select_obsolete(groups: dict[str, List[Tuple[int, Path]]],
                     keep: int) -> List[Path]:
    """Return the .pt paths that should be deleted (all but the newest `keep`
    per stream)."""
    obsolete: List[Path] = []
    for stream, entries in groups.items():
        if len(entries) <= keep:
            continue
        for _step, pt in entries[:-keep]:
            obsolete.append(pt)
    return obsolete


def _role_of(pt_path: Path) -> str:
    """'baseline' if filename contains `_baseline_`, else 'full'."""
    return "baseline" if "_baseline_" in pt_path.name else "full"


def _select_after_step(pt_paths: Iterable[Path], max_step: int,
                       only_role: str = "all") -> List[Path]:
    """.pt paths with step > max_step, optionally filtered to a role.

    only_role:
        'all'       → no role filter
        'full'      → only non-baseline checkpoints
        'baseline'  → only `*_baseline_*` checkpoints
    """
    obsolete: List[Path] = []
    for p in pt_paths:
        m = _STEP_RE.match(p.name)
        if not m:
            continue
        step = int(m.group("step"))
        if step <= max_step:
            continue
        if only_role != "all" and _role_of(p) != only_role:
            continue
        obsolete.append(p)
    return obsolete


def _companion_files(pt_path: Path) -> List[Path]:
    """Return all files sharing the stem of `pt_path` that exist on disk."""
    stem = pt_path.with_suffix("")             # strip .pt
    out: List[Path] = []
    for suf in _COMPANION_SUFFIXES:
        candidate = stem.with_suffix(stem.suffix + suf) if stem.suffix else stem.with_suffix(suf)
        # `.mem.json` is a compound suffix; with_suffix only swaps the last.
        # Build the path manually to be safe across both single + compound.
        candidate = Path(str(stem) + suf)
        if candidate.exists() and candidate.is_file():
            out.append(candidate)
    return out


# ────────────────────────────────────────────────────────────────────────
# Git plumbing (all best-effort)
# ────────────────────────────────────────────────────────────────────────

def _git_root(path: Path) -> Optional[Path]:
    try:
        r = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return Path(r.stdout.strip())
    except Exception:
        return None


def _git_is_tracked(repo: Path, file: Path) -> bool:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--error-unmatch",
             str(file.relative_to(repo))],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _git_rm_and_commit(repo: Path, files: List[Path], message: str) -> bool:
    """Stage deletions for tracked files; commit if anything was staged."""
    if not files:
        return False
    rel = [str(p.relative_to(repo)) for p in files
           if _git_is_tracked(repo, p)]
    if not rel:
        return False
    try:
        subprocess.run(["git", "-C", str(repo), "rm", "--quiet", *rel],
                       check=True, capture_output=True, text=True, timeout=120)
        subprocess.run(["git", "-C", str(repo), "commit", "-m", message],
                       check=True, capture_output=True, text=True, timeout=60)
        return True
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            f"[prune_ckpts] git rm/commit failed (rc={e.returncode}): "
            f"{e.stderr.strip()[:200]}\n")
        return False
    except Exception as e:
        sys.stderr.write(f"[prune_ckpts] git op error: {e}\n")
        return False


# ────────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────────

def prune_old_checkpoints(directories: List[Path], keep: int = 3,
                          use_git: bool = False,
                          verbose: bool = True) -> dict:
    """Delete obsolete checkpoints, keeping only the newest `keep` per stream.

    directories: list of directories to scan for .pt files (non-recursive).
                 Typically `Path(args.ckpt_dir)` and `lfs_checkpoints/`.
    keep:        retain the newest N checkpoints per stream.
    use_git:     when True, run `git rm` + `git commit` for tracked files.
                 No push — that's the caller's responsibility.
    verbose:     log to stderr.

    Returns a small summary dict.
    """
    summary = {"deleted_files": 0, "deleted_bytes": 0,
               "git_commits": 0, "streams_pruned": 0}

    all_pt: List[Path] = []
    for d in directories:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in d.glob("*.pt"):
            if p.is_file():
                all_pt.append(p)
    if not all_pt:
        if verbose:
            sys.stderr.write("[prune_ckpts] no .pt files found\n")
        return summary

    groups = _group_checkpoints(all_pt)
    obsolete_pt = _select_obsolete(groups, keep)
    if not obsolete_pt:
        if verbose:
            n_groups = len(groups)
            sys.stderr.write(
                f"[prune_ckpts] {n_groups} stream(s), nothing to prune "
                f"(keep={keep})\n")
        return summary
    summary["streams_pruned"] = len(
        {p.name.rsplit("_", 1)[0] for p in obsolete_pt})

    # Collect ALL files to delete (each .pt plus its companions)
    to_delete: List[Path] = []
    for pt in obsolete_pt:
        to_delete.extend(_companion_files(pt))

    if verbose:
        sys.stderr.write(
            f"[prune_ckpts] pruning {len(obsolete_pt)} checkpoint(s) "
            f"({len(to_delete)} files total, keep={keep})\n")
        for pt in obsolete_pt:
            sys.stderr.write(f"  - {pt.name}\n")

    # If git is in play, route deletions through `git rm` so the index
    # stays consistent.  Otherwise just unlink the files.
    repo: Optional[Path] = None
    if use_git and to_delete:
        repo = _git_root(to_delete[0].parent)

    if repo is not None:
        rotated = _git_rm_and_commit(
            repo, to_delete,
            f"chore: prune {len(obsolete_pt)} old checkpoint(s) (keep last {keep})")
        if rotated:
            summary["git_commits"] = 1
        else:
            # Fall through to plain unlink for anything `git rm` didn't catch
            for f in to_delete:
                if f.exists():
                    try:
                        size = f.stat().st_size
                        f.unlink()
                        summary["deleted_files"] += 1
                        summary["deleted_bytes"] += size
                    except Exception as e:
                        sys.stderr.write(
                            f"[prune_ckpts] failed to delete {f}: {e}\n")
    else:
        for f in to_delete:
            if not f.exists():
                continue
            try:
                size = f.stat().st_size
                f.unlink()
                summary["deleted_files"] += 1
                summary["deleted_bytes"] += size
            except Exception as e:
                sys.stderr.write(f"[prune_ckpts] failed to delete {f}: {e}\n")

    if verbose:
        mb = summary["deleted_bytes"] / 1048576
        sys.stderr.write(
            f"[prune_ckpts] freed {mb:.1f} MiB across "
            f"{summary['deleted_files']} files, "
            f"{summary['git_commits']} git commit(s)\n")
    return summary


def prune_after_step(directories: List[Path], max_step: int,
                     only_role: str = "all",
                     use_git: bool = False,
                     verbose: bool = True) -> dict:
    """Delete .pt files (+ companions) whose step > max_step.

    Use case: roll a stream back to a "best known" checkpoint by step
    number, e.g. `prune_after_step(['lfs_checkpoints'], max_step=6580,
    only_role='full', use_git=True)` deletes every full-stream .pt with
    step > 6580 and stages the deletions via `git rm` + commit.

    only_role:
        'all'       no role filter
        'full'      non-baseline checkpoints only
        'baseline'  `*_baseline_*` checkpoints only
    """
    summary = {"deleted_files": 0, "deleted_bytes": 0, "git_commits": 0}

    all_pt: List[Path] = []
    for d in directories:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in d.glob("*.pt"):
            if p.is_file():
                all_pt.append(p)
    obsolete = _select_after_step(all_pt, max_step=max_step, only_role=only_role)
    if not obsolete:
        if verbose:
            sys.stderr.write(
                f"[prune_after_step] no .pt > step {max_step} "
                f"(role={only_role})\n")
        return summary

    to_delete: List[Path] = []
    for pt in obsolete:
        to_delete.extend(_companion_files(pt))

    if verbose:
        sys.stderr.write(
            f"[prune_after_step] {len(obsolete)} .pt > step {max_step} "
            f"(role={only_role}); {len(to_delete)} files total\n")
        for pt in obsolete:
            sys.stderr.write(f"  - {pt.name}\n")

    repo = _git_root(to_delete[0].parent) if (use_git and to_delete) else None
    if repo is not None:
        rotated = _git_rm_and_commit(
            repo, to_delete,
            f"chore: prune {len(obsolete)} checkpoint(s) > step {max_step} "
            f"(role={only_role})")
        if rotated:
            summary["git_commits"] = 1
        else:
            for f in to_delete:
                if f.exists():
                    try:
                        size = f.stat().st_size
                        f.unlink()
                        summary["deleted_files"] += 1
                        summary["deleted_bytes"] += size
                    except Exception as e:
                        sys.stderr.write(
                            f"[prune_after_step] failed to delete {f}: {e}\n")
    else:
        for f in to_delete:
            if not f.exists():
                continue
            try:
                size = f.stat().st_size
                f.unlink()
                summary["deleted_files"] += 1
                summary["deleted_bytes"] += size
            except Exception as e:
                sys.stderr.write(
                    f"[prune_after_step] failed to delete {f}: {e}\n")

    if verbose:
        mb = summary["deleted_bytes"] / 1048576
        sys.stderr.write(
            f"[prune_after_step] freed {mb:.1f} MiB across "
            f"{summary['deleted_files']} files, "
            f"{summary['git_commits']} git commit(s)\n")
    return summary


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="Directories to scan (non-recursive)")
    ap.add_argument("--keep", type=int, default=3,
                    help="(rotation mode) keep newest N per stream. Default 3.")
    ap.add_argument("--max-step", type=int, default=None,
                    help="(step-cutoff mode) delete checkpoints with step > N. "
                         "When set, overrides --keep.")
    ap.add_argument("--only", default="all",
                    choices=["all", "full", "baseline"],
                    help="(step-cutoff mode) restrict to a role. Default all.")
    ap.add_argument("--git", action="store_true",
                    help="Stage deletions via `git rm` + commit locally")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.max_step is not None:
        s = prune_after_step(
            [Path(d) for d in args.dirs],
            max_step=args.max_step, only_role=args.only,
            use_git=args.git, verbose=not args.quiet,
        )
    else:
        s = prune_old_checkpoints(
            [Path(d) for d in args.dirs],
            keep=args.keep, use_git=args.git, verbose=not args.quiet,
        )
    return 0 if s["deleted_files"] >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
