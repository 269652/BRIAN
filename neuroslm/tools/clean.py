"""brian clean — delete unreferenced logs / checkpoints / docs.

The repo accumulates raw training logs (`logs/vast/*.log`), per-run
output folders (`logs/<YYYYMMDD-HHMMSS>_<arch>_<params>_<sha>/`),
per-step checkpoints (`lfs_checkpoints/*.pt|*.mem*|*.json`), and
one-off design markdowns (`docs/archive/*.md`) at a pace that quickly
bloats LFS and git history. This module finds files in those buckets
that are *not* referenced by anything that counts as a scientific
record — and deletes them.

A file is considered REFERENCED if any of the following is true:

  1. Its exact basename appears as a substring of any text file scanned
     (every `*.md`, `*.py`, `*.ipynb`, `*.json`, `*.toml`, `*.yaml`,
     `*.yml`, `*.txt` under the repo, excluding ``.git``, ``.venv``,
     ``__pycache__``, ``node_modules``, ``.pytest_cache``,
     ``neuroslm.egg-info`` and the bucket directories themselves).
  2. Its basename matches a glob token written in a scanned text file
     (so the ``20260614*_…_step2kof2k.log`` glob in
     ``docs/FINDINGS.md`` correctly protects
     ``af758c381388_arch_889M_…_step2kof2k.log``).
  3. Its basename minus extension appears verbatim as a token.

Always-keep guards (a candidate is dropped from the delete list even if
unreferenced):

  * ``*_best.*`` — best-known checkpoints survive rotation.
  * ``.gitkeep``, ``.gitignore``, ``README.md``, ``INSTRUCTIONS.md``.
  * Anchor docs (``FINDINGS.md``, ``CLI.md``, ``architecture.md`` …).
  * The N most-recent files per bucket (default 3) by mtime — so a
    currently-running experiment can't surprise-delete its own log.
  * Anything explicitly listed in ``brian.toml`` under
    ``[clean] extra_keep = [...]``.
  * Anything that is currently staged or has uncommitted changes in
    the git index.

Subcommands:

    brian clean logs           — operate on log files only
    brian clean checkpoints    — operate on checkpoints only
    brian clean docs           — operate on docs/archive markdown
    brian clean all            — all three buckets

Default mode is **dry-run** (prints what *would* be deleted, exit 0).
Pass ``--force`` to actually unlink.

Usage:

    python -m neuroslm.tools.clean logs           # dry-run
    python -m neuroslm.tools.clean logs --force   # actually delete
    python -m neuroslm.tools.clean all --force --keep-recent 5
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

# ── Reference scanner (extracted to neuroslm.references) ───────────────
# All the "what is referenced?" logic lives in one place now so the LFS
# pruner AND brian migrate can share it. The names below are re-exported
# at module level for back-compat with tests that still do
# ``from neuroslm.tools.clean import ReferenceIndex, build_reference_index``.
from neuroslm.references import (  # noqa: F401  (re-export)
    REPO_ROOT,
    ReferenceIndex,
    _FINDING_MARKERS,
    _MAX_SCAN_BYTES,
    _SKIP_DIRS,
    _SKIP_DIR_PREFIXES,
    _TEXT_SUFFIXES,
    _TOKEN_RE,
    _is_informative_glob,
    _iter_text_files,
    build_reference_index,
)

# Anchor files that NEVER appear on a delete list, regardless of bucket
# or reference-graph state. Match is by basename, case-insensitive.
_ANCHOR_BASENAMES: frozenset = frozenset({
    ".gitkeep", ".gitignore", ".gitattributes",
    "readme.md", "instructions.md", "license", "license.md",
    "contributing.md", "claude.md",
    # Scientific anchors under docs/
    "findings.md", "insights.md", "architecture.md", "cli.md",
    "brian.md", "formal_framework.md", "history.md", "changelog.md",
    "metrics.md", "technical_report.md", "harness.md",
    "emergent_topology.md", "ood_mechanisms.md",
    "dsl.md", "dsl_editor_setup.md", "dsl_nn_language.md",
    "dsl_subsystem_roadmap.md", "cdga.md",
    "thsd_formal_verification.md", "thsd_implementation_summary.md",
    "thsd_refactor_plan.md", "heatmap_evolution_plan.md",
    "features.md",
})


# ── Git-aware safety net ───────────────────────────────────────────────


def _git_protected_paths(root: Path = REPO_ROOT) -> Set[Path]:
    """Files that are staged or have uncommitted changes — never delete."""
    out: Set[Path] = set()
    try:
        # Staged + modified + untracked (so a freshly-saved log isn't nuked).
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root, capture_output=True, text=True, check=False,
        )
        for line in r.stdout.splitlines():
            if len(line) < 4:
                continue
            # Porcelain v1: "XY path" (or "XY path -> path" for renames).
            rest = line[3:]
            if " -> " in rest:
                rest = rest.split(" -> ", 1)[1]
            out.add((root / rest.strip().strip('"')).resolve())
    except (OSError, FileNotFoundError):
        pass
    return out


# ── Buckets ────────────────────────────────────────────────────────────


@dataclass
class Bucket:
    name: str
    description: str
    enumerate_fn: Callable[[Path], List[Path]]
    keep_predicate: Callable[[Path], bool]  # True = always keep

    def candidates(self, root: Path) -> List[Path]:
        return [p for p in self.enumerate_fn(root) if not self.keep_predicate(p)]


# ── logs bucket ────────────────────────────────────────────────────────


def _enumerate_logs(root: Path) -> List[Path]:
    out: List[Path] = []
    # vast.ai raw stdout (legacy flat layout)
    out.extend(sorted((root / "logs" / "vast").glob("*.log")))
    # analyzed insight markdowns are NOT enumerated here — they're docs.
    # Root-level stray training/debug logs
    for name in ("debug.log", "training.log", "training.err",
                 "_dep2.log", "inspect.log", "temp.log", "nul"):
        p = root / name
        if p.is_file():
            out.append(p)
    # Per-package debug log
    p = root / "docs" / "debug.log"
    if p.is_file():
        out.append(p)
    # Per-run log FOLDERS produced by the 0001 logs-layout migration
    # (``logs/<YYYYMMDD-HHMMSS>_<arch>_<params>[_<label>]_<instance>/``).
    # The whole folder is the unit of pruning: a folder is "referenced"
    # iff its full date-prefixed name appears verbatim somewhere in the
    # repo (the contained ``train.log`` basename is identical across
    # every run and so cannot be used as a discriminator). Folder names
    # are seeded into ``idx.exact`` via ``_FOLDER_TOKEN_RE`` in
    # ``neuroslm.references``. The folder must contain a ``train.log``
    # to count — anything else parked under ``logs/`` (vast/, archive/,
    # ad-hoc dirs the user created) is left alone.
    logs_dir = root / "logs"
    if logs_dir.is_dir():
        for child in sorted(logs_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name == "vast":
                continue  # legacy flat-layout subdir, walked above
            if (child / "train.log").is_file():
                out.append(child)
    return out


def _keep_logs(p: Path) -> bool:
    name = p.name.lower()
    if name in _ANCHOR_BASENAMES:
        return True
    # Per-instance benchmark dirs are kept until the parent gets cleaned.
    return False


# ── checkpoints bucket ─────────────────────────────────────────────────


def _enumerate_checkpoints(root: Path) -> List[Path]:
    ck = root / "lfs_checkpoints"
    if not ck.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(ck.iterdir()):
        if not p.is_file():
            continue
        if p.name in (".gitkeep", ".gitignore"):
            continue
        out.append(p)
    # Legacy `checkpoints/` dir, if it exists.
    legacy = root / "checkpoints"
    if legacy.is_dir():
        for p in sorted(legacy.iterdir()):
            if p.is_file() and p.suffix in (".pt", ".mem", ".json"):
                out.append(p)
    return out


def _keep_checkpoints(p: Path) -> bool:
    name = p.name.lower()
    if name in _ANCHOR_BASENAMES:
        return True
    # _best.* checkpoints (and their sidecar .mem / .json / .mem.json)
    # are always kept.
    stem_lower = p.name.lower()
    if "_best." in stem_lower or stem_lower.endswith("_best.pt"):
        return True
    # Sidecars whose stem ends with `_best`
    if re.search(r"_best\.(mem|json|mem\.json)$", stem_lower):
        return True
    return False


# ── docs bucket ────────────────────────────────────────────────────────


def _enumerate_docs(root: Path) -> List[Path]:
    """Conservative: only `docs/archive/` and one-off `docs/debug.log`-style.

    The active scientific docs (FINDINGS.md, architecture.md, etc.) are
    anchored. Anything under `docs/archive/` is a candidate; anything
    elsewhere under `docs/` is only a candidate if it isn't an anchor
    AND doesn't contain finding markers (the latter is enforced by
    `is_deletable`, not here).
    """
    out: List[Path] = []
    archive = root / "docs" / "archive"
    if archive.is_dir():
        for p in sorted(archive.rglob("*")):
            if p.is_file():
                out.append(p)
    return out


def _keep_docs(p: Path) -> bool:
    return p.name.lower() in _ANCHOR_BASENAMES


# ── bucket registry ────────────────────────────────────────────────────

BUCKETS: dict = {
    "logs": Bucket(
        name="logs",
        description="raw vast.ai stdout logs + stray *.log at the repo root",
        enumerate_fn=_enumerate_logs,
        keep_predicate=_keep_logs,
    ),
    "checkpoints": Bucket(
        name="checkpoints",
        description="lfs_checkpoints/*.pt + their .mem / .json sidecars",
        enumerate_fn=_enumerate_checkpoints,
        keep_predicate=_keep_checkpoints,
    ),
    "docs": Bucket(
        name="docs",
        description="docs/archive/* — historical design notes",
        enumerate_fn=_enumerate_docs,
        keep_predicate=_keep_docs,
    ),
}


# ── decision engine ────────────────────────────────────────────────────


def _most_recent(paths: Sequence[Path], n: int) -> Set[Path]:
    """Return resolved-paths of the N most recently-modified files."""
    try:
        ranked = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        ranked = list(paths)
    return {p.resolve() for p in ranked[:n]}


def _load_extra_keep(root: Path) -> Set[str]:
    """Parse `[clean] extra_keep = […]` from brian.toml if present."""
    cfg = root / "brian.toml"
    if not cfg.is_file():
        return set()
    try:
        try:
            import tomllib  # py311+
        except ImportError:  # pragma: no cover - legacy py
            import tomli as tomllib  # type: ignore[no-redef]
        data = tomllib.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return set()
    clean = data.get("clean", {}) or {}
    return set(clean.get("extra_keep", []) or [])


@dataclass
class CleanPlan:
    bucket: str
    delete: List[Path] = field(default_factory=list)
    keep_referenced: List[Path] = field(default_factory=list)
    keep_recent: List[Path] = field(default_factory=list)
    keep_anchor: List[Path] = field(default_factory=list)
    keep_git: List[Path] = field(default_factory=list)
    keep_finding: List[Path] = field(default_factory=list)
    keep_extra: List[Path] = field(default_factory=list)

    @property
    def bytes_to_free(self) -> int:
        total = 0
        for p in self.delete:
            try:
                if p.is_dir():
                    # Per-run log folders are sized by the union of
                    # their contained files (currently just train.log,
                    # but tomorrow's runs may drop metrics.json /
                    # config.toml / etc. into the same dir).
                    for sub in p.rglob("*"):
                        if sub.is_file():
                            try:
                                total += sub.stat().st_size
                            except OSError:
                                pass
                else:
                    total += p.stat().st_size
            except OSError:
                pass
        return total

    @property
    def total_kept(self) -> int:
        return (len(self.keep_referenced) + len(self.keep_recent)
                + len(self.keep_anchor) + len(self.keep_git)
                + len(self.keep_finding) + len(self.keep_extra))


def plan_for_bucket(
    bucket_name: str,
    idx: ReferenceIndex,
    *,
    root: Path = REPO_ROOT,
    keep_recent: int = 3,
    extra_keep: Optional[Set[str]] = None,
    git_protected: Optional[Set[Path]] = None,
) -> CleanPlan:
    """Compute what would be deleted in `bucket_name` without doing it."""
    if bucket_name not in BUCKETS:
        raise KeyError(f"unknown bucket {bucket_name!r}")
    bucket = BUCKETS[bucket_name]
    extra_keep = extra_keep or set()
    git_protected = git_protected if git_protected is not None else _git_protected_paths(root)

    candidates = bucket.candidates(root)
    recent = _most_recent(candidates, keep_recent)
    plan = CleanPlan(bucket=bucket_name)

    # extra_keep can be relative or basename; normalise to a set of (path, name).
    extra_keep_paths = {(root / k).resolve() for k in extra_keep}
    extra_keep_names = {Path(k).name for k in extra_keep}

    for p in candidates:
        resolved = p.resolve()
        name = p.name

        if bucket.keep_predicate(p):
            plan.keep_anchor.append(p)
            continue
        if resolved in extra_keep_paths or name in extra_keep_names:
            plan.keep_extra.append(p)
            continue
        if resolved in git_protected:
            plan.keep_git.append(p)
            continue
        if resolved in recent:
            plan.keep_recent.append(p)
            continue
        # Finding-doc protection (docs bucket only — relevant for md files).
        if p.suffix.lower() == ".md" and resolved in idx.finding_files:
            plan.keep_finding.append(p)
            continue
        if idx.references(name):
            plan.keep_referenced.append(p)
            continue
        plan.delete.append(p)

    return plan


def execute_plan(plan: CleanPlan, *, use_git: bool = True,
                 root: Path = REPO_ROOT) -> Tuple[int, int]:
    """Delete the planned paths. Returns (deleted_count, error_count).

    If ``use_git`` is true and the path is tracked, deletion is staged
    via ``git rm`` (so LFS pointers + history are updated correctly).
    Untracked files fall back to a plain ``Path.unlink``. Directories
    (per-run log folders) are removed recursively via ``git rm -rf``
    when tracked and ``shutil.rmtree`` otherwise.
    """
    import shutil

    deleted, errors = 0, 0
    tracked_cache: Optional[Set[str]] = None

    if use_git:
        try:
            r = subprocess.run(
                ["git", "ls-files"], cwd=root, capture_output=True,
                text=True, check=False,
            )
            tracked_cache = {line.strip() for line in r.stdout.splitlines() if line.strip()}
        except (OSError, FileNotFoundError):
            tracked_cache = set()

    for p in plan.delete:
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            rel = str(p)
        try:
            is_dir = p.is_dir()
            # Tracked-ness probe: ``git ls-files`` lists files only, so
            # a directory is "tracked" iff at least one descendant is.
            is_tracked = False
            if use_git and tracked_cache is not None:
                if is_dir:
                    prefix = rel + "/"
                    is_tracked = any(t.startswith(prefix) for t in tracked_cache)
                else:
                    is_tracked = rel in tracked_cache

            if is_tracked:
                cmd = (["git", "rm", "-rf", "--", rel] if is_dir
                       else ["git", "rm", "-f", "--", rel])
                r = subprocess.run(
                    cmd, cwd=root, capture_output=True, text=True, check=False,
                )
                if r.returncode != 0:
                    # Fall back to a plain filesystem delete so a hung
                    # or angry git won't block us.
                    if is_dir:
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        p.unlink(missing_ok=True)
            else:
                if is_dir:
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
            deleted += 1
        except OSError as e:
            print(f"  ! failed to delete {rel}: {e}", file=sys.stderr)
            errors += 1
    return deleted, errors


# ── pretty printing ────────────────────────────────────────────────────


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        n_f = n / 1024
        if n_f < 1024:
            return f"{n_f:.1f} {unit}"
        n //= 1024
    return f"{n} PiB"


# Choose ASCII-safe separators on legacy Windows consoles where stdout
# can't encode U+2500. We test by trying to encode the box-drawing char
# through stdout's encoding.
def _ascii_safe() -> bool:
    enc = (getattr(sys.stdout, "encoding", None) or "ascii").lower()
    try:
        "\u2500".encode(enc)
        return False
    except (UnicodeEncodeError, LookupError):
        return True


if _ascii_safe():
    _SEP = "-"
    _BULLET_DEL = "-"
    _BULLET_KEEP = "+"
else:
    _SEP = "\u2500"  # ─
    _BULLET_DEL = "-"
    _BULLET_KEEP = "+"


def print_plan(plan: CleanPlan, root: Path = REPO_ROOT, *,
               verbose: bool = False) -> None:
    bar = _SEP * 30
    print(f"\n{_SEP * 2} bucket: {plan.bucket} {bar}")
    print(f"  total candidates examined : {plan.total_kept + len(plan.delete)}")
    print(f"  would DELETE              : {len(plan.delete)}"
          f"  ({_fmt_size(plan.bytes_to_free)} reclaimable)")
    print(f"  kept * referenced         : {len(plan.keep_referenced)}")
    print(f"  kept * finding-md         : {len(plan.keep_finding)}")
    print(f"  kept * anchor / best      : {len(plan.keep_anchor)}")
    print(f"  kept * recent (mtime)     : {len(plan.keep_recent)}")
    print(f"  kept * git-dirty/staged   : {len(plan.keep_git)}")
    print(f"  kept * extra_keep (toml)  : {len(plan.keep_extra)}")

    if plan.delete:
        print(f"\n  {_SEP * 2} will delete {_SEP * 2}")
        for p in plan.delete:
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                rel = str(p)
            try:
                if p.is_dir():
                    total = 0
                    for sub in p.rglob("*"):
                        if sub.is_file():
                            try:
                                total += sub.stat().st_size
                            except OSError:
                                pass
                    rel = rel + "/"
                    sz = _fmt_size(total)
                else:
                    sz = _fmt_size(p.stat().st_size)
            except OSError:
                sz = "?"
            print(f"    {_BULLET_DEL} {rel}  ({sz})")
    if verbose:
        for label, items in (("referenced", plan.keep_referenced),
                             ("finding-md", plan.keep_finding),
                             ("anchor",     plan.keep_anchor),
                             ("recent",     plan.keep_recent),
                             ("git-dirty",  plan.keep_git),
                             ("extra_keep", plan.keep_extra)):
            if not items:
                continue
            print(f"\n  {_SEP * 2} kept * {label} {_SEP * 2}")
            for p in items:
                try:
                    rel = p.relative_to(root).as_posix()
                except ValueError:
                    rel = str(p)
                print(f"    {_BULLET_KEEP} {rel}")


# ── CLI ────────────────────────────────────────────────────────────────


def run(
    bucket_names: Sequence[str],
    *,
    force: bool = False,
    verbose: bool = False,
    keep_recent: int = 3,
    use_git: bool = True,
    root: Path = REPO_ROOT,
) -> int:
    """Run the clean pipeline. Returns process exit code."""
    if "all" in bucket_names:
        bucket_names = ["logs", "checkpoints", "docs"]

    # Pass --skip the bucket dirs themselves; their internal filenames
    # don't count as "references" to one another.
    skip = ("logs", "lfs_checkpoints", "checkpoints", "docs/archive")
    print(f"[clean] scanning references under {root} (skipping {skip}) ...")
    idx = build_reference_index(root, skip_dirs=skip)
    print(f"[clean] index: {len(idx.exact)} basenames "
          f"(exact-only matching), "
          f"{len(idx.finding_files)} finding-style md files")

    git_protected = _git_protected_paths(root)
    extra_keep = _load_extra_keep(root)

    total_delete, total_bytes = 0, 0
    plans: List[CleanPlan] = []
    for name in bucket_names:
        plan = plan_for_bucket(
            name, idx, root=root, keep_recent=keep_recent,
            extra_keep=extra_keep, git_protected=git_protected,
        )
        plans.append(plan)
        print_plan(plan, root=root, verbose=verbose)
        total_delete += len(plan.delete)
        total_bytes += plan.bytes_to_free

    rule = _SEP * 60
    print(f"\n{rule}")
    print(f"  GRAND TOTAL: {total_delete} files, "
          f"{_fmt_size(total_bytes)} reclaimable across {len(plans)} bucket(s)")
    if not force:
        print("  (dry-run -- re-run with --force to actually delete)")
        print(rule)
        return 0

    print("  --force given -- DELETING NOW ...")
    print(rule)
    total_done, total_err = 0, 0
    for plan in plans:
        d, e = execute_plan(plan, use_git=use_git, root=root)
        total_done += d
        total_err += e
        print(f"  {_BULLET_KEEP} {plan.bucket}: deleted {d}, errors {e}")
    print(f"\n  done: {total_done} deleted, {total_err} errors")
    return 1 if total_err else 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="brian clean",
        description="Find and delete unreferenced logs / checkpoints / docs.",
    )
    p.add_argument("bucket", nargs="+",
                   choices=sorted(list(BUCKETS) + ["all"]),
                   help="which buckets to clean")
    p.add_argument("--force", action="store_true",
                   help="actually delete (default is dry-run)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="also list every kept file with the reason it was kept")
    p.add_argument("--keep-recent", type=int, default=3,
                   help="number of most-recent files per bucket to always keep "
                        "(default 3)")
    p.add_argument("--no-git", action="store_true",
                   help="don't stage deletions via `git rm` — plain unlink only")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    return run(
        args.bucket,
        force=args.force,
        verbose=args.verbose,
        keep_recent=args.keep_recent,
        use_git=not args.no_git,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
