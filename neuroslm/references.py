"""Reference-aware repo scanner — the one place that knows what counts
as "referenced" anywhere in a NeuroSLM repo.

Why this is its own module:
    Before extraction, the reference index was buried inside
    ``neuroslm/tools/clean.py``. Both ``clean.py`` (delete-anything
    bucket janitor) and ``clean_lfs.py`` (LFS pruner) need it, and the
    upcoming ``brian migrate`` framework needs it too (a migration that
    moves logs into the new run-folder layout must only touch files
    that some doc/finding actually cites). Three callers ⇒ one module.

Public surface (KISS):
    * :class:`ReferenceIndex`           — the scan result
    * :func:`build_reference_index`     — the scanner
    * :data:`REPO_ROOT`                 — repo root constant
    * :data:`_FINDING_MARKERS`          — finding-doc heuristic tokens
    * :data:`_SKIP_DIR_PREFIXES`        — prefix-skip directory names
    * :data:`_MAX_SCAN_BYTES`           — per-file scan cap (1 MiB)

Everything else (regex, helper iterators, glob informativeness) is
underscore-prefixed but importable for tests.
"""
from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Set, Tuple


# ── Repo discovery ─────────────────────────────────────────────────────

# This module lives at ``neuroslm/references.py`` ⇒ repo root is two
# levels up (``neuroslm/`` → repo).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent


# ── Directory pruning ──────────────────────────────────────────────────

# Directories never scanned for references AND never enumerated for
# candidates. Anything inside these is invisible to ``clean``.
# NOTE: prefix-match — any dir whose name *starts with* one of these is
# skipped, so ``.venv-1``, ``.venv-old``, ``__pycache_old__`` all get
# pruned in one rule.
_SKIP_DIR_PREFIXES: Tuple[str, ...] = (
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    "neuroslm.egg-info",
    ".claude",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "site-packages",
    "dist",
    "build",
)

# Back-compat alias kept for any external code that imported the old
# name. New code should consult ``_SKIP_DIR_PREFIXES`` directly.
_SKIP_DIRS: frozenset = frozenset(_SKIP_DIR_PREFIXES)


# ── Scan caps ──────────────────────────────────────────────────────────

# Files larger than this are skipped during reference scanning. A real
# scientific record fits in KB; multi-MB text files are notebooks with
# inline outputs, large logs, or generated code that have no business
# in the protection index.
_MAX_SCAN_BYTES: int = 1 * 1024 * 1024  # 1 MiB


# ── File suffix gate ───────────────────────────────────────────────────

# Suffixes whose text content we scan for references. We *don't* read
# binary checkpoints — that would defeat the point.
_TEXT_SUFFIXES: frozenset = frozenset({
    ".md", ".py", ".ipynb", ".json", ".yaml", ".yml",
    ".toml", ".txt", ".rst", ".cfg", ".ini",
})


# ── Finding markers ────────────────────────────────────────────────────

# A markdown file that contains any of these markers is treated as a
# scientific record and is *itself* protected from ``clean docs`` even
# if nothing references its filename. Mirrors the style FINDINGS.md
# uses.
_FINDING_MARKERS: Tuple[str, ...] = (
    "✅ CONFIRMED",
    "❌ FALSIFIED",
    "🟡 PARTIAL",
    "🟠 PENDING",
    "**Hypothesis.**",
    "**Status.**",
    "**Spec.**",
    "## H",  # H1 — ..., H21 —, etc.
)


# ── Token regex ────────────────────────────────────────────────────────

# A token that *looks like* a filename we care about: a word with at
# least one allowed-relevant suffix. Permits glob metacharacters so
# patterns like ``20260614*_…_step2kof2k.log`` are captured.
_TOKEN_RE = re.compile(
    r"[A-Za-z0-9_*?\-\.]+\.(?:log|pt|mem|json|md|ipynb|txt|err|html|yaml|yml)\b",
    re.IGNORECASE,
)


# ── Reference index ────────────────────────────────────────────────────


@dataclass
class ReferenceIndex:
    """All literal and glob filename tokens found anywhere in the repo.

    A basename "is referenced" iff it passes ``references()`` — which
    bakes in three matching modes: exact basename, stem (basename minus
    extension), and glob (with permissive leading-``*`` and
    distinctive-segment fallbacks).
    """

    exact: Set[str] = field(default_factory=set)        # basenames seen verbatim
    globs: Set[str] = field(default_factory=set)        # tokens containing * or ?
    stems: Set[str] = field(default_factory=set)        # basename minus extension
    finding_files: Set[Path] = field(default_factory=set)  # md files w/ finding markers

    def references(self, basename: str) -> bool:
        if basename in self.exact:
            return True
        stem = basename.rsplit(".", 1)[0] if "." in basename else basename
        if stem in self.stems:
            return True
        for g in self.globs:
            # 1) Strict fnmatch as written.
            if fnmatch.fnmatch(basename, g):
                return True
            # 2) Permissive: also try with an implicit leading `*`, so a
            #    glob written as `20260614*_…_step2kof2k.log` (where the
            #    leading literal was the example author's run prefix)
            #    still matches any prefix.
            if not g.startswith("*") and fnmatch.fnmatch(basename, "*" + g):
                return True
            # 3) Distinctive-literal-segment match: split the glob on
            #    `*`/`?` and if ANY single literal segment of ≥8 chars
            #    appears in `basename`, treat it as a reference. This
            #    makes a finding-doc glob like
            #    ``20260614*_keep_me_glob_step2kof2k.log`` correctly
            #    protect ``af758c381388_keep_me_glob_step2kof2k.log``.
            for seg in re.split(r"[*?]+", g):
                if len(seg) >= 8 and seg in basename:
                    return True
        return False


# ── Glob informativeness gate ──────────────────────────────────────────


def _is_informative_glob(g: str) -> bool:
    """Reject globs that don't pin down a specific run.

    ``*.pt`` / ``*.log`` / ``*.mem.json`` etc. would otherwise match
    every checkpoint, silently neutering the entire reference filter.
    We require at least one literal segment of ≥4 characters BEFORE
    the final extension. (The extension itself doesn't count —
    ``pt``/``log`` are too generic to identify a single artifact.)
    """
    # Strip the final ".ext" so the extension can't act as the
    # distinctive segment.
    body = g.rsplit(".", 1)[0] if "." in g else g
    for seg in re.split(r"[*?]+", body):
        if len(seg.strip("._-")) >= 4:
            return True
    return False


# ── Filesystem walker ──────────────────────────────────────────────────


def _iter_text_files(root: Path, extra_skip_dirs: Iterable[str] = ()) -> Iterable[Path]:
    """Walk ``root`` yielding every text file we should scan.

    Directory pruning is **prefix-based** — any folder whose name
    starts with one of the configured prefixes is skipped (so
    ``.venv-1``, ``.venv-old``, etc. all get cut). Extra skip names
    passed via ``extra_skip_dirs`` are exact-match only.
    """
    extra = set(extra_skip_dirs)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in extra and not any(
                d.startswith(pfx) for pfx in _SKIP_DIR_PREFIXES
            )
        ]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in _TEXT_SUFFIXES:
                yield p


# ── Main entry point ───────────────────────────────────────────────────


def build_reference_index(
    root: Path = REPO_ROOT,
    skip_dirs: Iterable[str] = (),
    *,
    progress: bool = False,
    max_bytes: int = _MAX_SCAN_BYTES,
) -> ReferenceIndex:
    """Scan ``root`` for every basename-shaped token + finding markers.

    Files larger than ``max_bytes`` are silently skipped — those are
    notebooks-with-outputs / logs / generated dumps that bloat scan
    time without contributing real references. Pass ``progress=True``
    to print a one-line counter every 100 files so the user knows the
    scan is alive on large repos.
    """
    idx = ReferenceIndex()
    seen = 0
    skipped_big = 0
    for fp in _iter_text_files(root, extra_skip_dirs=skip_dirs):
        seen += 1
        if progress and seen % 100 == 0:
            print(f"[refs] scanned {seen} files...", flush=True)
        try:
            if fp.stat().st_size > max_bytes:
                skipped_big += 1
                continue
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        # Reference tokens
        for m in _TOKEN_RE.finditer(text):
            tok = m.group(0)
            if any(c in tok for c in "*?"):
                # Drop uninformative globs like `*.pt` that would match
                # every checkpoint in the repo.
                if _is_informative_glob(tok):
                    idx.globs.add(tok)
            else:
                idx.exact.add(tok)
                # Also store basename-only form so refs like
                # ``lfs_checkpoints/foo.pt`` register ``foo.pt``.
                base = Path(tok).name
                idx.exact.add(base)
                if "." in base:
                    idx.stems.add(base.rsplit(".", 1)[0])

        # Mark finding-style markdowns for docs-bucket protection
        if fp.suffix.lower() == ".md":
            if any(marker in text for marker in _FINDING_MARKERS):
                idx.finding_files.add(fp.resolve())

    if progress:
        print(f"[refs] scanned {seen} files "
              f"({skipped_big} skipped as >{max_bytes // 1024} KiB), "
              f"{len(idx.exact)} basenames, {len(idx.globs)} globs",
              flush=True)
    return idx


__all__ = [
    "REPO_ROOT",
    "ReferenceIndex",
    "build_reference_index",
    "_FINDING_MARKERS",
    "_MAX_SCAN_BYTES",
    "_SKIP_DIRS",
    "_SKIP_DIR_PREFIXES",
    "_TEXT_SUFFIXES",
    "_TOKEN_RE",
    "_is_informative_glob",
    "_iter_text_files",
]
