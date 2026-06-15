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

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple


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

# Per-run log folder names emitted by the 0001 logs-layout migration:
# ``<YYYYMMDD>-<HHMMSS>_<arch>_<params>[_<label>]_<instance>``. These
# carry NO file suffix, so ``_TOKEN_RE`` (which requires one of the
# allowed extensions) can never see them. Without this second regex,
# ``brian clean logs`` would have no way to detect a doc citing a run
# folder by name and would happily prune every per-run folder that
# isn't in the N-most-recent window. Pattern intentionally narrow —
# anchored on the date-prefix shape — so arbitrary tokens like
# ``foo_bar_baz`` don't get pulled into the reference set.
_FOLDER_TOKEN_RE = re.compile(
    r"\b\d{8}-\d{6}_[A-Za-z0-9][A-Za-z0-9_\-]+",
)


# ── Reference index ────────────────────────────────────────────────────


@dataclass
class ReferenceIndex:
    """All literal filename tokens found anywhere in the repo.

    Matching semantics (EXACT-ONLY)
    -------------------------------
    A basename "is referenced" iff it appears VERBATIM in ``self.exact``.
    Stems (basename minus extension) and glob tokens (containing
    ``*`` / ``?``) are NOT consulted. Path-prefixed citations like
    ``lfs_checkpoints/foo.pt`` register their basename ``foo.pt`` in
    ``exact`` (see :func:`build_reference_index`), so the doc author
    doesn't have to write the bare basename to protect a file.

    Why exact-only
    --------------
    The earlier three-matcher contract (exact / stem / glob with a
    distinctive-segment fallback) silently protected ~25 LFS pointers
    whose only "reference" was a glob example inside a docstring or a
    test-fixture filename that happened to share an ≥8-char substring
    with the production checkpoint. See the H22 forensic in
    ``docs/FINDINGS.md`` ("LFS prune keeps everything"). With the
    exact-only rule, a file is protected iff *some scanned text file
    spells its basename verbatim* — no surprises.

    The ``globs`` and ``stems`` fields are kept for back-compat with
    external diagnostics that may inspect ``ReferenceIndex`` directly,
    but :func:`build_reference_index` no longer populates them and
    :meth:`references` no longer consults them.
    Regression-pinned by ``tests/test_references_exact_only.py``.
    """

    exact: Set[str] = field(default_factory=set)        # basenames seen verbatim
    globs: Set[str] = field(default_factory=set)        # legacy, always empty
    stems: Set[str] = field(default_factory=set)        # legacy, always empty
    finding_files: Set[Path] = field(default_factory=set)  # md files w/ finding markers

    def references(self, basename: str) -> bool:
        """Return True iff ``basename`` appears verbatim in
        ``self.exact``. No stem or glob expansion is performed."""
        return basename in self.exact


# ── Glob informativeness gate (DEPRECATED) ─────────────────────────────


def _is_informative_glob(g: str) -> bool:
    """Deprecated under the EXACT-ONLY contract.

    Previously used to reject uninformative globs like ``*.pt`` /
    ``*.log`` before they could pollute :attr:`ReferenceIndex.globs`.
    Now that :func:`build_reference_index` drops glob tokens entirely
    and :meth:`ReferenceIndex.references` never consults globs, this
    function has no callers in the canonical path and always returns
    ``False`` so any legacy caller behaves as if every glob is
    uninformative (i.e. not a reference). Kept only so ``from
    neuroslm.references import _is_informative_glob`` doesn't break
    out-of-tree diagnostics. Regression-pinned by
    ``tests/test_references_exact_only.py``.
    """
    return False


# ── Filesystem walker ──────────────────────────────────────────────────


def _iter_text_files(
    root: Path,
    extra_skip_dirs: Iterable[str] = (),
    *,
    suffixes: Optional[Iterable[str]] = None,
) -> Iterable[Path]:
    """Walk ``root`` yielding every text file we should scan.

    Directory pruning is **prefix-based** — any folder whose name
    starts with one of the configured prefixes is skipped (so
    ``.venv-1``, ``.venv-old``, etc. all get cut). Extra skip names
    passed via ``extra_skip_dirs`` are exact-match only.

    ``suffixes`` (case-insensitive) restricts which file extensions
    are yielded. When ``None`` (default), the full :data:`_TEXT_SUFFIXES`
    set is used — back-compat with the existing ``brian clean`` flow.
    Pass e.g. ``{".md"}`` to scan only markdown (used by the LFS
    pruner: only scientific records should pin large binary blobs).
    """
    extra = set(extra_skip_dirs)
    if suffixes is None:
        allowed = _TEXT_SUFFIXES
    else:
        allowed = frozenset(s.lower() for s in suffixes)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in extra and not any(
                d.startswith(pfx) for pfx in _SKIP_DIR_PREFIXES
            )
        ]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in allowed:
                yield p


# ── Main entry point ───────────────────────────────────────────────────


def build_reference_index(
    root: Path = REPO_ROOT,
    skip_dirs: Iterable[str] = (),
    *,
    progress: bool = False,
    max_bytes: int = _MAX_SCAN_BYTES,
    text_suffixes: Optional[Iterable[str]] = None,
) -> ReferenceIndex:
    """Scan ``root`` for every basename-shaped token + finding markers.

    Files larger than ``max_bytes`` are silently skipped — those are
    notebooks-with-outputs / logs / generated dumps that bloat scan
    time without contributing real references. Pass ``progress=True``
    to print a one-line counter every 100 files so the user knows the
    scan is alive on large repos.

    ``text_suffixes`` (case-insensitive) restricts which file types
    are scanned for references. When ``None`` (default), the full
    :data:`_TEXT_SUFFIXES` set is used — back-compat with the
    existing ``brian clean logs/checkpoints/docs`` flow which needs
    cross-filetype awareness. The LFS pruner overrides this with
    ``{".md"}`` so only scientific records (FINDINGS.md,
    technical_report.md, archived findings) can protect a large
    binary blob — random docstring examples, test fixture names,
    JSON ood-result blobs, and CLI permission allow-lists no longer
    accidentally pin checkpoints. Regression-pinned by
    ``tests/test_references_exact_only.py::TestBuildReferenceIndexSuffixScope``.
    """
    idx = ReferenceIndex()
    seen = 0
    skipped_big = 0
    for fp in _iter_text_files(
        root, extra_skip_dirs=skip_dirs, suffixes=text_suffixes,
    ):
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

        # Reference tokens — EXACT-ONLY contract: glob tokens
        # (containing ``*`` / ``?``) are dropped on the floor; stems
        # are no longer derived. See ReferenceIndex docstring +
        # tests/test_references_exact_only.py for the rationale.
        for m in _TOKEN_RE.finditer(text):
            tok = m.group(0)
            if "*" in tok or "?" in tok:
                # Drop glob tokens entirely. The old behaviour added
                # them to ``idx.globs`` and matched them at lookup
                # time, which silently pinned files whose basenames
                # only shared a ≥8-char substring with the glob's
                # literal segments (see docs/FINDINGS.md "H22 LFS
                # prune keeps everything" forensic).
                continue
            idx.exact.add(tok)
            # Also store basename-only form so refs like
            # ``lfs_checkpoints/foo.pt`` register ``foo.pt`` in the
            # exact set — the doc author shouldn't have to spell out
            # the bare basename to protect a file.
            base = Path(tok).name
            idx.exact.add(base)

        # Per-run log folder names — same exact-only contract but with
        # a separate regex because folder names have no file suffix.
        # ``train.log`` itself is too generic to use as a folder key
        # (every run has one), so the folder is protected ONLY if its
        # full date-prefixed name appears verbatim in some scanned
        # text file.
        for m in _FOLDER_TOKEN_RE.finditer(text):
            idx.exact.add(m.group(0))

        # Mark finding-style markdowns for docs-bucket protection
        if fp.suffix.lower() == ".md":
            if any(marker in text for marker in _FINDING_MARKERS):
                idx.finding_files.add(fp.resolve())

    if progress:
        print(f"[refs] scanned {seen} files "
              f"({skipped_big} skipped as >{max_bytes // 1024} KiB), "
              f"{len(idx.exact)} basenames (exact-only matching; "
              "glob tokens dropped)",
              flush=True)
    return idx


__all__ = [
    "REPO_ROOT",
    "ReferenceIndex",
    "build_reference_index",
    "_FINDING_MARKERS",
    "_FOLDER_TOKEN_RE",
    "_MAX_SCAN_BYTES",
    "_SKIP_DIRS",
    "_SKIP_DIR_PREFIXES",
    "_TEXT_SUFFIXES",
    "_TOKEN_RE",
    "_is_informative_glob",
    "_iter_text_files",
]
