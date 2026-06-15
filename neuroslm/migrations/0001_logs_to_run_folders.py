"""0001 — copy referenced logs/vast/*.log into per-run folders.

Why:
    Early in the project every training run dumped a flat
    ``logs/vast/<sha>__neuroslm-full.log``. As the experiment count grew
    this became unusable: no date, no architecture, no step count, just
    a sha and a generic suffix. Recent runs use a richer name with an
    explicit UTC timestamp + sha + arch + step ratio, but the *old*
    files still sit alongside.

    This migration normalizes EVERYTHING into the per-run folder layout:

        logs/<YYYYMMDD>-<HHMMSS>_<arch>_<short-sha>/train.log

    Only logs whose basename is REFERENCED somewhere in the repo
    (docs, py, ipynb, ...) are copied. Unreferenced logs stay in
    ``logs/vast/`` so ``brian clean logs`` can decide what to do.
    Unparseable filenames are routed to ``logs/_unsorted_legacy/``.

How it stays idempotent:
    plan() inspects only ``logs/vast/`` (NOT ``logs/<folder>/...``).
    For each candidate it computes the destination path; if the file
    already exists there, no Op is emitted. Re-running plan() after
    apply() therefore yields ``[]`` and the migration shows APPLIED in
    ``brian migrate --list``.

Safety:
    apply() COPIES (shutil.copy2), it does NOT move. The source file
    remains in ``logs/vast/`` after migration. Pruning belongs to
    ``brian clean``, not to a migration.
"""
from __future__ import annotations

import datetime as _dt
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from neuroslm.migrations._framework import Context, Op


ID: str = "0001_logs_to_run_folders"
DESCRIPTION: str = (
    "Copy referenced logs/vast/*.log into "
    "logs/<date>_<arch>_<sha>/train.log folders"
)


# ── Filename parser ────────────────────────────────────────────────────


# New format example:
#   20260614T182653Z_07aba24be2bf_rcc_bowtie_889M_run_step920of10k.log
# The leading UTC timestamp + sha are anchored; everything between sha
# and the optional trailing _stepN[k|m]ofM[k|m] is treated as the
# architecture token (with underscores preserved).
#
# Step suffix grammar: ``_step <digits> [k|m|g] [of <digits> [k|m|g]]``
# Examples that must all parse:
#   _step920of10k     (raw → compact)
#   _step10kof10k     (compact → compact)
#   _step3540of10k    (raw → compact)
#   _step3kof3k       (compact → compact)  ← was broken pre-2026-06-15:
#                       the old grammar required ``of`` immediately
#                       after the first ``\d+`` group, so ``step3kof3k``
#                       fell into the arch token instead of being
#                       stripped. Fix: allow optional ``[kKmMgG]`` after
#                       each digit run on BOTH sides of ``of``.
_NEW_FMT = re.compile(
    r"^(?P<date>\d{8})T(?P<time>\d{6})Z_"
    r"(?P<sha>[0-9a-f]{8,16})_"
    r"(?P<arch>.+?)"
    r"(?:_step\d+[kKmMgG]?(?:of\d+[kKmMgG]?)?)?"
    r"\.log$",
    re.IGNORECASE,
)

# Legacy format example:
#   101ceb95a960__neuroslm-full.log
# (Two underscores separate the sha from the arch; no timestamp -> we
# fall back to file mtime.)
_LEGACY_FMT = re.compile(
    r"^(?P<sha>[0-9a-f]{8,16})__(?P<arch>[A-Za-z][\w\-]*)\.log$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _ParsedName:
    """Extracted metadata from a log basename."""
    date_token: str   # "YYYYMMDD-HHMMSS"  (UTC)
    arch: str         # architecture token; underscores preserved
    sha: str          # short SHA (8-16 hex chars)
    source: str       # "new" | "legacy"


def parse_log_basename(
    basename: str, mtime: Optional[float] = None,
) -> Optional[_ParsedName]:
    """Extract ``(date_token, arch, sha)`` from a log filename.

    Returns ``None`` when the filename matches no known pattern (so the
    caller can route it to ``logs/_unsorted_legacy/``). The legacy
    format has no embedded timestamp, so the caller must supply
    ``mtime`` (file mtime) — without it we also return None.
    """
    m = _NEW_FMT.match(basename)
    if m:
        return _ParsedName(
            date_token=f"{m['date']}-{m['time']}",
            arch=m['arch'].strip("_"),
            sha=m['sha'].lower(),
            source="new",
        )

    m = _LEGACY_FMT.match(basename)
    if m:
        if mtime is None:
            return None
        ts = _dt.datetime.fromtimestamp(mtime, _dt.timezone.utc)
        return _ParsedName(
            date_token=ts.strftime("%Y%m%d-%H%M%S"),
            arch=m['arch'],
            sha=m['sha'].lower(),
            source="legacy",
        )

    return None


def _new_folder_name(p: _ParsedName) -> str:
    """The destination folder name for a parsed log.

    Format: ``<YYYYMMDD>-<HHMMSS>_<arch>_<sha>``
    Example: ``20260614-182653_rcc_bowtie_889M_run_07aba24be2bf``
    """
    return f"{p.date_token}_{p.arch}_{p.sha}"


def _destination(root: Path, fp: Path) -> Path:
    """Compute the destination path for source log ``fp``.

    Parseable -> ``logs/<folder>/train.log``
    Unparseable -> ``logs/_unsorted_legacy/<original-basename>``
    """
    try:
        mtime: Optional[float] = fp.stat().st_mtime
    except OSError:
        mtime = None
    parsed = parse_log_basename(fp.name, mtime=mtime)
    if parsed is None:
        return root / "logs" / "_unsorted_legacy" / fp.name
    return root / "logs" / _new_folder_name(parsed) / "train.log"


# ── Migration protocol implementation ─────────────────────────────────


def plan(ctx: Context) -> List[Op]:
    """Return the list of copy operations needed to bring ``logs/vast/``
    in line with the per-run folder layout.

    Pure-ish: reads filesystem + ctx.refs, never writes.
    """
    ops: List[Op] = []
    vast_dir = ctx.root / "logs" / "vast"
    if not vast_dir.exists():
        return ops

    for fp in sorted(vast_dir.iterdir()):
        if not fp.is_file() or fp.suffix.lower() != ".log":
            continue
        # Reference gate: only copy logs that some doc/code mentions
        if not ctx.refs.references(fp.name):
            continue
        dst = _destination(ctx.root, fp)
        if dst.exists():
            continue  # idempotent
        # Note line tells the human at a glance which parser matched
        try:
            mtime = fp.stat().st_mtime
        except OSError:
            mtime = None
        parsed = parse_log_basename(fp.name, mtime=mtime)
        src_tag = parsed.source if parsed is not None else "unsorted"
        ops.append(Op(
            kind="copy",
            src=fp,
            dst=dst,
            note=f"format={src_tag}",
        ))
    return ops


def apply(ctx: Context, ops: List[Op]) -> int:
    """Execute the copy plan. Returns the number of files copied."""
    n = 0
    for op in ops:
        if op.kind != "copy" or op.src is None or op.dst is None:
            continue
        op.dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(op.src, op.dst)
        n += 1
    return n


__all__ = [
    "ID",
    "DESCRIPTION",
    "plan",
    "apply",
    "parse_log_basename",
]
