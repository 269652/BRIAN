# -*- coding: utf-8 -*-
"""Best-run detection, log quoting, and reference-locking for BRIAN logs.

Public API
----------
LogRef          — dataclass representing one .ln file (path + target + comment)
RunScore        — dataclass with the ranking metrics extracted from one log
write_ref       — create / overwrite a .ln file
read_ref        — parse a .ln file → LogRef
scan_refs       — walk a directory tree and return all LogRef objects found
locked_logs     — frozenset of absolute paths that must not be deleted
score_log       — extract RunScore from a log file path
find_best_log   — scan a log directory and return the path of the best run
update_best_run_pointer — scan logs, pick best, write .brian/best_run.ln
                          also writes .brian/checkpoint.ln with HF URL when found
extract_checkpoint_url  — parse the last [ckpt_push] HF URL from log text
quote_log       — create .brian/refs/<name>.ln protecting a log from deletion
unquote_log     — remove .brian/refs/<name>.ln

Constants
---------
BEST_RUN_LN     — relative path inside the repo root: ".brian/best_run.ln"
CHECKPOINT_LN   — relative path inside the repo root: ".brian/checkpoint.ln"
REFS_DIR        — relative path inside the repo root: ".brian/refs"

.ln file format
---------------
Lines starting with ``#`` are comments and are silently ignored.  The first
non-empty, non-comment line is the target path (stored relative to the repo
root).  Extra lines after the target are ignored.

Example::

    # best training run — gap_ratio 2.66
    # updated: 2026-06-17
    logs/20260617/SmolLM/170512_20_2000.log

Scoring / ranking
-----------------
Primary metric: lowest ``gap_ratio`` (min across all mid-OOD evaluations).
A run *with* any gap_ratio always beats a run with only train PPL.
Fallback metric (no OOD data): lowest final train PPL.
Secondary metric (explicit ``metric="ppl"``): lowest final train PPL regardless.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ── Constants ──────────────────────────────────────────────────────────────

BEST_RUN_LN: str = ".brian/best_run.ln"
CHECKPOINT_LN: str = ".brian/checkpoint.ln"
REFS_DIR: str = ".brian/refs"

# ── Regexes ────────────────────────────────────────────────────────────────

_MID_OOD_FULL_RE = re.compile(
    r"\[mid-ood\]\s+step\s+(?P<step>\d+):\s+\w+\s+ppl=(?P<ood_ppl>[\d.]+)"
    r"(?:\s+gap_ratio=(?P<gap_ratio>[\d.]+))?"
    r"(?:\s+\(train_ppl=(?P<train_ppl>[\d.]+)\))?"
)

_STEP_RE = re.compile(
    r"step\s+(?P<step>\d+)\s+\|\s+"
    r"loss\s+[\d.]+\s+\|\s+"
    r"lm\s+[\d.]+\s+\|\s+"
    r"ppl\s+(?P<ppl>[\d.]+)\s+\|"
)

# Matches: [ckpt_push] ✓ pushed stepN.pt[ (optimizer stripped)] → hf://...
_CKPT_PUSH_RE = re.compile(
    r"\[ckpt_push\] ✓ pushed \S+"  # filename
    r"(?:\s+\([^)]+\))?"           # optional note, e.g. "(optimizer stripped)"
    r"\s+→\s+"
    r"(?P<url>hf://\S+)"
)


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class LogRef:
    """A parsed .ln file."""
    path: Path    # path to the .ln file itself
    target: Path  # path to the referenced log (relative to repo root)
    comment: str = ""  # text of the first # comment line, stripped of leading #


@dataclass
class RunScore:
    """Ranking metrics extracted from one log file."""
    log_path: Path
    gap_ratio: Optional[float] = None    # best (min) gap_ratio across OOD evals
    train_ppl: Optional[float] = None    # final train PPL (last step line)
    ood_ppl: Optional[float] = None      # OOD PPL at the last mid-OOD eval
    step: int = 0                        # last step seen

    def sort_key(self, metric: str = "gap_ratio") -> float:
        """Lower is better.  None → +inf so scored runs always beat unscored."""
        if metric == "gap_ratio":
            # A run with any gap_ratio beats any run that only has train_ppl.
            # Encode that by using (0, gap_ratio) vs (1, train_ppl).
            # Handled in find_best_log; here just return the raw value.
            v = self.gap_ratio
        elif metric == "ood_ppl":
            v = self.ood_ppl
        else:  # "ppl"
            v = self.train_ppl
        return v if v is not None else float("inf")


# ── .ln file IO ────────────────────────────────────────────────────────────

def write_ref(ln_path: Path, target: Path, comment: str = "") -> LogRef:
    """Create or overwrite a .ln file pointing at *target*.

    *target* is stored as-is (caller decides whether relative or absolute).
    *comment* is written as the first ``#`` line when non-empty.
    """
    ln_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    if comment:
        lines.append(f"# {comment}")
    lines.append(str(target))
    ln_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return LogRef(path=ln_path, target=target, comment=comment)


def read_ref(ln_path: Path) -> LogRef:
    """Parse a .ln file and return a LogRef.

    Raises
    ------
    FileNotFoundError  if *ln_path* does not exist.
    ValueError         if the file contains no non-comment target line.
    """
    if not ln_path.is_file():
        raise FileNotFoundError(ln_path)
    lines = ln_path.read_text(encoding="utf-8").splitlines()
    comment = ""
    target_str = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if not comment:
                comment = stripped.lstrip("#").strip()
        else:
            target_str = stripped
            break
    if target_str is None:
        raise ValueError(f"{ln_path}: no target path found (only comments or empty)")
    return LogRef(path=ln_path, target=Path(target_str), comment=comment)


# ── Scanner ────────────────────────────────────────────────────────────────

def scan_refs(root: Path) -> List[LogRef]:
    """Return all LogRef objects found by walking *.ln files under *root*."""
    refs: List[LogRef] = []
    for ln_path in sorted(root.rglob("*.ln")):
        try:
            refs.append(read_ref(ln_path))
        except (ValueError, OSError):
            pass
    return refs


def locked_logs(root: Path) -> frozenset:
    """Return frozenset of absolute Path objects that must not be deleted.

    Any file referenced by a .ln file anywhere under *root* is locked,
    regardless of whether the referenced file currently exists.  Relative
    target paths are resolved relative to *root*.
    """
    locked: set = set()
    for ref in scan_refs(root):
        target = ref.target
        if not target.is_absolute():
            target = root / target
        locked.add(target.resolve())
    return frozenset(locked)


# ── Log scoring ────────────────────────────────────────────────────────────

def score_log(log_path: Path) -> Optional[RunScore]:
    """Extract a RunScore from *log_path*.

    Returns None if the file is empty or contains no parseable metrics.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    if not text.strip():
        return None

    # Collect all mid-OOD evaluations
    ood_matches = list(_MID_OOD_FULL_RE.finditer(text))
    gap_ratios: List[float] = []
    last_ood_ppl: Optional[float] = None
    last_train_ppl_from_ood: Optional[float] = None
    last_ood_step = 0
    for m in ood_matches:
        if m.group("gap_ratio") is not None:
            gap_ratios.append(float(m.group("gap_ratio")))
        last_ood_ppl = float(m.group("ood_ppl"))
        if m.group("train_ppl") is not None:
            last_train_ppl_from_ood = float(m.group("train_ppl"))
        last_ood_step = int(m.group("step"))

    # Collect all step lines for final train PPL and max step
    step_matches = list(_STEP_RE.finditer(text))
    final_train_ppl: Optional[float] = None
    last_step = 0
    if step_matches:
        last_m = step_matches[-1]
        final_train_ppl = float(last_m.group("ppl"))
        last_step = int(last_m.group("step"))

    # Nothing parseable at all
    if not step_matches and not ood_matches:
        return None

    best_gap = min(gap_ratios) if gap_ratios else None

    # Prefer train PPL from the last step line; fall back to the OOD annotation
    train_ppl = final_train_ppl if final_train_ppl is not None else last_train_ppl_from_ood
    step = max(last_step, last_ood_step)

    return RunScore(
        log_path=log_path,
        gap_ratio=best_gap,
        train_ppl=train_ppl,
        ood_ppl=last_ood_ppl,
        step=step,
    )


# ── Checkpoint URL extraction ─────────────────────────────────────────────

def extract_checkpoint_url(text: str) -> Optional[str]:
    """Return the last HF checkpoint URL from a training log, or None.

    Parses ``[ckpt_push] ✓ pushed stepN.pt … → hf://owner/repo/…`` lines
    emitted by :func:`neuroslm.checkpoint_push.push_checkpoint_to_hf`.
    Returns the last such URL (highest step pushed), or None if none found.
    """
    matches = list(_CKPT_PUSH_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group("url")


# ── Best-run detection ─────────────────────────────────────────────────────

def find_best_log(
    log_dir: Path,
    metric: str = "gap_ratio",
) -> Optional[Path]:
    """Scan all *.log files under *log_dir* and return the path of the best run.

    "Best" is determined by *metric*:
      ``"gap_ratio"`` — lowest gap_ratio (primary); a run with any gap_ratio
                        beats any run that only has train PPL.
      ``"ppl"``        — lowest final train PPL.
      ``"ood_ppl"``    — lowest OOD PPL at the last mid-OOD evaluation.

    Returns None if no scoreable log files are found.
    """
    scored: List[RunScore] = []
    for log_path in log_dir.rglob("*.log"):
        s = score_log(log_path)
        if s is not None:
            scored.append(s)

    if not scored:
        return None

    if metric == "gap_ratio":
        # Tier 1: runs with gap_ratio; Tier 2: runs with only train_ppl
        with_gap = [s for s in scored if s.gap_ratio is not None]
        if with_gap:
            best = min(with_gap, key=lambda s: s.gap_ratio)  # type: ignore[arg-type]
        else:
            best = min(scored, key=lambda s: s.sort_key("ppl"))
    else:
        best = min(scored, key=lambda s: s.sort_key(metric))

    return best.log_path


# ── Best-run pointer ───────────────────────────────────────────────────────

def update_best_run_pointer(
    root: Path,
    log_dir: Optional[Path] = None,
    metric: str = "gap_ratio",
) -> Optional[Path]:
    """Scan *log_dir* (default: *root*/logs), find best run, write .ln.

    Returns the absolute path of the best log, or None when no qualifying
    logs are found (in which case no .ln file is written/changed).
    """
    if log_dir is None:
        log_dir = root / "logs"

    best = find_best_log(log_dir, metric=metric)
    if best is None:
        return None

    # Store relative to root so the .ln file is portable
    try:
        rel = best.relative_to(root)
    except ValueError:
        rel = best  # absolute fallback when log_dir is outside root

    ln_path = root / BEST_RUN_LN
    write_ref(ln_path, rel, comment=f"auto: best {metric} — {rel.name}")

    # Also write checkpoint.ln if the log contains an HF checkpoint URL
    try:
        log_text = best.read_text(encoding="utf-8", errors="replace")
        hf_url = extract_checkpoint_url(log_text)
    except OSError:
        hf_url = None
    if hf_url:
        write_checkpoint_url(
            root,
            hf_url,
            comment=f"auto: best {metric} checkpoint — {rel.name}",
        )

    return best


# ── Checkpoint URL pointer (.brian/checkpoint.ln) ─────────────────────────
# HF URLs are NOT file paths; they are stored as raw strings in checkpoint.ln.
# Use write_checkpoint_url / read_checkpoint_url — NOT write_ref / read_ref —
# so that the URL survives round-tripping through pathlib on all platforms.

def write_checkpoint_url(root: Path, url: str, comment: str = "") -> None:
    """Write *url* (HF checkpoint URI) to ``.brian/checkpoint.ln``.

    The URL is stored as plain text so it is not normalised by :mod:`pathlib`.
    """
    ln_path = root / CHECKPOINT_LN
    ln_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    if comment:
        lines.append(f"# {comment}")
    lines.append(url)
    ln_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_checkpoint_url(root: Path) -> Optional[str]:
    """Return the HF checkpoint URL from ``.brian/checkpoint.ln``, or None.

    Reads the first non-comment, non-empty line from the file.
    """
    ln_path = root / CHECKPOINT_LN
    if not ln_path.is_file():
        return None
    for raw in ln_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            return line
    return None


# ── Quote / unquote ────────────────────────────────────────────────────────

def quote_log(
    root: Path,
    log_path: Path,
    name: str,
    comment: str = "",
) -> Path:
    """Protect *log_path* from deletion by creating .brian/refs/<name>.ln.

    Returns the path to the created .ln file.
    """
    try:
        rel = log_path.relative_to(root)
    except ValueError:
        rel = log_path  # already relative or absolute outside root

    ln_path = root / REFS_DIR / f"{name}.ln"
    write_ref(ln_path, rel, comment=comment or f"quoted: {rel.name}")
    return ln_path


def unquote_log(root: Path, name: str) -> None:
    """Remove .brian/refs/<name>.ln.

    Raises FileNotFoundError if the quote does not exist.
    """
    ln_path = root / REFS_DIR / f"{name}.ln"
    if not ln_path.is_file():
        raise FileNotFoundError(f"no quote named {name!r} at {ln_path}")
    ln_path.unlink()
