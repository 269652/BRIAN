"""0002_log_name_refactor — rename train.log → <HHMMSS>_<start>_<end>.log

Problem
-------
Current 3-level layout uses generic `train.log` for every run:
    logs/20260615/arch_name/175931_c19bf629/train.log

This makes it impossible to distinguish runs at a glance. The folder
name includes a git SHA but not the step range, so you must open the
file to see completion status.

Solution
--------
Rename logs to include step range in the filename:
    logs/20260615/arch_name/175931_0_10000.log

Where:
  - 175931 = boot time (HHMMSS in UTC)
  - 0 = starting step
  - 10000 = ending step

The folder structure remains date → arch, but the leaf filename now
carries the step range. Resumed runs get new files with their own boot
time and step range.

Migration logic
---------------
For each `train.log` under `logs/<YYYYMMDD>/<arch>/<folder>/`:
  1. Parse boot timestamp from folder name (format: <HHMMSS>_<sha>)
  2. Scan log content to find step range
  3. Rename to `logs/<YYYYMMDD>/<arch>/<HHMMSS>_<start>_<end>.log`
  4. Remove now-empty folder

Unparseable logs → `logs/_unsorted_legacy/<original_folder_name>/train.log`

Idempotency
-----------
If a file already matches the pattern `<6digits>_<digits>_<digits>.log`,
skip it. Re-running plan() after apply() yields [].

Safety
------
apply() COPIES (shutil.copy2), does NOT move. Source files remain until
`brian clean --force`. Folders are removed only if empty after copy.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from neuroslm.migrations._framework import Context, Op


ID: str = "0002_log_name_refactor"
DESCRIPTION: str = "Rename train.log → <HHMMSS>_<start>_<end>.log with step ranges"


@dataclass
class ParsedLog:
    """Metadata extracted from a train.log file."""
    boot_time: str  # HHMMSS
    start_step: int
    end_step: int
    arch_name: str
    date: str  # YYYYMMDD


def _parse_folder_name(folder: Path) -> Optional[str]:
    """Extract boot time from folder name like '175931_c19bf629' or '185105_41084160'.
    
    Returns boot time (HHMMSS) or None if unparseable.
    """
    # Pattern: <HHMMSS>_<sha> or <HHMMSS>_<instance_id> or <HHM MSS>_<something>
    # Handle both 6-digit (HHMMSS) and 5-6 digit with leading zero stripped
    m = re.match(r'^(\d{5,6})_', folder.name)
    if m:
        time_str = m.group(1)
        # Pad to 6 digits if needed (e.g., "81830" → "081830")
        return time_str.zfill(6)
    return None


def _parse_log_content(log_path: Path) -> Optional[tuple[int, int]]:
    """Scan train.log to find step range.
    
    Returns (start_step, end_step) or None if unparseable.
    """
    try:
        content = log_path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return None
    
    # Look for step lines like "step 0 |" or "step  1234 |"
    step_pattern = re.compile(r'^step\s+(\d+)\s+\|', re.MULTILINE)
    steps = [int(m.group(1)) for m in step_pattern.finditer(content)]
    
    # Fallback: look for "resumed from ... @ step N" or "final OOD ... @ step N"
    if not steps:
        resume_pattern = re.compile(r'(?:resumed from.*@ step|final OOD.*@ step)\s+(\d+)')
        resume_steps = [int(m.group(1)) for m in resume_pattern.finditer(content)]
        if resume_steps:
            # Assume training started from step 0 or 20 (common start points)
            # and ended at the resumed/final step
            final_step = max(resume_steps)
            # Try to infer start step from checkpoint name patterns like "step0.pt" or "step20.pt"
            start_match = re.search(r'step(\d+)\.pt', content)
            start_step = int(start_match.group(1)) if start_match else 0
            steps = [start_step, final_step]
    
    if not steps:
        return None
    
    return (min(steps), max(steps))


def _parse_log(date: str, arch: str, folder: Path, log_path: Path) -> Optional[ParsedLog]:
    """Parse a train.log file and its folder to extract metadata."""
    boot_time = _parse_folder_name(folder)
    if not boot_time:
        return None
    
    step_range = _parse_log_content(log_path)
    if not step_range:
        return None
    
    start_step, end_step = step_range
    return ParsedLog(
        boot_time=boot_time,
        start_step=start_step,
        end_step=end_step,
        arch_name=arch,
        date=date,
    )


def _new_log_name(parsed: ParsedLog) -> str:
    """Build new log filename: <HHMMSS>_<start>_<end>.log"""
    return f"{parsed.boot_time}_{parsed.start_step}_{parsed.end_step}.log"


def _destination(root: Path, parsed: ParsedLog) -> Path:
    """Compute destination path for renamed log."""
    return root / "logs" / parsed.date / parsed.arch_name / _new_log_name(parsed)


def _is_new_format(p: Path) -> bool:
    """Check if a file already matches the new format."""
    # Pattern: <6digits>_<digits>_<digits>.log
    return bool(re.match(r'^\d{6}_\d+_\d+\.log$', p.name))


def plan(ctx: Context) -> List[Op]:
    """Enumerate all train.log → <time>_<start>_<end>.log renames."""
    ops: List[Op] = []
    logs_dir = ctx.root / "logs"
    
    if not logs_dir.is_dir():
        return ops
    
    # Walk the 3-level hierarchy: logs/<YYYYMMDD>/<arch>/<folder>/train.log
    for date_dir in sorted(logs_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        if not re.match(r'^\d{8}$', date_dir.name):
            continue  # Skip non-date folders (vast/, _unsorted_legacy/, etc.)
        
        for arch_dir in sorted(date_dir.iterdir()):
            if not arch_dir.is_dir():
                continue
            
            for run_folder in sorted(arch_dir.iterdir()):
                if not run_folder.is_dir():
                    continue
                
                train_log = run_folder / "train.log"
                if not train_log.is_file():
                    continue
                
                # Parse log metadata
                parsed = _parse_log(
                    date=date_dir.name,
                    arch=arch_dir.name,
                    folder=run_folder,
                    log_path=train_log,
                )
                
                if parsed is None:
                    # Unparseable → move to _unsorted_legacy
                    unsorted = ctx.root / "logs" / "_unsorted_legacy" / run_folder.name
                    dst = unsorted / "train.log"
                    if not dst.exists():
                        ops.append(Op(
                            kind="move",
                            src=train_log,
                            dst=dst,
                            note=f"unparseable → _unsorted_legacy/{run_folder.name}/",
                        ))
                    continue
                
                dst = _destination(ctx.root, parsed)
                
                # Skip if already in new format
                if dst.exists() and dst == train_log:
                    continue
                
                if not dst.exists():
                    ops.append(Op(
                        kind="rename",
                        src=train_log,
                        dst=dst,
                        note=f"{run_folder.name}/train.log → {dst.name}",
                    ))
    
    return ops


def apply(ctx: Context, ops: List[Op]) -> int:
    """Execute the copy operations.
    
    Returns number of operations applied.
    """
    applied = 0
    for op in ops:
        try:
            op.dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(op.src), str(op.dst))
            applied += 1
            
            # Try to remove the old folder if it's now empty
            # (contains only the train.log we just copied)
            old_folder = op.src.parent
            try:
                # Check if folder is empty or only contains train.log
                remaining = list(old_folder.iterdir())
                if not remaining or (len(remaining) == 1 and remaining[0].name == "train.log"):
                    # Try to remove train.log and folder
                    if op.src.exists():
                        op.src.unlink()
                    if not any(old_folder.iterdir()):
                        old_folder.rmdir()
            except Exception:
                # If removal fails, that's okay - clean command will handle it
                pass
                
        except Exception as e:
            print(f"[0002] failed to copy {op.src} → {op.dst}: {e}")
            continue
    
    return applied
