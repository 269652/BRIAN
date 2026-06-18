"""Collect the personality-vector dimension (``PERSONALITY_DIM``).

The personality vector is a small per-model identity tensor used by the
neurochem subsystem. Its dimension is canonically defined as
``len(neuroslm.neurochem.personality.PERSONALITY_DIMS)`` — a tuple of
named axes (currently: curiosity, agreeableness, vigilance, patience,
hedonic_tone → 5 dims).

We import the constant live so that any future expansion of the
personality axes is automatically reflected in the README.
"""
from __future__ import annotations

import sys
from pathlib import Path

METRICS = ["PERSONALITY_DIM"]


def collect(root: Path) -> dict[str, str]:
    """Import N_PERSONALITY from the neurochem module."""
    metrics: dict[str, str] = {}

    # Make sure the workspace root is importable so we can pull the
    # canonical constant without spinning up a model.
    root_str = str(root)
    inserted = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        inserted = True
    try:
        try:
            from neuroslm.neurochem.personality import N_PERSONALITY  # type: ignore
            metrics["PERSONALITY_DIM"] = str(int(N_PERSONALITY))
        except Exception:
            # Fallback: parse the tuple definition directly so we don't
            # require importable neuroslm (e.g. minimal sandbox or when
            # heavy deps like torch can't load).
            import re

            src = root / "neuroslm" / "neurochem" / "personality.py"
            if src.exists():
                text = src.read_text(encoding="utf-8", errors="replace")
                m = re.search(
                    r"^\s*PERSONALITY_DIMS\s*=\s*\((.*?)\)",
                    text,
                    re.MULTILINE | re.DOTALL,
                )
                if m:
                    # Strip line comments, then split on commas.
                    body = re.sub(r"#[^\n]*", "", m.group(1))
                    entries = [
                        e.strip().strip(",")
                        for e in body.split(",")
                    ]
                    entries = [e for e in entries if e]
                    if entries:
                        metrics["PERSONALITY_DIM"] = str(len(entries))
    finally:
        if inserted:
            try:
                sys.path.remove(root_str)
            except ValueError:
                pass

    return metrics
