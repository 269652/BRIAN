"""Collect the highest hypothesis index from the ``hypothesis/`` ledger.

Each hypothesis is a markdown file at ``hypothesis/H<NNN>_<slug>.md``.
We surface the maximum index as ``LAST_HYPOTHESIS`` so the README's
"Hypothesis ledger H1-H${LAST_HYPOTHESIS}" link auto-updates whenever
a new hypothesis file is added.
"""
from __future__ import annotations

import re
from pathlib import Path

METRICS = ["LAST_HYPOTHESIS"]


def collect(root: Path) -> dict[str, str]:
    """Glob ``hypothesis/H*.md`` and return max index as string."""
    metrics: dict[str, str] = {}

    hyp_dir = root / "hypothesis"
    if not hyp_dir.is_dir():
        return metrics

    indices: list[int] = []
    for f in hyp_dir.glob("H*.md"):
        m = re.match(r"^H(\d+)", f.name)
        if m:
            indices.append(int(m.group(1)))

    if indices:
        metrics["LAST_HYPOTHESIS"] = str(max(indices))

    return metrics
