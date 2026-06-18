"""Collect Neural Flow Graph (NFG) topology metrics.

Three counts derived live from the canonical architecture sources:

* ``NFG_POPULATIONS`` — total population count = trunk + bio populations
  declared in ``architectures/{current,master}/arch.neuro`` via
  ``param_scope`` blocks. Reuses the same parsing logic as
  :mod:`arch_topology` so the numbers always agree.
* ``NFG_SYNAPSES`` — count of ``synapse <src> -> <dst> { ... }`` blocks
  in the active ``arch.neuro``.
* ``NFG_NT_SYSTEMS`` — count of source populations defined in
  ``lib/modules/nuclei.neuro`` (each NT-source nucleus drives one
  global modulator signal: VTA→DA, SN→DA, LC→NE, Raphe→5HT, NBM→ACh,
  NAcc→DA-gating).

All three values flow into the README template's ``${NFG_*}``
placeholders.
"""
from __future__ import annotations

import re
from pathlib import Path

METRICS = [
    "NFG_POPULATIONS",
    "NFG_SYNAPSES",
    "NFG_NT_SYSTEMS",
]


def _find_arch(root: Path) -> Path | None:
    """Return the active arch.neuro, preferring `current` over `master`."""
    for arch_dir in ("architectures/current", "architectures/master"):
        candidate = root / arch_dir / "arch.neuro"
        if candidate.exists():
            return candidate
    return None


def _count_populations(text: str) -> int:
    """Sum trunk + bio populations from `param_scope` blocks."""
    total = 0
    for scope in ("trunk", "bio"):
        m = re.search(
            rf"param_scope\s+{scope}\s*\{{[^}}]*populations:\s*\[([^\]]+)\]",
            text,
            re.DOTALL,
        )
        if m:
            pops = [p.strip() for p in m.group(1).split(",") if p.strip()]
            total += len(pops)
    return total


def _count_synapses(text: str) -> int:
    """Count `synapse <src> -> <dst>` declarations."""
    return len(re.findall(r"^synapse\s+\w+\s*->\s*\w+", text, re.MULTILINE))


def _count_nt_systems(root: Path) -> int | None:
    """Count source-nuclei populations in lib/modules/nuclei.neuro."""
    nuclei = root / "lib" / "modules" / "nuclei.neuro"
    if not nuclei.exists():
        return None
    text = nuclei.read_text(encoding="utf-8", errors="replace")
    return len(re.findall(r"^export\s+population\s+\w+", text, re.MULTILINE))


def collect(root: Path) -> dict[str, str]:
    """Read populations, synapses, and NT-source nuclei counts."""
    metrics: dict[str, str] = {}

    arch_path = _find_arch(root)
    if arch_path is not None:
        text = arch_path.read_text(encoding="utf-8", errors="replace")
        pops = _count_populations(text)
        if pops > 0:
            metrics["NFG_POPULATIONS"] = str(pops)
        syns = _count_synapses(text)
        if syns > 0:
            metrics["NFG_SYNAPSES"] = str(syns)

    nt = _count_nt_systems(root)
    if nt is not None:
        metrics["NFG_NT_SYSTEMS"] = str(nt)

    return metrics
