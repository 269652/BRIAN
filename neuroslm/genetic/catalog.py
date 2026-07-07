# -*- coding: utf-8 -*-
"""The full research-mechanics catalog — every `*.neuro` spec in the repo.

The repo already carries a rich semantic-description surface: each
``mechanics/*.neuro`` (and ``dynamics/*.neuro`` / ``structures/*.neuro``) block
declares a mechanic's ``summary``, ``category``, ``when_to_use``, ``not_for``,
``properties`` and ``references``. That *is* the human-facing "what it does /
when to use it" language. This module loads all of it through the existing
``mechanic_parser`` so the discovery loop can enumerate "all currently existing
research mechanics" (≈74, not a hand-kept list of 13) and treat every catalog
name as prior art.

Reuses ``neuroslm.dsl.mechanic_parser.parse_mechanic_file`` — the canonical
parser — rather than re-implementing block scanning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from neuroslm.dsl.mechanic_parser import MechanicSpec, parse_mechanic_file

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# The three trees that hold reusable neural mechanics in this repo.
_DEFAULT_DIRS = ("mechanics", "dynamics", "structures")


class MechanicCatalog:
    """An in-memory index of every parsed ``MechanicSpec``, keyed by name."""

    def __init__(self, specs: List[MechanicSpec]):
        # last spec wins on a name clash (deterministic: dirs scanned in order)
        self._by_name: Dict[str, MechanicSpec] = {}
        for s in specs:
            if s.name:
                self._by_name[s.name] = s

    def __len__(self) -> int:
        return len(self._by_name)

    def names(self) -> List[str]:
        return sorted(self._by_name.keys())

    def get(self, name: str) -> Optional[MechanicSpec]:
        return self._by_name.get(name)

    def specs(self) -> List[MechanicSpec]:
        return [self._by_name[n] for n in self.names()]

    def categories(self) -> List[str]:
        return sorted({s.category for s in self._by_name.values() if s.category})

    def by_category(self) -> Dict[str, List[MechanicSpec]]:
        out: Dict[str, List[MechanicSpec]] = {}
        for s in self.specs():
            out.setdefault(s.category or "uncategorized", []).append(s)
        return out

    def describe(self, name: str) -> str:
        s = self.get(name)
        if s is None:
            return f"unknown mechanic: {name!r}"
        parts = [f"{s.name} [{s.category or 'uncategorized'}]"]
        if s.summary:
            parts.append(s.summary.strip())
        wtu = (s.when_to_use or "").strip()
        if wtu:
            # first non-empty line of the when_to_use block
            first = next((ln.strip() for ln in wtu.splitlines() if ln.strip()), "")
            if first:
                parts.append("When to use: " + first)
        return " — ".join(parts)


def _iter_neuro_files(root: Path, dirs) -> List[Path]:
    files: List[Path] = []
    for d in dirs:
        base = root / d
        if base.is_dir():
            files.extend(sorted(base.glob("*.neuro")))
    return files


def load_catalog(root: Path = None, dirs=_DEFAULT_DIRS) -> MechanicCatalog:
    """Parse every ``*.neuro`` under the mechanic trees into a catalog."""
    root = Path(root) if root is not None else _REPO_ROOT
    specs: List[MechanicSpec] = []
    for f in _iter_neuro_files(root, dirs):
        try:
            specs.extend(parse_mechanic_file(f.read_text(encoding="utf-8")))
        except Exception:
            # a malformed file must never break enumeration of the rest
            continue
    return MechanicCatalog(specs)


def catalog_names(root: Path = None) -> set:
    """The set of every known mechanic name — prior art for the novelty gate."""
    return set(load_catalog(root).names())
