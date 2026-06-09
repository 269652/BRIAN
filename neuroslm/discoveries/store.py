# -*- coding: utf-8 -*-
"""File-backed stores for :class:`HypothesisRecord` and :class:`DiscoveryRecord`.

Layout maintained by the store::

    <root>/
      H001_phi_monotone.md             # one .md per record
      H002_ood_gap_decrease.md
      index.json                        # cache: list of (id, file, theorem, status)
      proofs/
        H001_phi_monotone.lean
        ...

The ``.md`` files are the source of truth — re-instantiating the store
always re-reads the directory so hand-edits survive. ``index.json`` is
rewritten on every ``save()`` to keep tooling that wants a quick
machine-readable listing happy.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Type, Union

from neuroslm.discoveries.records import (
    HypothesisRecord, DiscoveryRecord, _now_iso,
)


# ── shared base ────────────────────────────────────────────────────


class _RecordStore:
    """Common skeleton for the hypothesis + discovery stores.

    Subclasses must set:
      :attr:`_record_cls`    — the dataclass (``HypothesisRecord`` / …)
      :attr:`_id_prefix`     — ``"H"`` or ``"D"``
      :attr:`_kind`          — ``"hypothesis"`` / ``"discovery"`` (for index)
    """

    _record_cls: Type
    _id_prefix: str
    _kind: str

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "proofs").mkdir(exist_ok=True)

    # ── core operations ────────────────────────────────────────────

    def save(self, record) -> Path:
        """Write ``record.to_markdown()`` to disk and rewrite ``index.json``.

        Returns:
            Path of the ``.md`` file actually written.
        """
        if not isinstance(record, self._record_cls):
            raise TypeError(
                f"{type(self).__name__}.save expected "
                f"{self._record_cls.__name__}, got {type(record).__name__}"
            )
        # Stamp updated_at on hypothesis records so the on-disk file
        # always reflects the latest mutation; discoveries don't have
        # that field (they're append-only by intent).
        if hasattr(record, "updated_at"):
            record.updated_at = _now_iso()
        out_path = self.root / record.filename()
        out_path.write_text(record.to_markdown(), encoding="utf-8")
        self._rewrite_index()
        return out_path

    def get(self, record_id: str):
        """Load a single record by id."""
        for path in self._record_paths():
            head = self._peek_id(path)
            if head == record_id:
                return self._load_path(path)
        raise KeyError(record_id)

    def list_all(self) -> List:
        """Return every record in the store, sorted by id."""
        out = [self._load_path(p) for p in self._record_paths()]
        out.sort(key=lambda r: r.id)
        return out

    def next_id(self) -> str:
        """Smallest free id of the form ``<prefix>NNN`` (3-digit minimum)
        strictly greater than the maximum id currently on disk.

        This gives a stable temporal ordering even when ids are sparse
        on disk (e.g. an old discovery was deleted) — the engine never
        re-uses a slot, so the id timeline reflects discovery order.
        """
        max_n = 0
        for p in self._record_paths():
            head = self._peek_id(p)
            m = re.match(rf"^{self._id_prefix}(\d+)$", head or "")
            if m:
                n = int(m.group(1))
                max_n = max(max_n, n)
        return f"{self._id_prefix}{max_n + 1:03d}"

    # ── internals ──────────────────────────────────────────────────

    def _record_paths(self) -> List[Path]:
        """Every ``.md`` file in the root that matches our id prefix."""
        return [p for p in sorted(self.root.glob(f"{self._id_prefix}*.md"))
                if p.is_file()]

    def _load_path(self, path: Path):
        text = path.read_text(encoding="utf-8")
        return self._record_cls.from_markdown(text)

    def _peek_id(self, path: Path) -> Optional[str]:
        """Cheap header-only read — extract the ``id`` field without
        deserialising the whole record (matters once the store grows
        past a few hundred files)."""
        try:
            with path.open("r", encoding="utf-8") as fh:
                header = fh.read(2048)
        except OSError:
            return None
        m = re.search(r"\nid:\s*([^\s\n]+)", header)
        if not m:
            # Fallback to filename
            stem = path.stem
            return stem.split("_", 1)[0]
        return m.group(1).strip().strip('"').strip("'")

    def _rewrite_index(self) -> None:
        """Serialise the full record set to ``index.json`` — atomic via
        write-then-rename so concurrent readers never see a partial file."""
        records: List[Dict] = []
        for path in self._record_paths():
            rec = self._load_path(path)
            d = asdict(rec)
            records.append({
                "id": d["id"],
                "title": d.get("title", ""),
                "theorem_name": d.get("theorem_name", ""),
                "proof_status": d.get("proof_status", "missing"),
                "file": path.name,
                # Type-specific extra fields kept terse for grep-friendliness.
                **({"status": d["status"]}
                   if "status" in d else {}),
                **({"dna_integrated": d["dna_integrated"]}
                   if "dna_integrated" in d else {}),
            })
        index = {
            "kind": self._kind,
            "version": 1,
            "records": records,
        }
        idx_path = self.root / "index.json"
        tmp_path = idx_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(index, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(idx_path)


# ── concrete stores ────────────────────────────────────────────────


class HypothesisStore(_RecordStore):
    """Store for human-authored :class:`HypothesisRecord` files."""
    _record_cls = HypothesisRecord
    _id_prefix = "H"
    _kind = "hypothesis"


class DiscoveryStore(_RecordStore):
    """Store for engine-authored :class:`DiscoveryRecord` files."""
    _record_cls = DiscoveryRecord
    _id_prefix = "D"
    _kind = "discovery"
