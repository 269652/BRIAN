# -*- coding: utf-8 -*-
"""Record schemas — :class:`HypothesisRecord` and :class:`DiscoveryRecord`.

Both records share a YAML-front-matter + Markdown-body serialisation so
the on-disk format is human-editable, git-friendly, and the same shape
for human-authored hypotheses and engine-generated discoveries.

The Markdown layout is::

    ---
    id: H001
    title: ...
    theorem_name: Brian.PhiMonotone
    status: stated
    ...
    ---

    <Markdown body — typically the KaTeX statement of the claim,
    with prose context, intuition, and links to code/tests.>

Front-matter is parsed with a small, dependency-free YAML reader so the
package stays importable from cold-start CLI contexts without pulling
PyYAML into ``requirements.txt``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ── enums (string-typed for round-trip simplicity) ──────────────────

# A hypothesis moves through these states as we converge on it.
# ``draft``    — captured but not yet pinned to a precise theorem
# ``stated``   — a formal Lean theorem name + obligation exists
# ``proven``   — the obligation has been discharged (by Lean or argument)
# ``refuted``  — counter-example found or the claim was falsified
HYPOTHESIS_STATUSES = {"draft", "stated", "proven", "refuted"}

# A proof file (a ``.lean`` under ``proofs/``) moves through:
# ``missing``  — no proof file exists yet
# ``stub``     — a file exists that ends in ``sorry``
# ``compiles`` — Lean accepts the file (no errors) but ``sorry`` remains
# ``verified`` — Lean accepts the file AND no ``sorry`` is left
PROOF_STATUSES = {"missing", "stub", "compiles", "verified"}


# ── id regex — H001..H999+ for hypotheses, D001+ for discoveries ────
_HYPOTHESIS_ID_RE = re.compile(r"^H\d{3,}$")
_DISCOVERY_ID_RE  = re.compile(r"^D\d{3,}$")


def _now_iso() -> str:
    """UTC timestamp in ISO-8601 — used as a default for ``created_at``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── small dependency-free YAML front-matter parser ─────────────────
#
# We only need ``key: value`` lines and short list literals; this is not
# a general YAML parser, but it covers every field these records use
# and is fully deterministic.

_FRONT_MATTER_RE = re.compile(
    r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL,
)


def _yaml_dump(d: Dict[str, Any]) -> str:
    """Emit a minimal YAML representation of ``d`` (str / int / float /
    bool / None / list[str] / dict[str, float] only).  Deterministic
    ordering for git-friendly diffs."""
    lines: List[str] = []
    for key in sorted(d):
        val = d[key]
        if val is None:
            lines.append(f"{key}: null")
        elif isinstance(val, bool):
            lines.append(f"{key}: {'true' if val else 'false'}")
        elif isinstance(val, (int, float)):
            lines.append(f"{key}: {val}")
        elif isinstance(val, str):
            lines.append(f"{key}: {_yaml_scalar(val)}")
        elif isinstance(val, list):
            if not val:
                lines.append(f"{key}: []")
            else:
                rendered = ", ".join(_yaml_scalar(str(x)) for x in val)
                lines.append(f"{key}: [{rendered}]")
        elif isinstance(val, dict):
            # Flatten as inline JSON to keep diffs single-line.
            lines.append(f"{key}: {json.dumps(val, sort_keys=True)}")
        else:
            lines.append(f"{key}: {_yaml_scalar(str(val))}")
    return "\n".join(lines)


def _yaml_scalar(s: str) -> str:
    """Quote-if-needed for a YAML scalar — keep simple identifiers bare
    and quote anything with special characters."""
    if s == "":
        return '""'
    needs_quote = any(c in s for c in ":#'\"[]{},&*!|>%@`\n\t") \
                  or s.strip() != s \
                  or s.lower() in {"true", "false", "null", "yes", "no"}
    if needs_quote:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def _yaml_load(text: str) -> Dict[str, Any]:
    """Parse the minimal YAML emitted by :func:`_yaml_dump`. Robust to
    hand-edited files as long as they stick to the same shape."""
    result: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        result[key] = _yaml_parse_value(value)
    return result


def _yaml_parse_value(v: str) -> Any:
    """Inverse of :func:`_yaml_scalar` + list/JSON parsing."""
    if v == "" or v == "null":
        return None
    if v == "true":
        return True
    if v == "false":
        return False
    # JSON object literal
    if v.startswith("{") and v.endswith("}"):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    # List literal: [a, b, c] or ["a", "b", "c"]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        items: List[str] = []
        # Walk the inner string respecting quoted strings.
        buf: List[str] = []
        in_str: Optional[str] = None
        escape = False
        for ch in inner:
            if escape:
                buf.append(ch)
                escape = False
                continue
            if in_str:
                if ch == "\\":
                    escape = True
                elif ch == in_str:
                    in_str = None
                else:
                    buf.append(ch)
                continue
            if ch in ('"', "'"):
                in_str = ch
                continue
            if ch == ",":
                items.append("".join(buf).strip())
                buf = []
                continue
            buf.append(ch)
        tail = "".join(buf).strip()
        if tail:
            items.append(tail)
        return items
    # Quoted string
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
        return v[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    # Number
    try:
        if "." in v or "e" in v or "E" in v:
            return float(v)
        return int(v)
    except ValueError:
        pass
    return v


# ── HypothesisRecord ───────────────────────────────────────────────


@dataclass
class HypothesisRecord:
    """A human-authored formal claim about the BRIAN architecture.

    The record lives in ``hypothesis/<id>_<slug>.md`` and (optionally)
    has a sibling Lean proof at ``hypothesis/proofs/<id>_<slug>.lean``.
    Every field except ``statement_md`` round-trips through the YAML
    front-matter; ``statement_md`` is the Markdown body of the file.

    Attributes:
        id:            stable identifier matching ``^H\\d{3,}$``
                       (e.g. ``"H001"``)
        title:         short human title
        statement_md:  the formal claim in Markdown + KaTeX
        theorem_name:  Lean-canonical theorem name (e.g. ``Brian.PhiMonotone``)
        status:        one of :data:`HYPOTHESIS_STATUSES`
        references:    pointers into the docs (``["formal_framework.md §6.1"]``)
        code_refs:     source files realising the mechanism in code
        test_refs:     pytest files that exercise the claim empirically
        proof_path:    relative path to the ``.lean`` file (if emitted)
        proof_status:  one of :data:`PROOF_STATUSES`
        tags:          free-form tags for filtering
        created_at:    ISO-8601 timestamp (UTC)
        updated_at:    ISO-8601 timestamp (UTC)
    """

    id: str
    title: str
    statement_md: str
    theorem_name: str
    status: str = "draft"
    references: List[str] = field(default_factory=list)
    code_refs: List[str] = field(default_factory=list)
    test_refs: List[str] = field(default_factory=list)
    proof_path: Optional[str] = None
    proof_status: str = "missing"
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def __post_init__(self):
        if not _HYPOTHESIS_ID_RE.match(self.id or ""):
            raise ValueError(
                f"HypothesisRecord.id must match {_HYPOTHESIS_ID_RE.pattern}, "
                f"got {self.id!r}"
            )
        if self.status not in HYPOTHESIS_STATUSES:
            raise ValueError(
                f"HypothesisRecord.status must be in {sorted(HYPOTHESIS_STATUSES)}, "
                f"got {self.status!r}"
            )
        if self.proof_status not in PROOF_STATUSES:
            raise ValueError(
                f"HypothesisRecord.proof_status must be in {sorted(PROOF_STATUSES)}, "
                f"got {self.proof_status!r}"
            )

    # ── serialisation ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HypothesisRecord":
        return cls(**d)

    def slug(self) -> str:
        """A filename-safe slug from the title."""
        return _slugify(self.title)

    def filename(self) -> str:
        """The canonical ``.md`` filename for this record."""
        return f"{self.id}_{self.slug()}.md"

    def proof_filename(self) -> str:
        """The canonical ``.lean`` filename — same stem, ``.lean`` suffix."""
        return f"{self.id}_{self.slug()}.lean"

    def to_markdown(self) -> str:
        """Render as YAML-front-matter + Markdown body."""
        meta = self.to_dict()
        body = meta.pop("statement_md", "")
        return "---\n" + _yaml_dump(meta) + "\n---\n\n" + (body or "")

    @classmethod
    def from_markdown(cls, text: str) -> "HypothesisRecord":
        m = _FRONT_MATTER_RE.match(text)
        if not m:
            raise ValueError("HypothesisRecord: missing YAML front-matter")
        meta = _yaml_load(m.group(1))
        body = m.group(2).strip()
        meta["statement_md"] = body
        # Backfill optional fields that may be absent in hand-edited files.
        meta.setdefault("references", [])
        meta.setdefault("code_refs", [])
        meta.setdefault("test_refs", [])
        meta.setdefault("tags", [])
        meta.setdefault("status", "draft")
        meta.setdefault("proof_status", "missing")
        meta.setdefault("created_at", _now_iso())
        meta.setdefault("updated_at", _now_iso())
        # ``proof_path: null`` round-trips as Python ``None``; nothing to do.
        return cls.from_dict(meta)


# ── DiscoveryRecord ────────────────────────────────────────────────


@dataclass
class DiscoveryRecord:
    """An engine-authored discovery from the evolutionary loop.

    A discovery is the audit trail of a single admitted mutation:
    parent genome, mutation operations, fitness delta, generation, and
    the Lean theorem name we hold the mutation to. Only verified
    discoveries may be promoted into the genome via
    :func:`neuroslm.discoveries.splice.splice_discovery_into_dna`.

    The ``mutation_args_json`` field carries the *arguments* of each
    mutation op so the splice can reconstruct the DSL block to append.
    It's a JSON-encoded list of dicts to keep the serialised form
    homogeneous across mutation types.
    """

    id: str
    title: str
    mechanism_md: str
    mutation_chain: List[str]
    parent_dna_id: str
    fitness_before: Dict[str, float]
    fitness_after: Dict[str, float]
    generation: int
    theorem_name: str
    fitness_delta: Dict[str, float] = field(default_factory=dict)
    discovered_at: str = field(default_factory=_now_iso)
    proof_path: Optional[str] = None
    proof_status: str = "missing"
    dna_integrated: bool = False
    dna_integrated_at: Optional[str] = None
    hypergraph_delta_json: str = ""
    mutation_args_json: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not _DISCOVERY_ID_RE.match(self.id or ""):
            raise ValueError(
                f"DiscoveryRecord.id must match {_DISCOVERY_ID_RE.pattern}, "
                f"got {self.id!r}"
            )
        if self.proof_status not in PROOF_STATUSES:
            raise ValueError(
                f"DiscoveryRecord.proof_status must be in {sorted(PROOF_STATUSES)}, "
                f"got {self.proof_status!r}"
            )
        # Autocompute fitness_delta from before/after if the caller
        # didn't supply one — this is the most common case (engine
        # records `before` + `after` and lets the dataclass derive the
        # delta deterministically).
        if not self.fitness_delta:
            self.fitness_delta = {
                k: float(self.fitness_after.get(k, 0.0))
                   - float(self.fitness_before.get(k, 0.0))
                for k in set(self.fitness_before) | set(self.fitness_after)
            }

    # ── promotion ──────────────────────────────────────────────────

    def promote_to_dna(self, at: Optional[str] = None) -> None:
        """Flag this discovery as integrated into the genome.

        Refuses to flip the bit unless ``proof_status == "verified"`` —
        this is the single safeguard against unverified discoveries
        leaking into the lineage.

        Args:
            at: ISO-8601 timestamp. Defaults to :func:`_now_iso`.

        Raises:
            RuntimeError: if the proof has not been verified.
        """
        if self.proof_status != "verified":
            raise RuntimeError(
                f"DiscoveryRecord.{self.id}: cannot promote to DNA — "
                f"proof_status={self.proof_status!r}, must be 'verified'"
            )
        self.dna_integrated = True
        self.dna_integrated_at = at or _now_iso()

    # ── serialisation ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiscoveryRecord":
        # Strip fitness_delta so __post_init__ can recompute it (or
        # accept the on-disk value if the dict provided one).
        d = dict(d)
        # Make sure the order is deterministic — dataclass requires id first.
        return cls(**d)

    def slug(self) -> str:
        return _slugify(self.title)

    def filename(self) -> str:
        return f"{self.id}_{self.slug()}.md"

    def proof_filename(self) -> str:
        return f"{self.id}_{self.slug()}.lean"

    def to_markdown(self) -> str:
        meta = self.to_dict()
        body = meta.pop("mechanism_md", "")
        return "---\n" + _yaml_dump(meta) + "\n---\n\n" + (body or "")

    @classmethod
    def from_markdown(cls, text: str) -> "DiscoveryRecord":
        m = _FRONT_MATTER_RE.match(text)
        if not m:
            raise ValueError("DiscoveryRecord: missing YAML front-matter")
        meta = _yaml_load(m.group(1))
        body = m.group(2).strip()
        meta["mechanism_md"] = body
        # Backfill defaults for fields a hand-edited file may have dropped.
        meta.setdefault("mutation_chain", [])
        meta.setdefault("fitness_before", {})
        meta.setdefault("fitness_after", {})
        meta.setdefault("fitness_delta", {})
        meta.setdefault("tags", [])
        meta.setdefault("hypergraph_delta_json", "")
        meta.setdefault("mutation_args_json", "")
        meta.setdefault("dna_integrated", False)
        meta.setdefault("discovered_at", _now_iso())
        meta.setdefault("proof_status", "missing")
        return cls.from_dict(meta)


# ── helpers ────────────────────────────────────────────────────────


def _slugify(text: str) -> str:
    """Filename-safe slug: lowercase, non-alphanum → underscore."""
    if not text:
        return "unnamed"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    if not slug:
        return "unnamed"
    # Keep slugs short — long titles still produce something usable.
    return slug[:50]
