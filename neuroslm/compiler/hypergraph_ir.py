# -*- coding: utf-8 -*-
"""Hypergraph IR + SourceMap — Layer 3 of the DNA compiler.

The DSL is lifted into a hypergraph that is the *evolvable* semantic
substrate:

  - HyperNode  — a population, neurotransmitter, or the architecture decl
  - HyperEdge  — a synapse / modulation, connecting several member nodes

Every element records a ``span`` (start, end) into the original source —
its provenance. The ``SourceMap`` keeps the original source plus those
spans, which gives two guarantees the DNA encoder relies on:

  - ``render()``                 reproduces the DSL byte-for-byte
  - ``render_with_overrides()``  re-renders only mutated nodes, splicing
                                 their new text in while leaving every
                                 other byte of the file untouched

This is the AST + original-source pattern used by source-map-preserving
code generators: unchanged regions are byte-preserved via spans; mutated
regions are re-rendered from the IR. The unmutated genome therefore
round-trips bit-identically; a mutated genome differs exactly where it
was mutated.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from neuroslm.dsl.compiler import _parse_properties


Span = Tuple[int, int]


@dataclass
class HyperNode:
    """A vertex in the hypergraph (population / neurotransmitter / arch)."""
    id: str
    kind: str
    name: str
    attrs: Dict[str, str] = field(default_factory=dict)
    span: Span = (0, 0)

    def to_dict(self) -> Dict:
        return {"id": self.id, "kind": self.kind, "name": self.name,
                "attrs": dict(self.attrs), "span": list(self.span)}

    @classmethod
    def from_dict(cls, d: Dict) -> "HyperNode":
        return cls(id=d["id"], kind=d["kind"], name=d["name"],
                   attrs=dict(d.get("attrs", {})),
                   span=tuple(d.get("span", (0, 0))))


@dataclass
class HyperEdge:
    """A hyperedge connecting an ordered list of member node names."""
    id: str
    kind: str
    members: List[str] = field(default_factory=list)
    attrs: Dict[str, str] = field(default_factory=dict)
    span: Span = (0, 0)

    def to_dict(self) -> Dict:
        return {"id": self.id, "kind": self.kind, "members": list(self.members),
                "attrs": dict(self.attrs), "span": list(self.span)}

    @classmethod
    def from_dict(cls, d: Dict) -> "HyperEdge":
        return cls(id=d["id"], kind=d["kind"], members=list(d.get("members", [])),
                   attrs=dict(d.get("attrs", {})),
                   span=tuple(d.get("span", (0, 0))))


@dataclass
class SourceMap:
    """Original source plus per-element spans → bit-identical rendering."""
    source: str
    spans: Dict[str, Span] = field(default_factory=dict)

    def render(self) -> str:
        """Reproduce the original DSL byte-for-byte."""
        return self.source

    def render_with_overrides(self, overrides: Dict[str, str]) -> str:
        """Re-render, replacing the span of each overridden id with new text.

        Non-overridden bytes are emitted verbatim, so the output differs
        from the original only where an element was mutated.
        """
        if not overrides:
            return self.source
        # Collect (start, end, replacement) for ids we know spans for.
        edits = []
        for node_id, new_text in overrides.items():
            span = self.spans.get(node_id)
            if span is None:
                continue
            edits.append((span[0], span[1], new_text))
        edits.sort(key=lambda e: e[0])

        out: List[str] = []
        pos = 0
        for start, end, new_text in edits:
            if start < pos:
                continue  # overlapping / stale span — skip defensively
            out.append(self.source[pos:start])
            out.append(new_text)
            pos = end
        out.append(self.source[pos:])
        return "".join(out)

    def to_dict(self) -> Dict:
        return {"source": self.source,
                "spans": {k: list(v) for k, v in self.spans.items()}}

    @classmethod
    def from_dict(cls, d: Dict) -> "SourceMap":
        return cls(source=d["source"],
                   spans={k: tuple(v) for k, v in d.get("spans", {}).items()})


@dataclass
class HypergraphIR:
    """The full semantic hypergraph plus its source map."""
    nodes: List[HyperNode] = field(default_factory=list)
    hyperedges: List[HyperEdge] = field(default_factory=list)
    source_map: SourceMap = field(default_factory=lambda: SourceMap(""))

    def nodes_of_kind(self, kind: str) -> List[HyperNode]:
        return [n for n in self.nodes if n.kind == kind]

    def to_dict(self) -> Dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "hyperedges": [e.to_dict() for e in self.hyperedges],
            "source_map": self.source_map.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "HypergraphIR":
        return cls(
            nodes=[HyperNode.from_dict(n) for n in d.get("nodes", [])],
            hyperedges=[HyperEdge.from_dict(e) for e in d.get("hyperedges", [])],
            source_map=SourceMap.from_dict(d.get("source_map", {"source": ""})),
        )


# ── DSL -> hypergraph lifting ────────────────────────────────────────────

_NODE_PATTERNS = {
    "architecture":      re.compile(r"architecture\s+(\w+)\s*\{([^}]*)\}"),
    "neurotransmitter":  re.compile(r"neurotransmitter\s+(\w+)\s*\{([^}]*)\}"),
    "population":        re.compile(r"population\s+(\w+)\s*\{([^}]*)\}"),
}

_EDGE_PATTERNS = {
    "synapse":     re.compile(r"synapse\s+(\w+)\s*->\s*(\w+)\s*\{([^}]*)\}"),
    "modulation":  re.compile(r"modulation\s+(\w+)\s*->\s*(\w+)\s*\{([^}]*)\}"),
}


def lift_dsl_to_hypergraph(source: str) -> HypergraphIR:
    """Lift DSL source into a HypergraphIR with provenance spans."""
    nodes: List[HyperNode] = []
    hyperedges: List[HyperEdge] = []
    spans: Dict[str, Span] = {}

    for kind, pat in _NODE_PATTERNS.items():
        for m in pat.finditer(source):
            name = m.group(1)
            attrs = _parse_properties(m.group(2))
            nid = f"{kind}:{name}"
            span = m.span()
            nodes.append(HyperNode(id=nid, kind=kind, name=name,
                                   attrs=attrs, span=span))
            spans[nid] = span

    for kind, pat in _EDGE_PATTERNS.items():
        for m in pat.finditer(source):
            src, dst = m.group(1), m.group(2)
            attrs = _parse_properties(m.group(3))
            eid = f"{kind}:{src}->{dst}"
            span = m.span()
            hyperedges.append(HyperEdge(id=eid, kind=kind, members=[src, dst],
                                        attrs=attrs, span=span))
            spans[eid] = span

    return HypergraphIR(
        nodes=nodes,
        hyperedges=hyperedges,
        source_map=SourceMap(source=source, spans=spans),
    )
