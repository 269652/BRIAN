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
from neuroslm.dsl.training_config import _strip_comments


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


# ── DSL -> hypergraph lifting (multi-file) ───────────────────────────────

def lift_arch_to_hypergraph(arch_root, *, repo_root=None) -> HypergraphIR:
    """Lift a multi-file architecture (arch.neuro + imported modules) into
    a single :class:`HypergraphIR`.

    This is the visualisation source of truth: populations / synapses /
    modulations declared in ``modules/*.neuro`` are included alongside
    the top-level ``arch.neuro`` declarations.

    Args:
        arch_root: ``str`` or ``Path`` pointing at the architecture folder
                   (must contain ``arch.neuro``).
        repo_root: optional explicit repo root for ``@brian/`` / ``@lib/``
                   import resolution. When ``None`` the resolver walks
                   up from ``arch_root`` looking for ``pyproject.toml``.
                   Pass this explicitly when ``arch_root`` lives outside
                   the repo tree (e.g. a workspace tmp dir, vast.ai box,
                   colab) so ``@lib/equations`` resolves to
                   ``<repo>/lib/equations.neuro`` instead of failing.

    Returns:
        ``HypergraphIR`` whose ``source_map`` concatenates all module
        source texts (with a ``# --- file: <relative-path> ---`` marker
        between them) and records per-element spans into that
        concatenated source. Spans remain valid as long as the returned
        source is read verbatim from ``ir.source_map.source``.

    Notes:
        Architecture-level NT systems and synapses live in ``arch.neuro``;
        per-region populations live in ``modules/*.neuro``. Both are
        lifted; node ids stay unique because each ``HyperNode.name`` is
        unique per architecture (DSL semantics already enforce this).
    """
    from pathlib import Path as _Path
    from neuroslm.dsl.multifile import Resolver

    arch_root = _Path(arch_root)
    program = Resolver(arch_root, repo_root=repo_root).resolve()

    # Concatenate every module's text with a separator that includes the
    # relative path — this makes spans easy to debug while keeping all
    # declarations in a single string the regex lifter understands.
    parts: list[str] = []
    # Sort by (depth, name) so arch.neuro comes first, then modules
    # alphabetically.  This makes the concatenated source deterministic
    # which is important for stable spans across runs.
    sorted_paths = sorted(
        program.modules.keys(),
        key=lambda p: (0 if p.name == "arch.neuro" else 1, str(p).lower()),
    )
    for file_path in sorted_paths:
        ast = program.modules[file_path]
        try:
            rel = file_path.relative_to(arch_root.resolve()).as_posix()
        except ValueError:
            rel = file_path.name
        parts.append(f"# --- file: {rel} ---\n")
        # Read the raw file so we capture *everything* (including private
        # declarations and architecture blocks), not just exported names.
        try:
            parts.append(file_path.read_text(encoding="utf-8"))
        except OSError:
            # Fall back to whatever the resolver stashed if I/O fails
            # (defensive — Resolver already succeeded on this path).
            for decl in ast.exports.values():
                parts.append(decl)
            for decl in ast.private.values():
                parts.append(decl)
        parts.append("\n")

    combined = "".join(parts)
    ir = lift_dsl_to_hypergraph(combined)
    _apply_multi_cortex(combined, ir)
    _apply_param_scopes(combined, ir)
    return ir


# ── multi_cortex + param_scope lifting (block extraction) ───────────────
#
# The two helpers below pick up architectural features that aren't simple
# top-level declarations:
#
#   training { multi_cortex { ... } }   -> Slot A KL distillation +
#                                          Slot C NT-gated α inhibition
#   param_scope <name> { populations: [...] } -> anatomical / gradient
#                                                 grouping for clusters
#
# Both regexes locate the *opening* of the block; the body is then
# extracted by depth-balanced brace matching so nested dictionaries
# survive (e.g. ``inhibition: { ... }`` inside ``multi_cortex``).

_MULTI_CORTEX_RE = re.compile(r"multi_cortex\s*:\s*\{")
_PARAM_SCOPE_RE  = re.compile(r"param_scope\s+(\w+)\s*\{")


def _extract_balanced_block(source: str, brace_idx: int) -> Tuple[str, int]:
    """Return ``(body, end_idx)`` for the block opened at ``brace_idx``.

    ``source[brace_idx]`` must be ``'{'``. The returned ``body`` is the
    text *between* (not including) the outer braces; ``end_idx`` points
    one past the matching closing brace. The scan is string-literal aware
    so braces inside quoted strings don't disturb the depth counter.
    """
    if brace_idx >= len(source) or source[brace_idx] != "{":
        raise ValueError(f"expected '{{' at position {brace_idx}")
    depth = 0
    in_str: Optional[str] = None
    escaped = False
    for i in range(brace_idx, len(source)):
        ch = source[i]
        if escaped:
            escaped = False
            continue
        if in_str:
            if ch == "\\":
                escaped = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[brace_idx + 1:i], i + 1
    raise ValueError(f"unbalanced '{{' starting at position {brace_idx}")


def _unquote(value: str) -> str:
    """Strip surrounding ``"`` or ``'`` quotes from a DSL value string."""
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
        return v[1:-1]
    return v


def _parse_dsl_list(value: str) -> List[str]:
    """Parse ``[a, b, c]`` / ``["a", "b", "c"]`` into stripped, unquoted items."""
    from neuroslm.dsl.compiler import _split_top_level
    v = value.strip()
    if not (v.startswith("[") and v.endswith("]")):
        return []
    inner = v[1:-1]
    items: List[str] = []
    for piece in _split_top_level(inner):
        piece = _unquote(piece.strip())
        if piece:
            items.append(piece)
    return items


def _parse_dsl_object_list(value: str) -> List[Dict[str, str]]:
    """Parse ``[ { k1: v1, k2: v2 }, { ... } ]`` into a list of dicts.

    Used for the per-expert roster ``experts: [{id, domain, freeze}, ...]``
    inside the ``multi_cortex`` block. Comment-stripping must be done by
    the caller; this routine assumes a clean DSL string.

    Each inner brace block is parsed with the standard
    ``_parse_properties`` so quoting/spacing rules match the rest of the
    DSL surface. Unbalanced or malformed input returns ``[]`` rather
    than raising — the caller will fall back to ``domains:`` lifting.
    """
    v = value.strip()
    if not (v.startswith("[") and v.endswith("]")):
        return []
    inner = v[1:-1].strip()
    if not inner:
        return []
    out: List[Dict[str, str]] = []
    i, n = 0, len(inner)
    while i < n:
        # Skip whitespace and commas between objects
        while i < n and inner[i] in " \t\n\r,":
            i += 1
        if i >= n:
            break
        if inner[i] != "{":
            # Malformed — bail out rather than misparse.
            return []
        # Walk to matching '}' (string-aware)
        depth = 0
        in_str: Optional[str] = None
        esc = False
        start = i
        end = -1
        for j in range(i, n):
            ch = inner[j]
            if esc:
                esc = False
                continue
            if in_str:
                if ch == "\\":
                    esc = True
                elif ch == in_str:
                    in_str = None
                continue
            if ch in ('"', "'"):
                in_str = ch
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end < 0:
            return []
        obj_body = inner[start + 1:end]
        out.append(_parse_properties(obj_body))
        i = end + 1
    return out


def _apply_multi_cortex(source: str, ir: HypergraphIR) -> None:
    """Surface the GPT-2 cortex ensemble + Slot A/C edges from the
    ``training { multi_cortex { ... } }`` block.

    Effect on ``ir`` (idempotent — does nothing if the block is absent
    or ``enabled: false``):

    1. Every population named ``cortex_<domain>`` (where ``<domain>``
       appears in ``domains: [...]``) is **reclassified** from
       ``kind="population"`` to ``kind="cortex_expert"``. The GPT-2
       backbone tag (``weights``, ``freeze_weights``) is attached.
    2. A synthetic ``lm_trunk`` node (``kind="lm_trunk"``) is appended
       as the anchor for distillation / inhibition edges.
    3. If ``distillation_enabled: true`` — one ``HyperEdge`` of kind
       ``distillation`` per expert, member ``[cortex_<d>, lm_trunk]``,
       carrying ``lambda_max``, ``temperature``, ``gap_floor``,
       ``gap_ceiling`` attrs.
    4. If ``inhibition_enabled: true`` — one ``HyperEdge`` of kind
       ``inhibition`` per expert, member ``[lm_trunk, cortex_<d>]``,
       carrying ``ema_alpha`` and ``temperature``.

    These edges encode the BRIANHarness ``_cortex_fusion_aux_step`` aux
    losses so they show up in the diagram (see
    ``neuroslm/harness.py::_build_multi_cortex``).

    Comment-stripping
    -----------------
    The raw DSL source contains free-text comments with apostrophes
    (``# the expert's pretrained head``) and math notation
    (``# L_{ti} = ...``). The brace walker is string-aware but not
    comment-aware, so a stray apostrophe in a comment would put the
    walker into single-quote string mode and let the multi_cortex
    block "swallow" everything until the next apostrophe. That made
    distillation/inhibition edges invisible in the diagram.

    Fix: strip comments (replace ``# ...`` with spaces, preserving
    offsets) before the brace walk. ``_strip_comments`` from
    ``neuroslm.dsl.training_config`` already does this exactly.
    """
    # Strip comments so apostrophes / { / } inside `# ...` text can't
    # corrupt the brace walker or the property parser. Offsets are
    # preserved (comments become spaces), so spans stay accurate.
    source = _strip_comments(source)

    m = _MULTI_CORTEX_RE.search(source)
    if not m:
        return
    brace_idx = m.end() - 1
    try:
        body, end_idx = _extract_balanced_block(source, brace_idx)
    except ValueError:
        return

    props = _parse_properties(body)
    if _unquote(props.get("enabled", "false")).lower() != "true":
        return

    # New MoE roster path: `experts: [{id, domain, freeze}, ...]`.
    # Each entry's `id` (HF model id) becomes the per-expert `weights`
    # attr, and `freeze` becomes `freeze_weights`. This supersedes the
    # legacy block-level `weights: ""` field which only worked when
    # every expert shared one HF id.
    expert_specs = _parse_dsl_object_list(props.get("experts", ""))

    if expert_specs:
        domains = [_unquote(spec.get("domain", "")) for spec in expert_specs]
        domains = [d for d in domains if d]
        # Per-domain weight + freeze lookup
        per_expert: Dict[str, Dict[str, str]] = {}
        for spec in expert_specs:
            d = _unquote(spec.get("domain", ""))
            if not d:
                continue
            per_expert[d] = {
                "weights":        _unquote(spec.get("id", "")),
                "freeze_weights": _unquote(
                    spec.get("freeze", "true")
                ).lower(),
            }
    else:
        # Legacy single-weights path: every expert shares the block
        # level `weights:` / `freeze_weights:` fields.
        domains = _parse_dsl_list(props.get("domains", "[]"))
        block_weights = _unquote(props.get("weights", ""))
        block_freeze  = _unquote(props.get("freeze_weights", "false")).lower()
        per_expert = {
            d: {"weights": block_weights, "freeze_weights": block_freeze}
            for d in domains
        }

    if not domains:
        return

    span = (m.start(), end_idx)

    # 1. Reclassify cortex_<domain> populations -> cortex_expert
    for domain in domains:
        name = f"cortex_{domain}"
        info = per_expert.get(domain, {"weights": "", "freeze_weights": "false"})
        promoted = False
        for node in ir.nodes:
            if node.name == name and node.kind == "population":
                old_id = node.id
                node.kind = "cortex_expert"
                node.id = f"cortex_expert:{name}"
                node.attrs["domain"] = domain
                # Always tag weights/freeze from the roster so the
                # IR is honest about what each expert actually loads.
                node.attrs["weights"] = info["weights"]
                node.attrs["freeze_weights"] = info["freeze_weights"]
                # Re-key the span so round-trip rendering still works.
                if old_id in ir.source_map.spans:
                    ir.source_map.spans[node.id] = ir.source_map.spans.pop(old_id)
                promoted = True
                break
        if not promoted:
            # Multi-cortex declared a domain with no matching population
            # declaration — synthesise the expert anyway so the diagram
            # is honest about what BRIANHarness will instantiate.
            synth_id = f"cortex_expert:{name}"
            ir.nodes.append(HyperNode(
                id=synth_id,
                kind="cortex_expert",
                name=name,
                attrs={
                    "domain": domain,
                    "weights": info["weights"],
                    "freeze_weights": info["freeze_weights"],
                    "synthetic": "true",
                },
                span=span,
            ))
            ir.source_map.spans[synth_id] = span

    # 2. Synthetic LM trunk anchor (the main BRIAN language-model trunk)
    ir.nodes.append(HyperNode(
        id="lm_trunk",
        kind="lm_trunk",
        name="lm_trunk",
        attrs={
            "role": "language_model_trunk",
            "n_cortices": str(len(domains)),
            "synthetic": "true",
        },
        span=span,
    ))
    ir.source_map.spans["lm_trunk"] = span

    # 3. Distillation edges (Slot A): expert -> trunk
    if _unquote(props.get("distillation_enabled", "false")).lower() == "true":
        lambda_max  = _unquote(props.get("distillation_lambda_max",  ""))
        temperature = _unquote(props.get("distillation_temperature", ""))
        gap_floor   = _unquote(props.get("distillation_gap_floor",   ""))
        gap_ceiling = _unquote(props.get("distillation_gap_ceiling", ""))
        for domain in domains:
            name = f"cortex_{domain}"
            attrs: Dict[str, str] = {}
            if lambda_max:
                attrs["lambda_max"]  = lambda_max
            if temperature:
                attrs["temperature"] = temperature
            if gap_floor:
                attrs["gap_floor"]   = gap_floor
            if gap_ceiling:
                attrs["gap_ceiling"] = gap_ceiling
            edge_id = f"distillation:{name}->lm_trunk"
            ir.hyperedges.append(HyperEdge(
                id=edge_id,
                kind="distillation",
                members=[name, "lm_trunk"],
                attrs=attrs,
                span=span,
            ))
            ir.source_map.spans[edge_id] = span

    # 4. Inhibition edges (Slot C): trunk -> expert
    if _unquote(props.get("inhibition_enabled", "false")).lower() == "true":
        ema_alpha  = _unquote(props.get("inhibition_ema_alpha",  ""))
        inhib_temp = _unquote(props.get("inhibition_temperature", ""))
        for domain in domains:
            name = f"cortex_{domain}"
            attrs = {}
            if ema_alpha:
                attrs["ema_alpha"]   = ema_alpha
            if inhib_temp:
                attrs["temperature"] = inhib_temp
            edge_id = f"inhibition:lm_trunk->{name}"
            ir.hyperedges.append(HyperEdge(
                id=edge_id,
                kind="inhibition",
                members=["lm_trunk", name],
                attrs=attrs,
                span=span,
            ))
            ir.source_map.spans[edge_id] = span


def _apply_param_scopes(source: str, ir: HypergraphIR) -> None:
    """Annotate populations with their ``param_scope`` (the declarative
    gradient-isolation block).

    For every ``param_scope <name> { populations: [a, b, c], gradient: "..." }``
    in the combined source, each node whose ``name`` appears in the list
    gains ``attrs["param_scope"] = <name>`` (and ``attrs["gradient"]`` if
    a gradient mode is declared). Nodes not listed in any scope are left
    untouched so the renderer can route them to a fallback bucket.
    """
    # Strip comments so apostrophes / { / } inside `# ...` text can't
    # corrupt the brace walker (same reason as in _apply_multi_cortex).
    source = _strip_comments(source)
    name_to_node = {n.name: n for n in ir.nodes}

    for m in _PARAM_SCOPE_RE.finditer(source):
        scope_name = m.group(1)
        brace_idx = m.end() - 1
        try:
            body, _end = _extract_balanced_block(source, brace_idx)
        except ValueError:
            continue
        props = _parse_properties(body)
        pop_names = _parse_dsl_list(props.get("populations", "[]"))
        gradient = _unquote(props.get("gradient", ""))
        for pop_name in pop_names:
            node = name_to_node.get(pop_name)
            if node is None:
                continue
            node.attrs["param_scope"] = scope_name
            if gradient:
                node.attrs["gradient"] = gradient
