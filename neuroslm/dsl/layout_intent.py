# -*- coding: utf-8 -*-
"""LayoutIntent — semantic-driven layout inference for Neural Flow Graphs.

This module implements the four-stage layout compiler:

1. **Parse semantic graph** — ingest nodes, edges, NT systems, scopes, specs
   from the compiled NFG object.

2. **Infer layout roles** — classify each population into a semantic role
   (trunk, modulatory, memory, predictive, interoceptive, peripheral, etc.)
   using declared scopes first, then topology-based heuristics when absent.

3. **Generate layout constraints** — produce a LayoutIntent object containing
   envelope memberships, node positions/anchors, edge priorities, and orbit
   rails, all derived from the semantic graph rather than hardcoded names.

4. **Render style layer** — consumed by nfg._draw_main_graph() to apply
   colors, boxes, bands, and splines uniformly across any architecture.

Design principles:
- If a visual feature cannot be justified from the NFG object or explicit DSL
  layout metadata, it does not belong in the generic compiler.
- Preset-specific polish exists only as optional hints emitted by the DSL
  compiler (via `layout_hint`, `role`, `cluster` annotations), never as hidden
  renderer assumptions tied to population names.
- The same renderer must work for any BIOMIND architecture with consistent
  style and without fragile hardcoded memberships.

Usage:
    from neuroslm.dsl.layout_intent import infer_layout
    intent = infer_layout(nfg_graph)
    # intent.envelopes, intent.node_roles, intent.positions, ...
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from .nfg import NeuralFlowGraph, NFGNode, NFGEdge


# ═══════════════════════════════════════════════════════════════════════════
# 1. LayoutIntent schema — the normalized intermediate representation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EnvelopeDef:
    """A semantic subsystem envelope to be rendered around a group of nodes."""
    name: str                       # e.g. "trunk", "memory", "predictive_ctrl"
    members: List[str]              # population names belonging to this envelope
    fill_color: str = "#ecf0f1"     # CSS hex
    border_color: str = "#7f8c8d"   # CSS hex
    # Inferred from: param_scope, explicit `cluster` annotation, or topology


@dataclass
class NodeLayout:
    """Per-node layout intent."""
    name: str
    role: str                       # "trunk" | "modulatory" | "memory" | "predictive" | "interoceptive" | "peripheral" | "input" | "output"
    position: Optional[Tuple[float, float]] = None   # (x, y) if anchored
    anchor_to: Optional[str] = None                  # snap toward this node/cluster
    priority: float = 1.0                            # edge-drawing priority weight


@dataclass
class EdgeLayout:
    """Per-edge layout intent."""
    src: str
    tgt: str
    kind: str                       # "synapse" | "modulation"
    prominence: str = "normal"      # "primary" | "normal" | "demoted"
    # Primary = full routing; normal = standard; demoted = faint stub


@dataclass
class LayoutIntent:
    """Complete layout intent for a NeuralFlowGraph."""
    arch_name: str
    # Per-node roles and positions
    node_layouts: Dict[str, NodeLayout] = field(default_factory=dict)
    # Envelope definitions (derived from scopes + topology)
    envelopes: List[EnvelopeDef] = field(default_factory=list)
    # Per-edge prominence
    edge_layouts: List[EdgeLayout] = field(default_factory=list)
    # Derived spine (ordered list of trunk nodes from input to output)
    spine: List[str] = field(default_factory=list)
    # NT diamond positions (computed from modulation targets)
    nt_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Topology metrics — centrality, clustering, flow analysis
# ═══════════════════════════════════════════════════════════════════════════

def _compute_synapse_adjacency(g: "NeuralFlowGraph") -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Return (successors, predecessors) dicts for synapse edges only."""
    succ: Dict[str, Set[str]] = {}
    pred: Dict[str, Set[str]] = {}
    for e in g.edges:
        if e.kind == "synapse":
            succ.setdefault(e.src, set()).add(e.tgt)
            pred.setdefault(e.tgt, set()).add(e.src)
    return succ, pred


def _compute_degree_centrality(g: "NeuralFlowGraph") -> Dict[str, float]:
    """Normalized degree centrality over synapse edges."""
    succ, pred = _compute_synapse_adjacency(g)
    pops = [n.name for n in g.nodes if n.kind == "pop"]
    n = len(pops)
    if n <= 1:
        return {p: 1.0 for p in pops}
    centrality = {}
    for p in pops:
        deg = len(succ.get(p, set())) + len(pred.get(p, set()))
        centrality[p] = deg / (2 * (n - 1))
    return centrality


def _compute_betweenness_centrality(g: "NeuralFlowGraph") -> Dict[str, float]:
    """Approximate betweenness centrality (BFS-based, unweighted)."""
    succ, pred = _compute_synapse_adjacency(g)
    pops = [n.name for n in g.nodes if n.kind == "pop"]
    betweenness = {p: 0.0 for p in pops}
    
    for s in pops:
        # BFS from s
        dist = {s: 0}
        paths = {s: 1}
        queue = [s]
        order = []
        while queue:
            v = queue.pop(0)
            order.append(v)
            for w in succ.get(v, set()):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                    paths[w] = 0
                if dist[w] == dist[v] + 1:
                    paths[w] = paths.get(w, 0) + paths[v]
        # Accumulate betweenness — use all discovered nodes in order
        delta = {p: 0.0 for p in order}
        for w in reversed(order[1:]):
            for v in pred.get(w, set()):
                if v in dist and dist.get(w, 999) == dist[v] + 1:
                    delta[v] = delta.get(v, 0.0) + (paths.get(v, 1) / max(paths.get(w, 1), 1)) * (1 + delta.get(w, 0.0))
            if w in betweenness:
                betweenness[w] += delta.get(w, 0.0)
    
    # Normalize
    n = len(pops)
    norm = (n - 1) * (n - 2) if n > 2 else 1
    return {p: b / max(norm, 1) for p, b in betweenness.items()}


def _derive_spine_from_topology(g: "NeuralFlowGraph") -> List[str]:
    """Find the longest weighted path through the synapse DAG (the trunk)."""
    succ, pred = _compute_synapse_adjacency(g)
    pops = [n.name for n in g.nodes if n.kind == "pop"]
    
    # Find probable input nodes (low in-degree)
    in_deg = {p: len(pred.get(p, set())) for p in pops}
    sources = sorted([p for p in pops if in_deg[p] == 0], key=lambda p: p)
    if not sources:
        sources = sorted(pops, key=lambda p: in_deg[p])[:2]
    
    # Build edge weights
    edge_weights: Dict[Tuple[str, str], float] = {}
    for e in g.edges:
        if e.kind == "synapse":
            edge_weights[(e.src, e.tgt)] = e.weight
    
    best_path: List[str] = []
    best_score = -1.0
    
    for src in sources:
        # DFS with cycle detection
        stack = [(src, [src], 0.0)]
        while stack:
            node, path, score = stack.pop()
            extended = False
            for tgt in succ.get(node, set()):
                if tgt in path:
                    continue  # cycle
                w = edge_weights.get((node, tgt), 1.0)
                stack.append((tgt, path + [tgt], score + w))
                extended = True
            if not extended:
                if len(path) > len(best_path) or (len(path) == len(best_path) and score > best_score):
                    best_path = path
                    best_score = score
    
    return best_path


# ═══════════════════════════════════════════════════════════════════════════
# 3. Role inference — classify nodes semantically
# ═══════════════════════════════════════════════════════════════════════════

# Default role-to-color mapping (used when DSL doesn't specify colors)
ROLE_COLORS: Dict[str, Tuple[str, str]] = {
    "trunk":        ("#fef9e7", "#f39c12"),   # warm yellow band
    "memory":       ("#d5f5e3", "#27ae60"),   # green
    "predictive":   ("#fdf2f8", "#8e44ad"),   # purple
    "interoceptive":("#fce4ec", "#c0392b"),   # red-pink
    "self_model":   ("#d6eaf8", "#2980b9"),   # blue
    "cortical":     ("#fef9e7", "#f39c12"),   # orange
    "modulatory":   ("#fff3e0", "#e65100"),   # orange (nuclei)
    "peripheral":   ("#ecf0f1", "#7f8c8d"),   # gray
    "input":        ("#e3f2fd", "#1976d2"),   # blue
    "output":       ("#ffebee", "#c62828"),   # red
}


def _infer_role_from_scope(scope_name: str) -> str:
    """Map a declared param_scope name to a semantic role."""
    s = scope_name.lower()
    if "trunk" in s or "main" in s:
        return "trunk"
    if "bio" in s or "detach" in s:
        return "peripheral"
    if "memory" in s or "episodic" in s or "hippocampal" in s:
        return "memory"
    if "predict" in s or "forward" in s or "model" in s:
        return "predictive"
    if "intero" in s or "affect" in s or "emotion" in s:
        return "interoceptive"
    if "self" in s or "qualia" in s or "world" in s:
        return "self_model"
    if "cortex" in s or "cortical" in s:
        return "cortical"
    return "peripheral"


def _infer_role_from_topology(
    name: str,
    spine: List[str],
    centrality: Dict[str, float],
    is_nt: bool,
    in_deg: int,
    out_deg: int,
) -> str:
    """Topology-based role inference when no scope annotation exists."""
    if is_nt:
        return "modulatory"
    if name in spine:
        # Position on spine determines role
        idx = spine.index(name)
        frac = idx / max(len(spine) - 1, 1)
        if frac < 0.15:
            return "input"
        if frac > 0.85:
            return "output"
        return "trunk"
    # High centrality but not on spine → cortical/cognitive
    if centrality.get(name, 0) > 0.3:
        return "cortical"
    # Low in-degree, some out → likely input-adjacent
    if in_deg == 0 and out_deg > 0:
        return "input"
    # Low out-degree, some in → output-adjacent
    if out_deg == 0 and in_deg > 0:
        return "output"
    return "peripheral"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Position inference — semantic placement without hardcoded names
# ═══════════════════════════════════════════════════════════════════════════

def _compute_semantic_positions(
    g: "NeuralFlowGraph",
    node_roles: Dict[str, str],
    spine: List[str],
) -> Dict[str, Tuple[float, float]]:
    """Compute positions based on roles and spine membership."""
    pos: Dict[str, Tuple[float, float]] = {}
    
    # 1. Lay out spine left→right on y=0
    if spine:
        n = len(spine)
        x_span = 14.0  # total width
        x_start = -x_span / 2
        for i, name in enumerate(spine):
            x = x_start + (i / max(n - 1, 1)) * x_span
            pos[name] = (x, 0.0)
    
    # 2. Cluster non-spine nodes by role
    role_clusters: Dict[str, List[str]] = {}
    for n in g.nodes:
        if n.kind != "pop" or n.name in pos:
            continue
        role = node_roles.get(n.name, "peripheral")
        role_clusters.setdefault(role, []).append(n.name)
    
    # Role → (centroid_x, centroid_y) placement rules
    ROLE_CENTROIDS = {
        "memory":       (0.0, -2.5),
        "predictive":   (4.5, -2.5),
        "interoceptive":(-6.0, -2.5),
        "self_model":   (-3.5, -2.8),
        "cortical":     (0.0, 2.8),
        "modulatory":   (0.0, 4.0),
        "peripheral":   (0.0, -3.5),
        "input":        (-6.5, 0.0),
        "output":       (6.5, 0.0),
    }
    
    for role, members in role_clusters.items():
        cx, cy = ROLE_CENTROIDS.get(role, (0.0, 0.0))
        k = len(members)
        radius = 0.8 + 0.2 * k
        for i, name in enumerate(sorted(members)):
            if k == 1:
                pos[name] = (cx, cy)
            else:
                ang = 2 * math.pi * i / k
                pos[name] = (cx + radius * math.cos(ang), cy + radius * math.sin(ang))
    
    # 3. NT diamonds — place near centroid of their modulation targets
    for n in g.nodes:
        if n.kind != "nt":
            continue
        targets = [e.tgt for e in g.edges if e.kind == "modulation" and e.src == n.name and e.tgt in pos]
        if targets:
            cx = sum(pos[t][0] for t in targets) / len(targets)
            cy = sum(pos[t][1] for t in targets) / len(targets)
            # Place above the centroid
            pos[n.name] = (cx, max(cy + 1.5, 3.5))
        else:
            # Default NT position
            pos[n.name] = (0.0, 4.0)
    
    return pos


# ═══════════════════════════════════════════════════════════════════════════
# 5. Envelope generation — data-driven from scopes
# ═══════════════════════════════════════════════════════════════════════════

def _generate_envelopes(
    g: "NeuralFlowGraph",
    node_roles: Dict[str, str],
) -> List[EnvelopeDef]:
    """Generate envelope definitions from param_scopes and inferred roles."""
    envelopes: List[EnvelopeDef] = []
    seen_members: Set[str] = set()
    
    # 1. Envelopes from declared param_scopes
    for scope in g.param_scopes:
        scope_name = scope.get("name", "unnamed")
        members = [p for p in scope.get("populations", []) if p not in seen_members]
        if len(members) < 2:
            continue
        seen_members.update(members)
        
        role = _infer_role_from_scope(scope_name)
        fill, border = ROLE_COLORS.get(role, ("#ecf0f1", "#7f8c8d"))
        
        envelopes.append(EnvelopeDef(
            name=scope_name,
            members=members,
            fill_color=fill,
            border_color=border,
        ))
    
    # 2. Envelopes from inferred roles (for nodes not in any declared scope)
    role_groups: Dict[str, List[str]] = {}
    for name, role in node_roles.items():
        if name in seen_members:
            continue
        if role in ("trunk", "input", "output", "modulatory"):
            continue  # these are rendered differently (spine band, etc.)
        role_groups.setdefault(role, []).append(name)
    
    for role, members in role_groups.items():
        if len(members) < 2:
            continue
        fill, border = ROLE_COLORS.get(role, ("#ecf0f1", "#7f8c8d"))
        envelopes.append(EnvelopeDef(
            name=role,
            members=members,
            fill_color=fill,
            border_color=border,
        ))
    
    return envelopes


# ═══════════════════════════════════════════════════════════════════════════
# 6. Main entry point — infer_layout()
# ═══════════════════════════════════════════════════════════════════════════

def infer_layout(g: "NeuralFlowGraph") -> LayoutIntent:
    """Infer complete layout intent from a compiled NeuralFlowGraph.
    
    This is the main entry point. It:
    1. Derives the trunk spine from topology
    2. Computes centrality metrics
    3. Infers semantic roles for each node
    4. Generates envelope definitions from scopes + topology
    5. Computes semantic positions
    6. Determines edge prominence
    
    The returned LayoutIntent is a pure data object with no hardcoded
    architecture-specific assumptions.
    """
    intent = LayoutIntent(arch_name=g.arch_name)
    
    # Build scope membership lookup
    scope_of: Dict[str, str] = {}
    for scope in g.param_scopes:
        scope_name = scope.get("name", "")
        for pop in scope.get("populations", []):
            scope_of[pop] = scope_name
    
    # Compute topology metrics
    spine = _derive_spine_from_topology(g)
    intent.spine = spine
    
    centrality = _compute_betweenness_centrality(g)
    succ, pred = _compute_synapse_adjacency(g)
    
    # Infer roles
    for n in g.nodes:
        if n.kind == "nt":
            role = "modulatory"
        elif n.name in scope_of:
            role = _infer_role_from_scope(scope_of[n.name])
        else:
            in_deg = len(pred.get(n.name, set()))
            out_deg = len(succ.get(n.name, set()))
            role = _infer_role_from_topology(
                n.name, spine, centrality, n.kind == "nt", in_deg, out_deg
            )
        
        intent.node_layouts[n.name] = NodeLayout(
            name=n.name,
            role=role,
            priority=centrality.get(n.name, 0.5),
        )
    
    # Generate envelopes
    node_roles = {name: nl.role for name, nl in intent.node_layouts.items()}
    intent.envelopes = _generate_envelopes(g, node_roles)
    
    # Compute positions
    positions = _compute_semantic_positions(g, node_roles, spine)
    for name, (x, y) in positions.items():
        if name in intent.node_layouts:
            intent.node_layouts[name].position = (x, y)
    
    # Store NT positions separately
    for n in g.nodes:
        if n.kind == "nt" and n.name in positions:
            intent.nt_positions[n.name] = positions[n.name]
    
    # Determine edge prominence
    # Top-k modulations per target get "primary", rest get "demoted"
    MOD_TOP_K = 2
    tgt_mods: Dict[str, List["NFGEdge"]] = {}
    for e in g.edges:
        if e.kind == "modulation":
            tgt_mods.setdefault(e.tgt, []).append(e)
    
    primary_edges: Set[Tuple[str, str]] = set()
    for tgt, edges in tgt_mods.items():
        top = sorted(edges, key=lambda e: abs(e.weight), reverse=True)[:MOD_TOP_K]
        for e in top:
            primary_edges.add((e.src, e.tgt))
    
    for e in g.edges:
        if e.kind == "synapse":
            # Spine edges get primary prominence
            prom = "primary" if (e.src in spine and e.tgt in spine) else "normal"
        else:
            prom = "primary" if (e.src, e.tgt) in primary_edges else "demoted"
        
        intent.edge_layouts.append(EdgeLayout(
            src=e.src,
            tgt=e.tgt,
            kind=e.kind,
            prominence=prom,
        ))
    
    return intent


# ═══════════════════════════════════════════════════════════════════════════
# 7. Legacy compatibility — bridge to existing nfg.py renderer
# ═══════════════════════════════════════════════════════════════════════════

def intent_to_legacy_envelopes(intent: LayoutIntent) -> list:
    """Convert LayoutIntent envelopes to the legacy _SUBSYSTEM_ENVELOPES format."""
    return [
        (env.name, env.members, env.fill_color, env.border_color)
        for env in intent.envelopes
    ]


def intent_to_positions(intent: LayoutIntent) -> Dict[str, Tuple[float, float]]:
    """Extract positions dict from LayoutIntent."""
    pos = {}
    for name, nl in intent.node_layouts.items():
        if nl.position:
            pos[name] = nl.position
    pos.update(intent.nt_positions)
    return pos
