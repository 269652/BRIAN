# -*- coding: utf-8 -*-
"""Neural Flow Graph — compile an architecture into a typed graph (Python +
PNG) that visualises forward dataflow, transformation per node, synapses
with weights, and NT modulations.

The NFG is *richer than* `analyzer.render_topology()`:
  - Each node is labelled with its operation (ReLU / softmax / gated / ODE)
    parsed from the population equation, not just the name.
  - Edges carry the synapse weight + neurotransmitter type.
  - Modulations appear as coloured dashed edges from NT-nodes to their
    target populations (multiplicative=red, additive=blue), so the dual
    information channels (synaptic + neuromodulatory) are visible at a
    glance.
  - Layout is layered (sensory → motor) where the topology supports it,
    matching how you read a cognitive bowtie diagram.

The graph itself is exported as a *runnable Python module* — a dict-of-
dicts that `pickle.loads(...)` for a notebook or `networkx.DiGraph()`
for further analysis — so the NFG can be loaded without re-parsing the
.neuro DSL.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class NFGNode:
    name: str
    kind: str                      # "pop" | "nt"
    op: str = ""                   # "relu" | "softmax" | "gated" | "ode" | "linear" | "nt_dynamics"
    equation: Optional[str] = None
    properties: Dict = field(default_factory=dict)


@dataclass
class NFGEdge:
    src: str
    tgt: str
    kind: str                      # "synapse" | "modulation"
    weight: float = 1.0
    nt: Optional[str] = None       # neurotransmitter name (synapse: which NT, mod: source NT)
    effect: Optional[str] = None   # "multiplicative" | "additive" (modulation only)


@dataclass
class NeuralFlowGraph:
    arch_name: str
    nodes: List[NFGNode] = field(default_factory=list)
    edges: List[NFGEdge] = field(default_factory=list)

    def stats(self) -> Dict[str, int]:
        return {
            "n_populations": sum(1 for n in self.nodes if n.kind == "pop"),
            "n_neurotransmitters": sum(1 for n in self.nodes if n.kind == "nt"),
            "n_synapses": sum(1 for e in self.edges if e.kind == "synapse"),
            "n_modulations": sum(1 for e in self.edges if e.kind == "modulation"),
        }


# ── 1. Compile arch → NFG ─────────────────────────────────────────────

def _classify_op(equation: Optional[str], ode: Optional[str]) -> str:
    if ode:
        return "ode"
    if not equation:
        return "linear"
    e = equation.lower()
    if "softmax" in e and "relu" in e:
        return "softmax_relu"
    if "softmax" in e:
        return "softmax"
    if "sigmoid" in e and ("relu" in e or "max" in e):
        return "gated"
    if "tanh" in e:
        return "tanh"
    if "relu" in e or "max(0" in e:
        return "relu"
    return "linear"


def compile_nfg(arch_root) -> NeuralFlowGraph:
    """Build a Neural Flow Graph from an architecture folder."""
    from .multifile import compile_folder
    ir = compile_folder(Path(arch_root))

    # `ProgramIR` doesn't carry the architecture name directly — use the
    # folder name as a sensible label.
    name = Path(arch_root).name or "?"
    g = NeuralFlowGraph(arch_name=name)

    # Populations
    for pop in ir.populations:
        op = _classify_op(getattr(pop, "equation", None),
                          getattr(pop, "ode", None))
        g.nodes.append(NFGNode(
            name=pop.name, kind="pop", op=op,
            equation=pop.equation or pop.ode,
            properties={"count": pop.count, "dynamics": pop.dynamics},
        ))

    # Neurotransmitters as separate nodes (only those actually modulating)
    nt_used = {m.source_nt for m in ir.modulations}
    for nt in ir.neurotransmitter_systems:
        if nt.name in nt_used:
            g.nodes.append(NFGNode(
                name=nt.name, kind="nt", op="nt_dynamics",
                properties={"base": nt.base_concentration,
                            "release": nt.release_rate,
                            "reuptake": nt.reuptake_rate},
            ))

    # Synapses
    for syn in ir.synapses:
        g.edges.append(NFGEdge(
            src=syn.source, tgt=syn.target, kind="synapse",
            weight=float(syn.weight) if syn.weight is not None else 1.0,
            nt=syn.neurotransmitter,
        ))

    # Modulations — NT → population
    for mod in ir.modulations:
        g.edges.append(NFGEdge(
            src=mod.source_nt, tgt=mod.target_population, kind="modulation",
            weight=float(mod.gain) if mod.gain is not None else 1.0,
            nt=mod.source_nt, effect=mod.effect,
        ))

    return g


# ── 2. Layered layout — sensory (top) → motor (bottom) ────────────────

def _layered_positions(g: NeuralFlowGraph) -> Dict[str, Tuple[float, float]]:
    """BFS from source populations to assign each node a "depth" layer;
    NTs floated to the side."""
    # Build pop adjacency
    succ: Dict[str, List[str]] = {}
    pred: Dict[str, List[str]] = {}
    pop_names = [n.name for n in g.nodes if n.kind == "pop"]
    for e in g.edges:
        if e.kind != "synapse":
            continue
        succ.setdefault(e.src, []).append(e.tgt)
        pred.setdefault(e.tgt, []).append(e.src)

    sources = [n for n in pop_names if n not in pred]
    depth: Dict[str, int] = {n: 0 for n in sources}
    frontier = list(sources)
    while frontier:
        new_frontier = []
        for u in frontier:
            for v in succ.get(u, []):
                d = depth.get(u, 0) + 1
                if v not in depth or depth[v] < d:
                    depth[v] = d
                    new_frontier.append(v)
        frontier = new_frontier
    # Any disconnected pop → depth max+1
    if depth:
        max_d = max(depth.values())
    else:
        max_d = 0
    for n in pop_names:
        depth.setdefault(n, max_d + 1)

    # Bucket by depth
    by_depth: Dict[int, List[str]] = {}
    for n, d in depth.items():
        by_depth.setdefault(d, []).append(n)

    pos: Dict[str, Tuple[float, float]] = {}
    layers = sorted(by_depth)
    for d in layers:
        col = by_depth[d]
        col.sort()
        for i, n in enumerate(col):
            x = (i - (len(col) - 1) / 2) * 1.5
            y = -d * 1.2
            pos[n] = (x, y)

    # NTs along the right margin
    nts = [n.name for n in g.nodes if n.kind == "nt"]
    for i, nt in enumerate(nts):
        x = max([p[0] for p in pos.values()], default=0) + 3.5
        y = -(i / max(1, len(nts) - 1)) * (len(layers) - 1) * 1.2 if len(nts) > 1 else 0
        pos[nt] = (x, y)
    return pos


# ── 3. Render to PNG ──────────────────────────────────────────────────

# Anatomical region colors — overrides per-op coloring so the
# diagram reads as a brain map, not a math expression
_REGION_COLORS = {
    "input":      "#3498db",   # sensory, association   — blue
    "thalamic":   "#9b59b6",   # thalamus               — purple
    "cortex":     "#f39c12",   # pfc, acc, dmn, gws...  — orange
    "memory":     "#2ecc71",   # hippo, entorhinal      — green
    "subcort":    "#16a085",   # amygdala, insula, bg.. — teal
    "world":      "#7f8c8d",   # world, self_m, etc.    — slate
    "output":     "#e74c3c",   # motor                  — red
    "nuclei":     "#fdb6c8",   # vta, lc, raphe, etc.   — pink
}
_REGION_OF = {
    # input
    "sensory": "input", "association": "input",
    # thalamic
    "thalamus": "thalamic",
    # cortex (incl. workspace + adjacent)
    "pfc": "cortex", "acc": "cortex", "dmn": "cortex",
    "gws": "cortex", "claustrum": "cortex", "neural_geometry": "cortex",
    "qualia": "cortex", "thought_transformer": "cortex",
    "math_cortex": "cortex", "reasoning_cortex": "cortex",
    "language_cortex": "cortex",
    # memory
    "hippo": "memory", "entorhinal": "memory", "cerebellum": "memory",
    # subcortical
    "amygdala": "subcort", "insula": "subcort", "bg": "subcort",
    "forward_m": "subcort", "evaluator": "subcort",
    # world / self
    "world": "world", "self_m": "world",
    # output
    "motor": "output",
    # neuromod nuclei
    "vta": "nuclei", "nucleus_accumbens": "nuclei",
    "locus_coeruleus": "nuclei", "raphe_nuclei": "nuclei",
    "nucleus_basalis": "nuclei", "substantia_nigra": "nuclei",
}


def _try_graphviz_layout(G, prog: str = "dot"):
    """Try networkx's pygraphviz layout. Returns None if pygraphviz missing
    or if graphviz dot is not installed. Times out fast — never hangs."""
    # graphviz_layout shells out to `dot` — without graphviz installed on
    # PATH it'll fail with an unhelpful subprocess error or hang. Probe
    # cheaply first:
    import shutil
    if shutil.which("dot") is None:
        return None
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        return graphviz_layout(G, prog=prog)
    except Exception:
        try:
            from networkx.drawing.nx_agraph import graphviz_layout
            return graphviz_layout(G, prog=prog)
        except Exception:
            return None


def render_nfg(g: NeuralFlowGraph, output_path: str,
               figsize: Tuple[int, int] = (20, 14),
               layout: str = "auto") -> None:
    """Render via matplotlib + networkx — region-colored populations, hub-
    sized nodes, curved edges, distinct synapse/modulation styles.

    Args:
        layout: "auto" (prefer graphviz dot, else layered), "dot",
                "spring", "layered".
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    G = nx.DiGraph()
    for n in g.nodes:
        G.add_node(n.name, kind=n.kind, op=n.op)
    for e in g.edges:
        G.add_edge(e.src, e.tgt, kind=e.kind, weight=e.weight,
                   nt=e.nt, effect=e.effect)

    # Layout selection. Kamada-Kawai gives the cleanest balanced layout
    # for small (<100 node) graphs — handles nuclei + modulation edges
    # together so neuromod nodes pull toward their targets instead of
    # floating off to the margin (spring layout's typical failure).
    pos = None
    if layout in ("auto", "dot"):
        pos = _try_graphviz_layout(G, "dot")
    if pos is None and layout in ("auto", "kk"):
        try:
            pos = nx.kamada_kawai_layout(G, scale=4.0)
        except Exception:
            pos = None
    if pos is None and layout in ("auto", "spring"):
        pos = nx.spring_layout(G, seed=42, k=2.0, iterations=300)
    if pos is None:
        pos = _layered_positions(g)
    for n in g.nodes:
        pos.setdefault(n.name, (0.0, 0.0))

    # Compute fan size to size nodes by hub-importance
    fan: Dict[str, int] = {}
    for e in g.edges:
        if e.kind == "synapse":
            fan[e.src] = fan.get(e.src, 0) + 1
            fan[e.tgt] = fan.get(e.tgt, 0) + 1

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#fafbfc")

    # ── Population nodes — colored by anatomical region ──
    pop_nodes = [n for n in g.nodes if n.kind == "pop"]
    for node in pop_nodes:
        x, y = pos[node.name]
        region = _REGION_OF.get(node.name, "world")
        color = _REGION_COLORS[region]
        # Size proportional to fan-degree (hubs ~ 2x leaves)
        f = fan.get(node.name, 0)
        size = 1800 + 220 * min(f, 10)
        ax.scatter([x], [y], s=size, c=color, edgecolors="#2c3e50",
                   linewidths=1.8, zorder=3, alpha=0.9)
        ax.text(x, y, node.name, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white", zorder=4)
        # Op annotation below the node (small pill)
        ax.annotate(node.op, xy=(x, y), xytext=(0, -22),
                    textcoords="offset points",
                    ha="center", va="top", fontsize=6.5,
                    style="italic", color="#2c3e50", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec=color, lw=1.0, alpha=0.9))

    # ── NT nodes — yellow diamonds ──
    nt_nodes = [n for n in g.nodes if n.kind == "nt"]
    for node in nt_nodes:
        x, y = pos[node.name]
        ax.scatter([x], [y], s=1200, c="#fff3a0", marker="D",
                   edgecolors="#b7950b", linewidths=1.5, zorder=3, alpha=0.92)
        ax.text(x, y, node.name, ha="center", va="center", fontsize=7,
                fontweight="bold", color="#5d4501", zorder=4)

    # ── Synapse edges — curved black arrows, width ∝ weight ──
    for e in g.edges:
        if e.kind != "synapse":
            continue
        if e.src not in pos or e.tgt not in pos:
            continue
        x0, y0 = pos[e.src]; x1, y1 = pos[e.tgt]
        lw = max(0.7, e.weight * 1.8)
        # Curved (rad>0) so parallel edges don't overlap; sign alternates
        rad = 0.12 if (hash(e.src + e.tgt) & 1) else -0.12
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=12,
            color="#2c3e50", lw=lw, alpha=0.7,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=22, shrinkB=22, zorder=2)
        ax.add_patch(arrow)

    # ── Modulation edges — dashed, color by effect ──
    for e in g.edges:
        if e.kind != "modulation":
            continue
        if e.src not in pos or e.tgt not in pos:
            continue
        x0, y0 = pos[e.src]; x1, y1 = pos[e.tgt]
        color = "#c0392b" if e.effect == "multiplicative" else "#2874a6"
        rad = 0.18 if (hash(e.src + e.tgt) & 1) else -0.18
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=10,
            color=color, lw=1.0, alpha=0.65,
            linestyle=(0, (3, 2)),
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=18, shrinkB=22, zorder=1)
        ax.add_patch(arrow)

    # ── Legends — two columns: regions + edge kinds ──
    region_handles = [
        mpatches.Patch(color=col, label=name)
        for name, col in _REGION_COLORS.items()
    ]
    edge_handles = [
        mpatches.Patch(color="#2c3e50", label="synapse"),
        mpatches.Patch(color="#c0392b", label="modulation (multiplicative)"),
        mpatches.Patch(color="#2874a6", label="modulation (additive)"),
        mpatches.Patch(color="#fff3a0", label="neurotransmitter"),
    ]
    leg1 = ax.legend(handles=region_handles, loc="upper left", fontsize=8,
                     title="regions", framealpha=0.92, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=edge_handles, loc="lower left", fontsize=8,
              title="edges", framealpha=0.92, title_fontsize=9)

    stats = g.stats()
    ax.set_title(
        f"Neural Flow Graph — {g.arch_name}\n"
        f"{stats['n_populations']} populations · {stats['n_synapses']} synapses · "
        f"{stats['n_modulations']} modulations · {stats['n_neurotransmitters']} NTs",
        fontsize=12, pad=20)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#fafbfc")
    plt.close()


# ── 4. Emit Python source ─────────────────────────────────────────────

def emit_python(g: NeuralFlowGraph, output_path: str) -> None:
    """Write a runnable .py that reconstructs the NFG via plain dicts.

    Output file usage:
        from <module> import NFG
        # NFG['nodes'] / NFG['edges'] / NFG['stats']
        # or wrap in networkx:
        import networkx as nx
        G = nx.DiGraph()
        for n in NFG['nodes']: G.add_node(n['name'], **n)
        for e in NFG['edges']: G.add_edge(e['src'], e['tgt'], **e)
    """
    import json
    nodes = [vars(n) for n in g.nodes]
    edges = [vars(e) for e in g.edges]
    body = json.dumps({
        "arch": g.arch_name,
        "stats": g.stats(),
        "nodes": nodes,
        "edges": edges,
    }, indent=2, default=str)
    src = (
        f'# -*- coding: utf-8 -*-\n'
        f'"""Neural Flow Graph for the {g.arch_name} architecture.\n\n'
        f'Auto-generated by `brian compile nfg`. Load with:\n'
        f'    from {Path(output_path).stem} import NFG\n'
        f'    # NFG[\'nodes\'], NFG[\'edges\'], NFG[\'stats\']\n'
        f'"""\n'
        f'NFG = {body}\n'
    )
    Path(output_path).write_text(src, encoding="utf-8")
