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

def render_nfg(g: NeuralFlowGraph, output_path: str,
               figsize: Tuple[int, int] = (16, 11)) -> None:
    """Render via matplotlib + networkx. Layered layout, distinct edge
    styles for synapses vs modulations, op labels on nodes.
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    G = nx.DiGraph()
    for n in g.nodes:
        G.add_node(n.name, kind=n.kind, op=n.op)
    for e in g.edges:
        G.add_edge(e.src, e.tgt, kind=e.kind, weight=e.weight,
                   nt=e.nt, effect=e.effect)

    pos = _layered_positions(g)
    # Any node missing from pos (orphan) → put at origin
    for n in g.nodes:
        pos.setdefault(n.name, (0.0, 0.0))

    fig, ax = plt.subplots(figsize=figsize)

    # Population nodes — coloured by op
    op_colors = {
        "relu": "#a8d5e2", "softmax": "#f5cba7", "gated": "#d2b4de",
        "tanh": "#a9dfbf", "ode": "#f5b7b1", "softmax_relu": "#f8c471",
        "linear": "#d5dbdb",
    }
    pop_nodes = [n for n in g.nodes if n.kind == "pop"]
    for node in pop_nodes:
        x, y = pos[node.name]
        color = op_colors.get(node.op, "#bdc3c7")
        ax.scatter([x], [y], s=2400, c=color, edgecolors="#34495e",
                   linewidths=1.5, zorder=3)
        ax.text(x, y, node.name, ha="center", va="center", fontsize=8,
                fontweight="bold", zorder=4)
        ax.text(x, y - 0.35, node.op, ha="center", va="top", fontsize=6,
                style="italic", color="#566573", zorder=4)

    # NT nodes — yellow ellipses with NT name
    nt_nodes = [n for n in g.nodes if n.kind == "nt"]
    for node in nt_nodes:
        x, y = pos[node.name]
        ax.scatter([x], [y], s=1500, c="#fcf3cf", marker="D",
                   edgecolors="#b7950b", linewidths=1.2, zorder=3)
        ax.text(x, y, node.name, ha="center", va="center", fontsize=7,
                fontweight="bold", color="#7d6608", zorder=4)

    # Synapse edges — solid black, line width = weight
    for e in g.edges:
        if e.kind != "synapse":
            continue
        if e.src not in pos or e.tgt not in pos:
            continue
        x0, y0 = pos[e.src]; x1, y1 = pos[e.tgt]
        lw = max(0.5, e.weight * 1.4)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=lw,
                                    color="#2c3e50", shrinkA=22, shrinkB=22,
                                    alpha=0.75), zorder=2)
        # Edge label only when weight notably differs from 1.0 to reduce clutter
        if abs(e.weight - 1.0) > 0.05:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my, f"{e.weight:.1f}", fontsize=6,
                    color="#2c3e50",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.85), zorder=3)

    # Modulation edges — dashed coloured by effect
    for e in g.edges:
        if e.kind != "modulation":
            continue
        if e.src not in pos or e.tgt not in pos:
            continue
        x0, y0 = pos[e.src]; x1, y1 = pos[e.tgt]
        color = "#c0392b" if e.effect == "multiplicative" else "#2874a6"
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=0.9, color=color,
                                    linestyle="dashed", shrinkA=18, shrinkB=22,
                                    alpha=0.7), zorder=1)

    # Legend
    legend_handles = [
        mpatches.Patch(color="#a8d5e2", label="ReLU"),
        mpatches.Patch(color="#f5cba7", label="softmax"),
        mpatches.Patch(color="#d2b4de", label="gated"),
        mpatches.Patch(color="#a9dfbf", label="tanh"),
        mpatches.Patch(color="#f5b7b1", label="ODE"),
        mpatches.Patch(color="#fcf3cf", label="neurotransmitter"),
        mpatches.Patch(color="#c0392b", label="mod (multiplicative)"),
        mpatches.Patch(color="#2874a6", label="mod (additive)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=7,
              ncol=2, framealpha=0.9)

    stats = g.stats()
    ax.set_title(f"Neural Flow Graph — {g.arch_name}  "
                 f"({stats['n_populations']} pops, {stats['n_synapses']} syn, "
                 f"{stats['n_modulations']} mod, {stats['n_neurotransmitters']} NT)",
                 fontsize=10)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
