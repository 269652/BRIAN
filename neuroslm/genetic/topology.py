# -*- coding: utf-8 -*-
"""Graph-theoretic (geometric) analysis of an NGL execution flow graph.

The execution profile (``profile.py``) is projected into a weighted directed
graph — nodes are ops (weighted by compute), edges carry information flow as
capacity — and analysed with standard graph theory:

- **betweenness centrality** → which nodes route the most paths (bottlenecks),
- **articulation points** → cut vertices whose removal disconnects the graph,
- **max-flow / min-cut** → the tightest information bottleneck (source→sink),
- **algebraic connectivity** (Fiedler value) → how well-integrated the graph is.

From that geometry ``propose_edits`` suggests structural changes — bypass a cut
vertex, parallelise a bottleneck, prune a low-flow/high-compute edge — i.e. the
"geometrically beneficial shapes" to search first. This is the high-leverage,
well-founded alternative to a literal fluid-flow simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import networkx as nx

from neuroslm.genetic.profile import ExecutionProfile


def to_networkx(profile: ExecutionProfile) -> nx.DiGraph:
    G = nx.DiGraph()
    producer: Dict[str, int] = {}
    for n in profile.nodes:
        G.add_node(n.index, op=n.op, compute=n.flops, flow=n.flow)
    for n in profile.nodes:
        for r in n.ins:
            if r in producer:
                # capacity = the flow carried on this register (producer's output)
                cap = max(profile.nodes[producer[r]].flow, 1e-6)
                G.add_edge(producer[r], n.index, capacity=cap, reg=r)
        producer[n.out_reg] = n.index
    return G


@dataclass
class TopologyReport:
    betweenness: Dict[int, float] = field(default_factory=dict)
    articulation_points: List[int] = field(default_factory=list)
    algebraic_connectivity: float = 0.0
    min_cut_value: float = 0.0
    min_cut_edges: List[tuple] = field(default_factory=list)
    bottleneck_nodes: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "betweenness": {int(k): float(v) for k, v in self.betweenness.items()},
            "articulation_points": [int(x) for x in self.articulation_points],
            "algebraic_connectivity": float(self.algebraic_connectivity),
            "min_cut_value": float(self.min_cut_value),
            "min_cut_edges": [list(e) for e in self.min_cut_edges],
            "bottleneck_nodes": [int(x) for x in self.bottleneck_nodes],
        }


def _sources_and_sink(G: nx.DiGraph, profile: ExecutionProfile):
    sources = [n for n in G.nodes if G.in_degree(n) == 0]
    sink = None
    for n in profile.nodes:
        if n.out_reg == profile.out_reg:
            sink = n.index
    if sink is None and profile.nodes:
        sink = profile.nodes[-1].index
    return sources, sink


def analyze(profile: ExecutionProfile) -> TopologyReport:
    G = to_networkx(profile)
    rep = TopologyReport()
    if G.number_of_nodes() == 0:
        return rep

    try:
        rep.betweenness = nx.betweenness_centrality(G)
    except Exception:
        rep.betweenness = {n: 0.0 for n in G.nodes}

    UG = G.to_undirected()
    try:
        rep.articulation_points = sorted(nx.articulation_points(UG))
    except Exception:
        rep.articulation_points = []

    try:
        if UG.number_of_nodes() >= 2 and nx.is_connected(UG):
            rep.algebraic_connectivity = float(nx.algebraic_connectivity(UG))
    except Exception:
        rep.algebraic_connectivity = 0.0

    sources, sink = _sources_and_sink(G, profile)
    if sources and sink is not None:
        # super-source so multi-input graphs have a single max-flow source
        H = G.copy()
        super_src = -1
        H.add_node(super_src)
        for s in sources:
            if s != sink:
                H.add_edge(super_src, s, capacity=1e12)
        try:
            if super_src != sink and nx.has_path(H, super_src, sink):
                cut_val, (reach, _) = nx.minimum_cut(H, super_src, sink)
                rep.min_cut_value = float(cut_val)
                rep.min_cut_edges = [
                    (u, v) for u in reach for v in H.successors(u)
                    if v not in reach and u != super_src
                ]
        except Exception:
            rep.min_cut_value = 0.0

    # bottleneck = articulation points ∪ top-betweenness nodes
    top_bw = sorted(rep.betweenness, key=lambda k: -rep.betweenness[k])
    top_bw = [k for k in top_bw if rep.betweenness[k] > 0][:3]
    rep.bottleneck_nodes = sorted(set(rep.articulation_points) | set(top_bw))
    return rep


def propose_edits(profile: ExecutionProfile) -> List[dict]:
    """Suggest structural edits from the graph geometry."""
    rep = analyze(profile)
    G = to_networkx(profile)
    by_index = {n.index: n for n in profile.nodes}
    edits: List[dict] = []

    for node in rep.bottleneck_nodes:
        n = by_index.get(node)
        if n is None:
            continue
        kind = "parallelize" if n.flops >= profile.total_flops() / max(1, len(profile.nodes)) else "bypass"
        edits.append({
            "kind": kind,
            "target": node,
            "op": n.op,
            "reason": (f"node {node} ({n.op}) is a bottleneck "
                       f"(betweenness={rep.betweenness.get(node, 0):.2f}, "
                       f"flow={n.flow:.2f}, flops={n.flops:.0f}); "
                       f"{'add a parallel branch' if kind == 'parallelize' else 'route a skip path around it'}"),
        })

    # prune candidates: high compute, low flow (expensive but carries little info)
    avg_flow = sum(n.flow for n in profile.nodes) / max(1, len(profile.nodes))
    for n in profile.nodes:
        if n.flops > profile.total_flops() / max(1, len(profile.nodes)) and n.flow < 0.25 * avg_flow:
            edits.append({
                "kind": "prune",
                "target": n.index,
                "op": n.op,
                "reason": (f"node {n.index} ({n.op}) is heavy compute "
                           f"(flops={n.flops:.0f}) but low flow ({n.flow:.2f}) — "
                           f"a candidate to remove/replace"),
            })
    return edits
