# -*- coding: utf-8 -*-
"""Hypershape compiler + graph-theoretic analysis (Phases N10/N11).

N10 — `compile_hypershape(model)` lowers a DSL model into a typed
multigraph: nodes are ops/params, edges are shaped tensor flows, regions
label subsystems (one per transformer block, plus embed/head). A parallel
*adjoint* graph (reversed edges) represents gradient flow. Per the design
principle in docs/dsl_nn_language.md, the model is born as this graph, so
lowering is a structural read, not a re-derivation.

N11 — analysis over that graph: Fiedler algebraic connectivity, spectral
gap, degree centrality, articulation points (computation bottlenecks),
and a Φ-bipartition (integration estimate via normalized min-cut). These
are the levers for optimizing intelligence density / Φ / EI by inspecting
the model's mathematical structure.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional

import torch


# ── Graph types ────────────────────────────────────────────────────────

@dataclass
class HyperNode:
    id: str
    kind: str                 # embedding | attention | mlp | norm | head | block
    region: str               # subsystem label (e.g. "block0", "io")
    shape: Tuple = ()


@dataclass
class HyperEdge:
    src: str
    dst: str
    kind: str = "forward"     # forward | residual | adjoint
    shape: Tuple = ()


@dataclass
class HyperShape:
    nodes: List[HyperNode] = field(default_factory=list)
    edges: List[HyperEdge] = field(default_factory=list)

    # ── helpers ──
    def node_ids(self) -> List[str]:
        return [n.id for n in self.nodes]

    def _adjacency_sets(self, directed: bool = False) -> Dict[str, Set[str]]:
        adj: Dict[str, Set[str]] = {n.id: set() for n in self.nodes}
        for e in self.edges:
            adj[e.src].add(e.dst)
            if not directed:
                adj[e.dst].add(e.src)
        return adj

    def has_path(self, src: str, dst: str) -> bool:
        """Directed reachability src → dst."""
        adj = self._adjacency_sets(directed=True)
        seen, stack = set(), [src]
        while stack:
            u = stack.pop()
            if u == dst:
                return True
            if u in seen:
                continue
            seen.add(u)
            stack.extend(adj.get(u, ()))
        return False

    def adjoint(self) -> "HyperShape":
        """Gradient graph: every forward edge reversed (reverse-mode AD)."""
        rev = [HyperEdge(e.dst, e.src, "adjoint", e.shape) for e in self.edges]
        return HyperShape(nodes=list(self.nodes), edges=rev)

    def adjacency_matrix(self) -> torch.Tensor:
        ids = self.node_ids()
        idx = {nid: i for i, nid in enumerate(ids)}
        n = len(ids)
        A = torch.zeros(n, n)
        for e in self.edges:
            i, j = idx[e.src], idx[e.dst]
            A[i, j] = 1.0
            A[j, i] = 1.0   # undirected for spectral analysis
        return A


# ── N10: compile a DSL LM into a HyperShape ────────────────────────────

def compile_hypershape(model) -> HyperShape:
    """Lower a DSLLanguageModel to its typed graph.

    Structure: embed → [block0 → block1 → …] → norm_f → head, with a
    residual self-edge on each block (the residual stream) and one region
    per block. Generalises to any module exposing `.blocks` + `.embed` +
    `.lm_head`.
    """
    hs = HyperShape()
    hs.nodes.append(HyperNode("embed", "embedding", "io",
                              tuple(model.embed.shape)))

    n_blocks = len(getattr(model, "blocks", []))
    prev = "embed"
    for i in range(n_blocks):
        # Represent each block as an attention node + mlp node (the two
        # residual sub-layers), in its own region.
        region = f"block{i}"
        attn_id, mlp_id = f"{region}.attn", f"{region}.mlp"
        hs.nodes.append(HyperNode(attn_id, "attention", region))
        hs.nodes.append(HyperNode(mlp_id, "mlp", region))
        # forward path through the block
        hs.edges.append(HyperEdge(prev, attn_id, "forward"))
        hs.edges.append(HyperEdge(attn_id, mlp_id, "forward"))
        # residual skip edges (the residual stream bypasses each sub-layer)
        hs.edges.append(HyperEdge(prev, mlp_id, "residual"))
        prev = mlp_id

    hs.nodes.append(HyperNode("norm_f", "norm", "io",
                              tuple(model.gamma_f.shape)))
    hs.nodes.append(HyperNode("head", "head", "io",
                              tuple(model.lm_head.shape)))
    hs.edges.append(HyperEdge(prev, "norm_f", "forward"))
    hs.edges.append(HyperEdge("norm_f", "head", "forward"))
    return hs


# ── N11: graph-theoretic analysis ──────────────────────────────────────

def _normalized_laplacian(hs: HyperShape) -> torch.Tensor:
    A = hs.adjacency_matrix()
    deg = A.sum(dim=1)
    d_inv_sqrt = torch.diag((deg + 1e-8).rsqrt())
    n = A.shape[0]
    return torch.eye(n) - d_inv_sqrt @ A @ d_inv_sqrt


def fiedler_value(hs: HyperShape) -> float:
    """Second-smallest eigenvalue of the normalized Laplacian — algebraic
    connectivity. Higher = harder to bipartition = more integrated."""
    if len(hs.nodes) < 2:
        return 0.0
    eig = torch.linalg.eigvalsh(_normalized_laplacian(hs))
    return float(eig[1].clamp(min=0.0))


def spectral_gap(hs: HyperShape) -> float:
    """Gap between the two largest Laplacian eigenvalues — a stability /
    expander-quality indicator of the computation graph."""
    if len(hs.nodes) < 2:
        return 0.0
    eig = torch.linalg.eigvalsh(_normalized_laplacian(hs))
    return float((eig[-1] - eig[-2]).clamp(min=0.0))


def degree_centrality(hs: HyperShape) -> Dict[str, float]:
    """Normalized degree per node (fraction of other nodes it touches)."""
    adj = hs._adjacency_sets(directed=False)
    n = len(hs.nodes)
    denom = max(1, n - 1)
    return {nid: len(neigh) / denom for nid, neigh in adj.items()}


def articulation_points(hs: HyperShape) -> List[str]:
    """Cut vertices: nodes whose removal disconnects the (undirected)
    graph — the computation bottlenecks of the hypershape."""
    adj = hs._adjacency_sets(directed=False)
    ids = hs.node_ids()
    visited: Set[str] = set()
    disc: Dict[str, int] = {}
    low: Dict[str, int] = {}
    parent: Dict[str, Optional[str]] = {}
    aps: Set[str] = set()
    timer = [0]

    def dfs(u: str):
        visited.add(u)
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        children = 0
        for v in adj[u]:
            if v not in visited:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent.get(u) is None and children > 1:
                    aps.add(u)
                if parent.get(u) is not None and low[v] >= disc[u]:
                    aps.add(u)
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    import sys
    _old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(_old, len(ids) * 4 + 100))
    try:
        for nid in ids:
            if nid not in visited:
                parent[nid] = None
                dfs(nid)
    finally:
        sys.setrecursionlimit(_old)
    return sorted(aps)


def phi_bipartition(hs: HyperShape) -> Tuple[float, List[str], List[str]]:
    """Integration estimate via a spectral (Fiedler-vector) bipartition.

    Splits nodes by the sign of the Fiedler eigenvector, then scores
    integration as (edges crossing the cut) / (min part size) — high when
    the two halves are densely interconnected (hard to separate = high Φ).
    """
    ids = hs.node_ids()
    if len(ids) < 2:
        return 0.0, ids, []
    L = _normalized_laplacian(hs)
    eigvals, eigvecs = torch.linalg.eigh(L)
    fiedler_vec = eigvecs[:, 1]
    part_a = [ids[i] for i in range(len(ids)) if fiedler_vec[i] >= 0]
    part_b = [ids[i] for i in range(len(ids)) if fiedler_vec[i] < 0]
    if not part_a or not part_b:
        # Degenerate split — fall back to halving.
        mid = len(ids) // 2
        part_a, part_b = ids[:mid], ids[mid:]
    set_a = set(part_a)
    crossing = sum(1 for e in hs.edges
                   if (e.src in set_a) != (e.dst in set_a))
    score = crossing / max(1, min(len(part_a), len(part_b)))
    return float(score), part_a, part_b


def analyze(hs: HyperShape) -> Dict:
    """One-shot inspection summary of the hypershape."""
    score, _, _ = phi_bipartition(hs)
    return {
        "n_nodes": len(hs.nodes),
        "n_edges": len(hs.edges),
        "fiedler": fiedler_value(hs),
        "spectral_gap": spectral_gap(hs),
        "n_articulation_points": len(articulation_points(hs)),
        "phi_bipartition": score,
        "regions": sorted({n.region for n in hs.nodes}),
    }
