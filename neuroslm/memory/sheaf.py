"""Contextual Sheaf over the relational hypergraph.

The sheaf F assigns to each node U_i a local section (a partial belief about
the world from that node's perspective) and to each edge a restriction map
between adjacent sections. A *global section* is a consistent choice of
local sections that agrees across every restriction — i.e. a coherent
interpretation across all context patches.

We approximate sheaf cohomology in the following tractable way:

  • Each node carries an embedding ∈ ℝ^d (its local section).
  • Each edge carries a restriction map R_{ij} : V_i → V_j realised as a
    learned (or identity-initialised) linear map.
  • Inconsistency on edge (i,j) is ‖R_{ij} · v_i − v_j‖₂.
  • The Čech 1-cochain c_{ij} = R_{ij}·v_i − v_j defines a 1-cochain;
    the 1st cohomology H¹(F) is non-zero iff there exists a 1-cocycle
    not in the image of the 0-coboundary δ⁰. We test the practical
    proxy: the residual after orthogonal projection of c onto im(δ⁰)
    is non-zero. If it exceeds a threshold, we declare a contradiction.
  • A *global section* retrieval is the minimum-residual joint assignment
    of node values consistent with all restrictions; we obtain it via a
    few damped-Jacobi iterations averaging each node against its
    edge-projected neighbours.

This module is pure-Python/NumPy. It runs in the no-grad consolidation
path, not in the gradient graph.
"""
from __future__ import annotations
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Edge typing
# ─────────────────────────────────────────────────────────────────────────────

EDGE_CAUSAL    = "causal"     # actual-causation: A → B with strength α
EDGE_TEMPORAL  = "temporal"   # DNC link matrix L: A immediately precedes B
EDGE_QUALIA    = "qualia"     # Fisher-information-weighted feeling similarity
EDGE_SUPERSEDES = "supersedes"  # contradiction-resolution edge: A overrides B


@dataclass
class SheafSection:
    """A local section over a sub-collection of nodes.

    Practically: a set of node ids and the local belief vectors attached to
    each of them. The section is *consistent* iff every edge in the induced
    subgraph satisfies its restriction map within tolerance.
    """
    node_ids: List[str]
    values:   Dict[str, np.ndarray]      # node_id → local section vector
    edge_ids: List[str] = field(default_factory=list)
    consistency: float = 1.0             # 0 (contradiction) … 1 (perfect)
    h1_residual: float = 0.0             # H¹ proxy: residual cocycle norm
    timestamp:   float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# Sheaf consistency checker
# ─────────────────────────────────────────────────────────────────────────────

class SheafConsistencyChecker:
    """Detects contradictions in the relational hypergraph via H¹(F).

    Stores per-edge restriction maps R_{ij} (default = identity). For each
    edge we maintain the residual ‖R·v_i − v_j‖. The H¹ proxy aggregates
    edge residuals after projecting onto the coboundary image — large
    residuals that cannot be explained by a 0-coboundary correction
    indicate a true topological obstruction (contradiction).

    Two-node usage (the canonical "Alice likes coffee" / "Alice hates
    coffee" case): if the restriction map between two memories about
    the same entity drives one toward +preference and the other toward
    −preference, the 1-cocycle has no antiderivative, H¹ ≠ 0, and we
    emit a SUPERSEDES edge.
    """

    def __init__(self, d_emb: int, eps: float = 1e-6,
                 contradiction_threshold: float = 0.7):
        self.d_emb = d_emb
        self.eps = eps
        self.contradiction_threshold = contradiction_threshold
        # Restriction maps keyed by edge id (identity by default)
        self.R: Dict[str, np.ndarray] = {}
        # Inverse-variance weights per edge (sharper edges count more)
        self.W: Dict[str, float] = {}

    # ── restriction-map management ───────────────────────────────────────────

    def set_restriction(self, edge_id: str, mat: Optional[np.ndarray] = None,
                        weight: float = 1.0) -> None:
        """Register the linear restriction map for an edge. If None, identity."""
        if mat is None:
            self.R[edge_id] = np.eye(self.d_emb, dtype=np.float32)
        else:
            m = np.asarray(mat, dtype=np.float32)
            assert m.shape == (self.d_emb, self.d_emb), \
                f"restriction map must be ({self.d_emb}, {self.d_emb})"
            self.R[edge_id] = m
        self.W[edge_id] = float(weight)

    # ── cochain & residual ───────────────────────────────────────────────────

    def edge_residual(self, edge_id: str,
                      v_src: np.ndarray, v_dst: np.ndarray) -> float:
        """1-cochain entry: ‖R·v_src − v_dst‖₂ normalised by ‖v_dst‖."""
        R = self.R.get(edge_id)
        if R is None:
            R = np.eye(self.d_emb, dtype=np.float32)
        d = min(v_src.size, v_dst.size, self.d_emb)
        a = (R[:d, :d] @ v_src[:d]) - v_dst[:d]
        n_dst = float(np.linalg.norm(v_dst[:d])) + self.eps
        return float(np.linalg.norm(a) / n_dst)

    def h1_proxy(self, edges: List[Tuple[str, str, str]],
                 values: Dict[str, np.ndarray]) -> float:
        """Aggregate H¹ residual over a sub-collection of edges.

        edges: list of (edge_id, src_node_id, dst_node_id).
        values: node_id → vector.

        We collect the 1-cochain c_{ij} per edge, then project onto the
        coboundary image (i.e. the residual after subtracting the
        best-fit 0-coboundary correction). Returns the L2 norm of the
        non-coboundary part, normalised by the number of edges.
        """
        if not edges:
            return 0.0

        # Build the cochain vector (flattened over edges)
        cochains: List[np.ndarray] = []
        weights: List[float]       = []
        # Coboundary basis: for each unique node, a vector that adds δ to
        # source residuals and subtracts δ from dest residuals.
        node_idx: Dict[str, int] = {}
        for (_, s, d) in edges:
            if s not in node_idx: node_idx[s] = len(node_idx)
            if d not in node_idx: node_idx[d] = len(node_idx)

        # Per-edge cochain entries
        for eid, s, d in edges:
            vs, vd = values.get(s), values.get(d)
            if vs is None or vd is None:
                continue
            R = self.R.get(eid)
            if R is None:
                R = np.eye(self.d_emb, dtype=np.float32)
            dd = min(vs.size, vd.size, self.d_emb)
            cochain = (R[:dd, :dd] @ vs[:dd]) - vd[:dd]
            cochains.append(cochain)
            weights.append(self.W.get(eid, 1.0))

        if not cochains:
            return 0.0

        c_stack = np.stack(cochains, axis=0)              # (n_edges, d)
        w = np.asarray(weights, dtype=np.float32)[:, None]
        c_w = c_stack * np.sqrt(w)

        # Coboundary subspace: for each node m, a basis vector with +1 in
        # rows where the node is the source and −1 in rows where it's
        # the dest. We project c onto orthogonal complement of this
        # subspace; the residual is the H¹ proxy.
        n_e = len(cochains)
        n_n = len(node_idx)
        if n_n < 2:
            # Single node — no coboundary correction possible
            return float(np.linalg.norm(c_w) / (np.sqrt(n_e) + self.eps))

        # Build incidence matrix B ∈ ℝ^{n_edges × n_nodes}
        B = np.zeros((n_e, n_n), dtype=np.float32)
        for i, (eid, s, dd) in enumerate(edges):
            if eid not in self.R and not values.get(s) is None and not values.get(dd) is None:
                pass
            B[i, node_idx[s]] += 1.0
            B[i, node_idx[dd]] -= 1.0

        # Project c onto col(B) along each coordinate: least-squares for
        # δ such that B·δ ≈ c[:, k] for each k.
        # Residual r = c - B·δ.
        try:
            delta, *_ = np.linalg.lstsq(B, c_stack, rcond=None)
            residual = c_stack - B @ delta
        except np.linalg.LinAlgError:
            residual = c_stack

        h1 = float(np.linalg.norm(residual) / (np.sqrt(n_e) + self.eps))
        return h1

    # ── global section retrieval ─────────────────────────────────────────────

    def global_section(self,
                       node_ids: List[str],
                       values: Dict[str, np.ndarray],
                       edges: List[Tuple[str, str, str]],
                       n_iters: int = 4,
                       damping: float = 0.5) -> SheafSection:
        """Compute the maximum-consistency global section over a node set.

        Iterative damped Jacobi: each node's value is updated toward the
        mean of restriction-projected neighbour values. Converges to a
        sheaf section that minimises Σ_e ‖R·v_src − v_dst‖².

        Returns a SheafSection with consistency = exp(−H¹) ∈ (0, 1].
        """
        # Initialise from given values, padded/truncated to d_emb
        u: Dict[str, np.ndarray] = {}
        for nid in node_ids:
            v = values.get(nid)
            if v is None:
                u[nid] = np.zeros(self.d_emb, dtype=np.float32)
            else:
                vv = np.asarray(v, dtype=np.float32).flatten()
                if vv.size >= self.d_emb:
                    u[nid] = vv[:self.d_emb].copy()
                else:
                    pad = np.zeros(self.d_emb, dtype=np.float32)
                    pad[:vv.size] = vv
                    u[nid] = pad

        # Index edges by node for fast neighbour lookup
        out_edges: Dict[str, List[Tuple[str, str]]] = {nid: [] for nid in node_ids}
        in_edges:  Dict[str, List[Tuple[str, str]]] = {nid: [] for nid in node_ids}
        for eid, s, d in edges:
            if s in out_edges and d in in_edges:
                out_edges[s].append((eid, d))
                in_edges[d].append((eid, s))

        for _ in range(n_iters):
            new_u: Dict[str, np.ndarray] = {nid: u[nid].copy() for nid in node_ids}
            for nid in node_ids:
                contribs: List[np.ndarray] = []
                wts:      List[float]      = []
                # incoming: dst = nid, want u[nid] ≈ R·u[src]
                for eid, src in in_edges[nid]:
                    R = self.R.get(eid, np.eye(self.d_emb, dtype=np.float32))
                    contribs.append(R @ u[src])
                    wts.append(self.W.get(eid, 1.0))
                # outgoing: src = nid, want R·u[nid] ≈ u[dst]
                # → u[nid] ≈ R⁻¹·u[dst]; for identity R this is u[dst]
                for eid, dst in out_edges[nid]:
                    R = self.R.get(eid, np.eye(self.d_emb, dtype=np.float32))
                    try:
                        Rinv = np.linalg.pinv(R)
                    except np.linalg.LinAlgError:
                        Rinv = R.T
                    contribs.append(Rinv @ u[dst])
                    wts.append(self.W.get(eid, 1.0))

                if contribs:
                    W = np.asarray(wts, dtype=np.float32)
                    stack = np.stack(contribs, axis=0)
                    avg = (stack * W[:, None]).sum(axis=0) / (W.sum() + self.eps)
                    new_u[nid] = (1 - damping) * u[nid] + damping * avg
            u = new_u

        # Two complementary measures:
        #   h1_strict = pure cohomological H¹ (coboundary-projected)
        #   h1_raw    = pairwise raw inconsistency on the *original* values
        # We expose the *raw* measure as `h1_residual` because that is
        # what contradiction detection cares about (identity-restriction
        # opposites cancel under strict H¹ — see pairwise_inconsistency()).
        h1_strict = self.h1_proxy(edges, u)
        # Original values dict (before damped-Jacobi smoothing) for the raw
        # signal — Jacobi would erase the contradiction we want to catch.
        original_values: Dict[str, np.ndarray] = {}
        for nid in node_ids:
            v = values.get(nid)
            if v is None:
                original_values[nid] = np.zeros(self.d_emb, dtype=np.float32)
            else:
                vv = np.asarray(v, dtype=np.float32).flatten()
                if vv.size >= self.d_emb:
                    original_values[nid] = vv[:self.d_emb].copy()
                else:
                    pad = np.zeros(self.d_emb, dtype=np.float32)
                    pad[:vv.size] = vv
                    original_values[nid] = pad
        h1_raw = self.pairwise_inconsistency(edges, original_values)
        return SheafSection(
            node_ids=list(node_ids),
            values=u,
            edge_ids=[eid for eid, _, _ in edges],
            consistency=float(math.exp(-h1_raw)),
            h1_residual=float(h1_raw),
        )

    # ── pairwise inconsistency (raw cochain norm without coboundary fix) ─────

    def pairwise_inconsistency(self,
                                 edges: List[Tuple[str, str, str]],
                                 values: Dict[str, np.ndarray]) -> float:
        """Weighted-average raw per-edge residual ‖R·v_src − v_dst‖.

        This is *not* strict H¹ — it skips the coboundary projection that
        the cohomological H¹ does. Strict H¹ vanishes for two nodes linked
        by identity restriction maps even when their values are opposite,
        because the resulting cochain is a 0-coboundary (resolvable by
        shifting both nodes equally). For practical contradiction
        detection ("Alice likes coffee" vs "Alice hates coffee") we want
        the raw signal: how far apart are these node values under the
        edge constraints? Use this signal to gate SUPERSEDES emission.
        """
        if not edges:
            return 0.0
        total_w = 0.0
        total_r = 0.0
        for eid, s, d in edges:
            vs, vd = values.get(s), values.get(d)
            if vs is None or vd is None:
                continue
            w = self.W.get(eid, 1.0)
            r = self.edge_residual(eid, vs, vd)
            total_w += w
            total_r += w * r
        if total_w < self.eps:
            return 0.0
        return float(total_r / total_w)

    # ── contradiction detection ──────────────────────────────────────────────

    def is_contradiction(self, section: SheafSection) -> bool:
        """True iff the inconsistency signal exceeds the threshold.

        Uses the pairwise inconsistency stored on the section (set during
        global_section) rather than strict H¹, for the reasons explained
        in pairwise_inconsistency().
        """
        return section.h1_residual >= self.contradiction_threshold

    def resolve_contradiction(self,
                              section: SheafSection,
                              node_timestamps: Dict[str, float]
                              ) -> Tuple[str, str]:
        """Resolve by SUPERSEDES: the newer node overrides the older one.

        Returns (newer_id, older_id) — caller is expected to add a
        SUPERSEDES edge newer → older in its graph.
        """
        if len(section.node_ids) < 2:
            return ("", "")
        sorted_ids = sorted(section.node_ids,
                            key=lambda nid: node_timestamps.get(nid, 0.0))
        older, newer = sorted_ids[0], sorted_ids[-1]
        return (newer, older)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-typed edge accessor — used by hypergraph.py to dispatch edge math
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CausalEdge:
    """E_c: IIT 4.0 actual-causation strength α ∈ [0, 1]."""
    src: str
    dst: str
    alpha: float                    # actual-causation strength
    confidence: float = 1.0

@dataclass
class TemporalEdge:
    """E_t: DNC temporal link — A immediately precedes B regardless of clock."""
    src: str
    dst: str
    weight: float = 1.0

@dataclass
class QualiaEdge:
    """E_q: Fisher-information-weighted feeling similarity."""
    src: str
    dst: str
    fisher_dist: float              # smaller = more similar feeling
    feeling_similarity: float = 1.0


def fisher_information_distance(p: np.ndarray, q: np.ndarray,
                                  eps: float = 1e-6) -> float:
    """Symmetric Fisher-information distance between two (qualia) prob
    distributions or non-negative feature vectors. Implementation:
    arccos of Hellinger-equivalent dot product after sqrt normalisation —
    coincides with the Fisher–Rao metric on the simplex.
    """
    p = np.maximum(np.asarray(p, dtype=np.float32).flatten(), 0.0) + eps
    q = np.maximum(np.asarray(q, dtype=np.float32).flatten(), 0.0) + eps
    d = min(p.size, q.size)
    p = p[:d] / p[:d].sum()
    q = q[:d] / q[:d].sum()
    inner = float(np.clip((np.sqrt(p) * np.sqrt(q)).sum(), -1.0, 1.0))
    return float(2.0 * math.acos(inner))
