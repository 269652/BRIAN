"""Relational Memory Graph — multidimensional associative memory.

Replaces flat episodic buffers with a graph that encodes:
  - **Associative** edges (semantic similarity between memory embeddings)
  - **Causal** edges (event A preceded/caused event B)
  - **Temporal** edges (time proximity between episodes)
  - **Pattern** edges (recurring co-activation patterns)
  - **Affective** edges (shared NT state / emotional valence)

Each node stores:
  - content_vec:  d_sem-dimensional semantic embedding (from LanguageCortex)
  - nt_state:     neurotransmitter snapshot at encoding time
  - valence:      emotional valence scalar [-1, +1]
  - salience:     importance/reward score (from mesolimbic tagging)
  - timestamp:    when the episode was encoded
  - content:      raw text (for introspection / narrative generation)
  - tags:         free-form metadata

The graph supports efficient retrieval by the hippocampus module:
  - Cosine-similarity nearest-neighbor lookup (associative)
  - Graph walk along causal/temporal chains
  - Affective-state-conditioned filtering (retrieve memories matching current mood)
  - Spreading activation from a query node to related nodes

This module is purely NumPy/networkx (no torch dependency) so it can run
on CPU without device management concerns.
"""
from __future__ import annotations
import time
import numpy as np
import networkx as nx
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class EdgeType(str, Enum):
    ASSOCIATIVE = "associative"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    PATTERN = "pattern"
    AFFECTIVE = "affective"


@dataclass
class MemoryNode:
    """Data stored at each node in the relational graph."""
    node_id: int
    content: str
    content_vec: np.ndarray           # (d_sem,)
    nt_state: np.ndarray              # (n_nt,) — NT levels at encoding time
    valence: float = 0.0             # emotional valence [-1, +1]
    salience: float = 0.0            # importance / reward score
    timestamp: float = 0.0
    tags: list = field(default_factory=list)
    access_count: int = 0            # retrieval frequency (for consolidation)
    decay: float = 1.0               # memory strength (decays over time)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class RelationalMemoryGraph:
    """Thread-safe multidimensional relational memory graph."""

    def __init__(self, max_nodes: int = 8192,
                 assoc_threshold: float = 0.6,
                 temporal_window: float = 60.0,
                 decay_rate: float = 0.999):
        self.max_nodes = max_nodes
        self.assoc_threshold = assoc_threshold
        self.temporal_window = temporal_window
        self.decay_rate = decay_rate
        self.graph = nx.DiGraph()  # directed for causal edges
        self.lock = threading.Lock()
        self._next_id = 0
        # Fast lookup: numpy array of all content vecs for batch cosine sim
        self._vec_index: list[tuple[int, np.ndarray]] = []

    @property
    def size(self) -> int:
        return self.graph.number_of_nodes()

    def _rebuild_vec_index(self):
        """Rebuild the flat vector index from graph nodes."""
        self._vec_index = [
            (nid, self.graph.nodes[nid]["data"].content_vec)
            for nid in self.graph.nodes
        ]

    # ------------------------------------------------------------------
    # Encoding — store a new memory
    # ------------------------------------------------------------------
    def encode(self, content: str, content_vec: np.ndarray,
               nt_state: np.ndarray, valence: float = 0.0,
               salience: float = 0.0, tags: list | None = None,
               causal_parent: int | None = None) -> int:
        """Add a new memory node and automatically wire edges."""
        with self.lock:
            node_id = self._next_id
            self._next_id += 1

            node = MemoryNode(
                node_id=node_id,
                content=content,
                content_vec=np.asarray(content_vec, dtype=np.float32),
                nt_state=np.asarray(nt_state, dtype=np.float32),
                valence=float(valence),
                salience=float(salience),
                timestamp=time.time(),
                tags=tags or [],
            )
            self.graph.add_node(node_id, data=node)

            # --- Wire edges to existing nodes ---
            for other_id, other_vec in self._vec_index:
                sim = _cosine_sim(node.content_vec, other_vec)
                other_node: MemoryNode = self.graph.nodes[other_id]["data"]

                # Associative edge (bidirectional)
                if sim > self.assoc_threshold:
                    self.graph.add_edge(node_id, other_id,
                                        etype=EdgeType.ASSOCIATIVE,
                                        weight=sim)
                    self.graph.add_edge(other_id, node_id,
                                        etype=EdgeType.ASSOCIATIVE,
                                        weight=sim)

                # Temporal edge (if close in time)
                dt = abs(node.timestamp - other_node.timestamp)
                if dt < self.temporal_window:
                    temporal_w = 1.0 - dt / self.temporal_window
                    self.graph.add_edge(node_id, other_id,
                                        etype=EdgeType.TEMPORAL,
                                        weight=temporal_w)

                # Affective edge (similar NT state / valence)
                nt_sim = _cosine_sim(node.nt_state, other_node.nt_state)
                val_sim = 1.0 - abs(node.valence - other_node.valence) / 2.0
                affect_w = 0.5 * nt_sim + 0.5 * val_sim
                if affect_w > 0.6:
                    self.graph.add_edge(node_id, other_id,
                                        etype=EdgeType.AFFECTIVE,
                                        weight=affect_w)
                    self.graph.add_edge(other_id, node_id,
                                        etype=EdgeType.AFFECTIVE,
                                        weight=affect_w)

            # Causal edge (if a parent event is specified)
            if causal_parent is not None and causal_parent in self.graph:
                self.graph.add_edge(causal_parent, node_id,
                                    etype=EdgeType.CAUSAL,
                                    weight=1.0)

            # Update vector index
            self._vec_index.append((node_id, node.content_vec))

            # Prune if over capacity (remove lowest-salience, most-decayed)
            if self.graph.number_of_nodes() > self.max_nodes:
                self._prune_weakest()

            return node_id

    def _prune_weakest(self):
        """Remove the node with lowest salience × decay."""
        worst_id = None
        worst_score = float("inf")
        for nid in self.graph.nodes:
            n: MemoryNode = self.graph.nodes[nid]["data"]
            score = n.salience * n.decay
            if score < worst_score:
                worst_score = score
                worst_id = nid
        if worst_id is not None:
            self.graph.remove_node(worst_id)
            self._vec_index = [(i, v) for i, v in self._vec_index if i != worst_id]

    # ------------------------------------------------------------------
    # Retrieval — query memories by content, affect, or spreading activation
    # ------------------------------------------------------------------
    def query_associative(self, query_vec: np.ndarray, topk: int = 5,
                          nt_filter: np.ndarray | None = None,
                          valence_range: tuple[float, float] | None = None
                          ) -> list[MemoryNode]:
        """Return top-k most semantically similar nodes, optionally filtered."""
        query_vec = np.asarray(query_vec, dtype=np.float32)
        scored = []
        for nid, vec in self._vec_index:
            sim = _cosine_sim(query_vec, vec)
            node: MemoryNode = self.graph.nodes[nid]["data"]
            # Apply filters
            if nt_filter is not None:
                nt_sim = _cosine_sim(nt_filter, node.nt_state)
                if nt_sim < 0.3:
                    continue
            if valence_range is not None:
                if not (valence_range[0] <= node.valence <= valence_range[1]):
                    continue
            # Boost by salience and decay
            final_score = sim * (0.5 + 0.5 * node.salience) * node.decay
            scored.append((final_score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Increment access count for retrieved nodes
        results = []
        for _, node in scored[:topk]:
            node.access_count += 1
            results.append(node)
        return results

    def spreading_activation(self, seed_id: int, hops: int = 2,
                             topk: int = 8,
                             edge_types: set | None = None
                             ) -> list[MemoryNode]:
        """Walk the graph from a seed node via specified edge types.
        Returns nodes reachable within `hops`, ranked by accumulated weight."""
        if seed_id not in self.graph:
            return []
        if edge_types is None:
            edge_types = {EdgeType.ASSOCIATIVE, EdgeType.CAUSAL,
                          EdgeType.TEMPORAL, EdgeType.AFFECTIVE}

        visited: dict[int, float] = {seed_id: 1.0}
        frontier = {seed_id}
        for _ in range(hops):
            next_frontier = set()
            for nid in frontier:
                for _, neighbor, edata in self.graph.edges(nid, data=True):
                    if edata.get("etype") not in edge_types:
                        continue
                    w = edata.get("weight", 0.5) * visited[nid]
                    if neighbor not in visited or visited[neighbor] < w:
                        visited[neighbor] = w
                        next_frontier.add(neighbor)
            frontier = next_frontier

        # Remove seed, sort by activation, return top-k
        visited.pop(seed_id, None)
        ranked = sorted(visited.items(), key=lambda x: x[1], reverse=True)
        results = []
        for nid, _ in ranked[:topk]:
            node: MemoryNode = self.graph.nodes[nid]["data"]
            node.access_count += 1
            results.append(node)
        return results

    def query_causal_chain(self, node_id: int, direction: str = "forward",
                           max_depth: int = 5) -> list[MemoryNode]:
        """Follow causal edges forward or backward from a node."""
        if node_id not in self.graph:
            return []
        chain = []
        current = node_id
        for _ in range(max_depth):
            if direction == "forward":
                successors = [
                    n for _, n, d in self.graph.edges(current, data=True)
                    if d.get("etype") == EdgeType.CAUSAL
                ]
            else:
                successors = [
                    n for n, _, d in self.graph.in_edges(current, data=True)
                    if d.get("etype") == EdgeType.CAUSAL
                ]
            if not successors:
                break
            current = successors[0]
            chain.append(self.graph.nodes[current]["data"])
        return chain

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def decay_all(self, factor: float | None = None):
        """Apply time-based decay to all memories."""
        f = factor or self.decay_rate
        for nid in self.graph.nodes:
            self.graph.nodes[nid]["data"].decay *= f

    def tag_salience(self, node_id: int, reward: float, insight: str | None = None):
        """Update a node's salience (mesolimbic tagging)."""
        if node_id in self.graph:
            node: MemoryNode = self.graph.nodes[node_id]["data"]
            # EMA update of salience
            node.salience = 0.7 * node.salience + 0.3 * reward
            if insight:
                node.tags.append(f"insight:{insight}")

    def add_pattern_edge(self, id_a: int, id_b: int, pattern_strength: float):
        """Add a pattern edge (detected co-activation)."""
        if id_a in self.graph and id_b in self.graph:
            self.graph.add_edge(id_a, id_b,
                                etype=EdgeType.PATTERN,
                                weight=pattern_strength)
            self.graph.add_edge(id_b, id_a,
                                etype=EdgeType.PATTERN,
                                weight=pattern_strength)

    def get_node(self, node_id: int) -> MemoryNode | None:
        if node_id in self.graph:
            return self.graph.nodes[node_id]["data"]
        return None

    def all_nodes(self) -> list[MemoryNode]:
        return [self.graph.nodes[nid]["data"] for nid in self.graph.nodes]

    def stats(self) -> dict:
        return {
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "edge_types": {
                et.value: sum(1 for _, _, d in self.graph.edges(data=True)
                              if d.get("etype") == et)
                for et in EdgeType
            },
        }
