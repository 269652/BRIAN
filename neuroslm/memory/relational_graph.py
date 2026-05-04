"""Relational Memory Graph — multidimensional hierarchical associative memory.

Each node stores a memory episode with rich metadata. Edges between nodes
encode multiple relationship types simultaneously:

  SEMANTIC    — cosine similarity of content embeddings
  TEMPORAL    — recency proximity (time since encoding)
  MOOD        — NT-state cosine + valence distance (emotional congruence)
  CAUSAL      — directional: event A preceded / caused event B
  PATTERN     — detected co-activation (consolidated recurring patterns)
  CONTEXTUAL  — shared context tags (topic, task, episode ID)
  REWARD      — mesolimbic DA tagging: high-value memories get strong reward edges

Nodes are tagged by value through the mesolimbic reward system:
  tag_reward(node_id, da_level, reward_signal, insight)
This updates the node's salience with a dopaminergic EMA and adds a reward edge
to the most recently rewarded node (DA-mediated credit assignment).

Retrieval:
  query_associative()    — semantic nearest-neighbour with NT mood filter
  query_temporal()       — recency-biased retrieval
  query_mood()           — NT state + valence congruent memories
  spreading_activation() — multi-hop graph walk from seed node
  query_causal_chain()   — follow causal edges forward / backward

Training insight storage:
  store_insight(content, vec, nt_state, surprise, comprehension, valence)
  — writes to graph only when surprise * comprehension * novelty > threshold,
    then wires edges to contextually related nodes. This converts learning
    steps into persistent semantic memories.

Consolidation (from memory/consolidation.py):
  The graph exposes add_abstract_node() for the consolidation module to
  insert cluster centroids as semantic / abstract knowledge nodes.
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
    SEMANTIC    = "semantic"
    TEMPORAL    = "temporal"
    MOOD        = "mood"
    CAUSAL      = "causal"
    PATTERN     = "pattern"
    CONTEXTUAL  = "contextual"
    REWARD      = "reward"


@dataclass
class MemoryNode:
    node_id:      int
    content:      str
    content_vec:  np.ndarray           # (d_sem,)
    nt_state:     np.ndarray           # (n_nt,) NT levels at encoding time
    valence:      float  = 0.0         # emotional valence [-1, +1]
    arousal:      float  = 0.0         # arousal level (NE proxy)
    salience:     float  = 0.0         # importance (mesolimbic DA tagging)
    reward_value: float  = 0.0         # cumulative DA reward tag
    timestamp:    float  = 0.0         # wall-clock time at encoding
    write_index:  int    = 0           # monotonic write counter (for recency)
    tags:         list   = field(default_factory=list)
    access_count: int    = 0
    decay:        float  = 1.0         # memory strength (fades over time)
    is_abstract:  bool   = False       # True = consolidated semantic node


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class RelationalMemoryGraph:
    """Thread-safe multidimensional relational memory graph."""

    # Edge wiring thresholds
    SEMANTIC_THRESHOLD   = 0.55   # cosine sim for semantic edge
    TEMPORAL_WINDOW      = 120.0  # seconds: temporal proximity window
    MOOD_THRESHOLD       = 0.55   # combined mood similarity threshold
    REWARD_THRESHOLD     = 0.4    # DA reward level to wire reward edge

    def __init__(self, max_nodes: int = 8192, decay_rate: float = 0.9995):
        self.max_nodes  = max_nodes
        self.decay_rate = decay_rate
        self.graph      = nx.DiGraph()
        self.lock       = threading.Lock()
        self._next_id   = 0
        self._write_ctr = 0
        self._last_reward_node: Optional[int] = None  # for reward edge chaining
        # Fast cosine-lookup index: list of (node_id, vec)
        self._vec_index: list[tuple[int, np.ndarray]] = []

    @property
    def size(self) -> int:
        return self.graph.number_of_nodes()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, content: str,
               content_vec: np.ndarray,
               nt_state: np.ndarray,
               valence: float = 0.0,
               arousal: float = 0.0,
               salience: float = 0.0,
               reward: float = 0.0,
               tags: list | None = None,
               causal_parent: int | None = None,
               context_tags: list | None = None) -> int:
        """Add a new memory node and automatically wire all edge types."""
        with self.lock:
            nid = self._next_id
            self._next_id  += 1
            self._write_ctr += 1

            node = MemoryNode(
                node_id     = nid,
                content     = content,
                content_vec = np.asarray(content_vec, dtype=np.float32).flatten(),
                nt_state    = np.asarray(nt_state,    dtype=np.float32).flatten(),
                valence     = float(valence),
                arousal     = float(arousal),
                salience    = float(salience),
                reward_value= float(reward),
                timestamp   = time.time(),
                write_index = self._write_ctr,
                tags        = list(tags or []),
            )
            self.graph.add_node(nid, data=node)

            # --- Wire edges to all existing nodes ---
            for other_id, other_vec in self._vec_index:
                other: MemoryNode = self.graph.nodes[other_id]["data"]
                self._wire_edges(nid, node, other_id, other)

            # Causal edge (directed: parent → this node)
            if causal_parent is not None and causal_parent in self.graph:
                self.graph.add_edge(causal_parent, nid,
                                    etype=EdgeType.CAUSAL, weight=1.0)

            # Contextual edges (shared tags)
            if context_tags:
                for other_id, _ in self._vec_index:
                    other: MemoryNode = self.graph.nodes[other_id]["data"]
                    shared = set(context_tags) & set(other.tags)
                    if shared:
                        w = len(shared) / max(len(context_tags), 1)
                        self.graph.add_edge(nid, other_id,
                                            etype=EdgeType.CONTEXTUAL, weight=w)

            # Reward edge: link to last rewarded node if DA was high
            if reward > self.REWARD_THRESHOLD and self._last_reward_node is not None:
                if self._last_reward_node in self.graph:
                    self.graph.add_edge(self._last_reward_node, nid,
                                        etype=EdgeType.REWARD, weight=reward)

            self._vec_index.append((nid, node.content_vec))

            # Prune if over capacity
            if self.graph.number_of_nodes() > self.max_nodes:
                self._prune_weakest()

            return nid

    def _wire_edges(self, nid: int, node: MemoryNode,
                    other_id: int, other: MemoryNode):
        """Wire all applicable edge types between two nodes."""
        # --- Semantic edge ---
        sem = _cosine(node.content_vec, other.content_vec)
        if sem > self.SEMANTIC_THRESHOLD:
            self.graph.add_edge(nid, other_id, etype=EdgeType.SEMANTIC, weight=sem)
            self.graph.add_edge(other_id, nid, etype=EdgeType.SEMANTIC, weight=sem)

        # --- Temporal edge ---
        dt = abs(node.timestamp - other.timestamp)
        if dt < self.TEMPORAL_WINDOW:
            tw = 1.0 - dt / self.TEMPORAL_WINDOW
            self.graph.add_edge(nid, other_id, etype=EdgeType.TEMPORAL, weight=tw)
            self.graph.add_edge(other_id, nid, etype=EdgeType.TEMPORAL, weight=tw)

        # --- Mood edge: NT cosine + valence similarity ---
        d = min(node.nt_state.size, other.nt_state.size)
        nt_sim  = _cosine(node.nt_state[:d], other.nt_state[:d])
        val_sim = 1.0 - abs(node.valence - other.valence) / 2.0
        arl_sim = 1.0 - abs(node.arousal - other.arousal)
        mood_w  = 0.5 * nt_sim + 0.3 * val_sim + 0.2 * arl_sim
        if mood_w > self.MOOD_THRESHOLD:
            self.graph.add_edge(nid, other_id, etype=EdgeType.MOOD, weight=mood_w)
            self.graph.add_edge(other_id, nid, etype=EdgeType.MOOD, weight=mood_w)

    # ------------------------------------------------------------------
    # Mesolimbic reward tagging (DA-driven)
    # ------------------------------------------------------------------
    def tag_reward(self, node_id: int, da_level: float,
                   reward_signal: float, insight: str | None = None):
        """Update a node's reward value via dopaminergic EMA.
        Mirrors the mesolimbic reward circuit: high DA + positive RPE → stronger tag.
        """
        if node_id not in self.graph:
            return
        with self.lock:
            node: MemoryNode = self.graph.nodes[node_id]["data"]
            # DA-gated EMA: higher DA → faster salience update
            alpha = 0.2 + 0.4 * da_level
            node.reward_value = (1 - alpha) * node.reward_value + alpha * reward_signal
            node.salience     = max(node.salience, node.reward_value)
            if insight:
                node.tags.append(f"insight:{insight}")
            # Wire reward edge to previous high-value node
            if (reward_signal > self.REWARD_THRESHOLD
                    and self._last_reward_node is not None
                    and self._last_reward_node in self.graph):
                self.graph.add_edge(self._last_reward_node, node_id,
                                    etype=EdgeType.REWARD, weight=reward_signal)
            if reward_signal > self.REWARD_THRESHOLD:
                self._last_reward_node = node_id

    # legacy alias used by brain.py
    def tag_salience(self, node_id: int, reward: float, insight: str | None = None):
        self.tag_reward(node_id, da_level=0.5, reward_signal=reward, insight=insight)

    # ------------------------------------------------------------------
    # Insight storage (training → memory)
    # ------------------------------------------------------------------
    def store_insight(self, content: str, content_vec: np.ndarray,
                      nt_state: np.ndarray, surprise: float,
                      comprehension: float, valence: float,
                      da_level: float = 0.5,
                      causal_parent: int | None = None) -> int | None:
        """Store a training insight into the memory graph.
        Only writes when surprise * comprehension * novelty > threshold.
        Returns node_id or None if not written.
        """
        novelty = self._compute_novelty(content_vec)
        score   = surprise * comprehension * novelty
        if score < 0.05:
            return None
        salience = score * (0.5 + 0.5 * da_level)
        nid = self.encode(
            content     = content,
            content_vec = content_vec,
            nt_state    = nt_state,
            valence     = valence,
            arousal     = float(nt_state[1]) if nt_state.size > 1 else 0.5,
            salience    = salience,
            reward      = da_level * score,
            tags        = [f"insight", f"surprise={surprise:.2f}",
                           f"comprehension={comprehension:.2f}"],
            causal_parent = causal_parent,
        )
        return nid

    def _compute_novelty(self, vec: np.ndarray) -> float:
        if not self._vec_index:
            return 1.0
        vec = np.asarray(vec, dtype=np.float32).flatten()
        sims = [_cosine(vec, v) for _, v in self._vec_index[-256:]]
        return float(1.0 - max(sims)) if sims else 1.0

    # ------------------------------------------------------------------
    # Abstract / consolidated node insertion
    # ------------------------------------------------------------------
    def add_abstract_node(self, content: str, centroid_vec: np.ndarray,
                          nt_state: np.ndarray, salience: float = 0.5,
                          tags: list | None = None) -> int:
        """Insert a consolidated semantic/abstract node (from consolidation.py)."""
        with self.lock:
            nid = self._next_id
            self._next_id  += 1
            self._write_ctr += 1
            node = MemoryNode(
                node_id     = nid,
                content     = content,
                content_vec = np.asarray(centroid_vec, dtype=np.float32).flatten(),
                nt_state    = np.asarray(nt_state,     dtype=np.float32).flatten(),
                salience    = salience,
                timestamp   = time.time(),
                write_index = self._write_ctr,
                tags        = list(tags or []) + ["abstract"],
                is_abstract = True,
            )
            self.graph.add_node(nid, data=node)
            for other_id, other_vec in self._vec_index:
                other: MemoryNode = self.graph.nodes[other_id]["data"]
                self._wire_edges(nid, node, other_id, other)
            self._vec_index.append((nid, node.content_vec))
            if self.graph.number_of_nodes() > self.max_nodes:
                self._prune_weakest()
            return nid

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def query_associative(self, query_vec: np.ndarray, topk: int = 5,
                          nt_filter: np.ndarray | None = None,
                          valence_range: tuple[float, float] | None = None
                          ) -> list[MemoryNode]:
        """Semantic nearest-neighbour with optional NT mood filter."""
        query_vec = np.asarray(query_vec, dtype=np.float32).flatten()
        scored = []
        for nid, vec in self._vec_index:
            node: MemoryNode = self.graph.nodes[nid]["data"]
            sim = _cosine(query_vec, vec)
            if nt_filter is not None:
                d = min(nt_filter.size, node.nt_state.size)
                if _cosine(nt_filter[:d], node.nt_state[:d]) < 0.3:
                    continue
            if valence_range is not None:
                if not (valence_range[0] <= node.valence <= valence_range[1]):
                    continue
            final = sim * (0.5 + 0.5 * node.salience) * node.decay
            scored.append((final, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, node in scored[:topk]:
            node.access_count += 1
            results.append(node)
        return results

    def query_temporal(self, topk: int = 5,
                       recency_weight: float = 1.0) -> list[MemoryNode]:
        """Return most recent nodes, optionally weighted by salience."""
        nodes = [(n.write_index * recency_weight + n.salience,
                  self.graph.nodes[nid]["data"])
                 for nid, _ in self._vec_index
                 if nid in self.graph]
        nodes.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in nodes[:topk]]

    def query_mood(self, nt_state: np.ndarray, valence: float,
                   arousal: float, topk: int = 5) -> list[MemoryNode]:
        """Return memories with similar NT state and emotional valence."""
        nt_q = np.asarray(nt_state, dtype=np.float32).flatten()
        scored = []
        for nid, _ in self._vec_index:
            node: MemoryNode = self.graph.nodes[nid]["data"]
            d      = min(nt_q.size, node.nt_state.size)
            nt_sim = _cosine(nt_q[:d], node.nt_state[:d])
            v_sim  = 1.0 - abs(valence - node.valence) / 2.0
            a_sim  = 1.0 - abs(arousal - node.arousal)
            mood   = 0.5 * nt_sim + 0.3 * v_sim + 0.2 * a_sim
            scored.append((mood * node.decay, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, node in scored[:topk]:
            node.access_count += 1
            results.append(node)
        return results

    def spreading_activation(self, seed_id: int, hops: int = 2,
                             topk: int = 8,
                             edge_types: set | None = None
                             ) -> list[MemoryNode]:
        """Multi-hop walk from seed node across specified edge types."""
        if seed_id not in self.graph:
            return []
        if edge_types is None:
            edge_types = {EdgeType.SEMANTIC, EdgeType.CAUSAL,
                          EdgeType.TEMPORAL, EdgeType.MOOD,
                          EdgeType.PATTERN, EdgeType.REWARD}

        visited: dict[int, float] = {seed_id: 1.0}
        frontier = {seed_id}
        for _ in range(hops):
            nxt = set()
            for nid in frontier:
                for _, nb, edata in self.graph.edges(nid, data=True):
                    if edata.get("etype") not in edge_types:
                        continue
                    w = edata.get("weight", 0.5) * visited[nid]
                    if nb not in visited or visited[nb] < w:
                        visited[nb] = w
                        nxt.add(nb)
            frontier = nxt

        visited.pop(seed_id, None)
        ranked = sorted(visited.items(), key=lambda x: x[1], reverse=True)
        results = []
        for nid, _ in ranked[:topk]:
            node: MemoryNode = self.graph.nodes[nid]["data"]
            node.access_count += 1
            results.append(node)
        return results

    def query_causal_chain(self, node_id: int,
                           direction: str = "forward",
                           max_depth: int = 5) -> list[MemoryNode]:
        """Follow causal edges forward or backward from a node."""
        if node_id not in self.graph:
            return []
        chain   = []
        current = node_id
        for _ in range(max_depth):
            if direction == "forward":
                nexts = [n for _, n, d in self.graph.edges(current, data=True)
                         if d.get("etype") == EdgeType.CAUSAL]
            else:
                nexts = [n for n, _, d in self.graph.in_edges(current, data=True)
                         if d.get("etype") == EdgeType.CAUSAL]
            if not nexts:
                break
            current = nexts[0]
            chain.append(self.graph.nodes[current]["data"])
        return chain

    # ------------------------------------------------------------------
    # Pattern edge (co-activation)
    # ------------------------------------------------------------------
    def add_pattern_edge(self, id_a: int, id_b: int, strength: float):
        if id_a in self.graph and id_b in self.graph:
            self.graph.add_edge(id_a, id_b, etype=EdgeType.PATTERN, weight=strength)
            self.graph.add_edge(id_b, id_a, etype=EdgeType.PATTERN, weight=strength)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def decay_all(self, factor: float | None = None):
        f = factor or self.decay_rate
        for nid in list(self.graph.nodes):
            self.graph.nodes[nid]["data"].decay *= f

    def _prune_weakest(self):
        """Remove the node with lowest salience × decay (never prune abstract nodes)."""
        worst_id    = None
        worst_score = float("inf")
        for nid in self.graph.nodes:
            n: MemoryNode = self.graph.nodes[nid]["data"]
            if n.is_abstract:
                continue          # protect consolidated knowledge
            score = n.salience * n.decay + 0.1 * n.access_count
            if score < worst_score:
                worst_score = score
                worst_id    = nid
        if worst_id is not None:
            self.graph.remove_node(worst_id)
            self._vec_index = [(i, v) for i, v in self._vec_index if i != worst_id]

    def get_node(self, node_id: int) -> MemoryNode | None:
        return self.graph.nodes[node_id]["data"] if node_id in self.graph else None

    def all_nodes(self) -> list[MemoryNode]:
        return [self.graph.nodes[nid]["data"] for nid in self.graph.nodes]

    def stats(self) -> dict:
        n_by_type = {et.value: 0 for et in EdgeType}
        for _, _, d in self.graph.edges(data=True):
            et = d.get("etype")
            if et:
                n_by_type[et.value] += 1
        return {
            "n_nodes":       self.graph.number_of_nodes(),
            "n_edges":       self.graph.number_of_edges(),
            "n_abstract":    sum(1 for n in self.all_nodes() if n.is_abstract),
            "edge_types":    n_by_type,
            "mean_salience": float(np.mean([n.salience for n in self.all_nodes()])
                                   if self.size else 0.0),
        }
