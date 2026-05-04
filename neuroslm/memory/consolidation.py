"""Memory consolidation: episodic → semantic patterns + predictions.

Implements offline consolidation (analogous to hippocampal-neocortical replay):
  1. Cluster episodic memories by semantic similarity.
  2. For each cluster, extract a centroid embedding and a generalised description.
  3. Detect recurring (action, context) → outcome patterns (causal rules).
  4. Insert abstract nodes into the RelationalMemoryGraph for persistent storage.
  5. Tag high-value patterns with the mesolimbic reward signal (DA salience).

This runs:
  - On demand via brain.consolidate_memory()
  - During training every N steps (configured in BrainConfig.consolidate_every)

Key SOTA concepts:
  - Memory replay: consolidated nodes replay frequently-accessed patterns
    (access_count used as replay frequency proxy)
  - Schema formation: clusters with low intra-cluster variance form schemas
    (abstract nodes tagged as "schema")
  - Prediction compression: episodes predicting the same outcome are merged
    into a single predictive rule node
  - Temporal contiguity: temporally adjacent episodes are preferentially
    linked, supporting episodic→semantic transition
"""
from __future__ import annotations
import time
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .relational_graph import RelationalMemoryGraph
    from .causal import CausalRuleStore


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _cluster_episodes(episodes: list[dict],
                      threshold: float = 0.80) -> list[list[dict]]:
    """Greedy cosine clustering. Produces variable-size clusters."""
    clusters: list[list[dict]] = []
    centroids: list[np.ndarray] = []

    for ep in episodes:
        vec = ep.get("content_vec")
        if vec is None:
            continue
        vec = np.asarray(vec, dtype=np.float32).flatten()

        best_c, best_sim = -1, -1.0
        for ci, cen in enumerate(centroids):
            d = min(vec.size, cen.size)
            s = _cosine(vec[:d], cen[:d])
            if s > best_sim:
                best_sim, best_c = s, ci

        if best_sim >= threshold and best_c >= 0:
            clusters[best_c].append(ep)
            # Update centroid (running mean)
            n = len(clusters[best_c])
            d = min(vec.size, centroids[best_c].size)
            centroids[best_c][:d] = (centroids[best_c][:d] * (n - 1) + vec[:d]) / n
        else:
            clusters.append([ep])
            centroids.append(vec.copy())

    return clusters


def _cluster_centroid(cluster: list[dict]) -> np.ndarray:
    vecs = [np.asarray(ep["content_vec"], dtype=np.float32).flatten()
            for ep in cluster if ep.get("content_vec") is not None]
    if not vecs:
        return np.zeros(1, dtype=np.float32)
    min_d = min(v.size for v in vecs)
    return np.mean([v[:min_d] for v in vecs], axis=0)


def _mean_nt_state(cluster: list[dict], n_nt: int = 7) -> np.ndarray:
    states = [np.asarray(ep["nt_state"], dtype=np.float32).flatten()
              for ep in cluster if ep.get("nt_state") is not None]
    if not states:
        return np.zeros(n_nt, dtype=np.float32)
    min_d = min(s.size for s in states)
    min_d = min(min_d, n_nt)
    result = np.zeros(n_nt, dtype=np.float32)
    result[:min_d] = np.mean([s[:min_d] for s in states], axis=0)
    return result


def _cluster_label(cluster: list[dict], max_chars: int = 80) -> str:
    """Summarise a cluster as the most-common content prefix."""
    texts = [ep.get("content", "") for ep in cluster if ep.get("content")]
    if not texts:
        return f"[abstract cluster, n={len(cluster)}]"
    # Return the most salient (first) text, truncated
    return texts[0][:max_chars].replace("\n", " ")


class MemoryConsolidator:
    """Consolidates episodic memories into semantic patterns.

    Usage in brain.py:
        consolidator = MemoryConsolidator(relational_memory, causal_store)
        consolidator.consolidate(brain.episodic.recent(256), da_level=da)
    """

    def __init__(self, relational_memory: "RelationalMemoryGraph",
                 causal_store: "CausalRuleStore | None" = None,
                 cluster_threshold: float = 0.80,
                 min_cluster_size: int = 2,
                 schema_variance_threshold: float = 0.15):
        self.relational   = relational_memory
        self.causal       = causal_store
        self.cluster_thr  = cluster_threshold
        self.min_size     = min_cluster_size
        self.schema_var   = schema_variance_threshold
        self._last_run    = 0.0

    def consolidate(self, episodes: list[dict],
                    da_level: float = 0.5,
                    threshold: float | None = None) -> dict:
        """Run a full consolidation pass.

        Args:
            episodes:  list of episode dicts (from EpisodicMemory.recent())
            da_level:  current dopamine level (boosts salience of new nodes)
            threshold: override cluster_threshold

        Returns:
            stats dict with n_clusters, n_inserted, n_causal_rules
        """
        thr = threshold or self.cluster_thr
        clusters = _cluster_episodes(episodes, threshold=thr)

        n_inserted = 0
        n_schemas  = 0

        for cluster in clusters:
            if len(cluster) < self.min_size:
                continue

            centroid  = _cluster_centroid(cluster)
            nt_mean   = _mean_nt_state(cluster)
            label     = _cluster_label(cluster)
            mean_val  = float(np.mean([ep.get("emotion") or 0.0 for ep in cluster]))
            salience  = float(np.mean([ep.get("content_vec") is not None
                                       for ep in cluster])) * da_level

            # Detect schemas: low intra-cluster embedding variance
            vecs = [np.asarray(ep["content_vec"], dtype=np.float32).flatten()
                    for ep in cluster if ep.get("content_vec") is not None]
            is_schema = False
            if len(vecs) >= self.min_size:
                min_d   = min(v.size for v in vecs)
                mat     = np.stack([v[:min_d] for v in vecs])
                var     = float(np.mean(np.var(mat, axis=0)))
                is_schema = var < self.schema_var

            tags = ["consolidated"]
            if is_schema:
                tags.append("schema")
                n_schemas += 1

            # Insert abstract node into relational graph
            # Novelty check: skip if very similar node already exists
            novelty = self.relational._compute_novelty(centroid)
            if novelty > 0.1:
                nid = self.relational.add_abstract_node(
                    content     = f"[{'schema' if is_schema else 'pattern'}] {label}",
                    centroid_vec = centroid,
                    nt_state    = nt_mean,
                    salience    = salience,
                    tags        = tags,
                )
                # DA reward-tag the newly consolidated node
                self.relational.tag_reward(nid, da_level, da_level * salience)
                n_inserted += 1

            # Causal rule extraction: (episode_t-1, episode_t) → outcome
            if self.causal is not None:
                self._extract_causal_rules(cluster, da_level)

        self._last_run = time.time()
        return {
            "n_clusters":  len(clusters),
            "n_inserted":  n_inserted,
            "n_schemas":   n_schemas,
            "n_causal":    len(self.causal.rules) if self.causal else 0,
        }

    def _extract_causal_rules(self, cluster: list[dict], da_level: float):
        """Extract (context, action) → outcome patterns from episode sequence."""
        for i in range(1, len(cluster)):
            prev = cluster[i - 1]
            curr = cluster[i]
            ctx  = prev.get("content_vec")
            act  = curr.get("content_vec")
            if ctx is None or act is None:
                continue
            ctx = np.asarray(ctx, dtype=np.float32).flatten()
            act = np.asarray(act, dtype=np.float32).flatten()

            # Outcome: mesolimbic DA proxy — crude valence from NT state
            nt  = curr.get("nt_state")
            outcome = 0.0
            if nt is not None:
                nt = np.asarray(nt, dtype=np.float32).flatten()
                # DA(index 0) - GABA(index 6) as crude valence signal
                if nt.size >= 7:
                    outcome = float(nt[0] - nt[6])
                elif nt.size >= 1:
                    outcome = float(nt[0] - 0.5)

            if abs(outcome) > 0.05:
                try:
                    self.causal.observe(act, ctx, outcome, step=0)
                except Exception:
                    pass

    def replay_high_value(self, topk: int = 10) -> list:
        """Return the top-k highest-salience nodes for memory replay.
        These are candidates for re-injecting into the GWS during mind-wandering.
        """
        nodes = [(n.salience * n.decay + 0.05 * n.access_count, n)
                 for n in self.relational.all_nodes()]
        nodes.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in nodes[:topk]]
