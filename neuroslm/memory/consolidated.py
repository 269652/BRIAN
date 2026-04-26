"""Consolidated memory: deduplicates, compresses, and stores episodes as graph nodes."""
import numpy as np
import networkx as nx
import threading


def cosine_similarity(a, b=None):
    """Lightweight cosine similarity for two 2D arrays using numpy.
    If b is None, compute pairwise similarity between rows of a.
    """
    a = np.asarray(a, dtype=float)
    if b is None:
        b = a
    else:
        b = np.asarray(b, dtype=float)
    # Normalize rows
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(a_norm, b_norm.T)

class ConsolidatedMemory:
    def __init__(self):
        self.graph = nx.Graph()
        self.lock = threading.Lock()
        self.next_id = 0

    def add_node(self, content_vec, meta):
        with self.lock:
            node_id = self.next_id
            self.graph.add_node(node_id, **meta, content_vec=content_vec)
            self.next_id += 1
            return node_id

    def add_edge(self, id1, id2, weights):
        with self.lock:
            self.graph.add_edge(id1, id2, **weights)

    def consolidate(self, episodes, threshold=0.85):
        # Cluster similar episodes and add as nodes, connect with weighted edges
        if not episodes:
            return
        vecs = np.stack([ep['content_vec'] for ep in episodes])
        sims = cosine_similarity(vecs)
        used = set()
        for i, ep in enumerate(episodes):
            if i in used:
                continue
            cluster = [i]
            for j in range(i+1, len(episodes)):
                if sims[i, j] > threshold:
                    cluster.append(j)
                    used.add(j)
            # Merge cluster
            merged_vec = np.mean([episodes[k]['content_vec'] for k in cluster], axis=0)
            meta = {k: episodes[cluster[0]][k] for k in episodes[cluster[0]] if k != 'content_vec'}
            node_id = self.add_node(merged_vec, meta)
            # Connect to other nodes by similarity
            for other_id in self.graph.nodes:
                if other_id == node_id:
                    continue
                other_vec = self.graph.nodes[other_id]['content_vec']
                sim = cosine_similarity([merged_vec], [other_vec])[0,0]
                weights = {
                    'content_sim': float(sim),
                    'nt_sim': float(np.dot(meta.get('nt_state', np.zeros_like(merged_vec)), self.graph.nodes[other_id].get('nt_state', np.zeros_like(merged_vec)))),
                    'emotion_sim': float(meta.get('emotion', 0) == self.graph.nodes[other_id].get('emotion', 0)),
                    'temporal': abs(meta.get('timestamp', 0) - self.graph.nodes[other_id].get('timestamp', 0)),
                }
                self.add_edge(node_id, other_id, weights)

    def query(self, content_vec, topk=5):
        # Return top-k most similar nodes
        sims = []
        for node_id, data in self.graph.nodes(data=True):
            sim = cosine_similarity([content_vec], [data['content_vec']])[0,0]
            sims.append((sim, node_id))
        sims.sort(reverse=True)
        return [nid for _, nid in sims[:topk]]
