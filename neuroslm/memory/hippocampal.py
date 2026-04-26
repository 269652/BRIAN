"""Hippocampal enrichment: retrieves associative memories for GWS enrichment."""
import numpy as np

class HippocampalEnrichment:
    def __init__(self, consolidated_memory):
        self.consolidated_memory = consolidated_memory

    def enrich(self, gws_vec, nt_state=None, emotion=None, topk=5):
        # Query consolidated memory for associative nodes
        node_ids = self.consolidated_memory.query(gws_vec, topk=topk)
        enriched = []
        for nid in node_ids:
            node = self.consolidated_memory.graph.nodes[nid]
            # Optionally filter by NT/emotion similarity
            if nt_state is not None:
                sim = np.dot(nt_state, node.get('nt_state', np.zeros_like(gws_vec)))
                if sim < 0.5:
                    continue
            if emotion is not None and node.get('emotion', None) != emotion:
                continue
            enriched.append(node)
        return enriched
