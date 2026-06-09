# -*- coding: utf-8 -*-
"""Hot-path mutator — turn heatmap hot/cold paths into DNAPatch proposals.

Structural-plasticity policy (cf. tests/test_hypergraph_evolution.py):
  - HOT node  -> node_mutation : perturb the module's gene (explore where
    learning concentrates); delta magnitude scales with heat.
  - HOT edge  -> edge_strengthen : reinforce a heavily-used path.
  - COLD edge -> edge_prune : remove an inert path.

Proposals are pure data (`DNAPatch`); they are not applied here. They
flow to the formal gate (ImprovementGate/TripleGuard) and the Lean proof
backend, and only admitted proposals are written back to the genome.
"""
from __future__ import annotations
from typing import List

from neuroslm.compiler.ribosome import DNAPatch


def propose_mutations(
    heatmap,
    ir,
    *,
    hot_threshold: float = 0.7,
    cold_threshold: float = 0.1,
    step: int = 0,
    delta_dim: int = 16,
    delta_scale: float = 0.05,
) -> List[DNAPatch]:
    """Propose DNAPatch mutations for the heatmap's hot/cold paths."""
    norm = heatmap.normalized()
    id_to_node = {n.id: n for n in ir.nodes}
    id_to_edge = {e.id: e for e in ir.hyperedges}
    patches: List[DNAPatch] = []

    # HOT nodes -> node_mutation (delta scaled by heat).
    for nid in heatmap.hot_paths(hot_threshold, kind="node"):
        node = id_to_node.get(nid)
        if node is None:
            continue
        h = norm.get(nid, 0.0)
        patches.append(DNAPatch(
            version="1.0", step=step, kind="node_mutation",
            target=node.name,
            delta=[delta_scale * h] * delta_dim,
            metadata={"reason": "hot_path", "heat": h, "element_id": nid},
        ))

    # HOT edges -> edge_strengthen.
    for eid in heatmap.hot_paths(hot_threshold, kind="edge"):
        edge = id_to_edge.get(eid)
        if edge is None:
            continue
        h = norm.get(eid, 0.0)
        patches.append(DNAPatch(
            version="1.0", step=step, kind="edge_strengthen",
            target=eid,
            delta=[delta_scale * h],
            metadata={"reason": "hot_path", "heat": h, "members": edge.members},
        ))

    # COLD edges -> edge_prune.
    for eid in heatmap.cold_paths(cold_threshold, kind="edge"):
        edge = id_to_edge.get(eid)
        if edge is None:
            continue
        h = norm.get(eid, 0.0)
        patches.append(DNAPatch(
            version="1.0", step=step, kind="edge_prune",
            target=eid,
            delta=[],
            metadata={"reason": "cold_path", "heat": h, "members": edge.members},
        ))

    return patches
