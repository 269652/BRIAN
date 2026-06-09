# -*- coding: utf-8 -*-
"""TDD: hot-path mutator (L3).

Turns a TrainingHeatmap + IR into proposed DNAPatch mutations:
  - hot node  -> node_mutation (delta scaled by heat)
  - hot edge  -> edge_strengthen
  - cold edge -> edge_prune
These proposals feed the formal gate (L4) and Lean proof (L5).
"""
import pytest

from neuroslm.compiler.hypergraph_ir import lift_dsl_to_hypergraph
from neuroslm.evolution.heatmap import TrainingHeatmap
from neuroslm.evolution.mutator import propose_mutations


SAMPLE = (
    "architecture a { d_sem: 256 }\n"
    "neurotransmitter dopamine { base_concentration: 0.5 }\n"
    'population cortex { count: 512, dynamics: "rate_code" }\n'
    'population striatum { count: 256, dynamics: "rate_code" }\n'
    "synapse cortex -> striatum { weight: 0.5 }\n"
    "modulation dopamine -> striatum { gain: 1.2 }\n"
)


def _heatmap():
    hm = TrainingHeatmap()
    hm.update(
        {
            "population:cortex": 1.0,                 # hot node
            "population:striatum": 0.02,              # cold node
            "synapse:cortex->striatum": 0.95,        # hot edge
            "modulation:dopamine->striatum": 0.02,   # cold edge
        },
        kinds={
            "population:cortex": "node",
            "population:striatum": "node",
            "synapse:cortex->striatum": "edge",
            "modulation:dopamine->striatum": "edge",
        },
    )
    return hm


class TestProposeMutations:
    def test_hot_node_yields_node_mutation(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        patches = propose_mutations(_heatmap(), ir, step=1000, delta_dim=16)
        node_muts = [p for p in patches if p.kind == "node_mutation"]
        assert len(node_muts) == 1
        p = node_muts[0]
        assert p.target == "cortex"
        assert len(p.delta) == 16
        assert p.metadata["reason"] == "hot_path"
        assert p.step == 1000

    def test_hot_edge_yields_strengthen(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        patches = propose_mutations(_heatmap(), ir)
        strengthen = [p for p in patches if p.kind == "edge_strengthen"]
        assert len(strengthen) == 1
        assert strengthen[0].target == "synapse:cortex->striatum"

    def test_cold_edge_yields_prune(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        patches = propose_mutations(_heatmap(), ir)
        prune = [p for p in patches if p.kind == "edge_prune"]
        assert len(prune) == 1
        assert prune[0].target == "modulation:dopamine->striatum"
        assert prune[0].metadata["reason"] == "cold_path"

    def test_delta_scales_with_heat(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        # Two hot nodes with different heat -> larger heat => larger delta.
        hm = TrainingHeatmap()
        hm.update(
            {"population:cortex": 1.0, "population:striatum": 0.8},
            kinds={"population:cortex": "node", "population:striatum": "node"},
        )
        patches = propose_mutations(hm, ir, delta_scale=0.1, delta_dim=4)
        by_target = {p.target: p for p in patches if p.kind == "node_mutation"}
        assert by_target["cortex"].delta[0] > by_target["striatum"].delta[0]

    def test_no_proposals_when_all_uniform(self):
        """All-equal heat -> everything normalizes to 1.0 -> nothing cold."""
        ir = lift_dsl_to_hypergraph(SAMPLE)
        hm = TrainingHeatmap()
        hm.update(
            {"population:cortex": 0.5, "population:striatum": 0.5},
            kinds={"population:cortex": "node", "population:striatum": "node"},
        )
        patches = propose_mutations(hm, ir)
        assert all(p.kind != "edge_prune" for p in patches)

    def test_patches_are_dnapatch_instances(self):
        from neuroslm.compiler.ribosome import DNAPatch
        ir = lift_dsl_to_hypergraph(SAMPLE)
        patches = propose_mutations(_heatmap(), ir)
        assert patches and all(isinstance(p, DNAPatch) for p in patches)
        # Every patch serializes (for DNA storage / audit).
        assert all("kind" in p.to_dict() for p in patches)
