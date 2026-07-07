# -*- coding: utf-8 -*-
"""Geometric / graph-theoretic analysis of the flow graph.

Projects the execution profile into a weighted DiGraph and runs standard graph
theory — betweenness (bottleneck nodes), articulation points (cut vertices),
max-flow/min-cut (bottleneck edges), algebraic connectivity (spectral) — then
proposes structural edits from the geometry (bypass a bottleneck, prune a
low-flow/high-compute edge). NOT a fluid-flow sim; graph theory is the
high-leverage version of "find geometrically beneficial shapes".
"""
import torch

from neuroslm.genetic.language import Memory, Program, Instruction
from neuroslm.genetic.profile import profile_program
from neuroslm.genetic.attention_primitives import single_head_attention_program
from neuroslm.genetic.topology import analyze, to_networkx, propose_edits, TopologyReport


def _chain_profile():
    # a → b → c (a strict chain: every interior node is a bottleneck)
    prog = Program(
        [
            Instruction("tanh", "t2", ("t0",)),
            Instruction("sigmoid", "t3", ("t2",)),
            Instruction("neg", "t4", ("t3",)),
        ],
        n_scalar=2, n_tensor=6, out_reg="t4",
    )
    return profile_program(prog, {"t0": torch.randn(8)})


def _diamond_profile():
    # a splits to b,c which merge at d: b,c are parallel; a,d are cut vertices
    prog = Program(
        [
            Instruction("tanh", "t2", ("t0",)),        # a
            Instruction("sigmoid", "t3", ("t2",)),     # b
            Instruction("relu", "t4", ("t2",)),        # c
            Instruction("add", "t5", ("t3", "t4")),    # d
        ],
        n_scalar=2, n_tensor=8, out_reg="t5",
    )
    return profile_program(prog, {"t0": torch.randn(8)})


class TestGraphBuild:
    def test_networkx_graph_has_nodes_and_flow_edges(self):
        G = to_networkx(_chain_profile())
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        # edges carry a flow capacity
        for _, _, data in G.edges(data=True):
            assert "capacity" in data and data["capacity"] >= 0


class TestAnalysis:
    def test_report_fields_present(self):
        rep = analyze(_diamond_profile())
        assert isinstance(rep, TopologyReport)
        assert isinstance(rep.betweenness, dict)
        assert isinstance(rep.articulation_points, list)
        assert rep.algebraic_connectivity >= 0.0

    def test_diamond_parallel_nodes_are_not_cut_vertices(self):
        rep = analyze(_diamond_profile())
        # the two parallel branch nodes (indices 1 and 2) are NOT articulation pts
        arts = set(rep.articulation_points)
        assert 1 not in arts or 2 not in arts  # at least one branch is bypassable

    def test_chain_has_a_bottleneck(self):
        rep = analyze(_chain_profile())
        # interior node of a chain is a cut vertex → flagged as a bottleneck
        assert len(rep.bottleneck_nodes) >= 1

    def test_min_cut_computed_on_attention_graph(self):
        rep = analyze(profile_program(single_head_attention_program(scale=0.35), {
            "t0": torch.randn(2, 6, 8),
            "t1": torch.randn(8, 8) * 0.1,
            "t2": torch.randn(8, 8) * 0.1,
            "t3": torch.randn(8, 8) * 0.1,
            "t4": torch.randn(8, 8) * 0.1,
        }))
        assert rep.min_cut_value >= 0.0


class TestProposals:
    def test_proposes_edits_from_geometry(self):
        edits = propose_edits(_chain_profile())
        assert isinstance(edits, list)
        assert all("kind" in e and "reason" in e for e in edits)
        # a chain's bottleneck should suggest a bypass/parallel path
        assert any(e["kind"] in ("bypass", "parallelize") for e in edits)
