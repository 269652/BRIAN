# -*- coding: utf-8 -*-
"""TDD: Hypergraph IR + SourceMap (Layer 3).

The DSL is lifted into a hypergraph:
  - nodes      = populations, neurotransmitters, the architecture decl
  - hyperedges = synapses / modulations (connect several member nodes)

Each element carries a source span (provenance: which bytes it came from).
The SourceMap holds the original source plus those spans so that:
  - render()                 reproduces the DSL byte-for-byte
  - render_with_overrides()  re-renders only mutated nodes, leaving the
                             rest of the file untouched (evolvable codegen)

These tests pin the semantic-graph shape and the bit-identical render
contract the DNA encoder (Layer 4) builds on.
"""
import pytest

from neuroslm.compiler.hypergraph_ir import (
    HypergraphIR, HyperNode, HyperEdge, SourceMap, lift_dsl_to_hypergraph,
)


SAMPLE = """architecture demo { d_sem: 256, dt: 0.01 }

neurotransmitter dopamine { base_concentration: 0.5 }

population cortex { count: 512, dynamics: "rate_code" }
population striatum { count: 256, dynamics: "rate_code" }

synapse cortex -> striatum { weight: 0.5 }
modulation dopamine -> striatum { gain: 1.2 }
"""


class TestLiftToHypergraph:
    """DSL -> hypergraph extracts the right nodes and hyperedges."""

    def test_population_nodes_extracted(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        pops = ir.nodes_of_kind("population")
        names = {n.name for n in pops}
        assert names == {"cortex", "striatum"}

    def test_neurotransmitter_node_extracted(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        nts = ir.nodes_of_kind("neurotransmitter")
        assert {n.name for n in nts} == {"dopamine"}

    def test_architecture_node_extracted(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        arch = ir.nodes_of_kind("architecture")
        assert len(arch) == 1
        assert arch[0].name == "demo"

    def test_synapse_is_a_hyperedge_over_two_members(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        syn = [e for e in ir.hyperedges if e.kind == "synapse"]
        assert len(syn) == 1
        assert syn[0].members == ["cortex", "striatum"]

    def test_modulation_is_a_hyperedge(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        mod = [e for e in ir.hyperedges if e.kind == "modulation"]
        assert len(mod) == 1
        assert mod[0].members == ["dopamine", "striatum"]

    def test_population_node_carries_attrs(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        cortex = next(n for n in ir.nodes_of_kind("population") if n.name == "cortex")
        assert int(float(cortex.attrs["count"])) == 512

    def test_node_span_points_at_its_source_text(self):
        """A node's recorded span slices to its own declaration."""
        ir = lift_dsl_to_hypergraph(SAMPLE)
        cortex = next(n for n in ir.nodes_of_kind("population") if n.name == "cortex")
        start, end = cortex.span
        assert "cortex" in SAMPLE[start:end]
        assert SAMPLE[start:end].startswith("population")


class TestSourceMapBitIdentity:
    """SourceMap.render() reproduces the DSL byte-for-byte."""

    def test_render_is_bit_identical(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        assert ir.source_map.render() == SAMPLE

    def test_render_identical_for_unicode_and_comments(self):
        src = (
            "# Φ > 0 comment — non-ascii\n"
            'architecture u { d_sem: 256 }\n'
            "population p { count: 8 }  # trailing\n"
        )
        ir = lift_dsl_to_hypergraph(src)
        assert ir.source_map.render() == src

    def test_sourcemap_roundtrips_through_dict(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        d = ir.source_map.to_dict()
        sm2 = SourceMap.from_dict(d)
        assert sm2.render() == SAMPLE


class TestHypergraphSerialization:
    """The whole IR serializes to a dict and back without loss."""

    def test_ir_roundtrips_through_dict(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        d = ir.to_dict()
        ir2 = HypergraphIR.from_dict(d)
        assert {n.name for n in ir2.nodes_of_kind("population")} == {"cortex", "striatum"}
        assert ir2.source_map.render() == SAMPLE
        assert len([e for e in ir2.hyperedges if e.kind == "synapse"]) == 1

    def test_to_dict_is_json_serializable(self):
        import json
        ir = lift_dsl_to_hypergraph(SAMPLE)
        # Must not raise.
        blob = json.dumps(ir.to_dict())
        assert isinstance(blob, str) and len(blob) > 0


class TestEvolvableRender:
    """render_with_overrides re-renders only mutated nodes."""

    def test_override_changes_only_target_span(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        cortex = next(n for n in ir.nodes_of_kind("population") if n.name == "cortex")
        new_decl = 'population cortex { count: 1024, dynamics: "rate_code" }'
        rendered = ir.source_map.render_with_overrides({cortex.id: new_decl})

        # The mutated declaration is present...
        assert "count: 1024" in rendered
        # ...and everything else is untouched.
        assert "striatum" in rendered
        assert "modulation dopamine -> striatum { gain: 1.2 }" in rendered
        # The original cortex count is gone (replaced, not appended).
        assert "count: 512" not in rendered

    def test_no_overrides_is_bit_identical(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        assert ir.source_map.render_with_overrides({}) == SAMPLE
