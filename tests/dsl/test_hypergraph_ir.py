# -*- coding: utf-8 -*-
"""TDD Tests for Hypergraph IR (Phase 2)

Tests for constructing and validating hypergraph representations
of simplicial complexes with sheaf bundles.
"""
import pytest
from neuroslm.dsl.thsd_ir import (
    ComplexIR,
    SheafStalkIR,
    TopologyIR,
    CohomologyIR,
    DynamicsIR,
)
from neuroslm.dsl.hypergraph_ir import (
    SimplexNode,
    HypergraphEdge,
    HypergraphIR,
    HypergraphBuilder,
)


class TestSimplexNodeConstruction:
    """Test SimplexNode dataclass."""

    def test_simplex_node_creation(self):
        """Create a 0-simplex (vertex) node."""
        node = SimplexNode(
            id="v1",
            dimension=0,
            name="LanguageCortex_input",
            stalk_dim=512,
            metadata={"region": "sensory"},
        )
        assert node.id == "v1"
        assert node.dimension == 0
        assert node.name == "LanguageCortex_input"
        assert node.stalk_dim == 512
        assert node.metadata["region"] == "sensory"

    def test_simplex_node_edge(self):
        """Create a 1-simplex (edge) node."""
        node = SimplexNode(
            id="e1",
            dimension=1,
            name="sensory_to_thalamus",
            stalk_dim=256,
        )
        assert node.dimension == 1
        assert node.stalk_dim == 256

    def test_simplex_node_face(self):
        """Create a 2-simplex (face) node."""
        node = SimplexNode(
            id="f1",
            dimension=2,
            name="sensory_thalamus_cortex",
            stalk_dim=384,
        )
        assert node.dimension == 2


class TestHypergraphEdgeConstruction:
    """Test HypergraphEdge dataclass."""

    def test_hypergraph_edge_face_relation(self):
        """Create edge representing face relation."""
        edge = HypergraphEdge(
            id="face_e1_f1",
            src_simplex="e1",
            dst_simplex="f1",
            kind="faces",
            weight=1.0,
            metadata={"orientation": "positive"},
        )
        assert edge.src_simplex == "e1"
        assert edge.dst_simplex == "f1"
        assert edge.kind == "faces"

    def test_hypergraph_edge_coboundary(self):
        """Create edge representing coboundary relation."""
        edge = HypergraphEdge(
            id="cobound_v1_e1",
            src_simplex="v1",
            dst_simplex="e1",
            kind="coboundary",
            weight=1.0,
        )
        assert edge.kind == "coboundary"

    def test_hypergraph_edge_coupling(self):
        """Create edge representing sheaf coupling."""
        edge = HypergraphEdge(
            id="couple_f1_f2",
            src_simplex="f1",
            dst_simplex="f2",
            kind="coupling",
            weight=0.8,
            metadata={"sheaf_section": "syntactic_layer"},
        )
        assert edge.kind == "coupling"
        assert edge.weight == 0.8


class TestHypergraphIRConstruction:
    """Test HypergraphIR dataclass."""

    def test_hypergraph_ir_creation(self):
        """Create basic hypergraph IR."""
        nodes = {
            "v1": SimplexNode("v1", 0, "input", 512),
            "v2": SimplexNode("v2", 0, "output", 512),
        }
        edges = {
            "e1": HypergraphEdge("e1", "v1", "v2", "faces", 1.0),
        }
        hypergraph = HypergraphIR(
            name="SimpleGraph",
            nodes=nodes,
            edges=edges,
            dimension=1,
        )
        assert hypergraph.name == "SimpleGraph"
        assert len(hypergraph.nodes) == 2
        assert len(hypergraph.edges) == 1
        assert hypergraph.dimension == 1

    def test_hypergraph_ir_with_spectral_gap(self):
        """Hypergraph should track spectral gap."""
        nodes = {
            "v1": SimplexNode("v1", 0, "input", 512),
        }
        hypergraph = HypergraphIR(
            name="TonnetzGraph",
            nodes=nodes,
            edges={},
            dimension=0,
            spectral_gap=0.3,
        )
        assert hypergraph.spectral_gap == 0.3

    def test_hypergraph_ir_with_phi(self):
        """Hypergraph should track integrated information."""
        nodes = {}
        hypergraph = HypergraphIR(
            name="PhiGraph",
            nodes=nodes,
            edges={},
            dimension=0,
            phi_value=0.75,
        )
        assert hypergraph.phi_value == 0.75


class TestHypergraphBuilder:
    """Test HypergraphBuilder for constructing hypergraphs from THSD IR."""

    def test_builder_from_complex_ir(self):
        """Build hypergraph from ComplexIR."""
        complex_ir = ComplexIR(
            name="TestComplex",
            stalk=SheafStalkIR(
                representation_dim=512,
                fisher_information_metric="information_geometry",
            ),
            topology=TopologyIR(
                kind="Tonnetz",
                spectral_gap=0.3,
                dimension=8,
            ),
        )
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        assert hypergraph.name == "TestComplex"
        assert hypergraph.dimension == 8
        assert hypergraph.spectral_gap == 0.3

    def test_builder_creates_root_simplex(self):
        """Builder should create root simplex for the complex."""
        complex_ir = ComplexIR(
            name="RootComplex",
            stalk=SheafStalkIR(representation_dim=256, fisher_information_metric="ig"),
        )
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        # Should have at least a root node
        assert len(hypergraph.nodes) > 0
        root_node = hypergraph.nodes.get("RootComplex")
        assert root_node is not None
        assert root_node.name == "RootComplex"

    def test_builder_tracks_cohomology(self):
        """Builder should embed cohomological constraints."""
        complex_ir = ComplexIR(
            name="CohomologyComplex",
            stalk=SheafStalkIR(representation_dim=512, fisher_information_metric="ig"),
            formal_spec=CohomologyIR(
                cohomology_floor=0.01,
                phi_target=0.8,
            ),
        )
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        assert hypergraph.cohomology_floor == 0.01
        assert hypergraph.phi_target == 0.8

    def test_hypergraph_validates_spectral_gap(self):
        """Hypergraph should validate spectral gap constraint."""
        nodes = {"v1": SimplexNode("v1", 0, "node", 512)}
        hypergraph = HypergraphIR(
            name="ValidSpectralGap",
            nodes=nodes,
            edges={},
            dimension=0,
            spectral_gap=0.3,
        )
        assert hypergraph.validate()

    def test_hypergraph_rejects_negative_spectral_gap(self):
        """Hypergraph should reject invalid spectral gap."""
        nodes = {"v1": SimplexNode("v1", 0, "node", 512)}
        with pytest.raises(ValueError, match="spectral_gap.*positive"):
            HypergraphIR(
                name="InvalidSpectralGap",
                nodes=nodes,
                edges={},
                dimension=0,
                spectral_gap=-0.5,
            )


class TestHypergraphTopology:
    """Test topological properties of hypergraph."""

    def test_boundary_operator(self):
        """Compute boundary of a simplex."""
        nodes = {
            "v1": SimplexNode("v1", 0, "v1", 256),
            "v2": SimplexNode("v2", 0, "v2", 256),
            "e1": SimplexNode("e1", 1, "e1", 512),
        }
        edges = {
            "b1": HypergraphEdge("b1", "v1", "e1", "faces", 1.0),
            "b2": HypergraphEdge("b2", "v2", "e1", "faces", 1.0),
        }
        hypergraph = HypergraphIR(
            name="BoundaryTest",
            nodes=nodes,
            edges=edges,
            dimension=1,
        )

        # Boundary of e1 should include v1 and v2
        boundary = hypergraph.boundary("e1")
        assert "v1" in boundary
        assert "v2" in boundary

    def test_coboundary_operator(self):
        """Compute coboundary of a simplex."""
        nodes = {
            "v1": SimplexNode("v1", 0, "v1", 256),
            "e1": SimplexNode("e1", 1, "e1", 512),
            "f1": SimplexNode("f1", 2, "f1", 768),
        }
        edges = {
            "cobound": HypergraphEdge("cobound", "e1", "f1", "coboundary", 1.0),
        }
        hypergraph = HypergraphIR(
            name="CoboundaryTest",
            nodes=nodes,
            edges=edges,
            dimension=2,
        )

        # Coboundary of e1 should include f1
        coboundary = hypergraph.coboundary("e1")
        assert "f1" in coboundary

    def test_hypergraph_dimension_from_nodes(self):
        """Hypergraph dimension should match max node dimension."""
        nodes = {
            "v0": SimplexNode("v0", 0, "v0", 256),
            "e1": SimplexNode("e1", 1, "e1", 512),
            "f2": SimplexNode("f2", 2, "f2", 768),
        }
        hypergraph = HypergraphIR(
            name="DimensionTest",
            nodes=nodes,
            edges={},
        )
        assert hypergraph.dimension == 2


class TestHypergraphIntegration:
    """Integration tests for full hypergraph workflow."""

    def test_build_from_thsd_complex_and_validate(self):
        """End-to-end: build hypergraph from THSD IR and validate."""
        complex_ir = ComplexIR(
            name="LanguageCortex",
            stalk=SheafStalkIR(
                representation_dim=512,
                fisher_information_metric="information_geometry",
                local_constraints=["predictive_consistency"],
            ),
            topology=TopologyIR(
                kind="Tonnetz",
                spectral_gap=0.3,
                dimension=8,
            ),
            formal_spec=CohomologyIR(
                cohomology_floor=0.01,
                phi_target=0.8,
            ),
        )

        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        # Validate all constraints
        assert hypergraph.validate()
        assert hypergraph.spectral_gap == 0.3
        assert hypergraph.phi_target == 0.8
        assert hypergraph.dimension >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
