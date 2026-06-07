# -*- coding: utf-8 -*-
"""TDD Tests for THSD End-to-End Integration (Phase 4)

Full integration tests: arch.neuro → THSD IR → Hypergraph IR → Validation
"""
import pytest
from neuroslm.dsl.compiler import NeuroMLCompiler
from neuroslm.dsl.hypergraph_ir import HypergraphBuilder


class TestTHSDDSLToHypergraph:
    """Test complete DSL parsing to hypergraph conversion."""

    def test_simple_complex_dsl_to_hypergraph(self):
        """Parse minimal complex and convert to hypergraph."""
        dsl = """
        complex SimpleBrain {
            stalk {
                representation_dim: 256,
                fisher_information_metric: "information_geometry"
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        assert len(ir.thsd_complexes) == 1

        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        assert hypergraph.name == "SimpleBrain"
        assert len(hypergraph.nodes) == 1
        assert hypergraph.nodes["SimpleBrain"].stalk_dim == 256

    def test_complex_with_topology_dsl_to_hypergraph(self):
        """Parse complex with topology to hypergraph."""
        dsl = """
        complex TonnetzBrain {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            topology {
                kind: "Tonnetz",
                spectral_gap: 0.3,
                dimension: 8
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        assert hypergraph.spectral_gap == 0.3
        assert hypergraph.dimension == 8
        assert hypergraph.validate()

    def test_complex_with_formal_spec_dsl_to_hypergraph(self):
        """Parse complex with formal_spec to hypergraph."""
        dsl = """
        complex ConstrainedBrain {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            formal_spec {
                cohomology_floor: 0.01,
                phi_target: 0.8
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        assert hypergraph.cohomology_floor == 0.01
        assert hypergraph.phi_target == 0.8

    def test_complex_with_full_spec_dsl_to_hypergraph(self):
        """Parse complete THSD complex to hypergraph."""
        dsl = """
        complex FullBrain {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry",
                local_constraints: ["predictive_consistency"]
            },
            topology {
                kind: "Tonnetz",
                spectral_gap: 0.3,
                dimension: 8,
                coherence_threshold: 0.95
            },
            formal_spec {
                cohomology_floor: 0.01,
                phi_target: 0.8,
                phi_method: "geometric_IIT4",
                information_bottleneck: {
                    enabled: true,
                    compression_ratio: 0.7,
                    prediction_lower_bound: 0.95
                }
            },
            dynamics {
                emission {
                    trigger: "surprise_head(threshold=0.8)",
                    payload_dim: 64,
                    lifetime_steps: 100
                }
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        assert len(ir.thsd_complexes) == 1

        complex_ir = ir.thsd_complexes[0]
        assert complex_ir.name == "FullBrain"
        assert complex_ir.stalk is not None
        assert complex_ir.topology is not None
        assert complex_ir.formal_spec is not None
        assert complex_ir.dynamics is not None

        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)
        assert hypergraph.validate()


class TestMultipleCOmplexes:
    """Test parsing and integrating multiple complexes."""

    def test_multiple_complexes_each_to_hypergraph(self):
        """Parse multiple complexes and convert each independently."""
        dsl = """
        complex Cortex {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            }
        }
        complex Thalamus {
            stalk {
                representation_dim: 256,
                fisher_information_metric: "information_geometry"
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        assert len(ir.thsd_complexes) == 2

        builder = HypergraphBuilder()
        for complex_ir in ir.thsd_complexes:
            hypergraph = builder.from_complex_ir(complex_ir)
            assert hypergraph.validate()
            assert hypergraph.nodes[complex_ir.name] is not None


class TestConstraintPropagation:
    """Test that constraints propagate correctly through pipeline."""

    def test_spectral_gap_propagates_dsl_to_hypergraph(self):
        """Spectral gap from DSL → THSD IR → Hypergraph."""
        dsl = """
        complex SpectralBrain {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            topology {
                kind: "Tonnetz",
                spectral_gap: 0.42
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        assert complex_ir.topology.spectral_gap == 0.42

        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)
        assert hypergraph.spectral_gap == 0.42

    def test_phi_target_propagates_dsl_to_hypergraph(self):
        """Phi target from DSL → THSD IR → Hypergraph."""
        dsl = """
        complex PhiBrain {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            formal_spec {
                phi_target: 0.73
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        assert complex_ir.formal_spec.phi_target == 0.73

        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)
        assert hypergraph.phi_target == 0.73

    def test_cohomology_propagates_dsl_to_hypergraph(self):
        """Cohomology constraints propagate through pipeline."""
        dsl = """
        complex CohomologyBrain {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            formal_spec {
                cohomology_floor: 0.005
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        assert complex_ir.formal_spec.cohomology_floor == 0.005

        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)
        assert hypergraph.cohomology_floor == 0.005


class TestErrorHandlingIntegration:
    """Test error propagation through full pipeline."""

    def test_invalid_spectral_gap_caught_at_parse(self):
        """Invalid spectral gap should fail at parse time."""
        dsl = """
        complex BadBrain {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            topology {
                kind: "Tonnetz",
                spectral_gap: -0.1
            }
        }
        """
        with pytest.raises(ValueError, match="spectral_gap.*positive"):
            NeuroMLCompiler.compile(dsl)

    def test_invalid_phi_target_caught_at_parse(self):
        """Invalid phi_target should fail at parse time."""
        dsl = """
        complex BadPhi {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            formal_spec {
                phi_target: 2.0
            }
        }
        """
        with pytest.raises(ValueError, match="phi_target.*\\[0, 1\\]"):
            NeuroMLCompiler.compile(dsl)


class TestTopologicalInvariants:
    """Test that topological invariants are preserved."""

    def test_hypergraph_dimension_from_nodes(self):
        """Hypergraph dimension should match max node dimension."""
        dsl = """
        complex DimensionalBrain {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            topology {
                kind: "Tonnetz",
                spectral_gap: 0.3,
                dimension: 12
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        assert hypergraph.dimension == 12

    def test_stalk_dimension_preserved(self):
        """Stalk representation dimension should be preserved."""
        dsl = """
        complex StalkBrain {
            stalk {
                representation_dim: 768,
                fisher_information_metric: "information_geometry"
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        root_node = hypergraph.nodes[complex_ir.name]
        assert root_node.stalk_dim == 768


class TestRoundTripFidelity:
    """Test that information is preserved through full pipeline."""

    def test_dsl_parse_hypergraph_roundtrip(self):
        """DSL → THSD IR → Hypergraph should preserve all metadata."""
        dsl = """
        complex RoundTripBrain {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry",
                local_constraints: ["predictive_consistency", "stability"]
            },
            topology {
                kind: "Tonnetz",
                spectral_gap: 0.35,
                dimension: 10,
                coherence_threshold: 0.92
            },
            formal_spec {
                cohomology_floor: 0.015,
                phi_target: 0.82,
                phi_method: "geometric_IIT4"
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]

        # Check THSD IR preservation
        assert complex_ir.name == "RoundTripBrain"
        assert complex_ir.stalk.representation_dim == 512
        assert "predictive_consistency" in complex_ir.stalk.local_constraints
        assert "stability" in complex_ir.stalk.local_constraints
        assert complex_ir.topology.spectral_gap == 0.35
        assert complex_ir.topology.dimension == 10
        assert complex_ir.formal_spec.cohomology_floor == 0.015
        assert complex_ir.formal_spec.phi_target == 0.82

        # Check Hypergraph IR preservation
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)
        assert hypergraph.name == "RoundTripBrain"
        assert hypergraph.spectral_gap == 0.35
        assert hypergraph.dimension == 10
        assert hypergraph.cohomology_floor == 0.015
        assert hypergraph.phi_target == 0.82


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
