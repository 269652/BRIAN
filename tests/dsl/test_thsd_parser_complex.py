# -*- coding: utf-8 -*-
"""TDD Tests for THSD Complex Block Parsing

Tests for parsing `complex` blocks (simplicial complexes with sheaf-stalks)
from DSL code into THSD IR.
"""
import pytest
from neuroslm.dsl.compiler import NeuroMLCompiler
from neuroslm.dsl.thsd_ir import (
    ComplexIR,
    SheafStalkIR,
    TopologyIR,
    CohomologyIR,
    DynamicsIR,
)


class TestComplexBasicParsing:
    """Test basic complex block structure."""

    def test_complex_block_parses_minimal(self):
        """Parse minimal complex block."""
        dsl = """
        complex LanguageCortex {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        assert ir.thsd_complexes, "No THSD complexes found in IR"
        assert len(ir.thsd_complexes) == 1
        assert ir.thsd_complexes[0].name == "LanguageCortex"

    def test_complex_stalk_definition(self):
        """Parse sheaf stalk with all required fields."""
        dsl = """
        complex LanguageCortex {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry",
                local_constraints: ["predictive_consistency", "gradient_stability"]
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]

        assert complex_ir.stalk.representation_dim == 512
        assert complex_ir.stalk.fisher_information_metric == "information_geometry"
        assert "predictive_consistency" in complex_ir.stalk.local_constraints
        assert "gradient_stability" in complex_ir.stalk.local_constraints

    def test_complex_topology_tonnetz(self):
        """Parse Tonnetz topology specification."""
        dsl = """
        complex LanguageCortex {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            topology {
                kind: "Tonnetz",
                spectral_gap: 0.3,
                dimension: 8,
                coherence_threshold: 0.95
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]

        assert complex_ir.topology.kind == "Tonnetz"
        assert complex_ir.topology.spectral_gap == 0.3
        assert complex_ir.topology.dimension == 8
        assert complex_ir.topology.coherence_threshold == 0.95

    def test_complex_formal_spec_cohomology(self):
        """Parse formal_spec with cohomological constraints."""
        dsl = """
        complex LanguageCortex {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            formal_spec {
                cohomology_floor: 0.01,
                phi_target: 0.8,
                phi_method: "geometric_IIT4"
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]

        assert complex_ir.formal_spec.cohomology_floor == 0.01
        assert complex_ir.formal_spec.phi_target == 0.8
        assert complex_ir.formal_spec.phi_method == "geometric_IIT4"

    def test_complex_formal_spec_information_bottleneck(self):
        """Parse information bottleneck (NEMORI) configuration."""
        dsl = """
        complex LanguageCortex {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            formal_spec {
                cohomology_floor: 0.01,
                phi_target: 0.8,
                information_bottleneck: {
                    enabled: true,
                    compression_ratio: 0.7,
                    prediction_lower_bound: 0.95
                }
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]

        assert complex_ir.formal_spec.information_bottleneck.enabled is True
        assert complex_ir.formal_spec.information_bottleneck.compression_ratio == 0.7
        assert complex_ir.formal_spec.information_bottleneck.prediction_lower_bound == 0.95

    def test_complex_dynamics_emission(self):
        """Parse dynamics emission kernel specification."""
        dsl = """
        complex LanguageCortex {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
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
        complex_ir = ir.thsd_complexes[0]

        assert complex_ir.dynamics.emission.trigger == "surprise_head(threshold=0.8)"
        assert complex_ir.dynamics.emission.payload_dim == 64
        assert complex_ir.dynamics.emission.lifetime_steps == 100

    def test_complex_dynamics_release(self):
        """Parse dynamics release operator specification."""
        dsl = """
        complex LanguageCortex {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            dynamics {
                release {
                    rule: "rank_one_update",
                    learning_rate: 0.001,
                    target: "parameter_counts"
                }
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]

        assert complex_ir.dynamics.release.rule == "rank_one_update"
        assert complex_ir.dynamics.release.learning_rate == 0.001
        assert complex_ir.dynamics.release.target == "parameter_counts"

    def test_complex_dynamics_nemori(self):
        """Parse NEMORI (predictive forgetting) configuration."""
        dsl = """
        complex LanguageCortex {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            dynamics {
                nemori {
                    enabled: true,
                    consolidation_interval: 1000,
                    forgetting_floor: 0.01
                }
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]

        assert complex_ir.dynamics.nemori.enabled is True
        assert complex_ir.dynamics.nemori.consolidation_interval == 1000
        assert complex_ir.dynamics.nemori.forgetting_floor == 0.01

    def test_complex_full_integration(self):
        """Parse complete complex block with all sections."""
        dsl = """
        complex LanguageCortex {
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
                },
                release {
                    rule: "rank_one_update",
                    learning_rate: 0.001,
                    target: "parameter_counts"
                },
                nemori {
                    enabled: true,
                    consolidation_interval: 1000,
                    forgetting_floor: 0.01
                }
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)

        assert len(ir.thsd_complexes) == 1
        complex_ir = ir.thsd_complexes[0]
        assert complex_ir.name == "LanguageCortex"
        assert complex_ir.stalk is not None
        assert complex_ir.topology is not None
        assert complex_ir.formal_spec is not None
        assert complex_ir.dynamics is not None


class TestMultipleComplexes:
    """Test parsing multiple complex blocks."""

    def test_multiple_complexes_in_dsl(self):
        """Parse multiple distinct complex blocks."""
        dsl = """
        complex LanguageCortex {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            }
        }
        complex ThalamicRelay {
            stalk {
                representation_dim: 256,
                fisher_information_metric: "information_geometry"
            }
        }
        complex PrefrontalCortex {
            stalk {
                representation_dim: 384,
                fisher_information_metric: "information_geometry"
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)

        assert len(ir.thsd_complexes) == 3
        names = [c.name for c in ir.thsd_complexes]
        assert "LanguageCortex" in names
        assert "ThalamicRelay" in names
        assert "PrefrontalCortex" in names


class TestComplexErrors:
    """Test error handling for malformed complex blocks."""

    def test_complex_missing_stalk_raises_error(self):
        """Missing stalk definition should raise error."""
        dsl = """
        complex LanguageCortex {
            topology {
                kind: "Tonnetz",
                spectral_gap: 0.3
            }
        }
        """
        with pytest.raises(ValueError, match="stalk.*required"):
            NeuroMLCompiler.compile(dsl)

    def test_complex_invalid_spectral_gap_raises_error(self):
        """Invalid spectral_gap value should raise error."""
        dsl = """
        complex LanguageCortex {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            topology {
                kind: "Tonnetz",
                spectral_gap: -0.5
            }
        }
        """
        with pytest.raises(ValueError, match="spectral_gap.*positive"):
            NeuroMLCompiler.compile(dsl)

    def test_complex_invalid_phi_target_raises_error(self):
        """Phi target outside [0, 1] should raise error."""
        dsl = """
        complex LanguageCortex {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            formal_spec {
                phi_target: 1.5
            }
        }
        """
        with pytest.raises(ValueError, match="phi_target.*\\[0, 1\\]"):
            NeuroMLCompiler.compile(dsl)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
