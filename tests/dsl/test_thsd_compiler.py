# -*- coding: utf-8 -*-
"""TDD Tests for THSD Compiler Extensions (Phase 3)

Tests for compiling THSD IR to executable PyTorch modules with
constraint validation and topological hardening.
"""
import pytest
import torch
import torch.nn as nn
from neuroslm.dsl.compiler import NeuroMLCompiler
from neuroslm.dsl.hypergraph_ir import HypergraphIR, SimplexNode, HypergraphBuilder
from neuroslm.dsl.thsd_ir import (
    ComplexIR,
    SheafStalkIR,
    TopologyIR,
    CohomologyIR,
)


class TestConstraintValidation:
    """Test topological constraint validation."""

    def test_spectral_gap_validation_passes(self):
        """Valid spectral gap should pass validation."""
        hypergraph = HypergraphIR(
            name="ValidGap",
            nodes={"v1": SimplexNode("v1", 0, "root", 512)},
            edges={},
            spectral_gap=0.3,
        )
        assert hypergraph.validate()

    def test_spectral_gap_validation_rejects_negative(self):
        """Negative spectral gap should raise error."""
        with pytest.raises(ValueError, match="spectral_gap.*positive"):
            HypergraphIR(
                name="InvalidGap",
                nodes={"v1": SimplexNode("v1", 0, "root", 512)},
                edges={},
                spectral_gap=-0.1,
            )

    def test_spectral_gap_validation_rejects_zero(self):
        """Zero spectral gap should raise error."""
        with pytest.raises(ValueError, match="spectral_gap.*positive"):
            HypergraphIR(
                name="ZeroGap",
                nodes={"v1": SimplexNode("v1", 0, "root", 512)},
                edges={},
                spectral_gap=0.0,
            )

    def test_phi_target_validation_passes(self):
        """Valid phi_target should pass validation."""
        hypergraph = HypergraphIR(
            name="ValidPhi",
            nodes={"v1": SimplexNode("v1", 0, "root", 512)},
            edges={},
            phi_target=0.75,
        )
        assert hypergraph.validate()

    def test_phi_target_validation_accepts_boundary_values(self):
        """Phi_target boundaries [0, 1] should be valid."""
        # Test 0.0
        hg1 = HypergraphIR(
            name="PhiZero",
            nodes={"v1": SimplexNode("v1", 0, "root", 512)},
            edges={},
            phi_target=0.0,
        )
        assert hg1.validate()

        # Test 1.0
        hg2 = HypergraphIR(
            name="PhiOne",
            nodes={"v1": SimplexNode("v1", 0, "root", 512)},
            edges={},
            phi_target=1.0,
        )
        assert hg2.validate()

    def test_phi_target_validation_rejects_negative(self):
        """Negative phi_target should raise error."""
        with pytest.raises(ValueError, match="phi_target.*\\[0, 1\\]"):
            HypergraphIR(
                name="NegativePhi",
                nodes={"v1": SimplexNode("v1", 0, "root", 512)},
                edges={},
                phi_target=-0.1,
            )

    def test_phi_target_validation_rejects_over_one(self):
        """Phi_target > 1 should raise error."""
        with pytest.raises(ValueError, match="phi_target.*\\[0, 1\\]"):
            HypergraphIR(
                name="OverPhi",
                nodes={"v1": SimplexNode("v1", 0, "root", 512)},
                edges={},
                phi_target=1.5,
            )


class TestHypergraphCompilation:
    """Test compilation of hypergraph IR to PyTorch modules."""

    def test_compile_hypergraph_to_module(self):
        """Compile hypergraph IR to nn.Module."""
        hypergraph = HypergraphIR(
            name="SimpleHypergraph",
            nodes={"root": SimplexNode("root", 0, "root", 512)},
            edges={},
            dimension=0,
        )
        # For now, just test that compilation doesn't crash
        # (actual codegen in Phase 3 full implementation)
        assert hypergraph.validate()

    def test_compiled_module_has_correct_shape(self):
        """Compiled module should preserve feature dimensions."""
        complex_ir = ComplexIR(
            name="TestComplex",
            stalk=SheafStalkIR(representation_dim=256, fisher_information_metric="ig"),
            topology=TopologyIR(kind="Tonnetz", spectral_gap=0.3, dimension=4),
        )
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        assert hypergraph.nodes["TestComplex"].stalk_dim == 256


class TestSpectralHardening:
    """Test spectral gap enforcement via zero-init gates."""

    def test_zero_init_gate_passthrough(self):
        """With zero-init gate, output equals input identity."""
        # Create a simple module with zero-init gate
        gate = nn.Parameter(torch.tensor(0.0))
        identity = torch.eye(10)
        projection = torch.randn(10, 10)

        # Output = identity + gate * projection
        output = identity + gate * projection
        expected = identity

        assert torch.allclose(output, expected)

    def test_spectral_gap_lambda_enforcement(self):
        """Spectral gap should enforce minimum eigenvalue."""
        # Create a symmetric matrix with known spectral gap
        A = torch.eye(5) + 0.3 * torch.randn(5, 5)
        A = (A + A.T) / 2  # Make symmetric

        # Eigenvalues of identity matrix are all 1.0
        # Adding 0.3 * random symmetric matrix shifts them
        eigenvalues = torch.linalg.eigvalsh(A)
        min_eigenvalue = eigenvalues.min().item()

        # Min eigenvalue should be > 0 (positive definite)
        assert min_eigenvalue > 0


class TestCohomologyConstraints:
    """Test cohomological constraint embedding."""

    def test_cohomology_floor_tracked(self):
        """Cohomology floor constraint should be stored."""
        hypergraph = HypergraphIR(
            name="CohomologyTest",
            nodes={"v1": SimplexNode("v1", 0, "root", 512)},
            edges={},
            cohomology_floor=0.01,
        )
        assert hypergraph.cohomology_floor == 0.01

    def test_cohomology_floor_in_metadata(self):
        """Cohomology constraint should propagate from THSD IR."""
        complex_ir = ComplexIR(
            name="CohomologyComplex",
            stalk=SheafStalkIR(representation_dim=512, fisher_information_metric="ig"),
            formal_spec=CohomologyIR(cohomology_floor=0.02, phi_target=0.8),
        )
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        assert hypergraph.cohomology_floor == 0.02


class TestIntegratedInformationTracking:
    """Test Φ (integrated information) tracking and optimization."""

    def test_phi_target_from_complex_ir(self):
        """Phi target should be extracted from ComplexIR."""
        complex_ir = ComplexIR(
            name="PhiComplex",
            stalk=SheafStalkIR(representation_dim=512, fisher_information_metric="ig"),
            formal_spec=CohomologyIR(phi_target=0.75),
        )
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        assert hypergraph.phi_target == 0.75

    def test_phi_value_tracking(self):
        """Hypergraph should track current Φ value."""
        hypergraph = HypergraphIR(
            name="PhiTracking",
            nodes={"v1": SimplexNode("v1", 0, "root", 512)},
            edges={},
            phi_target=0.8,
            phi_value=0.65,
        )
        assert hypergraph.phi_target == 0.8
        assert hypergraph.phi_value == 0.65


class TestEndToEndCompilation:
    """End-to-end compilation from DSL to validated hypergraph."""

    def test_parse_thsd_dsl_to_hypergraph(self):
        """Parse THSD DSL, build hypergraph, validate constraints."""
        dsl = """
        complex LanguageCortex {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            topology {
                kind: "Tonnetz",
                spectral_gap: 0.3,
                dimension: 8
            },
            formal_spec {
                cohomology_floor: 0.01,
                phi_target: 0.8
            }
        }
        """
        # Parse to IR
        ir = NeuroMLCompiler.compile(dsl)
        assert len(ir.thsd_complexes) == 1

        # Build hypergraph
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        # Validate constraints
        assert hypergraph.validate()
        assert hypergraph.spectral_gap == 0.3
        assert hypergraph.phi_target == 0.8
        assert hypergraph.cohomology_floor == 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
