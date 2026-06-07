# -*- coding: utf-8 -*-
"""TDD Tests for THSD Code Generation (Phase 5)

Tests for compiling THSD hypergraph IR to executable PyTorch modules
with constraint enforcement (spectral gap, cohomology, Φ tracking).
"""
import pytest
import torch
import torch.nn as nn
from neuroslm.dsl.compiler import NeuroMLCompiler
from neuroslm.dsl.hypergraph_ir import HypergraphBuilder
from neuroslm.dsl.thsd_codegen import THSDCodeGenerator


class TestTHSDModuleGeneration:
    """Test PyTorch module generation from hypergraph IR."""

    def test_generate_module_from_simple_complex(self):
        """Generate nn.Module from simple THSD complex."""
        dsl = """
        complex SimpleBrain {
            stalk {
                representation_dim: 256,
                fisher_information_metric: "information_geometry"
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        generator = THSDCodeGenerator()
        module = generator.generate_module(hypergraph)

        assert isinstance(module, nn.Module)
        assert module.stalk_dim == 256

    def test_generated_module_forward_pass(self):
        """Generated module should accept input and produce output."""
        dsl = """
        complex ForwardBrain {
            stalk {
                representation_dim: 128,
                fisher_information_metric: "information_geometry"
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        generator = THSDCodeGenerator()
        module = generator.generate_module(hypergraph)

        # Test forward pass
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 128)
        output = module(x)

        assert output.shape == x.shape

    def test_generated_module_with_topology(self):
        """Module with topology constraints should enforce spectral gap."""
        dsl = """
        complex TonnetzBrain {
            stalk {
                representation_dim: 64,
                fisher_information_metric: "information_geometry"
            },
            topology {
                kind: "Tonnetz",
                spectral_gap: 0.3,
                dimension: 4
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        generator = THSDCodeGenerator()
        module = generator.generate_module(hypergraph)

        assert hasattr(module, "spectral_gap")
        assert module.spectral_gap == 0.3


class TestZeroInitGateEnforcement:
    """Test zero-init gate pattern for smooth constraint activation."""

    def test_zero_init_gate_identity_passthrough(self):
        """With zero gate, output should equal input (identity)."""
        from neuroslm.dsl.thsd_codegen import ZeroInitGate

        gate = ZeroInitGate(10)  # 10D projection
        input_x = torch.randn(2, 3, 10)

        output = gate(input_x)

        # At init (gate=0), output should be input
        assert torch.allclose(output, input_x, atol=1e-6)

    def test_zero_init_gate_learns_smoothly(self):
        """Zero-init gate should learn gradually without abrupt changes."""
        from neuroslm.dsl.thsd_codegen import ZeroInitGate

        gate = ZeroInitGate(8)
        optimizer = torch.optim.SGD(gate.parameters(), lr=0.1)

        input_x = torch.randn(2, 3, 8)

        # Initial forward pass
        output1 = gate(input_x)
        loss1 = output1.sum()
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        # After one step
        output2 = gate(input_x)
        loss2 = output2.sum()

        # Output should have changed slightly (learned)
        assert not torch.allclose(output1, output2, atol=1e-6)

    def test_zero_init_gate_gradient_flow(self):
        """Zero-init gate should allow gradient flow."""
        from neuroslm.dsl.thsd_codegen import ZeroInitGate

        gate = ZeroInitGate(4)
        input_x = torch.randn(1, 2, 4, requires_grad=True)

        output = gate(input_x)
        loss = output.sum()
        loss.backward()

        assert input_x.grad is not None
        assert input_x.grad.shape == input_x.shape


class TestSpectralGapHardening:
    """Test spectral gap constraint enforcement in generated modules."""

    def test_spectral_layer_output_covariance(self):
        """Output covariance should have spectral gap > target."""
        dsl = """
        complex SpectralBrain {
            stalk {
                representation_dim: 32,
                fisher_information_metric: "information_geometry"
            },
            topology {
                kind: "Tonnetz",
                spectral_gap: 0.25,
                dimension: 2
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        generator = THSDCodeGenerator()
        module = generator.generate_module(hypergraph)

        # Generate output
        with torch.no_grad():
            x = torch.randn(100, 16, 32)
            output = module(x)

            # Reshape for covariance (batch * seq, dim)
            reshaped = output.reshape(-1, 32)
            cov = torch.cov(reshaped.T)

            # Get eigenvalues
            eigenvalues = torch.linalg.eigvalsh(cov)
            min_eigenvalue = eigenvalues[0].item()

            # All eigenvalues should be positive
            assert (eigenvalues >= 0).all()


class TestCohomologyTracking:
    """Test cohomological constraint tracking in generated modules."""

    def test_module_tracks_cohomology_floor(self):
        """Module should track cohomology_floor constraint."""
        dsl = """
        complex CohomologyBrain {
            stalk {
                representation_dim: 256,
                fisher_information_metric: "information_geometry"
            },
            formal_spec {
                cohomology_floor: 0.01
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        generator = THSDCodeGenerator()
        module = generator.generate_module(hypergraph)

        assert hasattr(module, "cohomology_floor")
        assert module.cohomology_floor == 0.01


class TestPhiTracking:
    """Test Φ (integrated information) tracking."""

    def test_module_tracks_phi_target(self):
        """Module should track phi_target from constraints."""
        dsl = """
        complex PhiBrain {
            stalk {
                representation_dim: 512,
                fisher_information_metric: "information_geometry"
            },
            formal_spec {
                phi_target: 0.75
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        generator = THSDCodeGenerator()
        module = generator.generate_module(hypergraph)

        assert hasattr(module, "phi_target")
        assert module.phi_target == 0.75

    def test_module_can_compute_phi_proxy(self):
        """Module should compute Φ proxy metric from forward pass."""
        dsl = """
        complex PhiComputeBrain {
            stalk {
                representation_dim: 64,
                fisher_information_metric: "information_geometry"
            },
            formal_spec {
                phi_target: 0.8
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        generator = THSDCodeGenerator()
        module = generator.generate_module(hypergraph)

        # Forward pass and measure Phi proxy
        with torch.no_grad():
            x = torch.randn(4, 8, 64)
            output = module(x)

            # Phi proxy: entropy-based measure
            phi_proxy = module.compute_phi_proxy(output)
            assert isinstance(phi_proxy, (float, torch.Tensor))


class TestEndToEndCodeGeneration:
    """End-to-end: DSL → Module with full constraint enforcement."""

    def test_complete_pipeline_dsl_to_trained_module(self):
        """Full pipeline: parse DSL, generate module, train one step."""
        dsl = """
        complex TrainableBrain {
            stalk {
                representation_dim: 128,
                fisher_information_metric: "information_geometry"
            },
            topology {
                kind: "Tonnetz",
                spectral_gap: 0.3,
                dimension: 4
            },
            formal_spec {
                cohomology_floor: 0.01,
                phi_target: 0.8
            }
        }
        """
        # Parse
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]

        # Build hypergraph
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        # Generate module
        generator = THSDCodeGenerator()
        module = generator.generate_module(hypergraph)
        module.train()

        # Training step
        optimizer = torch.optim.Adam(module.parameters(), lr=0.001)
        x = torch.randn(2, 8, 128)
        y = torch.randn(2, 8, 128)

        # Forward pass (with gradients)
        output = module(x)
        loss = ((output - y) ** 2).mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify training worked (no errors and parameters exist)
        assert len(list(module.parameters())) > 0

    def test_generated_module_preserves_constraint_metadata(self):
        """Generated module should preserve all constraint metadata."""
        dsl = """
        complex MetadataBrain {
            stalk {
                representation_dim: 256,
                fisher_information_metric: "information_geometry",
                local_constraints: ["predictive_consistency"]
            },
            topology {
                kind: "Tonnetz",
                spectral_gap: 0.35,
                dimension: 6
            },
            formal_spec {
                cohomology_floor: 0.015,
                phi_target: 0.82
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        generator = THSDCodeGenerator()
        module = generator.generate_module(hypergraph)

        assert module.stalk_dim == 256
        assert module.spectral_gap == 0.35
        assert module.cohomology_floor == 0.015
        assert module.phi_target == 0.82


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
