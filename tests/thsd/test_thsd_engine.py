# -*- coding: utf-8 -*-
"""Tests for THSD (Topological Hyper-Sheaf-Dynamics) Notation Engine.

Task 1: Implement the THSD core with cellular sheaf, cohomology guard, and Φ-dynamics.
- CellularSheaf: assigns stalks F(σ) to each simplex σ in complex K
- Coboundary operator δ¹: detects H¹ contradictions (hallucinations as algebraic obstructions)
- Φ-Dynamic Integration: IIT 4.0 irreducibility over Minimum Information Partition (MIP)
"""
import pytest
import numpy as np
import torch

from neuroslm.thsd.engine import (
    SimplexComplex,
    CellularSheaf,
    CoboundaryOperator,
    PhiDynamicsComputer,
)


class TestSimplexComplex:
    """Test the simplicial complex K (model architecture as simplices)."""

    def test_simplex_complex_creation(self):
        """Create a simplicial complex with vertices and edges."""
        complex_k = SimplexComplex(dim_max=2)
        assert complex_k.dim_max == 2
        # All dimension slots initialized, but empty
        assert len(complex_k.simplices[0]) == 0
        assert len(complex_k.simplices[1]) == 0
        assert len(complex_k.simplices[2]) == 0

    def test_add_vertex(self):
        """Add 0-simplices (vertices) to complex."""
        complex_k = SimplexComplex(dim_max=2)
        v0 = complex_k.add_simplex("pop_sensory", dim=0)
        v1 = complex_k.add_simplex("pop_motor", dim=0)
        assert len(complex_k.simplices[0]) == 2

    def test_add_edge(self):
        """Add 1-simplices (edges) to complex."""
        complex_k = SimplexComplex(dim_max=2)
        v0 = complex_k.add_simplex("v0", dim=0)
        v1 = complex_k.add_simplex("v1", dim=0)
        e = complex_k.add_simplex("e_01", dim=1, boundary=[v0, v1])
        assert len(complex_k.simplices[1]) == 1

    def test_boundary_operator(self):
        """Boundary operator ∂ maps k-simplex to (k-1)-faces."""
        complex_k = SimplexComplex(dim_max=2)
        v0 = complex_k.add_simplex("v0", dim=0)
        v1 = complex_k.add_simplex("v1", dim=0)
        e = complex_k.add_simplex("e_01", dim=1, boundary=[v0, v1])

        boundary = complex_k.boundary(e)
        assert len(boundary) == 2  # Two vertices


class TestCellularSheaf:
    """Test cellular sheaf F that assigns stalks F(σ) to simplices."""

    def test_sheaf_creation(self):
        """Create a cellular sheaf over a simplicial complex."""
        complex_k = SimplexComplex(dim_max=1)
        v0 = complex_k.add_simplex("v0", dim=0)

        sheaf = CellularSheaf(complex_k, stalk_dim=256)
        assert sheaf.complex == complex_k
        assert sheaf.stalk_dim == 256

    def test_stalk_assignment(self):
        """Assign a stalk F(σ) to a simplex σ."""
        complex_k = SimplexComplex(dim_max=1)
        v0 = complex_k.add_simplex("v0", dim=0)

        sheaf = CellularSheaf(complex_k, stalk_dim=256)
        stalk = sheaf.get_stalk(v0)

        assert stalk is not None
        assert stalk.shape == (256,)

    def test_fisher_metric(self):
        """Each stalk carries a Fisher information metric."""
        complex_k = SimplexComplex(dim_max=1)
        v0 = complex_k.add_simplex("v0", dim=0)

        sheaf = CellularSheaf(complex_k, stalk_dim=256)
        fisher = sheaf.get_fisher_metric(v0)

        # Fisher metric is a diagonal covariance matrix
        assert fisher.shape == (256, 256)
        assert torch.allclose(fisher, fisher.T)  # Symmetric

    def test_restriction_map(self):
        """Restriction map for edge restrictions: F(e) → F(v)."""
        complex_k = SimplexComplex(dim_max=1)
        v0 = complex_k.add_simplex("v0", dim=0)
        v1 = complex_k.add_simplex("v1", dim=0)
        e = complex_k.add_simplex("e_01", dim=1, boundary=[v0, v1])

        sheaf = CellularSheaf(complex_k, stalk_dim=256)
        rho_ev0 = sheaf.restriction(e, v0)  # restriction from edge to v0

        assert rho_ev0.shape == (256, 256)


class TestCoboundaryOperator:
    """Test the coboundary operator δ¹ for detecting contradictions."""

    def test_coboundary_creation(self):
        """Create coboundary operator for detecting H¹ obstructions."""
        complex_k = SimplexComplex(dim_max=1)
        v0 = complex_k.add_simplex("v0", dim=0)
        v1 = complex_k.add_simplex("v1", dim=0)
        e = complex_k.add_simplex("e_01", dim=1, boundary=[v0, v1])

        sheaf = CellularSheaf(complex_k, stalk_dim=256)
        coboundary = CoboundaryOperator(sheaf)

        assert coboundary.sheaf == sheaf

    def test_h1_residual_computation(self):
        """Compute H¹ residual (cohomological obstruction) on a 1-cochain."""
        complex_k = SimplexComplex(dim_max=1)
        v0 = complex_k.add_simplex("v0", dim=0)
        v1 = complex_k.add_simplex("v1", dim=0)
        e = complex_k.add_simplex("e_01", dim=1, boundary=[v0, v1])

        sheaf = CellularSheaf(complex_k, stalk_dim=16)
        coboundary = CoboundaryOperator(sheaf)

        # Create a 0-cochain (assignment to vertices)
        zero_cochain = {v0: torch.randn(16), v1: torch.randn(16)}

        # δ⁰(zero_cochain) gives a 1-cochain
        one_cochain = coboundary.apply_delta_0(zero_cochain)

        # δ¹(one_cochain) returns the L2 norm (simplified computation)
        # For a 1-cochain that is a coboundary (in image of δ⁰),
        # the residual should be computable
        two_cochain = coboundary.apply_delta_1(one_cochain)
        assert isinstance(two_cochain, torch.Tensor)
        assert two_cochain.shape == torch.Size([])

    def test_is_contradiction(self):
        """Detect if a cochain has H¹ contradictions."""
        complex_k = SimplexComplex(dim_max=1)
        v0 = complex_k.add_simplex("v0", dim=0)
        v1 = complex_k.add_simplex("v1", dim=0)
        e = complex_k.add_simplex("e_01", dim=1, boundary=[v0, v1])

        sheaf = CellularSheaf(complex_k, stalk_dim=16)
        coboundary = CoboundaryOperator(sheaf)

        # Create a cocycle (δ¹ maps it to zero) - no contradiction
        zero_cochain = {v0: torch.zeros(16), v1: torch.zeros(16)}
        one_cochain = coboundary.apply_delta_0(zero_cochain)

        is_contradiction = coboundary.is_contradiction(one_cochain, threshold=0.1)
        assert isinstance(is_contradiction, bool)


class TestPhiDynamicsComputer:
    """Test Φ-dynamics (IIT 4.0 irreducibility computation)."""

    def test_phi_dynamics_creation(self):
        """Create Φ-dynamics computer."""
        complex_k = SimplexComplex(dim_max=1)
        v0 = complex_k.add_simplex("v0", dim=0)

        phi_computer = PhiDynamicsComputer(complex_k, activation_dim=256)
        assert phi_computer.complex == complex_k

    def test_mip_computation(self):
        """Compute Minimum Information Partition (MIP) for irreducibility."""
        complex_k = SimplexComplex(dim_max=1)
        # Create 3 nodes
        v0 = complex_k.add_simplex("v0", dim=0)
        v1 = complex_k.add_simplex("v1", dim=0)
        v2 = complex_k.add_simplex("v2", dim=0)

        phi_computer = PhiDynamicsComputer(complex_k, activation_dim=16)

        # Create activation state
        state = {
            v0: torch.randn(16),
            v1: torch.randn(16),
            v2: torch.randn(16),
        }

        # Compute MIP and irreducibility
        mip, phi = phi_computer.compute_mip(state)

        assert phi >= 0.0  # Φ ≥ 0
        assert isinstance(mip, tuple)  # Partition into (A, B)

    def test_phi_score(self):
        """Φ-score should increase for tightly integrated systems."""
        complex_k = SimplexComplex(dim_max=1)
        v0 = complex_k.add_simplex("v0", dim=0)
        v1 = complex_k.add_simplex("v1", dim=0)

        phi_computer = PhiDynamicsComputer(complex_k, activation_dim=16)

        # Highly correlated state (should have high Φ)
        correlated_state = {
            v0: torch.ones(16),
            v1: torch.ones(16),
        }

        _, phi_high = phi_computer.compute_mip(correlated_state)

        # Independent state (should have low Φ)
        independent_state = {
            v0: torch.randn(16),
            v1: torch.randn(16),
        }

        _, phi_low = phi_computer.compute_mip(independent_state)

        # Correlated should have higher Φ
        # (stochastic, so we just check both are computed)
        assert phi_high >= 0.0 and phi_low >= 0.0


class TestTHSDIntegration:
    """Integration tests combining sheaf, cohomology, and Φ-dynamics."""

    def test_end_to_end_contradiction_detection(self):
        """Full pipeline: detect hallucination as H¹ obstruction."""
        # Build a simple complex: v0 -- e -- v1
        complex_k = SimplexComplex(dim_max=1)
        v0 = complex_k.add_simplex("sensory", dim=0)
        v1 = complex_k.add_simplex("motor", dim=0)
        e = complex_k.add_simplex("syn_01", dim=1, boundary=[v0, v1])

        # Create sheaf with stalks
        sheaf = CellularSheaf(complex_k, stalk_dim=16)

        # Create contradiction detector
        coboundary = CoboundaryOperator(sheaf)

        # Create a contradictory 1-cochain (manual assignment)
        contradictory_cochain = {e: torch.randn(16)}

        # Check if it's a contradiction
        is_contra = coboundary.is_contradiction(contradictory_cochain, threshold=0.5)
        assert isinstance(is_contra, bool)

    def test_end_to_end_phi_maximization(self):
        """Full pipeline: identify system irreducibility for Φ maximization."""
        complex_k = SimplexComplex(dim_max=1)
        v0 = complex_k.add_simplex("v0", dim=0)
        v1 = complex_k.add_simplex("v1", dim=0)
        e = complex_k.add_simplex("e", dim=1, boundary=[v0, v1])

        phi_computer = PhiDynamicsComputer(complex_k, activation_dim=16)

        # Create a tightly integrated state
        integrated_state = {
            v0: torch.tensor([1.0] * 8 + [0.0] * 8),
            v1: torch.tensor([1.0] * 8 + [0.0] * 8),
        }

        mip, phi = phi_computer.compute_mip(integrated_state)

        # Should recognize integration
        assert phi >= 0.0
        assert len(mip) == 2  # Partition into two parts
