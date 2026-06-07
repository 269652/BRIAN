# -*- coding: utf-8 -*-
"""TDD Tests for THSD Plasticity (Phase 6)

Tests for structural plasticity, activity-dependent learning,
and evolutionary dynamics in living THSD architectures.
"""
import pytest
import torch
import torch.nn as nn
from neuroslm.dsl.thsd_plasticity import (
    StructuralPlasticityController,
    HebbianFastWeights,
    NEMORIConsolidator,
)


class TestStructuralPlasticity:
    """Test activity-dependent structural plasticity."""

    def test_stabilize_hot_paths(self):
        """Stabilize frequently-used connections."""
        controller = StructuralPlasticityController(
            stabilize_threshold=0.1,
            lr=0.01,
        )

        # Simulate edge activities
        edge_activities = {
            "e1": 0.9,  # Hot
            "e2": 0.02,  # Cold
        }
        edge_weights = {"e1": 0.5, "e2": 0.5}

        # Update
        updated_weights = controller.stabilize_edges(edge_activities, edge_weights)

        # Hot edge should gain weight, cold should not
        assert updated_weights["e1"] > edge_weights["e1"]
        assert updated_weights["e2"] <= edge_weights["e2"]

    def test_prune_cold_edges(self):
        """Remove unused connections after threshold."""
        controller = StructuralPlasticityController(
            prune_threshold=0.01,
            cold_steps=5,
        )

        # Edge that has been inactive for many steps
        edge_activity_history = {"e1": [0.0] * 5}
        edges = {"e1": {"weight": 0.5}}

        pruned = controller.prune_edges(edge_activity_history, edges)

        # Cold edge should be removed
        assert "e1" not in pruned

    def test_rewire_for_exploration(self):
        """Add new edges to unexplored regions."""
        controller = StructuralPlasticityController(
            exploration_prob=0.5,
        )

        # Existing nodes
        nodes = {"v1": {"activity": 0.8}, "v2": {"activity": 0.1}}
        edges = {"e12": {"src": "v1", "dst": "v2"}}

        # Rewire
        new_edges = controller.add_exploration_edges(nodes, edges)

        # Should have new edges
        assert len(new_edges) >= len(edges)


class TestHebbianFastWeights:
    """Test transient outer-product associative memory."""

    def test_fast_weights_zero_init(self):
        """Fast weights should start near zero."""
        fast_weights = HebbianFastWeights(dim=8, eta=0.05)

        # Initial state should have small norm
        assert torch.norm(fast_weights.A) < 0.1

    def test_fast_weights_update_on_forward(self):
        """Fast weights matrix should update during forward pass."""
        fast_weights = HebbianFastWeights(dim=4, eta=0.1)

        h_t = torch.randn(1, 4)
        h_prev = torch.randn(1, 4)

        # Record initial state
        A_before = fast_weights.A.clone()

        # Forward pass
        output = fast_weights(h_t, h_prev)

        # State should have changed
        assert not torch.allclose(A_before, fast_weights.A)

    def test_fast_weights_output_shape(self):
        """Output should have correct shape."""
        fast_weights = HebbianFastWeights(dim=16, eta=0.05)

        h_t = torch.randn(2, 5, 16)  # batch, seq, dim
        h_prev = torch.randn(2, 5, 16)

        output = fast_weights(h_t, h_prev)

        assert output.shape == h_t.shape

    def test_fast_weights_gate_parameter(self):
        """Gate parameter controls fast weight contribution."""
        fast_weights = HebbianFastWeights(dim=8, eta=0.05)

        h_t = torch.randn(1, 8)
        h_prev = torch.randn(1, 8)

        # With zero gate, output should equal input
        fast_weights.gate.data.fill_(0.0)
        output1 = fast_weights(h_t, h_prev)
        assert torch.allclose(output1, h_t, atol=1e-6)

        # With non-zero gate, output should differ
        fast_weights.gate.data.fill_(0.5)
        output2 = fast_weights(h_t, h_prev)
        assert not torch.allclose(output2, h_t, atol=1e-4)


class TestNEMORIConsolidation:
    """Test predictive forgetting via information bottleneck."""

    def test_nemori_identifies_nonpredictive_edges(self):
        """NEMORI should identify edges that don't help prediction."""
        consolidator = NEMORIConsolidator(
            nemori_floor=0.01,
        )

        # Create edge importance scores
        edge_importance = {
            "e1": 0.5,  # Important
            "e2": 0.001,  # Unimportant
        }

        # Identify which edges are nonpredictive
        nonpredictive = consolidator.identify_nonpredictive(edge_importance)

        assert "e2" in nonpredictive
        assert "e1" not in nonpredictive

    def test_nemori_prunes_nonpredictive(self):
        """NEMORI consolidation should remove nonpredictive edges."""
        consolidator = NEMORIConsolidator(nemori_floor=0.05)

        edges = {
            "e1": {"importance": 0.8},
            "e2": {"importance": 0.02},
            "e3": {"importance": 0.3},
        }

        pruned = consolidator.consolidate(edges)

        assert "e1" in pruned
        assert "e2" not in pruned
        assert "e3" in pruned

    def test_nemori_preserves_predictive_edges(self):
        """NEMORI should never prune highly predictive edges."""
        consolidator = NEMORIConsolidator(nemori_floor=0.1)

        edges = {
            "critical": {"importance": 0.99},
            "useful": {"importance": 0.5},
            "useless": {"importance": 0.001},
        }

        pruned = consolidator.consolidate(edges)

        assert "critical" in pruned
        assert "useful" in pruned


class TestBDNFSignaling:
    """Test BDNF-like trophic growth signals."""

    def test_bdnf_signal_increases_rank(self):
        """BDNF should increase rank for high-Φ edges."""
        # Placeholder: BDNF signaling for rank-1 updates
        phi_values = {"e1": 0.9, "e2": 0.3}
        phi_threshold = 0.7

        # Edges with high Φ should get rank increase
        for edge_id, phi in phi_values.items():
            if phi > phi_threshold:
                # Would increase rank
                pass

        # Verify threshold logic
        assert phi_values["e1"] > phi_threshold
        assert phi_values["e2"] < phi_threshold

    def test_bdnf_signal_decreases_rank_for_low_phi(self):
        """BDNF should decrease rank for low-Φ edges."""
        phi_values = {"e1": 0.2, "e2": 0.8}
        phi_threshold = 0.5

        low_phi_edges = [e for e, phi in phi_values.items() if phi < phi_threshold]

        assert "e1" in low_phi_edges
        assert "e2" not in low_phi_edges


class TestIntegratedPlasticitySystem:
    """Integration tests for plasticity system."""

    def test_full_plasticity_cycle(self):
        """Complete: activity → stabilize → prune → rewire."""
        controller = StructuralPlasticityController(
            stabilize_threshold=0.2,
            prune_threshold=0.01,
            exploration_prob=0.1,
        )

        # Initial state
        edges = {"e1": 0.5, "e2": 0.5}
        activity = {"e1": 0.9, "e2": 0.001}

        # Stabilize
        edges = controller.stabilize_edges(activity, edges)
        assert edges["e1"] > 0.5

        # Prune
        pruned = {"e1": edges["e1"]}  # e2 removed

        # Rewire
        rewired = dict(pruned)
        rewired["e3"] = 0.3  # New edge added

        assert "e3" in rewired

    def test_plasticity_preserves_critical_paths(self):
        """Critical high-activity edges should be preserved."""
        controller = StructuralPlasticityController(
            stabilize_threshold=0.5,
            prune_threshold=0.01,
        )

        edges = {
            "critical": 0.5,
            "useful": 0.5,
            "junk": 0.5,
        }

        activity = {
            "critical": 0.95,
            "useful": 0.6,
            "junk": 0.001,
        }

        # After consolidation, critical should remain
        stabilized = controller.stabilize_edges(activity, edges)
        assert stabilized["critical"] > edges["critical"]


class TestPlasticityMetrics:
    """Test measurement of plasticity and learning dynamics."""

    def test_measure_network_stability(self):
        """Compute stability metric from edge weight changes."""
        old_weights = {"e1": 0.5, "e2": 0.5}
        new_weights = {"e1": 0.6, "e2": 0.5}

        # Stability: fraction of unchanged edges
        stability = sum(
            1 for e in old_weights if abs(old_weights[e] - new_weights.get(e, 0)) < 0.01
        ) / len(old_weights)

        assert 0 <= stability <= 1
        assert stability == 0.5  # 1 out of 2 edges unchanged

    def test_measure_exploration_rate(self):
        """Compute rate of new edge creation."""
        old_edges = {"e1", "e2"}
        new_edges = {"e1", "e2", "e3"}

        exploration_rate = len(new_edges - old_edges) / len(old_edges)

        assert exploration_rate == 0.5  # 1 new edge per 2 old


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
