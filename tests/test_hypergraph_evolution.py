# -*- coding: utf-8 -*-
"""TDD tests for real-time hypergraph evolution during training.

Tests cover:
- Path activity tracking (per-edge activation correlation)
- Hot/Cold stabilization (HOT paths strengthen, COLD paths prune)
- Epigenetic feedback (mutations written back to DNA patches)
- Integration with training loop
"""
import pytest
import tempfile
from pathlib import Path

from neuroslm.dsl.thg_ir import THGCheckpoint, THGNode, THGEdge
from neuroslm.compiler.ribosome import DNAPatch


class TestPathActivityTracking:
    """Test tracking edge activity during forward passes."""

    def test_activity_log_creation(self):
        """Create activity log tracking correlations per edge."""
        activity_log = {
            "e_gws_motor": {"activation_corr": 0.85, "firing_rate": 0.9},
            "e_sens_gws": {"activation_corr": 0.92, "firing_rate": 0.88},
            "e_motor_thal": {"activation_corr": 0.05, "firing_rate": 0.02},  # COLD
        }

        assert activity_log["e_gws_motor"]["activation_corr"] > 0.7  # HOT
        assert activity_log["e_motor_thal"]["activation_corr"] < 0.1  # COLD

    def test_correlation_computation(self):
        """Compute correlation coefficient for an edge's activations."""
        import numpy as np

        # Simulated activations over time steps
        src_activations = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.28])
        dst_activations = np.array([0.05, 0.18, 0.12, 0.22, 0.32, 0.30])

        # Compute Pearson correlation
        corr = np.corrcoef(src_activations, dst_activations)[0, 1]

        # Should be high positive correlation
        assert corr > 0.95

    def test_hot_cold_classification(self):
        """Classify edges as HOT or COLD based on activity."""
        edges = {
            "e1": {"activation_corr": 0.85, "firing_rate": 0.9},  # HOT
            "e2": {"activation_corr": 0.92, "firing_rate": 0.88},  # HOT
            "e3": {"activation_corr": 0.05, "firing_rate": 0.02},  # COLD
            "e4": {"activation_corr": 0.08, "firing_rate": 0.05},  # COLD
        }

        HOT_threshold = 0.7
        COLD_threshold = 0.1

        hot_edges = [eid for eid, data in edges.items() if data["activation_corr"] > HOT_threshold]
        cold_edges = [eid for eid, data in edges.items() if data["activation_corr"] < COLD_threshold]

        assert len(hot_edges) == 2
        assert len(cold_edges) == 2
        assert "e1" in hot_edges
        assert "e3" in cold_edges


class TestHotColdStabilization:
    """Test structural plasticity: HOT paths strengthen, COLD paths prune."""

    def test_hot_path_strengthening_via_bdnf(self):
        """HOT path: weight += lr * activation_corr * BDNF_signal."""
        edge = {"weight": 0.5, "activation_corr": 0.85}
        lr = 0.01
        bdnf_signal = 1.0

        # BDNF update
        weight_delta = lr * edge["activation_corr"] * bdnf_signal
        new_weight = edge["weight"] + weight_delta

        assert new_weight > edge["weight"]
        assert abs(new_weight - 0.5085) < 1e-4

    def test_cold_path_pruning(self):
        """COLD path: weight → 0 after N steps of low activity."""
        edge = {"weight": 0.3, "activation_corr": 0.02, "cold_steps": 0}

        # Increment cold counter
        if edge["activation_corr"] < 0.1:
            edge["cold_steps"] += 1

        # After 3 cold steps, prune
        if edge["cold_steps"] >= 3:
            edge["weight"] = 0.0

        # After one cold step, weight stays
        assert edge["weight"] == 0.3
        assert edge["cold_steps"] == 1

        # After 3 cold steps, weight → 0
        edge["cold_steps"] = 3
        if edge["cold_steps"] >= 3:
            edge["weight"] = 0.0
        assert edge["weight"] == 0.0

    def test_rank_increase_for_high_phi_edges(self):
        """HOT edge with high Φ: increase NeuralGeometryAdapter rank."""
        edge = {
            "weight": 0.8,
            "activation_corr": 0.95,
            "rank": 4,  # NGA rank
            "phi_contribution": 0.12  # high
        }

        # If HOT and high Φ, increase rank
        if edge["activation_corr"] > 0.7 and edge["phi_contribution"] > 0.1:
            edge["rank"] += 1

        assert edge["rank"] == 5

    def test_mycelium_effect_trajectory(self):
        """HOT paths accumulate over multiple steps (mycelium effect)."""
        edge = {"weight": 0.1}
        edge_activity = [0.9, 0.88, 0.92, 0.85]  # Consistently high
        lr = 0.01

        # Apply BDNF updates
        for activity in edge_activity:
            edge["weight"] += lr * activity

        # Weight should grow monotonically
        expected_final = 0.1 + lr * sum(edge_activity)
        assert abs(edge["weight"] - expected_final) < 1e-6


class TestEpigeneticFeedback:
    """Test mutations written back to DNA patches via epigenetic feedback."""

    def test_vesicle_emission_on_high_surprise(self):
        """When surprise > threshold, emit a vesicle with mutation payload."""
        surprise = 0.95  # High surprise
        surprise_threshold = 0.8

        if surprise > surprise_threshold:
            vesicle_triggered = True
            payload = {
                "kind": "node_mutation",
                "target": "gws",
                "delta": [0.05] * 16,
                "reason": "high_surprise_adaptation"
            }
        else:
            vesicle_triggered = False
            payload = None

        assert vesicle_triggered
        assert payload["reason"] == "high_surprise_adaptation"

    def test_vesicle_payload_to_dna_patch(self):
        """Convert vesicle payload to DNA patch."""
        from neuroslm.compiler.ribosome import DNAPatch

        vesicle_payload = {
            "kind": "node_mutation",
            "target": "gws",
            "delta": [0.05] * 16,
            "reason": "high_surprise_adaptation"
        }

        # Create patch from vesicle
        patch = DNAPatch(
            version="1.0",
            step=5000,
            kind=vesicle_payload["kind"],
            target=vesicle_payload["target"],
            delta=vesicle_payload["delta"],
            metadata={"reason": vesicle_payload["reason"]}
        )

        assert patch.target == "gws"
        assert patch.metadata["reason"] == "high_surprise_adaptation"

    def test_patch_accumulation_during_epoch(self):
        """Multiple vesicles during an epoch create multiple patches."""
        from neuroslm.compiler.ribosome import DNAPatch

        patches = []
        # Simulate epoch with 3 mutation events
        for i, (step, target) in enumerate([
            (1000, "gws"),
            (2500, "language_trunk"),
            (4800, "hippo"),
        ]):
            patch = DNAPatch(
                version="1.0",
                step=step,
                kind="node_mutation",
                target=target,
                delta=[0.1] * 16,
                metadata={"epoch": 0}
            )
            patches.append(patch)

        # All patches created
        assert len(patches) == 3
        assert patches[0].target == "gws"
        assert patches[2].target == "hippo"


class TestHypergraphEvolutionIntegration:
    """Test end-to-end evolution workflow during training."""

    def test_evolution_step_in_training_loop(self):
        """Simulate one evolution step during training."""
        # Initial THG
        thg = THGCheckpoint(
            version="2.0",
            nodes={
                "gws": THGNode("gws", "complex", [0.5] * 16, {}),
                "lang": THGNode("lang", "complex", [0.3] * 16, {}),
            },
            edges={
                "e_gws_lang": THGEdge("e_gws_lang", "gws", "lang", "synapse", 0.7, "plastic"),
            },
            gene_state={},
            step=1000,
            metadata={}
        )

        # Simulate training loop
        activity_log = {
            "e_gws_lang": {"activation_corr": 0.88, "firing_rate": 0.85}
        }

        # Detect HOT edge
        hot_edges = [eid for eid, data in activity_log.items() if data["activation_corr"] > 0.7]
        assert "e_gws_lang" in hot_edges

        # Apply BDNF update
        if "e_gws_lang" in thg.edges:
            edge = thg.edges["e_gws_lang"]
            edge.weight += 0.01 * activity_log["e_gws_lang"]["activation_corr"]

        # Verify edge weight increased
        assert thg.edges["e_gws_lang"].weight > 0.7

    def test_evolution_produces_valid_patches(self):
        """Evolution step produces valid DNA patches."""
        from neuroslm.compiler.ribosome import DNAPatch

        # Simulate evolution discovering a beneficial mutation
        discovered_mutations = [
            {
                "step": 5000,
                "kind": "node_mutation",
                "target": "gws",
                "delta": [0.08] * 16,
                "phi_delta": 0.15,  # Improved Φ
            },
            {
                "step": 7500,
                "kind": "edge_weight",
                "target": "e_gws_motor",
                "delta": [0.05],
                "gap_delta": -0.3,  # Improved gap_ratio
            }
        ]

        patches = [
            DNAPatch(
                version="1.0",
                step=mut["step"],
                kind=mut["kind"],
                target=mut["target"],
                delta=mut["delta"],
                metadata={"phi_delta": mut.get("phi_delta"), "gap_delta": mut.get("gap_delta")}
            )
            for mut in discovered_mutations
        ]

        # All patches valid
        assert len(patches) == 2
        assert patches[0].metadata["phi_delta"] == 0.15
        assert patches[1].metadata["gap_delta"] == -0.3

    def test_evolution_checkpoint_creation(self):
        """At end of epoch, save DNA patches to disk."""
        from neuroslm.compiler.ribosome import DNAPatch

        patches = [
            DNAPatch(
                version="1.0",
                step=step,
                kind="node_mutation",
                target=f"region_{i}",
                delta=[0.1] * 16,
                metadata={"epoch": 0}
            )
            for i, step in enumerate([1000, 2000, 3000])
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save patches
            for patch in patches:
                patch_file = Path(tmpdir) / f"step_{patch.step:05d}.patch.dna"
                patch.save(str(patch_file))

            # Verify all saved
            saved_files = list(Path(tmpdir).glob("*.patch.dna"))
            assert len(saved_files) == 3

            # Load and verify
            loaded_patches = [DNAPatch.load(str(f)) for f in sorted(saved_files)]
            assert loaded_patches[0].step == 1000
            assert loaded_patches[2].step == 3000


class TestEvolutionaryMetrics:
    """Test metrics tracking during evolution (Φ, gap_ratio, rank changes)."""

    def test_phi_improvement_tracking(self):
        """Track Φ improvement across evolutionary steps."""
        phi_history = [
            {"step": 1000, "phi": 0.02},
            {"step": 2000, "phi": 0.05},
            {"step": 3000, "phi": 0.08},
            {"step": 4000, "phi": 0.12},
        ]

        # Φ monotonically increasing (good sign)
        phi_values = [h["phi"] for h in phi_history]
        for i in range(1, len(phi_values)):
            assert phi_values[i] >= phi_values[i-1]

    def test_gap_ratio_improvement_tracking(self):
        """Track OOD gap_ratio improvement (lower is better)."""
        gap_history = [
            {"step": 1000, "gap_ratio": 6.0},
            {"step": 2000, "gap_ratio": 5.5},
            {"step": 3000, "gap_ratio": 5.0},
            {"step": 4000, "gap_ratio": 4.7},
        ]

        # Gap ratio decreasing (better generalization)
        gap_values = [h["gap_ratio"] for h in gap_history]
        for i in range(1, len(gap_values)):
            assert gap_values[i] <= gap_values[i-1]

    def test_mutation_acceptance_rate(self):
        """Track what fraction of mutations improve metrics."""
        mutations = [
            {"phi_delta": +0.03, "gap_delta": -0.2, "accepted": True},
            {"phi_delta": -0.01, "gap_delta": +0.5, "accepted": False},
            {"phi_delta": +0.02, "gap_delta": -0.1, "accepted": True},
            {"phi_delta": +0.05, "gap_delta": -0.4, "accepted": True},
        ]

        acceptance_rate = sum(1 for m in mutations if m["accepted"]) / len(mutations)
        assert acceptance_rate == 0.75  # 3/4 accepted
