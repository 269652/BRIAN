# -*- coding: utf-8 -*-
"""TDD Tests for THSD Evolutionary Integration (Phase 7)

Tests for mutation operators, DNA checkpointing, and fitness-driven evolution.
"""
import pytest
import json
import tempfile
from pathlib import Path
from neuroslm.dsl.compiler import NeuroMLCompiler
from neuroslm.dsl.hypergraph_ir import HypergraphBuilder
from neuroslm.dsl.thsd_evolution import (
    ThsdMutationOperator,
    ThgCheckpoint,
    FitnessEvaluator,
    EvolutionaryLoop,
)


class TestDNACheckpointing:
    """Test serialization and deserialization of architecture state."""

    def test_thg_checkpoint_create_from_hypergraph(self):
        """Create checkpoint from hypergraph IR."""
        dsl = """
        complex SimpleBrain {
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
                phi_target: 0.75
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        # Create checkpoint
        checkpoint = ThgCheckpoint.from_hypergraph(hypergraph, step=0)

        assert checkpoint is not None
        assert checkpoint.name == "SimpleBrain"
        assert checkpoint.step == 0
        assert len(checkpoint.nodes) > 0

    def test_thg_checkpoint_save_and_load(self):
        """Save and load checkpoint preserves structure."""
        dsl = """
        complex TestBrain {
            stalk {
                representation_dim: 64,
                fisher_information_metric: "information_geometry"
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        checkpoint1 = ThgCheckpoint.from_hypergraph(hypergraph, step=100)

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            checkpoint1.save(str(path))

            # Load
            checkpoint2 = ThgCheckpoint.load(str(path))

            assert checkpoint2.name == checkpoint1.name
            assert checkpoint2.step == checkpoint1.step
            assert len(checkpoint2.nodes) == len(checkpoint1.nodes)

    def test_thg_checkpoint_roundtrip_preserves_data(self):
        """Save and load preserves all node/edge data."""
        dsl = """
        complex RoundtripBrain {
            stalk {
                representation_dim: 256,
                fisher_information_metric: "information_geometry"
            },
            topology {
                kind: "Tonnetz",
                spectral_gap: 0.35,
                dimension: 6
            },
            formal_spec {
                phi_target: 0.8,
                cohomology_floor: 0.01
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        checkpoint1 = ThgCheckpoint.from_hypergraph(hypergraph, step=50)
        initial_node_count = len(checkpoint1.nodes)
        initial_edge_count = len(checkpoint1.edges)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "roundtrip.json"
            checkpoint1.save(str(path))
            checkpoint2 = ThgCheckpoint.load(str(path))

            assert len(checkpoint2.nodes) == initial_node_count
            assert len(checkpoint2.edges) == initial_edge_count


class TestMutationOperators:
    """Test structural mutations on hypergraph IR."""

    def test_add_node_mutation(self):
        """Add a new node to hypergraph."""
        dsl = """
        complex MutableBrain {
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

        checkpoint = ThgCheckpoint.from_hypergraph(hypergraph, step=0)
        initial_count = len(checkpoint.nodes)

        mutator = ThsdMutationOperator()
        mutator.add_node(checkpoint, dimension=0, stalk_dim=64)

        assert len(checkpoint.nodes) == initial_count + 1

    def test_rewire_edge_mutation(self):
        """Rewire an existing edge."""
        dsl = """
        complex RewirableBrain {
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

        checkpoint = ThgCheckpoint.from_hypergraph(hypergraph, step=0)

        if len(checkpoint.edges) > 0:
            first_edge_id = list(checkpoint.edges.keys())[0]
            first_edge = checkpoint.edges[first_edge_id]
            old_src = first_edge["src"]

            mutator = ThsdMutationOperator()
            mutator.modify_edge_weight(checkpoint, first_edge_id, new_weight=0.5)

            # Edge should still exist but weight changed
            assert first_edge_id in checkpoint.edges
            assert checkpoint.edges[first_edge_id]["weight"] == 0.5

    def test_parameter_mutation(self):
        """Mutate stalk parameters."""
        dsl = """
        complex ParamBrain {
            stalk {
                representation_dim: 128,
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

        checkpoint = ThgCheckpoint.from_hypergraph(hypergraph, step=0)
        old_spectral_gap = checkpoint.metadata.get("spectral_gap", 0.3)

        mutator = ThsdMutationOperator()
        mutator.mutate_spectral_gap(checkpoint, delta=0.05)

        new_spectral_gap = checkpoint.metadata.get("spectral_gap", 0.3)
        assert new_spectral_gap > old_spectral_gap


class TestFitnessEvaluation:
    """Test fitness evaluation from THSD constraints."""

    def test_fitness_evaluator_creates_score(self):
        """Fitness evaluator computes fitness from constraints."""
        dsl = """
        complex FitnessBrain {
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
                phi_target: 0.75,
                cohomology_floor: 0.01
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        checkpoint = ThgCheckpoint.from_hypergraph(hypergraph, step=0)

        evaluator = FitnessEvaluator()
        fitness = evaluator.evaluate(
            checkpoint,
            task_loss=0.5,
            phi_value=0.7,
            h1_norm=0.02,
            spectral_gap_value=0.32,
        )

        assert isinstance(fitness, dict)
        assert "total" in fitness
        assert "task_loss" in fitness
        assert "phi_violation" in fitness
        assert "cohomology_violation" in fitness

    def test_fitness_higher_when_constraints_satisfied(self):
        """Fitness should be higher when constraints are satisfied."""
        dsl = """
        complex ConstrainedBrain {
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
                phi_target: 0.8,
                cohomology_floor: 0.01
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        checkpoint = ThgCheckpoint.from_hypergraph(hypergraph, step=0)
        evaluator = FitnessEvaluator()

        # Good constraints
        fitness_good = evaluator.evaluate(
            checkpoint,
            task_loss=0.3,
            phi_value=0.8,  # Meets target
            h1_norm=0.005,  # Below floor
            spectral_gap_value=0.35,  # Exceeds minimum
        )

        # Bad constraints
        fitness_bad = evaluator.evaluate(
            checkpoint,
            task_loss=0.3,
            phi_value=0.5,  # Below target
            h1_norm=0.05,  # Above floor
            spectral_gap_value=0.2,  # Below minimum
        )

        assert fitness_good["total"] > fitness_bad["total"]


class TestEvolutionaryLoop:
    """Test the main evolutionary loop."""

    def test_evolutionary_loop_initializes(self):
        """Evolutionary loop can be initialized."""
        dsl = """
        complex EvolvingBrain {
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

        loop = EvolutionaryLoop(
            initial_hypergraph=hypergraph,
            mutation_rate=0.1,
            fitness_threshold=0.7,
        )

        assert loop is not None
        assert loop.generation == 0

    def test_evolutionary_loop_applies_mutations(self):
        """Loop applies mutations and tracks generations."""
        dsl = """
        complex EvolvingBrain {
            stalk {
                representation_dim: 128,
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

        loop = EvolutionaryLoop(
            initial_hypergraph=hypergraph,
            mutation_rate=0.5,  # High mutation rate for testing
            fitness_threshold=0.5,
        )

        # Simulate one generation
        fitness = {
            "total": 0.6,
            "task_loss": 0.4,
            "phi_violation": 0.05,
            "cohomology_violation": 0.01,
        }

        loop.step(fitness_metrics=fitness)

        assert loop.generation == 1

    def test_evolutionary_loop_saves_checkpoint(self):
        """Loop saves checkpoint at specified intervals."""
        dsl = """
        complex CheckpointBrain {
            stalk {
                representation_dim: 64,
                fisher_information_metric: "information_geometry"
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        with tempfile.TemporaryDirectory() as tmpdir:
            loop = EvolutionaryLoop(
                initial_hypergraph=hypergraph,
                mutation_rate=0.1,
                fitness_threshold=0.7,
                checkpoint_dir=tmpdir,
                checkpoint_interval=1,
            )

            fitness = {"total": 0.6}
            loop.step(fitness_metrics=fitness)

            # Check that checkpoint was saved
            checkpoint_files = list(Path(tmpdir).glob("*.json"))
            assert len(checkpoint_files) > 0


class TestIntegrationTrainingLoop:
    """Integration: evolution loop + training."""

    def test_training_loop_with_evolution(self):
        """Training loop applies mutations based on fitness."""
        dsl = """
        complex TrainingBrain {
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
                phi_target: 0.75
            }
        }
        """
        ir = NeuroMLCompiler.compile(dsl)
        complex_ir = ir.thsd_complexes[0]
        builder = HypergraphBuilder()
        hypergraph = builder.from_complex_ir(complex_ir)

        loop = EvolutionaryLoop(
            initial_hypergraph=hypergraph,
            mutation_rate=0.1,
            fitness_threshold=0.6,
        )

        # Simulate 3 training steps
        for step in range(3):
            # Training metrics
            fitness = {
                "total": 0.5 + (step * 0.1),  # Improving fitness
                "task_loss": 0.5 - (step * 0.05),
                "phi_violation": 0.05,
                "cohomology_violation": 0.01,
            }

            # Check if we should mutate
            should_mutate = fitness["total"] > loop.fitness_threshold

            if should_mutate and step > 0:
                loop.step(fitness_metrics=fitness)

        # Verify generation counter incremented
        assert loop.generation >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
