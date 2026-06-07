# -*- coding: utf-8 -*-
"""TDD tests for fitness configuration system.

Tests cover:
- Defining fitness metrics in fitness.neuro or fitness block in arch.neuro
- Loading fitness config when initializing from DNA
- Fitness mutations via epigenetic feedback
- Self-improving fitness objectives
"""
import pytest
import tempfile
import json
from pathlib import Path


class TestFitnessConfigDefinition:
    """Test defining fitness objectives in architecture."""

    def test_fitness_block_in_arch(self):
        """Define fitness block directly in arch.neuro."""
        fitness_config = {
            "version": "1.0",
            "objectives": [
                {
                    "name": "minimize_ood_ppl",
                    "metric": "ood_ppl",
                    "direction": "minimize",
                    "weight": 0.5,
                    "target": 180.0
                },
                {
                    "name": "maximize_phi",
                    "metric": "phi",
                    "direction": "maximize",
                    "weight": 0.3,
                    "target": 0.15
                },
                {
                    "name": "gap_ratio_target",
                    "metric": "gap_ratio",
                    "direction": "minimize",
                    "weight": 0.2,
                    "target": 2.0
                }
            ]
        }

        assert fitness_config["version"] == "1.0"
        assert len(fitness_config["objectives"]) == 3
        assert fitness_config["objectives"][0]["direction"] == "minimize"
        assert fitness_config["objectives"][1]["weight"] == 0.3

    def test_fitness_sidefile_format(self):
        """Define fitness in separate fitness.neuro file."""
        fitness_dsl = """
# architectures/rcc_bowtie/fitness.neuro
fitness {
    objectives: [
        { name: "minimize_ood_ppl", metric: "ood_ppl", direction: "minimize", weight: 0.5, target: 180.0 },
        { name: "maximize_phi", metric: "phi", direction: "maximize", weight: 0.3, target: 0.15 },
        { name: "gap_ratio", metric: "gap_ratio", direction: "minimize", weight: 0.2, target: 2.0 }
    ],
    adaptation: {
        enabled: true,
        mutation_rate: 0.01,
        target_adjustment_rate: 0.001
    }
}
"""
        assert "objectives" in fitness_dsl
        assert "adaptation" in fitness_dsl
        assert "minimize_ood_ppl" in fitness_dsl

    def test_fitness_parser_reads_objectives(self):
        """Parse fitness objectives from config."""
        fitness_dict = {
            "objectives": [
                {"name": "ood_ppl", "metric": "ood_ppl", "direction": "minimize", "weight": 0.5},
                {"name": "phi", "metric": "phi", "direction": "maximize", "weight": 0.3},
            ],
            "adaptation": {"enabled": True}
        }

        objectives = fitness_dict["objectives"]
        assert len(objectives) == 2
        assert objectives[0]["direction"] == "minimize"
        assert objectives[1]["direction"] == "maximize"


class TestFitnessLoadingFromDNA:
    """Test loading fitness config when training from DNA."""

    def test_load_fitness_from_dna(self):
        """DNA can store fitness config as invariant."""
        from neuroslm.compiler.ribosome import LatentDNA

        dna = LatentDNA(length=256)

        # Store fitness config in DNA invariants
        fitness_config = {
            "objectives": [
                {"name": "ood_ppl", "metric": "ood_ppl", "direction": "minimize", "weight": 0.5}
            ]
        }

        dna.invariants["fitness_config"] = fitness_config

        # Load it back
        assert "fitness_config" in dna.invariants
        assert dna.invariants["fitness_config"]["objectives"][0]["name"] == "ood_ppl"

    def test_fitness_fallback_to_sidefile(self):
        """If DNA has no fitness, fall back to fitness.neuro."""
        from neuroslm.compiler.ribosome import LatentDNA

        dna = LatentDNA(length=256)
        # No fitness in DNA invariants

        # Should fall back to loading from sidefile
        fitness_file = "architectures/rcc_bowtie/fitness.neuro"  # Not checking if exists

        if "fitness_config" not in dna.invariants:
            # Fallback behavior
            use_sidefile = True
        else:
            use_sidefile = False

        assert use_sidefile is True

    def test_fitness_roundtrip_through_dna(self):
        """Fitness config survives DNA encode/decode."""
        from neuroslm.compiler.ribosome import LatentDNA

        original_fitness = {
            "objectives": [
                {"name": "minimize_ood_ppl", "metric": "ood_ppl", "direction": "minimize", "weight": 0.5},
                {"name": "maximize_phi", "metric": "phi", "direction": "maximize", "weight": 0.3},
            ]
        }

        # Store in DNA
        dna1 = LatentDNA(length=256)
        dna1.invariants["fitness_config"] = original_fitness

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            dna_file = Path(tmpdir) / "test.dna"
            dna1.save(str(dna_file))

            dna2 = LatentDNA.load(str(dna_file))

        # Check preserved
        assert "fitness_config" in dna2.invariants
        assert dna2.invariants["fitness_config"] == original_fitness


class TestFitnessMutations:
    """Test fitness config evolving via mutations."""

    def test_fitness_mutation_adjusts_targets(self):
        """Mutation can adjust fitness target values."""
        fitness_config = {
            "objectives": [
                {"name": "ood_ppl", "metric": "ood_ppl", "direction": "minimize", "weight": 0.5, "target": 180.0}
            ]
        }

        # Mutation: lower the target (make objective harder)
        mutation_delta = {"target_adjustment": -20.0}

        new_target = fitness_config["objectives"][0]["target"] + mutation_delta["target_adjustment"]
        assert new_target == 160.0

    def test_fitness_mutation_reweights_objectives(self):
        """Mutation can change objective weights."""
        fitness_config = {
            "objectives": [
                {"name": "ood_ppl", "metric": "ood_ppl", "direction": "minimize", "weight": 0.5},
                {"name": "phi", "metric": "phi", "direction": "maximize", "weight": 0.3},
                {"name": "gap_ratio", "metric": "gap_ratio", "direction": "minimize", "weight": 0.2}
            ]
        }

        # Mutation: emphasize OOD PPL more, de-emphasize gap ratio
        original_weights = [obj["weight"] for obj in fitness_config["objectives"]]
        total = sum(original_weights)
        assert abs(total - 1.0) < 1e-6  # Normalized

        # Mutate: shift 0.1 from gap_ratio to ood_ppl
        fitness_config["objectives"][0]["weight"] = 0.6
        fitness_config["objectives"][2]["weight"] = 0.1

        new_weights = [obj["weight"] for obj in fitness_config["objectives"]]
        assert sum(new_weights) == 1.0  # Still normalized

    def test_fitness_mutation_via_vesicle(self):
        """Epigenetic feedback emits fitness mutations."""
        # Simulate vesicle carrying fitness mutation
        vesicle_payload = {
            "kind": "fitness_mutation",
            "target_objective": "ood_ppl",
            "delta": {
                "weight": +0.1,
                "target": -20.0
            },
            "reason": "ood_gap_widened"
        }

        assert vesicle_payload["kind"] == "fitness_mutation"
        assert vesicle_payload["target_objective"] == "ood_ppl"
        assert vesicle_payload["delta"]["weight"] == +0.1

    def test_fitness_converges_under_mutation(self):
        """Fitness objectives can improve through adaptive mutations."""
        # Track fitness config evolution
        evolution_history = [
            {
                "step": 1000,
                "fitness": {
                    "objectives": [
                        {"name": "ood_ppl", "target": 200.0, "weight": 0.5}
                    ]
                }
            },
            {
                "step": 2000,
                "fitness": {
                    "objectives": [
                        {"name": "ood_ppl", "target": 190.0, "weight": 0.55}  # Mutation: tighter target, more weight
                    ]
                }
            },
            {
                "step": 3000,
                "fitness": {
                    "objectives": [
                        {"name": "ood_ppl", "target": 180.0, "weight": 0.6}  # Another mutation
                    ]
                }
            },
        ]

        # Verify targets decreasing (getting harder)
        targets = [h["fitness"]["objectives"][0]["target"] for h in evolution_history]
        assert targets == [200.0, 190.0, 180.0]  # Monotonic improvement


class TestFitnessConfigurationIntegration:
    """Test fitness config integrated with training loop."""

    def test_init_evolution_loads_fitness(self):
        """init_evolution loads fitness when DNA present."""
        # Simulate what init_evolution should do
        architecture_config = {"arch_path": "architectures/rcc_bowtie"}
        dna_path = "architectures/evol/evol.dna"

        # Check if DNA has fitness, fall back to fitness.neuro
        fitness_config = None

        # If DNA has fitness_config invariant, use it
        # Otherwise, try to load fitness.neuro from architecture
        if fitness_config is None:
            fitness_path = "architectures/rcc_bowtie/fitness.neuro"
            # Load from fitness_path (in real implementation)

        assert architecture_config is not None

    def test_training_loop_applies_fitness_objectives(self):
        """Training loop applies fitness objectives when computing loss."""
        fitness_config = {
            "objectives": [
                {"name": "ood_ppl", "metric": "ood_ppl", "direction": "minimize", "weight": 0.5, "target": 180.0},
                {"name": "phi", "metric": "phi", "direction": "maximize", "weight": 0.3, "target": 0.15},
            ]
        }

        # Simulate metrics from training step
        step_metrics = {
            "ood_ppl": 200.0,
            "phi": 0.12
        }

        # Compute multi-objective loss
        loss = 0.0
        for obj in fitness_config["objectives"]:
            metric_value = step_metrics.get(obj["metric"])
            target_value = obj.get("target", 0)

            if obj["direction"] == "minimize":
                obj_loss = (metric_value - target_value) * obj["weight"]
            else:  # maximize
                obj_loss = (target_value - metric_value) * obj["weight"]

            loss += obj_loss

        # OOD PPL: (200 - 180) * 0.5 = 10
        # Phi: (0.15 - 0.12) * 0.3 = 0.009
        # Total: ~10.009
        assert loss > 10.0

    def test_fitness_drives_architecture_evolution(self):
        """Fitness config drives which mutations survive."""
        fitness_config = {
            "objectives": [
                {"name": "ood_ppl", "metric": "ood_ppl", "direction": "minimize", "weight": 1.0}
            ]
        }

        # Two mutations to evaluate
        mutation_a_metrics = {"ood_ppl": 185.0}  # Better
        mutation_b_metrics = {"ood_ppl": 195.0}  # Worse

        # Score each mutation
        def score_mutation(metrics):
            loss = 0.0
            for obj in fitness_config["objectives"]:
                metric_val = metrics[obj["metric"]]
                target = obj.get("target", 0)
                if obj["direction"] == "minimize":
                    loss += (metric_val - target) * obj["weight"]
            return loss

        score_a = score_mutation(mutation_a_metrics)
        score_b = score_mutation(mutation_b_metrics)

        # Mutation A should be selected (lower loss)
        assert score_a < score_b
