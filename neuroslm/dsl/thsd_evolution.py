# -*- coding: utf-8 -*-
"""THSD Evolutionary Integration (Phase 7)

DNA checkpointing, mutation operators, and fitness-driven evolution loop.
"""
from __future__ import annotations
import json
import torch
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from neuroslm.dsl.hypergraph_ir import HypergraphIR


@dataclass
class ThgCheckpoint:
    """Topological Hypergraph Checkpoint — serializable architecture state.

    Stores: nodes, edges, metadata, step counter. Can be saved/loaded as JSON.
    """

    name: str
    step: int
    nodes: Dict[str, Dict[str, Any]]
    edges: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]

    @staticmethod
    def from_hypergraph(
        hypergraph: HypergraphIR, step: int = 0
    ) -> ThgCheckpoint:
        """Create checkpoint from hypergraph IR.

        Args:
            hypergraph: HypergraphIR instance
            step: Evolutionary step number

        Returns:
            ThgCheckpoint with all topology and metadata
        """
        nodes = {}
        for node_id, node in hypergraph.nodes.items():
            nodes[node_id] = {
                "id": node.id,
                "dimension": node.dimension,
                "name": node.name,
                "stalk_dim": node.stalk_dim,
                "metadata": node.metadata or {},
            }

        edges = {}
        for edge_id, edge in hypergraph.edges.items():
            edges[edge_id] = {
                "id": edge.id,
                "src": edge.src_simplex,
                "dst": edge.dst_simplex,
                "kind": edge.kind,
                "weight": float(edge.weight) if isinstance(edge.weight, (int, float, torch.Tensor)) else edge.weight,
                "metadata": edge.metadata or {},
            }

        metadata = {
            "name": hypergraph.name,
            "dimension": hypergraph.dimension,
            "spectral_gap": float(hypergraph.spectral_gap) if hypergraph.spectral_gap else None,
            "phi_target": float(hypergraph.phi_target) if hypergraph.phi_target else None,
            "cohomology_floor": float(hypergraph.cohomology_floor) if hypergraph.cohomology_floor else None,
        }

        return ThgCheckpoint(
            name=hypergraph.name,
            step=step,
            nodes=nodes,
            edges=edges,
            metadata=metadata,
        )

    def save(self, path: str) -> None:
        """Save checkpoint to JSON file.

        Args:
            path: File path to save to
        """
        data = {
            "name": self.name,
            "step": self.step,
            "nodes": self.nodes,
            "edges": self.edges,
            "metadata": self.metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> ThgCheckpoint:
        """Load checkpoint from JSON file.

        Args:
            path: File path to load from

        Returns:
            Loaded ThgCheckpoint
        """
        with open(path, "r") as f:
            data = json.load(f)

        return ThgCheckpoint(
            name=data["name"],
            step=data["step"],
            nodes=data["nodes"],
            edges=data["edges"],
            metadata=data["metadata"],
        )


class ThsdMutationOperator:
    """Mutation operators for THSD hypergraph topology."""

    def __init__(self):
        self.mutation_count = 0

    def add_node(
        self,
        checkpoint: ThgCheckpoint,
        dimension: int = 0,
        stalk_dim: int = 64,
        name: Optional[str] = None,
    ) -> None:
        """Add a new simplex node to hypergraph.

        Args:
            checkpoint: ThgCheckpoint to mutate
            dimension: Simplex dimension (0=vertex, 1=edge, etc.)
            stalk_dim: Stalk dimension for new node
            name: Optional name for node
        """
        node_id = f"node_{len(checkpoint.nodes)}_{self.mutation_count}"
        self.mutation_count += 1

        checkpoint.nodes[node_id] = {
            "id": node_id,
            "dimension": dimension,
            "name": name or f"simplex_{dimension}_{node_id}",
            "stalk_dim": stalk_dim,
            "metadata": {},
        }

    def remove_node(self, checkpoint: ThgCheckpoint, node_id: str) -> None:
        """Remove a node and its incident edges.

        Args:
            checkpoint: ThgCheckpoint to mutate
            node_id: ID of node to remove
        """
        if node_id not in checkpoint.nodes:
            return

        # Remove incident edges
        edges_to_remove = [
            eid
            for eid, e in checkpoint.edges.items()
            if e["src"] == node_id or e["dst"] == node_id
        ]
        for eid in edges_to_remove:
            del checkpoint.edges[eid]

        # Remove node
        del checkpoint.nodes[node_id]

    def modify_edge_weight(
        self,
        checkpoint: ThgCheckpoint,
        edge_id: str,
        new_weight: float,
    ) -> None:
        """Modify edge weight (connection strength).

        Args:
            checkpoint: ThgCheckpoint to mutate
            edge_id: ID of edge to modify
            new_weight: New weight value
        """
        if edge_id in checkpoint.edges:
            checkpoint.edges[edge_id]["weight"] = max(0.0, min(1.0, new_weight))

    def mutate_spectral_gap(
        self, checkpoint: ThgCheckpoint, delta: float = 0.01
    ) -> None:
        """Mutate spectral gap constraint.

        Args:
            checkpoint: ThgCheckpoint to mutate
            delta: Change in spectral gap (can be positive or negative)
        """
        # Use `or default` semantics: `.get(k, d)` only returns `d` when
        # the key is *absent* — but the parser may store an explicit
        # `None` when a `formal_spec` block omits the field. Treat
        # `None` as "use the default" so unspecified constraints get
        # mutated from a sane starting point rather than crashing.
        current = checkpoint.metadata.get("spectral_gap")
        if current is None:
            current = 0.3
        new_gap = max(0.01, min(0.5, current + delta))
        checkpoint.metadata["spectral_gap"] = new_gap

    def mutate_phi_target(
        self, checkpoint: ThgCheckpoint, delta: float = 0.01
    ) -> None:
        """Mutate Phi target constraint.

        Args:
            checkpoint: ThgCheckpoint to mutate
            delta: Change in Phi target
        """
        # See ``mutate_spectral_gap`` for the None-handling rationale.
        current = checkpoint.metadata.get("phi_target")
        if current is None:
            current = 0.75
        new_phi = max(0.0, min(1.0, current + delta))
        checkpoint.metadata["phi_target"] = new_phi


class FitnessEvaluator:
    """Evaluate fitness from THSD constraints + task metrics."""

    def __init__(
        self,
        phi_weight: float = 1.0,
        cohomology_weight: float = 1.0,
        spectral_weight: float = 1.0,
        task_weight: float = 1.0,
    ):
        """Initialize fitness evaluator.

        Args:
            phi_weight: Weight for Phi constraint violation
            cohomology_weight: Weight for cohomology constraint violation
            spectral_weight: Weight for spectral gap violation
            task_weight: Weight for task loss
        """
        self.phi_weight = phi_weight
        self.cohomology_weight = cohomology_weight
        self.spectral_weight = spectral_weight
        self.task_weight = task_weight

    def evaluate(
        self,
        checkpoint: ThgCheckpoint,
        task_loss: float = 0.5,
        phi_value: float = 0.5,
        h1_norm: float = 0.05,
        spectral_gap_value: float = 0.3,
    ) -> Dict[str, float]:
        """Evaluate fitness combining constraints and task metrics.

        Args:
            checkpoint: Architecture checkpoint
            task_loss: Task loss value (lower is better)
            phi_value: Current Phi value
            h1_norm: Current H¹ norm (cohomology)
            spectral_gap_value: Current spectral gap

        Returns:
            Fitness dict with components and total
        """
        phi_target = checkpoint.metadata.get("phi_target", 0.75)
        cohomology_floor = checkpoint.metadata.get("cohomology_floor", 0.01)
        spectral_minimum = 0.25  # Min acceptable spectral gap

        # Component violations (lower = better)
        phi_violation = abs(phi_value - phi_target)
        cohomology_violation = max(0.0, h1_norm - cohomology_floor)
        spectral_violation = max(0.0, spectral_minimum - spectral_gap_value)

        # Task loss (already normalized)
        task_component = task_loss

        # Combined fitness (higher = better)
        phi_fitness = 1.0 / (1.0 + self.phi_weight * phi_violation)
        cohomology_fitness = 1.0 / (1.0 + self.cohomology_weight * cohomology_violation)
        spectral_fitness = 1.0 / (1.0 + self.spectral_weight * spectral_violation)
        task_fitness = 1.0 / (1.0 + self.task_weight * task_component)

        # Weighted combination
        total_fitness = (
            phi_fitness + cohomology_fitness + spectral_fitness + task_fitness
        ) / 4.0

        return {
            "total": total_fitness,
            "phi_violation": phi_violation,
            "phi_fitness": phi_fitness,
            "cohomology_violation": cohomology_violation,
            "cohomology_fitness": cohomology_fitness,
            "spectral_violation": spectral_violation,
            "spectral_fitness": spectral_fitness,
            "task_loss": task_component,
            "task_fitness": task_fitness,
        }


class EvolutionaryLoop:
    """Main evolutionary loop for architecture search."""

    def __init__(
        self,
        initial_hypergraph: HypergraphIR,
        mutation_rate: float = 0.1,
        fitness_threshold: float = 0.7,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
    ):
        """Initialize evolutionary loop.

        Args:
            initial_hypergraph: Starting architecture
            mutation_rate: Probability of applying mutation per generation
            fitness_threshold: Minimum fitness to apply mutations
            checkpoint_dir: Directory for saving checkpoints
            checkpoint_interval: Save checkpoint every N generations
        """
        self.checkpoint = ThgCheckpoint.from_hypergraph(initial_hypergraph, step=0)
        self.mutation_rate = mutation_rate
        self.fitness_threshold = fitness_threshold
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.generation = 0
        self.mutator = ThsdMutationOperator()
        self.evaluator = FitnessEvaluator()
        self.fitness_history = []

    def step(self, fitness_metrics: Dict[str, float]) -> None:
        """Execute one evolutionary step.

        Args:
            fitness_metrics: Fitness evaluation results
        """
        self.generation += 1

        # Track fitness
        self.fitness_history.append(fitness_metrics)

        # Apply mutations if fitness exceeds threshold
        if fitness_metrics.get("total", 0.0) > self.fitness_threshold:
            import random

            if random.random() < self.mutation_rate:
                # Randomly select mutation type
                mutation_types = [
                    "spectral_gap",
                    "phi_target",
                    "edge_weight",
                ]
                mutation = random.choice(mutation_types)

                if mutation == "spectral_gap":
                    self.mutator.mutate_spectral_gap(
                        self.checkpoint, delta=random.uniform(-0.05, 0.05)
                    )
                elif mutation == "phi_target":
                    self.mutator.mutate_phi_target(
                        self.checkpoint, delta=random.uniform(-0.05, 0.05)
                    )
                elif mutation == "edge_weight":
                    if self.checkpoint.edges:
                        edge_id = random.choice(list(self.checkpoint.edges.keys()))
                        self.mutator.modify_edge_weight(
                            self.checkpoint,
                            edge_id,
                            new_weight=random.uniform(0.1, 0.9),
                        )

        # Save checkpoint if interval reached
        if (
            self.checkpoint_dir
            and self.generation % self.checkpoint_interval == 0
        ):
            checkpoint_path = (
                Path(self.checkpoint_dir)
                / f"checkpoint_gen_{self.generation}.json"
            )
            self.checkpoint.save(str(checkpoint_path))

    def get_current_checkpoint(self) -> ThgCheckpoint:
        """Get current checkpoint state.

        Returns:
            Current ThgCheckpoint
        """
        return self.checkpoint

    def get_fitness_history(self) -> list:
        """Get fitness history.

        Returns:
            List of fitness metrics per generation
        """
        return self.fitness_history
