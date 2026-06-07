# -*- coding: utf-8 -*-
"""Task 3: Epigenetic Feedback and Mycelium Plasticity.

Hot-path reinforcement via neuro-vesicles → genomic rewriting.
Causal emergence (NIS+) abstracts hyper-neuron internal complexity.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn


@dataclass
class MyceliumEffect:
    """Activity-dependent path stabilization (grass shortcuts).

    Mimics mycelial networks in fungi: frequently-used paths strengthen,
    creating efficient shortcuts. Unused paths eventually prune.
    """

    stabilize_threshold: float = 0.7
    prune_threshold: float = 0.01
    learning_rate: float = 0.01

    def step(self, thg, activity_log: Dict[str, float]):
        """Apply one step of mycelium plasticity to THG-IR."""
        # Update edge weights based on activity
        for edge_id, edge in list(thg.edges.items()):
            activity = activity_log.get(edge_id, 0.0)

            if activity > self.stabilize_threshold:
                # HOT path: strengthen
                edge.weight += self.learning_rate * activity
            elif edge.weight < self.prune_threshold and activity < 0.1:
                # COLD path: mark for pruning (set weight to near-zero)
                edge.weight *= 0.5

        return thg


@dataclass
class EpigenesisController:
    """Epigenetic rewriting: translate vesicle signals to genomic changes."""

    def compute_vesicle_emission(self, thg, activity_log: Dict[str, float]) -> List[Dict]:
        """Determine which vesicles should emit based on activity."""
        vesicles = []

        # Vesicles emit from high-activity nodes/edges
        for node_id, node in thg.nodes.items():
            activity = activity_log.get(node_id, 0.0)
            if activity > 0.8:
                # Emit a vesicle
                vesicles.append(
                    {
                        "source": node_id,
                        "activity": activity,
                        "payload_type": "reinforcement",
                    }
                )

        return vesicles

    def create_protein_payload(
        self, target_node: str, delta_embedding: List[float]
    ) -> Dict:
        """Create a protein payload (graph edit instruction) from vesicle."""
        return {
            "target_node": target_node,
            "delta_embedding": delta_embedding,
            "type": "graph_edit",
        }

    def rewrite_genes(self, gene_state: Dict, activity_signal: float) -> Dict:
        """Rewrite gene expression rates based on activity feedback.

        High activity → increase learning rates, baseline neurotransmitter production.
        """
        new_state = dict(gene_state)

        # Update learning rate
        if "learning_rate" in new_state:
            new_state["learning_rate"] = new_state["learning_rate"] * (
                1 + 0.01 * activity_signal
            )

        # Update baseline neurotransmitter levels
        if "baseline_nt" in new_state:
            new_state["baseline_nt"] = new_state["baseline_nt"] * (
                1 + 0.005 * activity_signal
            )

        # Add new genes based on novelty/activity
        if activity_signal > 0.8 and "plasticity_factor" not in new_state:
            new_state["plasticity_factor"] = 0.1

        return new_state


class NISPlus(nn.Module):
    """Neural Information Squeezer Plus: abstract internal complexity to conscious variable.

    Maps d_internal-dimensional internal network state to d_conscious=1 "conscious variable"
    that captures all behaviorally-relevant information via a bottleneck.

    Implements causal emergence: the d=1 conscious variable has more causal power
    than the raw d_internal activations (paradoxically).
    """

    def __init__(
        self, internal_dim: int = 256, conscious_dim: int = 1, hidden_dim: int = 64
    ):
        super().__init__()
        self.internal_dim = internal_dim
        self.conscious_dim = conscious_dim

        # Projection network: internal → conscious
        # Bottleneck structure forces information compression
        self.projection = nn.Sequential(
            nn.Linear(internal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, conscious_dim),
            nn.Tanh(),  # Bounded output
        )

    def project_to_conscious(self, internal_state: torch.Tensor) -> torch.Tensor:
        """Project internal network state to conscious variable.

        Args:
            internal_state: (batch, internal_dim) tensor

        Returns:
            conscious_var: (batch, conscious_dim) tensor
        """
        return self.projection(internal_state)

    def forward(self, x):
        """Alias for project_to_conscious."""
        return self.project_to_conscious(x)
