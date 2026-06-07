# -*- coding: utf-8 -*-
"""THSD Structural Plasticity (Phase 6)

Living architecture with activity-dependent learning, Hebbian updates,
and predictive forgetting (NEMORI).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Set, Any, Optional


class StructuralPlasticityController:
    """Activity-dependent structural plasticity for evolving networks.

    Stabilizes hot paths (frequent use), prunes cold paths (disuse),
    and rewires for exploration (novelty).
    """

    def __init__(
        self,
        stabilize_threshold: float = 0.2,
        prune_threshold: float = 0.01,
        cold_steps: int = 5,
        exploration_prob: float = 0.1,
        lr: float = 0.01,
    ):
        self.stabilize_threshold = stabilize_threshold
        self.prune_threshold = prune_threshold
        self.cold_steps = cold_steps
        self.exploration_prob = exploration_prob
        self.lr = lr

    def stabilize_edges(
        self, edge_activities: Dict[str, float], edge_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Increase weight of hot (high-activity) edges.

        weight += lr * activity if activity > threshold
        """
        updated = {}
        for edge_id, weight in edge_weights.items():
            activity = edge_activities.get(edge_id, 0.0)
            if activity > self.stabilize_threshold:
                updated[edge_id] = weight + self.lr * activity
            else:
                updated[edge_id] = weight
        return updated

    def prune_edges(
        self,
        edge_activity_history: Dict[str, list],
        edges: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Remove edges that have been inactive for cold_steps."""
        pruned = {}
        for edge_id, edge_data in edges.items():
            history = edge_activity_history.get(edge_id, [])
            # Check if cold for cold_steps
            if len(history) >= self.cold_steps:
                recent = history[-self.cold_steps :]
                if all(a < self.prune_threshold for a in recent):
                    continue  # Prune this edge
            pruned[edge_id] = edge_data
        return pruned

    def add_exploration_edges(
        self,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add random new edges with exploration_prob."""
        new_edges = dict(edges)
        node_ids = list(nodes.keys())

        # With exploration_prob, add new edge
        if len(node_ids) >= 2 and torch.rand(1).item() < self.exploration_prob:
            new_id = f"e_explore_{len(new_edges)}"
            new_edges[new_id] = {"weight": 0.1}

        return new_edges


class HebbianFastWeights(nn.Module):
    """Transient associative memory via outer-product Hebbian updates.

    Implements: A ← (1-η)A + η(h_t ⊗ h_prev)
    Output: h_out = h_t + gate * A @ h_in
    """

    def __init__(self, dim: int, eta: float = 0.05, decay: float = 0.95):
        super().__init__()
        self.dim = dim
        self.eta = eta
        self.decay = decay

        # Fast weight matrix (not trained)
        self.register_buffer("A", torch.zeros(dim, dim))

        # Gate parameter (learns whether to use fast weights)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        h_t: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Update and apply fast weights.

        Args:
            h_t: Current hidden state (*, dim)
            h_prev: Previous hidden state (*, dim)

        Returns:
            h_out = h_t + gate * (A @ h_t)
        """
        # Reshape for Hebbian update
        batch_shape = h_t.shape[:-1]
        h_t_flat = h_t.reshape(-1, self.dim)  # (*, dim)
        h_prev_flat = h_prev.reshape(-1, self.dim)

        # Hebbian update: A ← decay*A + eta * (h_t ⊗ h_prev)
        with torch.no_grad():
            # Outer product (dim, dim)
            outer_prod = torch.mm(h_t_flat.T, h_prev_flat)  # (dim, *)*(*, dim)
            # Normalize by batch size
            outer_prod = outer_prod / max(1, h_t_flat.shape[0])
            # Update A
            self.A = self.decay * self.A + self.eta * outer_prod

        # Apply fast weights: output = input + gate * A @ input
        fast_component = torch.mm(h_t_flat, self.A.T)  # (*, dim)
        output_flat = h_t_flat + self.gate * fast_component
        output = output_flat.reshape(*batch_shape, self.dim)

        return output

    def compute_phi_proxy(self, h: torch.Tensor) -> float:
        """Compute Φ proxy from fast weight activity."""
        h_flat = h.reshape(-1, self.dim)
        # Φ proxy: frobenius norm of A weighted by mean activation
        a_norm = torch.norm(self.A, p="fro").item()
        h_mean = torch.norm(h_flat.mean(0)).item()
        phi_proxy = min(1.0, (a_norm * h_mean) / self.dim)
        return phi_proxy


class NEMORIConsolidator:
    """Predictive forgetting via information bottleneck.

    Prunes edges that carry no predictive information about future
    targets while preserving task-relevant structure.
    """

    def __init__(self, nemori_floor: float = 0.01):
        self.nemori_floor = nemori_floor

    def identify_nonpredictive(
        self, edge_importance: Dict[str, float]
    ) -> Set[str]:
        """Identify edges below importance threshold.

        Args:
            edge_importance: Dict of edge_id -> importance score

        Returns:
            Set of edge IDs to remove
        """
        nonpredictive = set()
        for edge_id, importance in edge_importance.items():
            if importance < self.nemori_floor:
                nonpredictive.add(edge_id)
        return nonpredictive

    def consolidate(self, edges: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate by removing low-importance edges.

        Args:
            edges: Dict of edge_id -> edge_data (with 'importance' field)

        Returns:
            Consolidated edges
        """
        consolidated = {}
        for edge_id, edge_data in edges.items():
            importance = edge_data.get("importance", 0.0)
            if importance >= self.nemori_floor:
                consolidated[edge_id] = edge_data
        return consolidated

    def compute_compression_ratio(
        self, original_edges: Dict, pruned_edges: Dict
    ) -> float:
        """Compute information compression achieved by pruning.

        Args:
            original_edges: Original edge set
            pruned_edges: Pruned edge set

        Returns:
            Ratio: len(pruned) / len(original)
        """
        if len(original_edges) == 0:
            return 1.0
        return len(pruned_edges) / len(original_edges)
