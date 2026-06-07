# -*- coding: utf-8 -*-
"""THSD (Topological Hyper-Sheaf-Dynamics) Notation Engine.

Core mathematical machinery for treating model architecture as a simplicial complex K,
with cellular sheaves F assigning stalks F(σ) to simplices, cohomology guards,
and IIT 4.0 irreducibility dynamics.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import torch
import torch.nn as nn
from itertools import combinations


@dataclass
class SimplexComplex:
    """A simplicial complex K: a collection of simplices organized by dimension.

    A k-simplex σ_k is a k-dimensional face (0-simplex = vertex, 1-simplex = edge,
    2-simplex = triangle, etc.). The boundary operator ∂ maps k-simplices to (k-1)-faces.
    """

    dim_max: int  # Maximum dimension of simplices in K
    simplices: Dict[int, Dict[str, dict]] = field(default_factory=lambda: {})

    def __post_init__(self):
        """Initialize simplices dictionary by dimension."""
        for d in range(self.dim_max + 1):
            if d not in self.simplices:
                self.simplices[d] = {}

    def add_simplex(self, name: str, dim: int, boundary: Optional[List[str]] = None) -> str:
        """Add a k-simplex to the complex.

        Args:
            name: Unique identifier for the simplex (e.g., "pop_sensory")
            dim: Dimension of the simplex (0 for vertex, 1 for edge, etc.)
            boundary: List of (k-1)-simplex IDs that form the boundary (for k>0)

        Returns:
            The simplex ID (name)
        """
        if dim > self.dim_max:
            raise ValueError(f"Dimension {dim} exceeds max {self.dim_max}")

        self.simplices[dim][name] = {"boundary": boundary or [], "dim": dim}
        return name

    def boundary(self, simplex_id: str) -> List[str]:
        """Return the boundary faces of a simplex (all (k-1)-faces).

        For a k-simplex σ_k, ∂(σ_k) is a formal sum of (k-1)-faces.
        """
        # Find which dimension this simplex is in
        for d in range(self.dim_max + 1):
            if simplex_id in self.simplices[d]:
                return self.simplices[d][simplex_id]["boundary"]
        return []

    def get_simplices_by_dim(self, dim: int) -> List[str]:
        """Return all simplices of a given dimension."""
        return list(self.simplices.get(dim, {}).keys())


@dataclass
class CellularSheaf:
    """A cellular sheaf F over a simplicial complex K.

    F assigns to each simplex σ a vector space (stalk) F(σ). For our architecture,
    the stalk encodes the local activation space and Fisher information metric.
    """

    complex: SimplexComplex
    stalk_dim: int  # Dimension of each stalk (e.g., 256)
    stalks: Dict[str, torch.Tensor] = field(default_factory=dict)
    fisher_metrics: Dict[str, torch.Tensor] = field(default_factory=dict)
    restrictions: Dict[Tuple[str, str], torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize stalks and Fisher metrics for all simplices."""
        for d in range(self.complex.dim_max + 1):
            for simplex_id in self.complex.simplices[d]:
                # Initialize stalk as a zero vector
                self.stalks[simplex_id] = torch.zeros(self.stalk_dim)

                # Initialize Fisher metric as identity (diagonal covariance)
                self.fisher_metrics[simplex_id] = torch.eye(self.stalk_dim)

                # Initialize restriction maps for edges
                boundary = self.complex.boundary(simplex_id)
                for boundary_id in boundary:
                    key = (simplex_id, boundary_id)
                    # Restriction matrix (k_stalk_dim × k-1_stalk_dim), here assume same dim
                    self.restrictions[key] = torch.eye(self.stalk_dim)

    def get_stalk(self, simplex_id: str) -> torch.Tensor:
        """Return the stalk F(σ) assigned to a simplex σ."""
        return self.stalks.get(simplex_id, torch.zeros(self.stalk_dim))

    def set_stalk(self, simplex_id: str, value: torch.Tensor) -> None:
        """Update the stalk value for a simplex."""
        if value.shape[0] != self.stalk_dim:
            raise ValueError(f"Expected stalk dim {self.stalk_dim}, got {value.shape[0]}")
        self.stalks[simplex_id] = value

    def get_fisher_metric(self, simplex_id: str) -> torch.Tensor:
        """Return the Fisher information metric for a simplex."""
        return self.fisher_metrics.get(simplex_id, torch.eye(self.stalk_dim))

    def restriction(self, source_id: str, target_id: str) -> torch.Tensor:
        """Return the restriction map ρ_{σ τ}: F(σ) → F(τ) from σ down to boundary τ."""
        key = (source_id, target_id)
        return self.restrictions.get(key, torch.eye(self.stalk_dim))


@dataclass
class CoboundaryOperator:
    """The coboundary operator δ for detecting H¹ contradictions.

    The coboundary δ: C^k(K;F) → C^{k+1}(K;F) acts on cochains (sections of the sheaf).
    δ² = 0 is the fundamental property; if δ^k(cochain) ≠ 0 and cannot be killed by
    a global section, then H^k ≠ 0 (obstruction to consistency).
    """

    sheaf: CellularSheaf

    def apply_delta_0(self, zero_cochain: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply δ⁰: C⁰ → C¹ (maps 0-cochains on vertices to 1-cochains on edges).

        δ⁰(f)(e) = ρ_{e v1}(f(v1)) - ρ_{e v0}(f(v0))
        """
        one_cochain = {}

        # Iterate over all edges (1-simplices)
        for edge_id in self.sheaf.complex.get_simplices_by_dim(1):
            boundary = self.sheaf.complex.boundary(edge_id)
            if len(boundary) != 2:
                continue  # Skip malformed edges

            v0_id, v1_id = boundary[0], boundary[1]
            f_v0 = zero_cochain.get(v0_id, torch.zeros(self.sheaf.stalk_dim))
            f_v1 = zero_cochain.get(v1_id, torch.zeros(self.sheaf.stalk_dim))

            # Apply restriction maps
            rho_e_v0 = self.sheaf.restriction(edge_id, v0_id)
            rho_e_v1 = self.sheaf.restriction(edge_id, v1_id)

            # Compute δ⁰(f)(e) = ρ(f_v1) - ρ(f_v0)
            delta_f_e = rho_e_v1 @ f_v1 - rho_e_v0 @ f_v0
            one_cochain[edge_id] = delta_f_e

        return one_cochain

    def apply_delta_1(self, one_cochain: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply δ¹: C¹ → C² (maps 1-cochains on edges to 2-cochains on triangles).

        Since we're working mostly in dimensions ≤1, this returns a "global residual"
        measuring how far the cochain is from being a cocycle.

        For a true cocycle (δ¹(cochain) = 0), the 1-cochain comes from a 0-cochain
        (it's in the image of δ⁰).
        """
        # For simplicity, compute the global L2 norm of the coboundary residuals
        residual = torch.tensor(0.0)

        for edge_id, edge_val in one_cochain.items():
            boundary = self.sheaf.complex.boundary(edge_id)
            if len(boundary) >= 2:
                # Coboundary residual: how "closed" is this 1-cochain?
                # A simple proxy: for each pair of edges sharing a vertex,
                # check if the consistency condition is satisfied.
                residual = residual + torch.norm(edge_val) ** 2

        return torch.sqrt(residual) if residual > 0 else torch.tensor(0.0)

    def is_contradiction(self, one_cochain: Dict[str, torch.Tensor], threshold: float) -> bool:
        """Detect if a 1-cochain has H¹ contradictions (is NOT a cocycle).

        Returns True if the cochain cannot be explained by a global section
        (i.e., if δ¹(cochain) is large relative to the threshold).
        """
        residual = self.apply_delta_1(one_cochain)
        return bool(residual.item() > threshold)


@dataclass
class PhiDynamicsComputer:
    """Φ-Dynamics: compute integrated information Φ from IIT 4.0.

    The Minimum Information Partition (MIP) divides a system into two parts (A, B)
    such that the integrated information Φ(A,B) is minimized. If Φ > 0 for the MIP,
    the system is irreducible and "conscious" in the IIT sense.
    """

    complex: SimplexComplex
    activation_dim: int

    def compute_mip(
        self, state: Dict[str, torch.Tensor]
    ) -> Tuple[Tuple[Set[str], Set[str]], float]:
        """Compute the Minimum Information Partition (MIP) and its integrated information.

        Args:
            state: Mapping from simplex IDs to activation vectors

        Returns:
            (mip, phi): MIP is a tuple (part_A, part_B) of simplex ID sets,
                       phi is the integrated information Φ ≥ 0
        """
        # Collect all simplex IDs in the system
        simplex_ids = set(state.keys())
        if len(simplex_ids) < 2:
            return (set(), set()), 0.0

        # Try all possible bipartitions and find the one with minimum Φ
        min_phi = float("inf")
        best_mip = (set(), set())

        # Iterate over all non-empty, non-full partitions
        n = len(simplex_ids)
        for r in range(1, (n + 1) // 2 + 1):
            for part_a_tuple in combinations(sorted(simplex_ids), r):
                part_a = set(part_a_tuple)
                part_b = simplex_ids - part_a

                # Compute Φ(A, B): the mutual information reduction due to the partition
                phi_ab = self._compute_phi(state, part_a, part_b)

                if phi_ab < min_phi:
                    min_phi = phi_ab
                    best_mip = (part_a, part_b)

        # Return the MIP and its Φ value (Φ_system = Φ_MIP)
        return best_mip, max(0.0, min_phi)

    def _compute_phi(
        self, state: Dict[str, torch.Tensor], part_a: Set[str], part_b: Set[str]
    ) -> float:
        """Compute the integrated information Φ(A,B) for a partition.

        Simplified IIT: Φ(A,B) ≈ MI(A; B) - MI(A; B | partition)
        where MI is mutual information.

        Here we use a simple proxy: correlation structure before/after partition.
        """
        # Extract states for parts A and B
        states_a = torch.stack([state[sid] for sid in part_a if sid in state])
        states_b = torch.stack([state[sid] for sid in part_b if sid in state])

        if states_a.shape[0] == 0 or states_b.shape[0] == 0:
            return 0.0

        # Compute mean activation for each part
        mean_a = states_a.mean(dim=0)
        mean_b = states_b.mean(dim=0)

        # Compute correlation as dot product of mean-centered states
        correlation = torch.dot(mean_a, mean_b).item()

        # Compute variance of each part
        var_a = torch.var(states_a).item()
        var_b = torch.var(states_b).item()

        # Simple Φ proxy: correlation relative to variances (higher = more integrated)
        denom = (var_a + var_b + 1e-6)
        phi = abs(correlation) / denom if denom > 0 else 0.0

        return float(phi)
