# -*- coding: utf-8 -*-
"""Hypergraph Intermediate Representation (Phase 2)

Represents simplicial complexes and sheaf bundles as hypergraphs.
Transforms THSD IR into topological graph structures for compilation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from neuroslm.dsl.thsd_ir import (
    ComplexIR,
    SheafStalkIR,
    TopologyIR,
    CohomologyIR,
    DynamicsIR,
)


@dataclass
class SimplexNode:
    """A simplex σᵈᵢ in the simplicial complex K.

    Represents a single simplex (vertex, edge, face, etc.) with
    its associated representation space (stalk).
    """
    id: str
    dimension: int  # 0=vertex, 1=edge, 2=face, ...
    name: str
    stalk_dim: int  # Dimensionality of local representation space
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.dimension < 0:
            raise ValueError(f"dimension must be >= 0, got {self.dimension}")
        if self.stalk_dim <= 0:
            raise ValueError(f"stalk_dim must be positive, got {self.stalk_dim}")


@dataclass
class HypergraphEdge:
    """Edge between two simplices in the hypergraph.

    Represents face relations, coboundary relations, or sheaf couplings.
    """
    id: str
    src_simplex: str  # ID of source simplex
    dst_simplex: str  # ID of destination simplex
    kind: str  # "faces" | "coboundary" | "coupling" | "sheaf_section"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")


@dataclass
class HypergraphIR:
    """Complete hypergraph representation of a simplicial complex with sheaves.

    Encodes the topological structure, node-stalk assignments, edges,
    and all topological invariants (spectral gap, cohomology, Φ).
    """
    name: str
    nodes: Dict[str, SimplexNode]  # simplex_id -> SimplexNode
    edges: Dict[str, HypergraphEdge]  # edge_id -> HypergraphEdge
    dimension: int = 0  # Maximum dimension of simplices
    spectral_gap: Optional[float] = None  # λ₁ (Fiedler value)
    phi_target: Optional[float] = None  # Φ target (IIT 4.0)
    phi_value: Optional[float] = None  # Current Φ value
    cohomology_floor: Optional[float] = None  # min ‖H¹‖
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Auto-compute dimension from nodes if not specified
        if self.dimension == 0 and self.nodes:
            self.dimension = max((n.dimension for n in self.nodes.values()), default=0)

        # Validate spectral gap
        if self.spectral_gap is not None and self.spectral_gap <= 0:
            raise ValueError(f"spectral_gap must be positive, got {self.spectral_gap}")

        # Validate phi values
        if self.phi_target is not None and not 0 <= self.phi_target <= 1:
            raise ValueError(f"phi_target must be in [0, 1], got {self.phi_target}")
        if self.phi_value is not None and not 0 <= self.phi_value <= 1:
            raise ValueError(f"phi_value must be in [0, 1], got {self.phi_value}")

    def validate(self) -> bool:
        """Validate all topological constraints."""
        if self.spectral_gap is not None and self.spectral_gap <= 0:
            raise ValueError(f"spectral_gap must be positive")
        if self.phi_target is not None and not 0 <= self.phi_target <= 1:
            raise ValueError(f"phi_target must be in [0, 1]")
        return True

    def boundary(self, simplex_id: str) -> Set[str]:
        """Compute boundary of a simplex (lower-dimensional faces).

        Returns set of simplex IDs that form the boundary.
        """
        boundary_ids = set()
        for edge in self.edges.values():
            if edge.kind == "faces" and edge.dst_simplex == simplex_id:
                boundary_ids.add(edge.src_simplex)
        return boundary_ids

    def coboundary(self, simplex_id: str) -> Set[str]:
        """Compute coboundary of a simplex (higher-dimensional faces).

        Returns set of simplex IDs that have this simplex in their boundary.
        """
        coboundary_ids = set()
        for edge in self.edges.values():
            if edge.kind == "coboundary" and edge.src_simplex == simplex_id:
                coboundary_ids.add(edge.dst_simplex)
        return coboundary_ids


class HypergraphBuilder:
    """Construct hypergraph IR from THSD IR."""

    def from_complex_ir(self, complex_ir: ComplexIR) -> HypergraphIR:
        """Build hypergraph from ComplexIR.

        Creates a root simplex for the complex and embeds all topological
        constraints into the hypergraph structure.
        """
        nodes: Dict[str, SimplexNode] = {}
        edges: Dict[str, HypergraphEdge] = {}

        # Create root simplex node
        root_node = SimplexNode(
            id=complex_ir.name,
            dimension=0,
            name=complex_ir.name,
            stalk_dim=complex_ir.stalk.representation_dim,
            metadata={
                "fisher_metric": complex_ir.stalk.fisher_information_metric,
                "constraints": complex_ir.stalk.local_constraints,
            },
        )
        nodes[complex_ir.name] = root_node

        # Create hypergraph with constraints from topology
        dimension = 0
        spectral_gap = None
        if complex_ir.topology:
            dimension = complex_ir.topology.dimension
            spectral_gap = complex_ir.topology.spectral_gap

        # Extract cohomological constraints
        phi_target = None
        cohomology_floor = None
        if complex_ir.formal_spec:
            phi_target = complex_ir.formal_spec.phi_target
            cohomology_floor = complex_ir.formal_spec.cohomology_floor

        hypergraph = HypergraphIR(
            name=complex_ir.name,
            nodes=nodes,
            edges=edges,
            dimension=dimension,
            spectral_gap=spectral_gap,
            phi_target=phi_target,
            cohomology_floor=cohomology_floor,
            metadata={
                "topology_kind": complex_ir.topology.kind if complex_ir.topology else None,
                "dynamics": complex_ir.dynamics is not None,
            },
        )

        return hypergraph
