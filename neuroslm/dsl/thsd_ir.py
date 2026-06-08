# -*- coding: utf-8 -*-
"""THSD Intermediate Representation (IR)

Dataclasses for Topological Hyper-Sheaf Dynamics (THSD) concepts:
- SimplexIR: A simplex σᵈᵢ in the simplicial complex K
- SheafStalkIR: Local stalk F(σ) with representation space + constraints
- TopologyIR: Topological structure (Tonnetz manifold, spectral gap, etc.)
- CohomologyIR: Cohomological constraints H¹(K; F)
- DynamicsIR: Dynamic operators (vesicles, mutations, forgetting)
- FormalSpecIR: Mathematical specification of invariants and loss
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class SheafStalkIR:
    """Local stalk F(σ) of sheaf bundle over a simplex.

    Represents the local representation space and constraints at a point
    in the simplicial complex.
    """
    representation_dim: int
    fisher_information_metric: str  # e.g., "information_geometry"
    local_constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyIR:
    """Topological specification of a simplicial complex.

    Encodes manifold structure (Tonnetz), spectral properties,
    and coherence requirements.
    """
    kind: str  # "Tonnetz" | "flat" | "hyperbolic"
    spectral_gap: float  # λ₁ (Fiedler value) for spectral hardening
    dimension: int  # Dimension of the manifold
    coherence_threshold: float = 0.95  # Minimum coherence

    def __post_init__(self):
        if self.spectral_gap <= 0:
            raise ValueError(f"spectral_gap must be positive, got {self.spectral_gap}")
        if not 0 <= self.coherence_threshold <= 1:
            raise ValueError(f"coherence_threshold must be in [0, 1], got {self.coherence_threshold}")


@dataclass
class InformationBottleneckIR:
    """NEMORI (Predictive Forgetting) configuration.

    Implements information bottleneck objective:
    min I(X; Z) s.t. I(Z; Y) ≥ I_target
    """
    enabled: bool = False
    compression_ratio: float = 0.7  # How much to compress
    prediction_lower_bound: float = 0.95  # Minimum prediction accuracy to maintain


@dataclass
class CohomologyIR:
    """Cohomological constraints H¹(K; F).

    Specifies topological consistency requirements and penalty weights.
    """
    cohomology_floor: float = 0.01  # min H¹ violation allowed
    phi_target: float = 0.8  # Φ target (IIT 4.0 integrated information)
    phi_method: str = "geometric_IIT4"
    information_bottleneck: InformationBottleneckIR = field(
        default_factory=InformationBottleneckIR
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.phi_target <= 1:
            raise ValueError(f"phi_target must be in [0, 1], got {self.phi_target}")
        if not 0 <= self.cohomology_floor <= 1:
            raise ValueError(f"cohomology_floor must be in [0, 1], got {self.cohomology_floor}")


@dataclass
class EmissionKernelIR:
    """Emission kernel P_emit: vesicle synthesis trigger.

    Specifies when and how vesicles are created based on network state.
    """
    trigger: str  # e.g., "surprise_head(threshold=0.8)" or "always"
    payload_dim: int  # Dimensionality of vesicle payload
    lifetime_steps: int  # How long vesicle persists


@dataclass
class ReleaseOperatorIR:
    """Release operator R_rule: how vesicles modify architecture.

    Specifies the mutation rule applied when vesicles dock.
    """
    rule: str  # e.g., "rank_one_update" | "parameter_mutation" | "topology_edit"
    learning_rate: float
    target: str  # e.g., "parameter_counts" | "weights" | "topology"


@dataclass
class NEMORIConsolidatorIR:
    """NEMORI: Predictive forgetting during sleep/consolidation.

    Prunes non-predictive edges and nodes from the graph.
    """
    enabled: bool = True
    consolidation_interval: int = 1000  # Steps between consolidation runs
    forgetting_floor: float = 0.01  # Minimum prediction loss allowed


@dataclass
class DynamicsIR:
    """Dynamic operators: vesicles, mutations, and plasticity.

    Encodes how the architecture evolves during training.
    """
    emission: Optional[EmissionKernelIR] = None
    release: Optional[ReleaseOperatorIR] = None
    nemori: Optional[NEMORIConsolidatorIR] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplexIR:
    """Simplicial complex with sheaf-stalk and constraints.

    Represents a region of the architecture as a formal simplicial complex σᵈᵢ
    with local representation space (stalk), topology, and dynamics.
    """
    name: str
    stalk: SheafStalkIR
    topology: Optional[TopologyIR] = None
    formal_spec: Optional[CohomologyIR] = None
    dynamics: Optional[DynamicsIR] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate all constraints."""
        if self.topology and self.topology.spectral_gap <= 0:
            raise ValueError(f"Invalid spectral gap in {self.name}")
        if self.formal_spec and not 0 <= self.formal_spec.phi_target <= 1:
            raise ValueError(f"Invalid phi_target in {self.name}")
        return True


@dataclass
class SheafIR:
    """Sheaf bundle over simplicial complex.

    Defines sections (coherent layers) and consistency constraints.
    """
    name: str
    base_complex: str  # Reference to ComplexIR.name
    sections: List[Dict[str, Any]] = field(default_factory=list)  # {name, dimension, ...}
    consistency_check: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LossEquationIR:
    """Grand unified loss equation.

    Encodes: min_θ,V L_LM + λL_FE - βΦ(K) + γ‖H¹(F)‖
    """
    equation: str  # Mathematical formula
    lambda_weight: float = 0.02  # L_FE coupling
    beta_weight: float = 0.5  # Φ maximization strength
    gamma_weight: float = 1.0  # Cohomology penalty
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConvergenceCriteriaIR:
    """Convergence criteria for training with THSD objectives."""
    phi_min: float = 0.75
    cohomology_max: float = 0.02
    gap_ratio_max: float = 2.0
    steps_to_verify: int = 500


@dataclass
class FormalSpecIR:
    """Complete formal specification of THSD model.

    Top-level specification block that defines mathematical objectives
    and convergence criteria.
    """
    loss_equation: Optional[LossEquationIR] = None
    convergence: Optional[ConvergenceCriteriaIR] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologicalInvariantsIR:
    """Topological invariants that must be satisfied.

    Tracks λ₁ (spectral gap), H¹ (cohomology), Φ (integrated information).
    """
    spectral_gaps: Dict[str, float] = field(default_factory=dict)  # complex_name -> λ₁
    cohomology_errors: Dict[str, float] = field(default_factory=dict)  # complex_name -> ‖H¹‖
    phi_values: Dict[str, float] = field(default_factory=dict)  # complex_name -> Φ
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TheoryOfMindIR:
    """Theory-of-Mind formal IR — per-agent belief stalks over a
    finite, indexed agent set.

    The ToM stalk ``F(σ_ToM(a))`` for agent ``a`` is a vector in
    ``ℝ^{d_belief}`` encoding the model's belief state about ``a``:
    what ``a`` is believed to know, want, and feel. The sheaf over
    the indexed family ``{a_1, …, a_{max_agents}}`` is the model's
    full first-order belief model.

    Order
    -----
    * ``order=1`` — first-order ToM: "what does X believe?"
      ``stalk_dim() == d_belief``.
    * ``order=2`` — second-order: "what does X believe Y believes?"
      ``stalk_dim() == d_belief * (max_agents + 1)`` so the
      per-agent belief carries one slot per believed-about agent
      plus a self slot.
    * ``order=k`` — k-th order; ``stalk_dim()`` grows polynomially
      (``d_belief * (max_agents + 1)^(order - 1)``). The linter
      warns above order 3 because the compute cost explodes.

    False-belief task
    -----------------
    The Sally-Anne / Smarties task is a standard ToM benchmark
    (Wimmer & Perner 1983). Enabling ``false_belief_enabled`` adds
    a gating threshold above which the model maintains the
    *agent's* outdated belief separately from the *world's* current
    state — required to predict where Sally will look for the marble.

    See also: ``docs/formal_framework.md`` §7 (added P1).
    """

    d_belief: int = 64
    max_agents: int = 16
    belief_decay: float = 0.95
    order: int = 1
    false_belief_enabled: bool = False
    false_belief_threshold: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.d_belief <= 0:
            raise ValueError(
                f"d_belief must be > 0, got {self.d_belief!r}"
            )
        if self.max_agents <= 0:
            raise ValueError(
                f"max_agents must be > 0, got {self.max_agents!r}"
            )
        if not (0.0 < self.belief_decay <= 1.0):
            raise ValueError(
                f"belief_decay must be in (0, 1], got {self.belief_decay!r}"
            )
        if self.order < 1:
            raise ValueError(
                f"order must be >= 1 (use no theory_of_mind block to "
                f"disable ToM), got {self.order!r}"
            )
        if self.false_belief_enabled and not (
                0.0 <= self.false_belief_threshold <= 1.0):
            raise ValueError(
                f"false_belief_threshold must be in [0, 1], "
                f"got {self.false_belief_threshold!r}"
            )

    def stalk_dim(self) -> int:
        """Per-agent belief stalk dimension.

        Polynomial growth in order so second-order ToM allocates
        ``(max_agents + 1)`` belief slots per agent (one per
        believed-about agent plus a self slot), and higher orders
        chain that. Codegen uses this to size buffers.
        """
        return self.d_belief * (self.max_agents + 1) ** (self.order - 1)
