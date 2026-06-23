# -*- coding: utf-8 -*-
"""Minimal NeuroML DSL compiler for circuit validation.

This is a stub implementation that validates basic DSL syntax.
The full compiler pipeline (lexer -> parser -> semantic analyzer -> IR gen)
is implemented separately in phase 1.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Mapping


@dataclass
class NodeIR:
    """Minimal IR node for validation."""
    pass


@dataclass
class EquationDefnIR(NodeIR):
    """User-defined reusable equation definition."""
    name: str
    params: List[str]
    formula: str
    id: str = ""
    exported: bool = False

    def __post_init__(self):
        if self.id == "":
            self.id = self.name


@dataclass
class PopulationIR(NodeIR):
    name: str
    count: int
    id: str = ""
    dynamics: str = "rate_code"
    timescale: float = 0.01
    capacity: float = 1.0
    resting: float = 0.0
    output_dim: int = None
    properties: Dict = None
    # Phase 7 Stage 1 — algebraic equation override. When present, this
    # takes precedence over the `dynamics` enum during codegen. The string
    # is parsed lazily by codegen/equations.py.
    equation: str = None
    # Phase 7 Stage 2 — ODE override. `dV/dt = ...` or `coef * dV/dt = ...`.
    # Mutually exclusive with `equation`; if both set, ODE wins (more
    # specific). Like `equation:`, this is parsed lazily.
    ode: str = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.id == "":
            self.id = self.name


@dataclass
class SynapseIR(NodeIR):
    source: str
    target: str
    id: str = ""
    weight: float = None
    neurotransmitter: str = None
    binding_rate: float = None
    unbinding_rate: float = None
    max_conductance: float = None
    plasticity_rule: str = None
    learning_rate: float = None
    properties: Dict = None
    # Phase 7 Stage 1 — algebraic transmission equation, e.g.
    # `y = g * sigmoid(W @ x_pre)`. None → fall back to linear default.
    equation: str = None
    # §14 — reference to a `feature.endpoint` whose impl class supplies
    # the edge function. Format: ``"<feature_name>.<endpoint_name>"`` or
    # short form ``"<feature_name>"`` when the feature has exactly one
    # endpoint. ``None`` keeps the canonical ``weight * (x_pre @ W)``
    # transmission for backwards compatibility.
    feature_ref: Optional[str] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class NeurotransmitterSystemIR(NodeIR):
    name: str
    id: str = ""
    base_concentration: float = 0.0
    release_rate: float = None
    reuptake_rate: float = None
    diffusion_rate: float = None
    receptors: Dict = None
    properties: Dict = None

    def __post_init__(self):
        if self.receptors is None:
            self.receptors = {}
        if self.properties is None:
            self.properties = {}


@dataclass
class ModulationIR(NodeIR):
    source_nt: str
    target_population: str
    id: str = ""
    effect: str = "multiplicative"
    gain: float = 1.0
    offset: float = 0.0
    receptor_type: str = None
    desensitization_tau: float = None
    properties: Dict = None
    # Phase 7 Stage 1 — explicit modulation equation, e.g.
    # `gain = 1 + k * c` (c = NT concentration). None → fall back to
    # the legacy `effect` + `gain` form.
    equation: str = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class GeneIR(NodeIR):
    """A declarative gene — wired into the GeneticOrchestrator as a
    `FixedGeneSpec` at harness build time.

    Fields:
        name:         human-readable identifier
        target:       module to modulate (one of the orchestrator's
                       target_modules). "*" → all modules.
        constitutive: True → always-on; False → trigger-gated
        trigger:      dict like {"surprise_above": 0.3, "mat_above": 0.5}
                       Only meaningful when constitutive=False.
        effects:      dict of effect-kind → {NT-name: magnitude}
                       Recognised kinds (see neurochem.genetics.EFFECT_*):
                         "nt_baseline_offset"  — additive baseline shift
                         "receptor_tau_shift"  — push τ_decay toward 1.0
                                                   (reuptake blockade)
                         "nt_release_gain"     — multiply release amounts
    """
    name: str
    id: str = ""
    target: str = ""
    constitutive: bool = False
    trigger: Dict = None
    effects: Dict = None
    properties: Dict = None

    def __post_init__(self):
        if self.trigger is None: self.trigger = {}
        if self.effects is None: self.effects = {}
        if self.properties is None: self.properties = {}


@dataclass
class ProteinIR(NodeIR):
    """A learnable protein payload — the latent vector the GeneticLibrary
    optimises to maximise Φ.

    Fields:
        name:         identifier
        payload_dim:  vector length (becomes `d_pay` on the orchestrator)
        init:         "zero" | "small_normal" (ReZero discipline)
        optimize_for: "phi" | "lm_loss" | metric name; selects which
                       auxiliary loss pulls the payload
    """
    name: str
    id: str = ""
    payload_dim: int = 16
    init: str = "zero"
    optimize_for: str = "phi"
    properties: Dict = None

    def __post_init__(self):
        if self.properties is None: self.properties = {}


@dataclass
class MetricIR(NodeIR):
    """A computed metric whose value is exposed at one or more node tags
    so the rest of the architecture (gene triggers, schedulers, observers)
    can read it without paying for compute everywhere.

    Fields:
        name:           identifier (e.g. "phi", "mat", "surprise")
        compute:        "lm_logits" | "iit_proxy" | "external"
        expose_at:      list of node tags
                          {"lm_head", "gws", "pfc", "trunk",
                           "gene_trigger", "all"}
        every_n_steps:  recompute cadence (1 = every step)
    """
    name: str
    id: str = ""
    compute: str = "lm_logits"
    expose_at: List[str] = None
    every_n_steps: int = 1
    properties: Dict = None

    def __post_init__(self):
        if self.expose_at is None: self.expose_at = []
        if self.properties is None: self.properties = {}


@dataclass
class FormalSpecIR(NodeIR):
    name: str
    id: str = ""
    spec_type: str = "generic"
    properties: Dict = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class SheetIR(NodeIR):
    name: str
    id: str = ""
    contradiction_threshold: float = 0.3
    mechanism: str = "h1_cohomology_proxy"
    action: str = "supersedes"
    properties: Dict = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


# ═════════════════════════════════════════════════════════════════════════════
# DSL v2.0 — Complex, Workspace, Vesicle, Sieve, Manifold, MutationKernel
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ManifoldIR(NodeIR):
    """Topological constraint on a complex's latent space (e.g., Tonnetz)."""
    kind: str                    # "Tonnetz" | "flat"
    dim: int
    spectral_gap: float
    properties: Dict = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class ComplexSubstrateIR(NodeIR):
    """A Maximal Substrate — the primary computational unit of v2.0."""
    name: str
    id: str = ""
    topology: Optional["ManifoldIR"] = None
    trunk: str = ""              # e.g. "PredictiveCoding(layers: 12)"
    sieve: Optional[str] = None  # e.g. "MotifRejection(gnorm_threshold: 3.0)"
    evolution_policy: Optional[str] = None
    geometric_priors: List[str] = None
    genetic_library: List["GeneIR"] = None
    mutation_kernel: Optional["MutationKernelIR"] = None
    properties: Dict = None

    def __post_init__(self):
        if self.geometric_priors is None:
            self.geometric_priors = []
        if self.genetic_library is None:
            self.genetic_library = []
        if self.properties is None:
            self.properties = {}


@dataclass
class WorkspaceIR(NodeIR):
    """Global Workspace — three-stage partitioned dynamics (Gatekeeping, Integration, Broadcasting)."""
    name: str
    id: str = ""
    dynamics: str = ""           # e.g. "SAPHIRE(synergy_ratio: 0.8)"
    ignition: str = ""           # e.g. "Adaptive(ema_window: 100)"
    sheaf: Optional[str] = None  # e.g. "ConsistencyChecker(cohomology: H1)"
    properties: Dict = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class VesicleIR(NodeIR):
    """Mobile neuro-vesicle with content vector and lifetime."""
    name: str
    id: str = ""
    trigger: str = "always"      # "Surprise_Head(threshold: 0.8)" | "always"
    lifetime: int = 16
    content_dim: int = 16
    payload: str = "structural_edit"
    properties: Dict = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class SieveIR(NodeIR):
    """Topological filter — extracts divergence motifs and projects them orthogonal."""
    name: str
    id: str = ""
    kind: str = "MotifRejection"
    gnorm_threshold: float = 3.0
    properties: Dict = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class MutationKernelIR(NodeIR):
    """Event-based graph editing engine (mobile evolutionary agents)."""
    kind: str = "NeuroVesicle"    # "NeuroVesicle" | "GradientPatch"
    trigger: str = "always"       # "Surprise_Head(threshold: 0.8)" | "always"
    id: str = ""
    properties: Dict = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class CircuitIR(NodeIR):
    name: str
    id: str = ""
    populations: List[PopulationIR] = None
    neurotransmitter_systems: List[NeurotransmitterSystemIR] = None
    synapses: List[SynapseIR] = None
    modulations: List[ModulationIR] = None
    formal_specs: List[FormalSpecIR] = None
    sheaf_specs: List[SheetIR] = None
    properties: Dict = None

    def __post_init__(self):
        if self.populations is None:
            self.populations = []
        if self.neurotransmitter_systems is None:
            self.neurotransmitter_systems = []
        if self.synapses is None:
            self.synapses = []
        if self.modulations is None:
            self.modulations = []
        if self.formal_specs is None:
            self.formal_specs = []
        if self.sheaf_specs is None:
            self.sheaf_specs = []
        if self.properties is None:
            self.properties = {}

    @property
    def nodes(self):
        return (self.populations + self.neurotransmitter_systems + self.synapses +
                self.modulations + self.formal_specs + self.sheaf_specs)


class _ParamRef:
    """Marker for a feature.params value that is a bare Python identifier
    rather than a literal — emitted unquoted by codegen so the generated
    ``__init__`` can pass through a constructor argument like ``d_sem``.

    Example::

        params: { d_model: d_sem, max_seq_len: 2048 }

    parses to ``{"d_model": _ParamRef("d_sem"), "max_seq_len": 2048}``,
    and codegen lowers it to ``ImplClass(d_model=d_sem, max_seq_len=2048)``
    where ``d_sem`` is the runtime trunk width passed to the generated
    circuit's ``__init__``.

    Why a class (vs a string sentinel): keeps `isinstance(v, _ParamRef)`
    clean in codegen and avoids any chance of a user-supplied string
    colliding with a sentinel prefix.
    """
    __slots__ = ("expr",)

    def __init__(self, expr: str) -> None:
        self.expr = expr

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return f"_ParamRef({self.expr!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, _ParamRef) and self.expr == other.expr

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.expr))


# Identifier regex used to decide whether a params value is a bare
# Python name. Kept here (not in the parse function) so codegen tests
# can import it for symmetry checks.
_BARE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Identifiers we never treat as variable refs even if they look like
# bare names — they're language-level literals or reserved keywords.
_PARAM_LITERAL_KEYWORDS = {"true", "false", "none", "null"}


@dataclass
class FeatureIR(NodeIR):
    """Toggleable mechanism — names an equation + carries an active flag.

    Grammar::

        feature <name> {
            equation: <equation_name>      # must resolve at compile time
            active:   true | false         # required
            impl:     "<dotted.python.path>" # optional — implementation class
            endpoints: {                   # optional — wiring surfaces
                <ep_name>: {
                    kind: "edge" | "modulator" | "transform",
                    inputs: [<arg_names>],
                    output: <name>,
                    params: { ... }
                },
                ...
            }
            params:   { k: v, ... }        # optional — feature-level defaults
        }

    Why this is a separate IR node (not a flag on EquationDefnIR):
    a mechanism's math is reusable across architectures, but its
    on/off state, implementation binding, and wiring surfaces are
    per-arch and per-experiment. The feature block is the single edit
    point for an ablation run.

    Pinned by ``tests/dsl/test_feature_block.py`` and
    ``tests/dsl/test_feature_endpoints.py``.
    """
    name: str = ""
    id: str = ""
    equation_ref: str = ""
    active: bool = False
    # Dotted Python path to the implementation class, e.g.
    # ``"neuroslm.modules.hyperbolic_attention.HyperbolicMultiHeadAttention"``.
    # Empty string means documentation-only (no runtime wiring).
    impl: str = ""
    # Wiring surfaces this feature exposes. Empty list = feature can be
    # declared but not referenced from a synapse / modulation / etc.
    endpoints: List["FeatureEndpointIR"] = None
    params: Dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if self.endpoints is None:
            self.endpoints = []
        if not self.id:
            self.id = self.name


@dataclass
class FeatureEndpointIR(NodeIR):
    """Single wiring surface exposed by a feature.

    A feature's implementation class may expose several callable surfaces
    (e.g. a hyperbolic-attention module exposes ``edge`` for synapse
    routing AND ``modulator`` for neurotransmitter-gated variants).
    Each surface is one ``FeatureEndpointIR``.

    Fields:
        name: endpoint identifier, e.g. ``"edge"``.
        kind: one of ``"edge"`` (synapse-shaped), ``"modulator"``
            (multiplicative gain), ``"transform"`` (population-level
            in-place update). Validated against the synapse/modulation
            it gets wired into.
        inputs: ordered list of argument names the impl's forward expects.
        output: name of the produced tensor (matches the equation's LHS).
        params: optional endpoint-level overrides on top of the feature's
            top-level ``params`` dict.
    """
    name: str = ""
    id: str = ""
    kind: str = "edge"
    inputs: List[str] = None
    output: str = ""
    params: Dict = None

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = []
        if self.params is None:
            self.params = {}
        if not self.id:
            self.id = self.name


# ──────────────────────────────────────────────────────────────────────
# MechanicIR — richer spec block for reusable mechanics library
# (see neuroslm/dsl/mechanic_parser.py for the full grammar)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MechanicIR(NodeIR):
    """IR node for a `mechanic NAME { ... }` block.

    A mechanic block is a richer version of a feature: it carries the
    full mathematical specification, implementation binding, parameter
    schema, usage guidance, empirical evidence, and formal references in
    one self-contained unit.  Mechanics live in ``mechanics/*.neuro`` and
    are imported into any arch via::

        import { name } from "@mechanics/name"

    The compiler records the MechanicIR in the program and makes it
    available to codegen, documentation generators, and arch analysers.
    The actual runtime wiring still uses the existing ``regularization``,
    ``training``, or ``feature`` blocks — MechanicIR is the specification
    layer; those are the activation layer.

    Pinned by ``tests/dsl/test_mechanic_parser.py``.
    """
    name: str = ""
    id: str = ""
    category: str = ""
    summary: str = ""
    equation: str = ""
    impl: str = ""
    loss_fn: str = ""
    zero_init: bool = False
    params: Dict = None
    properties: Dict = None
    when_to_use: str = ""
    not_for: str = ""
    empirical_evidence: Dict = None
    formal_proof: str = ""
    references: List[str] = None
    exported: bool = False

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if self.properties is None:
            self.properties = {}
        if self.empirical_evidence is None:
            self.empirical_evidence = {}
        if self.references is None:
            self.references = []
        if not self.id:
            self.id = self.name


# ──────────────────────────────────────────────────────────────────────
# Expert / Funnel / Distillation / Warmup / ModuleInstance IRs
#
# These five blocks form the LanguageCortex DSL surface (2026-06-15).
# They lift the LM-trunk + expert-ensemble + teacher-cutoff design
# pattern out of the flat ``multi_cortex { experts: [...] cfd_* ... }``
# config block into named, referenceable, validatable declarations.
#
# Wiring graph at compile time:
#
#     [expert MathExpert] ─┐
#     [expert CodeExpert] ─┼─▶ [funnel ensemble] ──▶ [population lm_trunk]
#     [expert LangExpert] ─┘            │
#                                       └─method─▶ [distillation cfd]
#
#     [warmup teacher_cutoff] ──target──▶ [funnel ensemble]
#
# The compiler validates every cross-block reference (input experts
# exist, target population exists, distillation method exists) so a
# typo never reaches the vast.ai box. Pinned by:
#
#   tests/dsl/test_expert_block.py
#   tests/dsl/test_distillation_block.py
#   tests/dsl/test_funnel_block.py
#   tests/dsl/test_warmup_block.py
#   tests/dsl/test_module_instantiation.py
#   tests/dsl/test_language_cortex_lib.py
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ExpertIR(NodeIR):
    """Declarative pretrained expert backbone.

    Grammar::

        expert <name> {
            model:      "<hf_id_or_alias>"      # required
            role:       "<routing_role>"        # required, unique across arch
            d_out:      <int>                   # optional, 0 = auto-detect
            frozen:     true | false            # optional, default true
            dtype:      "float32"|"float16"|"bfloat16"
            device:     "<torch_device>"        # empty = trunk device
            pool:       "last_token"|"mean"|"cls"
            cache:      "<path>"                # supports %key% / $ENV
            auth_token: "<str_or_$ENV>"
        }
    """
    name: str = ""
    id: str = ""
    model: str = ""
    role: str = ""
    d_out: int = 0
    frozen: bool = True
    dtype: str = "float32"
    device: str = ""
    pool: str = "last_token"
    cache: str = ""
    auth_token: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = self.name


@dataclass
class DistillationIR(NodeIR):
    """Named distillation method (referenced by ``funnel.method``).

    Grammar::

        distillation <name> {
            method:            "capacity_funneled"|"vanilla_kd"|"fitnet"
            temperature:       <float>           # > 0
            alpha:             <float>           # in [0, 1]
            bottleneck:        <int>             # 0 = no bottleneck
            loss:              "kl_div"|"mse"|"cosine"
            # CFD-specific (ignored for other methods)
            topk_start:        <int>             # <= topk_end
            topk_end:          <int>
            topk_anneal_steps: <int>
            temperature_floor: <float>
        }
    """
    name: str = ""
    id: str = ""
    method: str = "vanilla_kd"
    temperature: float = 4.0
    alpha: float = 0.5
    bottleneck: int = 0
    loss: str = "kl_div"
    topk_start: int = 4
    topk_end: int = 32
    topk_anneal_steps: int = 10000
    temperature_floor: float = 1.0

    def __post_init__(self):
        if not self.id:
            self.id = self.name


@dataclass
class FunnelIR(NodeIR):
    """Wires N experts to a single trunk population via a bottlenecked
    projection + gating + optional distillation.

    Grammar::

        funnel <name> {
            inputs:        [<ExpertName>, ...]   # required, non-empty
            target:        <population_name>     # required
            d_bottleneck:  <int>                 # 0 = no bottleneck
            gate:          "mean"|"topk2"|"softmax_router"|"attention"
            method:        <DistillationName>    # optional
        }
    """
    name: str = ""
    id: str = ""
    inputs: List[str] = None
    target: str = ""
    d_bottleneck: int = 0
    gate: str = "mean"
    method_ref: str = ""

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = []
        if not self.id:
            self.id = self.name


@dataclass
class WarmupRuleIR(NodeIR):
    """One condition row inside a ``warmup`` block's ``rules: [...]``."""
    metric: str = ""
    op: str = ">="
    value: float = 0.0
    window: int = 1


@dataclass
class WarmupIR(NodeIR):
    """Rule-driven teacher-cutoff controller.

    Grammar::

        warmup <name> {
            target:     <funnel_name>          # which funnel to detach
            action:     "detach"|"anneal_alpha"|"gate_to_zero"
            combinator: "any"|"all"            # default "any"
            rules: [
                { metric: <m>, op: <op>, value: <v>, window: <w> }, ...
            ]
        }
    """
    name: str = ""
    id: str = ""
    target: str = ""
    action: str = "detach"
    combinator: str = "any"
    rules: List["WarmupRuleIR"] = None

    def __post_init__(self):
        if self.rules is None:
            self.rules = []
        if not self.id:
            self.id = self.name


@dataclass
class ModuleInstanceIR(NodeIR):
    """``module <name> = <Lib> { ... }`` — instantiation of a lib module.

    The compiler stores the call-site params here; the
    ``compile_with_lib`` entry point reads the lib's template body and
    expands it with these params substituted via the
    ``%key%`` interpolation engine, then re-parses the expanded text
    into the same ProgramIR.
    """
    name: str = ""
    id: str = ""
    lib: str = ""
    params: Dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if not self.id:
            self.id = self.name


@dataclass
class ProgramIR(NodeIR):
    id: str = ""
    # Architecture metadata (from architecture { ... } block in arch.neuro)
    architecture: Optional[Dict[str, Any]] = None
    equation_decls: List[EquationDefnIR] = None
    populations: List[PopulationIR] = None
    neurotransmitter_systems: List[NeurotransmitterSystemIR] = None
    synapses: List[SynapseIR] = None
    modulations: List[ModulationIR] = None
    circuits: List[CircuitIR] = None
    formal_specs: List[FormalSpecIR] = None
    sheaf_specs: List[SheetIR] = None
    # §6.5 genetics — empty by default so legacy archs are unchanged.
    genes: List["GeneIR"] = None
    proteins: List["ProteinIR"] = None
    metrics: List["MetricIR"] = None
    # v2.0 DSL primitives
    complexes: List[ComplexSubstrateIR] = None
    workspaces: List[WorkspaceIR] = None
    vesicles: List[VesicleIR] = None
    sieves: List[SieveIR] = None
    # Toggleable mechanisms (2026-06-12). Each feature names an
    # equation from the lib and carries an active flag — flip one
    # bit to enable/disable a mechanism for an ablation run.
    # Pinned by tests/dsl/test_feature_block.py.
    features: List["FeatureIR"] = None
    # LanguageCortex DSL surface (2026-06-15). The five lists below
    # form the expert-ensemble + LM-trunk + teacher-cutoff wiring
    # graph. Pinned by tests/dsl/test_{expert,distillation,funnel,
    # warmup,module_instantiation,language_cortex_lib}_block.py.
    experts: List["ExpertIR"] = None
    distillations: List["DistillationIR"] = None
    funnels: List["FunnelIR"] = None
    warmups: List["WarmupIR"] = None
    module_instances: List["ModuleInstanceIR"] = None
    # THSD (Topological Hyper-Sheaf Dynamics) primitives
    thsd_complexes: List["ComplexIR"] = None  # From thsd_ir.py
    thsd_sheaves: List["SheafIR"] = None
    thsd_formal_spec: Optional["FormalSpecIR"] = None  # Note: different from FormalSpecIR above

    def __post_init__(self):
        if self.equation_decls is None:
            self.equation_decls = []
        if self.populations is None:
            self.populations = []
        if self.neurotransmitter_systems is None:
            self.neurotransmitter_systems = []
        if self.synapses is None:
            self.synapses = []
        if self.modulations is None:
            self.modulations = []
        if self.circuits is None:
            self.circuits = []
        if self.formal_specs is None:
            self.formal_specs = []
        if self.sheaf_specs is None:
            self.sheaf_specs = []
        if self.genes is None:
            self.genes = []
        if self.proteins is None:
            self.proteins = []
        if self.metrics is None:
            self.metrics = []
        if self.complexes is None:
            self.complexes = []
        if self.workspaces is None:
            self.workspaces = []
        if self.vesicles is None:
            self.vesicles = []
        if self.sieves is None:
            self.sieves = []
        if self.features is None:
            self.features = []
        # LanguageCortex DSL surface defaults
        if self.experts is None:
            self.experts = []
        if self.distillations is None:
            self.distillations = []
        if self.funnels is None:
            self.funnels = []
        if self.warmups is None:
            self.warmups = []
        if self.module_instances is None:
            self.module_instances = []
        # THSD initialization
        if self.thsd_complexes is None:
            self.thsd_complexes = []
        if self.thsd_sheaves is None:
            self.thsd_sheaves = []

    @property
    def nodes(self):
        return (self.equation_decls + self.populations + self.neurotransmitter_systems + self.synapses +
                self.modulations + self.formal_specs + self.sheaf_specs +
                self.genes + self.proteins + self.metrics +
                self.complexes + self.workspaces + self.vesicles + self.sieves)


class NeuroMLError(Exception):
    """Base DSL error."""
    pass


def _split_top_level(s: str) -> List[str]:
    """Split on `,` or newline, but only at depth 0 (outside strings/parens).

    Lets equation values like `"y = max(0, x)"` survive the split intact.
    """
    out, buf = [], []
    depth = 0
    in_str = None  # None, '"', or "'"
    for ch in s:
        if in_str:
            buf.append(ch)
            if ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif (ch == "," or ch == "\n") and depth == 0:
            piece = "".join(buf).strip()
            if piece:
                out.append(piece)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def _parse_properties(props_str: str) -> Dict:
    """Parse key: value, key: value style property strings (handles multi-line).

    Quote- and paren-aware: equation values with commas/colons survive.
    """
    if not props_str:
        return {}
    result = {}
    for pair in _split_top_level(props_str):
        if ':' not in pair:
            continue
        key, value = pair.split(':', 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            result[key] = value
    return result


class NeuroMLCompiler:
    """Minimal compiler for DSL validation."""

    @staticmethod
    def compile(source: str) -> ProgramIR:
        """Parse DSL source and return IR.

        This is a stub that validates basic syntax and extracts all field values.
        Full compilation (lexer → parser → semantic analyzer → IR gen) from Phase 1 available separately.
        """
        if not source or len(source) < 10:
            raise NeuroMLError("Empty or invalid source")

        # Basic validation: check for at least one declaration (including THSD blocks)
        declarations = ['population', 'equation', 'architecture', 'synapse', 'neurotransmitter', 'complex', 'sheaf', 'workspace']
        if not any(f'{k} ' in source for k in declarations):
            raise NeuroMLError("Missing required declarations")

        # Extract equation definitions
        eq_defs = []
        eq_pattern = r'(?:export\s+)?equation\s+(\w+)\s*\{([^}]+)\}'
        for match in re.finditer(eq_pattern, source):
            name, props = match.groups()
            props_dict = _parse_properties(props)
            params_str = props_dict.get('params', '[]').strip('[]').replace('"', '').replace("'", '')
            params = [p.strip() for p in params_str.split(',') if p.strip()]
            formula = props_dict.get('formula', '').strip('"\'')
            is_exported = 'export equation' in source[max(0, match.start()-20):match.start()]
            eq_defs.append(EquationDefnIR(
                name=name, params=params, formula=formula,
                id=name, exported=is_exported
            ))

        # Extract populations with all fields: count, dynamics, timescale, capacity
        pops = []
        pop_pattern = r'population\s+(\w+)\s*\{([^}]+)\}'
        for match in re.finditer(pop_pattern, source):
            name, props = match.groups()
            props_dict = _parse_properties(props)
            count_val = props_dict.get('count', 256)
            count = int(float(count_val)) if isinstance(count_val, str) else int(count_val)
            dynamics = props_dict.get('dynamics', 'rate_code').strip('"\'')
            timescale = float(props_dict.get('timescale', 0.01))
            capacity = float(props_dict.get('capacity', 1.0))
            equation = props_dict.get('equation')
            if equation is not None:
                equation = equation.strip().strip('"\'')
            ode = props_dict.get('ode')
            if ode is not None:
                ode = ode.strip().strip('"\'')
            pops.append(PopulationIR(
                name=name, count=count, id=name,
                dynamics=dynamics, timescale=timescale, capacity=capacity,
                equation=equation, ode=ode,
            ))

        # Extract synapse mentions with weight and neurotransmitter
        synapses = []
        syn_pattern = r'synapse\s+(\w+)\s*->\s*(\w+)(?:\s*\{([^}]*)\})?'
        for match in re.finditer(syn_pattern, source):
            src, tgt, props_str = match.groups()
            props_dict = _parse_properties(props_str) if props_str else {}
            weight = None
            if 'weight' in props_dict:
                try:
                    weight = float(props_dict.get('weight'))
                except (ValueError, TypeError):
                    weight = None  # Could be 'learnable' or other non-numeric string
            nt = props_dict.get('neurotransmitter', '').strip('"\'') if 'neurotransmitter' in props_dict else None
            equation = props_dict.get('equation')
            if equation is not None:
                equation = equation.strip().strip('"\'')
            # §14 — optional feature-endpoint reference; if set, codegen
            # routes the edge function through the named feature.
            feature_ref = props_dict.get('feature')
            if feature_ref is not None:
                feature_ref = feature_ref.strip().strip('"\'')
                if not feature_ref:
                    feature_ref = None
            synapses.append(SynapseIR(
                source=src, target=tgt, id=f"{src}_{tgt}",
                weight=weight, neurotransmitter=nt,
                equation=equation,
                feature_ref=feature_ref,
            ))

        # Extract neurotransmitters with kinetics: base_concentration, release_rate, reuptake_rate, diffusion_rate
        nts = []
        nt_pattern = r'neurotransmitter\s+(\w+)\s*\{([^}]*)\}'
        for match in re.finditer(nt_pattern, source):
            name, props = match.groups()
            props_dict = _parse_properties(props)
            base_conc = float(props_dict.get('base_concentration', 0.0))
            release_rate = float(props_dict.get('release_rate', 0.0)) if 'release_rate' in props_dict else None
            reuptake_rate = float(props_dict.get('reuptake_rate', 0.0)) if 'reuptake_rate' in props_dict else None
            diffusion_rate = float(props_dict.get('diffusion_rate', 0.0)) if 'diffusion_rate' in props_dict else None
            nts.append(NeurotransmitterSystemIR(
                name=name, id=name,
                base_concentration=base_conc,
                release_rate=release_rate,
                reuptake_rate=reuptake_rate,
                diffusion_rate=diffusion_rate
            ))

        # Extract modulations with effect and gain
        mods = []
        mod_pattern = r'modulation\s+(\w+)\s*->\s*(\w+)(?:\s*\{([^}]*)\})?'
        for match in re.finditer(mod_pattern, source):
            nt, pop, props_str = match.groups()
            props_dict = _parse_properties(props_str) if props_str else {}
            effect = props_dict.get('effect', 'multiplicative').strip('"\'')
            gain = float(props_dict.get('gain', 1.0))
            equation = props_dict.get('equation')
            if equation is not None:
                equation = equation.strip().strip('"\'')
            mods.append(ModulationIR(
                source_nt=nt, target_population=pop, id=f"{nt}_{pop}",
                effect=effect, gain=gain,
                equation=equation,
            ))

        # Extract sheaf specs with contradiction_threshold and mechanism
        sheaves = []
        sheaf_pattern = r'sheaf\s+(\w+)\s*\{([^}]*)\}'
        for match in re.finditer(sheaf_pattern, source):
            name, props = match.groups()
            props_dict = _parse_properties(props)
            threshold = float(props_dict.get('contradiction_threshold', 0.3))
            mechanism = props_dict.get('mechanism', 'h1_cohomology_proxy').strip('"\'')
            sheaves.append(SheetIR(
                name=name, id=name,
                contradiction_threshold=threshold,
                mechanism=mechanism
            ))

        # Extract formal specs
        formal_specs = []
        formal_pattern = r'formal_spec\s+(\w+)\s*\{([^}]*)\}'
        for match in re.finditer(formal_pattern, source):
            name, props = match.groups()
            props_dict = _parse_properties(props)
            spec_type = props_dict.get('rule', 'generic').strip('"\'')
            formal_specs.append(FormalSpecIR(
                name=name, id=name,
                spec_type=spec_type,
                properties=props_dict
            ))

        # §6.5 genetics — balanced-brace extraction for gene/protein/metric
        # since the bodies contain nested `{ ... }` (effects, trigger).
        genes = _extract_genes(source)
        proteins = _extract_proteins(source)
        metrics = _extract_metrics(source)

        # Parse THSD blocks FIRST (new topology-aware parser)
        from neuroslm.dsl.thsd_parser import THSDParser
        thsd_complexes, thsd_sheaves = THSDParser.parse_dsl_for_thsd(source)

        # v2.0 DSL — extract complex, workspace, vesicle, sieve blocks
        # (only if THSD parser didn't find them)
        if not thsd_complexes:
            complexes = _extract_complexes(source)
        else:
            complexes = []  # Skip v2.0 parsing when THSD found complexes
        workspaces = _extract_workspaces(source)
        vesicles = _extract_vesicles(source)
        sieves = _extract_sieves(source)

        # Feature blocks — toggleable mechanisms with equation refs.
        # Each block must reference an equation defined elsewhere in the
        # source (either inline or imported from lib/). Unresolved refs
        # raise at compile time so a typo never reaches the vast.ai box.
        features = _extract_features(source, eq_defs)

        # LanguageCortex DSL surface (2026-06-15) — declarative
        # expert/teacher/CFD wiring. Each block is independently
        # parseable; cross-block references validated below.
        experts = _extract_experts(source)
        distillations = _extract_distillations(source)
        funnels = _extract_funnels(source)
        warmups = _extract_warmups(source)
        module_instances = _extract_module_instances(source)

        ir = ProgramIR(
            id="circuit",
            equation_decls=eq_defs,
            populations=pops,
            neurotransmitter_systems=nts,
            synapses=synapses,
            modulations=mods,
            formal_specs=formal_specs,
            sheaf_specs=sheaves,
            genes=genes,
            proteins=proteins,
            metrics=metrics,
            complexes=complexes,
            workspaces=workspaces,
            vesicles=vesicles,
            sieves=sieves,
            features=features,
            experts=experts,
            distillations=distillations,
            funnels=funnels,
            warmups=warmups,
            module_instances=module_instances,
        )

        # Attach THSD blocks to IR
        ir.thsd_complexes = thsd_complexes
        ir.thsd_sheaves = thsd_sheaves

        # Cross-block reference validation for LanguageCortex surface.
        # (Funnel inputs must reference declared experts, etc.)
        _validate_languagecortex_refs(ir)

        return ir

    @staticmethod
    def compile_with_lib(
        source: str,
        *,
        lib_root: Optional[Path] = None,
        lib_search_path: Optional[List[Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> "ProgramIR":
        """Compile DSL source with library-module expansion.

        Pipeline:
          1. Parse ``module <name> = <Lib> { params }`` instantiations.
          2. For each instance, search ``lib_search_path`` (or
             ``[lib_root]`` if only ``lib_root`` was passed) for a
             matching ``<Lib>.neuro`` file.
          3. Expand the lib's triple-quoted body template via
             ``resolve_interpolation`` using the instance's params
             (merged with any ``config`` passed by the caller) and ``env``.
          4. Append the expanded text to the source.
          5. Re-call ``NeuroMLCompiler.compile()`` on the augmented source.

        ``lib_root`` defaults to ``<cwd>/lib/`` (the repo-root shared
        library directory, 2026-06-15 layout).
        ``lib_search_path`` may pass multiple roots (e.g. arch-local
        ``lib/`` first, repo-shared ``/lib/`` second).
        """
        from neuroslm.dsl.interpolation import resolve_interpolation, InterpolationError

        # Build the search path. Explicit `lib_search_path` wins;
        # otherwise fall back to `[lib_root]` or the default location.
        if lib_search_path is not None:
            search_roots = [Path(p) for p in lib_search_path]
        elif lib_root is not None:
            search_roots = [Path(lib_root)]
        else:
            search_roots = [Path.cwd() / "lib"]

        config = dict(config or {})
        env = dict(env or {})

        # Discover instantiations BEFORE compile (so we can expand the source first).
        instances = _extract_module_instances(source)
        expanded = source

        for inst in instances:
            # Search each root in order; first hit wins.
            lib_path: Optional[Path] = None
            for root in search_roots:
                candidate = root / f"{inst.lib}.neuro"
                if candidate.exists():
                    lib_path = candidate
                    break
            if lib_path is None:
                raise NeuroMLError(
                    f"module instantiation '{inst.name}' references unknown lib "
                    f"'{inst.lib}' (searched: "
                    f"{[str(r) for r in search_roots]})"
                )
            lib_text = lib_path.read_text(encoding="utf-8")

            # Parse `export module <Lib> { params: {...}, body: "..." }`
            defaults, body_template = _parse_lib_module(lib_text, inst.lib)

            # Merge: defaults < caller-config < instance-params.
            merged: Dict[str, Any] = {}
            merged.update(defaults)
            merged.update(config)
            merged.update(inst.params)

            # Coerce Python lists to bare comma-joined identifier
            # sequences so that a template like ``inputs: [%experts%]``
            # expands to ``inputs: [A, B, C]`` (not ``[['A','B','C']]``).
            for k, v in list(merged.items()):
                if isinstance(v, list):
                    merged[k] = ", ".join(str(x) for x in v)

            try:
                expanded_body = resolve_interpolation(
                    body_template, config=merged, env=env
                )
            except InterpolationError as exc:
                raise NeuroMLError(
                    f"failed to expand module '{inst.name}' (lib={inst.lib}): {exc}"
                ) from exc

            expanded = expanded + "\n\n" + expanded_body

        return NeuroMLCompiler.compile(expanded)

    @staticmethod
    def compile_file(filepath: str) -> ProgramIR:
        """Compile DSL from file.

        First runs the linter to validate syntax and semantics. Raises
        NeuroMLError on linting errors before attempting compilation.
        """
        from pathlib import Path
        from neuroslm.dsl.neuro_linter import NeuroLinter, Severity

        filepath = Path(filepath)
        if not filepath.exists():
            raise NeuroMLError(f"File not found: {filepath}")

        # Run linter first to catch errors early
        linter = NeuroLinter(filepath)
        diagnostics = linter.lint()

        # Fail on structural/semantic errors
        errors = [d for d in diagnostics if d.severity == Severity.ERROR]
        if errors:
            error_msg = "\n".join([f"  {d.file.name}:{d.line}:{d.col} {d.message}" for d in errors])
            raise NeuroMLError(
                f"Linting failed: {len(errors)} error(s)\n{error_msg}"
            )

        # Warn about reference errors but don't fail (imported symbols may not be locally visible)
        warnings = [d for d in diagnostics if d.severity == Severity.WARNING]
        if warnings and len(warnings) <= 5:  # Only show if not too many
            import sys
            for w in warnings:
                print(f"  warning: {w.file.name}:{w.line}:{w.col} {w.message}", file=sys.stderr)

        # Now compile the source
        with open(filepath, 'r') as f:
            source = f.read()
        return NeuroMLCompiler.compile(source)


# ── §6.5 genetics extractors (balanced-brace; nested {} allowed) ──────

def _slice_balanced_brace(s: str, open_idx: int) -> Tuple[str, int]:
    """Return (body, end_idx_exclusive) for the matching `{` at `open_idx`."""
    depth = 0
    in_str = None
    i = open_idx
    while i < len(s):
        ch = s[i]
        if in_str:
            if ch == in_str: in_str = None
        elif ch in ('"', "'"):
            in_str = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[open_idx + 1: i], i + 1
        i += 1
    raise NeuroMLError(f"unbalanced braces starting at {open_idx}")


def _iter_named_blocks(source: str, keyword: str):
    """Yield (name, body) for every `<keyword> <name> { ... }` block."""
    pat = re.compile(rf'\b{re.escape(keyword)}\s+(\w+)\s*\{{')
    for m in pat.finditer(source):
        open_idx = m.end() - 1
        try:
            body, _ = _slice_balanced_brace(source, open_idx)
        except NeuroMLError:
            continue
        yield m.group(1), body


def _parse_nt_dict(raw: str) -> Dict[str, float]:
    """Parse `{ "5HT": 0.10, "DA": 0.20 }` → dict of NT→magnitude."""
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        raw = raw[1:-1]
    out: Dict[str, float] = {}
    for piece in _split_top_level(raw):
        if ":" not in piece:
            continue
        k, v = piece.split(":", 1)
        k = k.strip().strip('"\'')
        try:
            out[k] = float(v.strip())
        except ValueError:
            continue
    return out


def _parse_trigger_block(raw: str) -> Dict:
    """Parse `{ surprise_above: 0.30, mat_above: 0.55 }`."""
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        raw = raw[1:-1]
    out: Dict = {}
    for piece in _split_top_level(raw):
        if ":" not in piece:
            continue
        k, v = piece.split(":", 1)
        k = k.strip()
        v = v.strip()
        if v.startswith("{") and v.endswith("}"):
            out[k] = _parse_nt_dict(v)
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v.strip('"\'')
    return out


def _parse_effects_block(raw: str) -> Dict[str, Dict[str, float]]:
    """Parse `{ nt_baseline_offset: { "5HT": 0.10 }, receptor_tau_shift: { ... } }`."""
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        raw = raw[1:-1]
    out: Dict[str, Dict[str, float]] = {}
    # Re-walk to split on top-level commas/newlines, keeping nested braces
    pieces = []
    buf = []
    depth = 0
    in_str = None
    for ch in raw:
        if in_str:
            buf.append(ch)
            if ch == in_str: in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch; buf.append(ch); continue
        if ch == "{": depth += 1; buf.append(ch); continue
        if ch == "}": depth -= 1; buf.append(ch); continue
        if (ch == "," or ch == "\n") and depth == 0:
            piece = "".join(buf).strip()
            if piece: pieces.append(piece)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail: pieces.append(tail)
    for piece in pieces:
        if ":" not in piece:
            continue
        k, v = piece.split(":", 1)
        k = k.strip()
        out[k] = _parse_nt_dict(v.strip())
    return out


def _extract_genes(source: str) -> List["GeneIR"]:
    out: List[GeneIR] = []
    for name, body in _iter_named_blocks(source, "gene"):
        props = _parse_properties(body)
        target = (props.get("target") or "").strip().strip('"\'')
        constitutive = (props.get("constitutive", "false").lower() in ("true", "1", "yes"))
        trigger_raw = _extract_subblock(body, "trigger")
        effects_raw = _extract_subblock(body, "effects")
        trigger = _parse_trigger_block(trigger_raw) if trigger_raw else {}
        effects = _parse_effects_block(effects_raw) if effects_raw else {}
        out.append(GeneIR(
            name=name, id=name, target=target,
            constitutive=constitutive, trigger=trigger, effects=effects,
            properties=props,
        ))
    return out


def _extract_proteins(source: str) -> List["ProteinIR"]:
    out: List[ProteinIR] = []
    for name, body in _iter_named_blocks(source, "protein"):
        props = _parse_properties(body)
        payload_dim = int(float(props.get("payload_dim", 16)))
        init = props.get("init", "zero").strip('"\'')
        optimize_for = props.get("optimize_for", "phi").strip('"\'')
        out.append(ProteinIR(
            name=name, id=name,
            payload_dim=payload_dim, init=init, optimize_for=optimize_for,
            properties=props,
        ))
    return out


def _extract_metrics(source: str) -> List["MetricIR"]:
    out: List[MetricIR] = []
    for name, body in _iter_named_blocks(source, "metric"):
        props = _parse_properties(body)
        compute = props.get("compute", "lm_logits").strip('"\'')
        every_n_steps = int(float(props.get("every_n_steps", 1)))
        expose_raw = (props.get("expose_at") or "").strip()
        if expose_raw.startswith("[") and expose_raw.endswith("]"):
            expose_raw = expose_raw[1:-1]
        expose_at = [x.strip().strip('"\'') for x in expose_raw.split(",") if x.strip()]
        out.append(MetricIR(
            name=name, id=name,
            compute=compute, expose_at=expose_at,
            every_n_steps=every_n_steps,
            properties=props,
        ))
    return out


def _extract_subblock(body: str, key: str) -> str:
    """Find `<key>: { ... }` inside `body` and return the brace body."""
    pat = re.compile(rf'\b{re.escape(key)}\s*:\s*\{{')
    m = pat.search(body)
    if not m:
        return ""
    open_idx = m.end() - 1
    try:
        sub, _ = _slice_balanced_brace(body, open_idx)
    except NeuroMLError:
        return ""
    return "{" + sub + "}"


# ── v2.0 DSL extractors (complex, workspace, vesicle, sieve) ──────────────

def _extract_complexes(source: str) -> List[ComplexSubstrateIR]:
    """Extract all `complex Name { ... }` blocks (v2.0 style only).

    Skip THSD-style complexes (those with dict-type fields like
    stalk { ... }, topology { ... }) — those are handled by THSDParser.
    """
    out: List[ComplexSubstrateIR] = []
    for name, body in _iter_named_blocks(source, "complex"):
        # Check if this is THSD syntax by looking for nested block patterns
        # THSD blocks have: stalk { ... }, topology { ... }, formal_spec { ... }, etc.
        # v2.0 blocks have: topology: "...", trunk: "...", etc. (key: value pairs)
        # Simple heuristic: if body contains "stalk {" or "topology {", it's THSD
        is_thsd = (
            'stalk {' in body or
            'topology {' in body or
            'formal_spec {' in body or
            'dynamics {' in body
        )
        if is_thsd:
            # This is THSD syntax — skip it, let THSDParser handle it
            continue

        props = _parse_properties(body)

        # Parse topology (Tonnetz or flat)
        topology_str = props.get("topology", "").strip().strip('"\'')
        topology = None
        if topology_str:
            # Simple parsing: "Tonnetz(dim: 256, spectral_gap: 0.05)"
            if topology_str.startswith("Tonnetz"):
                # Extract params from Tonnetz(...)
                m = re.search(r'Tonnetz\s*\(\s*dim\s*:\s*(\d+)\s*,\s*spectral_gap\s*:\s*([\d.]+)', topology_str)
                if m:
                    topology = ManifoldIR(
                        kind="Tonnetz",
                        dim=int(m.group(1)),
                        spectral_gap=float(m.group(2))
                    )

        trunk = props.get("trunk", "").strip().strip('"\'')
        sieve = props.get("sieve", "").strip().strip('"\'') or None
        evolution_policy = props.get("evolution_policy", "").strip().strip('"\'') or None

        # Parse genetic_library (nested gene blocks)
        genetic_library = []
        # Try to find genetic_library { ... } block
        gl_match = re.search(r'genetic_library\s*\{', body)
        if gl_match:
            open_idx = gl_match.end() - 1
            try:
                gl_body, _ = _slice_balanced_brace(body, open_idx)
                for gene_name, gene_body in _iter_named_blocks(gl_body, "gene"):
                    gene_props = _parse_properties(gene_body)
                    target = (gene_props.get("target") or "").strip().strip('"\'')
                    rate = float(gene_props.get("rate", 0.0))
                    genetic_library.append(GeneIR(
                        name=gene_name,
                        id=gene_name,
                        target=target,
                        effects={"rate": rate}
                    ))
            except NeuroMLError:
                pass

        # Parse mutation_kernel (nested block)
        mutation_kernel = None
        mk_match = re.search(r'mutation_kernel\s*\{', body)
        if mk_match:
            open_idx = mk_match.end() - 1
            try:
                mk_body, _ = _slice_balanced_brace(body, open_idx)
                mk_props = _parse_properties(mk_body)
                mk_kind = mk_props.get("kind", "NeuroVesicle").strip().strip('"\'')
                mk_trigger = mk_props.get("trigger", "always").strip().strip('"\'')
                mutation_kernel = MutationKernelIR(
                    kind=mk_kind,
                    trigger=mk_trigger,
                    id=f"mk_{name}"
                )
            except NeuroMLError:
                pass

        out.append(ComplexSubstrateIR(
            name=name,
            id=name,
            topology=topology,
            trunk=trunk,
            sieve=sieve,
            evolution_policy=evolution_policy,
            genetic_library=genetic_library,
            mutation_kernel=mutation_kernel
        ))
    return out


def _extract_workspaces(source: str) -> List[WorkspaceIR]:
    """Extract all `workspace Name { ... }` blocks."""
    out: List[WorkspaceIR] = []
    for name, body in _iter_named_blocks(source, "workspace"):
        props = _parse_properties(body)
        dynamics = props.get("dynamics", "").strip().strip('"\'')
        ignition = props.get("ignition", "").strip().strip('"\'')
        sheaf = props.get("sheaf", "").strip().strip('"\'') or None

        out.append(WorkspaceIR(
            name=name,
            id=name,
            dynamics=dynamics,
            ignition=ignition,
            sheaf=sheaf
        ))
    return out


def _extract_features(source: str,
                      eq_defs: "List[EquationDefnIR]") -> List["FeatureIR"]:
    """Extract all ``feature Name { equation: ..., active: ..., ... }`` blocks.

    Validates at compile time:
      - ``active`` is mandatory and parses to a Python bool.
      - ``equation`` is mandatory and must reference a known equation
        (either defined inline or imported from ``lib/`` — both end up
        in ``eq_defs`` after the multifile resolver has run).

    Optional fields:
      - ``impl``: dotted Python path to the implementation class.
      - ``endpoints``: ``{ name: { kind, inputs, output, params } }``
        block declaring wiring surfaces consumable by ``synapse`` etc.
      - ``params``: feature-level default kwargs for the impl.

    Pinned by ``tests/dsl/test_feature_block.py`` (core) and
    ``tests/dsl/test_feature_endpoints.py`` (impl + endpoints).
    """
    out: List[FeatureIR] = []
    by_name: Dict[str, int] = {}  # name → index in out, for override merge
    known_equations = {e.name for e in eq_defs}
    # Strip end-of-line comments BEFORE scanning for feature blocks.
    # Otherwise a docstring like ``# `feature foo { params: {...} }` re-
    # declaration`` matches the feature regex and the brace-slicer pulls
    # the wrong body. Stripping preserves character offsets so any future
    # error messages still point at the right line.
    source = _strip_line_comments(source)
    for name, body in _iter_named_blocks(source, "feature"):
        # Body is already comment-free because we stripped the whole
        # source above; no second strip needed.
        props = _parse_properties(body)
        # ── mandatory: equation ref ──
        if "equation" not in props:
            raise NeuroMLError(
                f"feature {name!r}: missing required field `equation` "
                f"(must reference an exported equation by name)"
            )
        equation_ref = props["equation"].strip().strip('"\'')
        if equation_ref not in known_equations:
            raise NeuroMLError(
                f"feature {name!r}: equation reference {equation_ref!r} "
                f"is not defined; known equations: "
                f"{sorted(known_equations) or '[]'}"
            )
        # ── mandatory: active flag ──
        if "active" not in props:
            raise NeuroMLError(
                f"feature {name!r}: missing required field `active` "
                f"(true|false) — the toggle is the whole point of "
                f"the block"
            )
        active_raw = props["active"].strip().strip('"\'').lower()
        if active_raw == "true":
            active = True
        elif active_raw == "false":
            active = False
        else:
            raise NeuroMLError(
                f"feature {name!r}: `active` must be `true` or `false`, "
                f"got {props['active']!r}"
            )
        # ── optional: impl (dotted Python path) ──
        impl = props.get("impl", "").strip().strip('"\'')
        # ── optional: endpoints block ──
        endpoints = _parse_feature_endpoints(
            name, props.get("endpoints", "").strip()
        )
        # ── optional: params dict ──
        params = _parse_feature_params_dict(props.get("params", "").strip())
        new_feat = FeatureIR(
            name=name,
            id=name,
            equation_ref=equation_ref,
            active=active,
            impl=impl,
            endpoints=endpoints,
            params=params,
        )
        # ── §14 override merge ──
        # A feature may be declared more than once: the canonical
        # pattern is "lib defines the default + impl + endpoints; arch
        # overrides only the `active` toggle (and optionally a few
        # params)". Later blocks win per-field; fields the override
        # leaves implicit inherit from the earlier declaration so an
        # arch override never has to repeat the whole impl spec.
        # The compile_folder pipeline also emits some sources twice for
        # THSD context, so dedup-with-merge is the only correct policy.
        existing_idx = by_name.get(name)
        if existing_idx is None:
            by_name[name] = len(out)
            out.append(new_feat)
        else:
            prev = out[existing_idx]
            merged_params = dict(prev.params or {})
            merged_params.update(new_feat.params or {})
            merged_endpoints = (
                new_feat.endpoints
                if new_feat.endpoints
                else prev.endpoints
            )
            out[existing_idx] = FeatureIR(
                name=prev.name,
                id=prev.id,
                equation_ref=(
                    new_feat.equation_ref or prev.equation_ref
                ),
                active=new_feat.active,
                impl=new_feat.impl or prev.impl,
                endpoints=merged_endpoints,
                params=merged_params,
            )
    return out


def _parse_feature_params_dict(raw: str) -> Dict:
    """Parse a ``{ k: v, k: v }`` literal into a dict.

    Coercion rules (most-specific wins):

    * ``true`` / ``false`` (case-insensitive) → Python ``bool``.
    * Bare identifiers like ``d_sem`` (matching :data:`_BARE_IDENT_RE`
      and not in :data:`_PARAM_LITERAL_KEYWORDS`) → :class:`_ParamRef`
      so codegen emits them unquoted in the impl constructor call.
      This is how a feature spec wires its ``d_model`` to the trunk
      width without hard-coding it::

          params: { d_model: d_sem, max_seq_len: 2048 }

    * Numeric strings → ``int`` or ``float`` (``.`` or ``e`` → float).
    * Everything else stays as a string (preserving the user's quotes
      already stripped).
    """
    if not raw:
        return {}
    raw = _strip_line_comments(raw)
    inner = raw.lstrip("{").rstrip("}").strip()
    out: Dict = {}
    for k, v in _parse_properties(inner).items():
        v_stripped = v.strip()
        # If the value was quoted, force-string mode: keep the literal
        # without interpretation. Lets users opt out of bool/ident
        # coercion by writing ``schedule: "geometric"`` vs the bare
        # ``schedule: geometric`` (which would become a _ParamRef).
        is_quoted = (
            len(v_stripped) >= 2
            and v_stripped[0] in ("'", '"')
            and v_stripped[-1] == v_stripped[0]
        )
        v_clean = v_stripped.strip('"\'')
        if is_quoted:
            out[k] = v_clean
            continue
        v_low = v_clean.lower()
        if v_low == "true":
            out[k] = True
            continue
        if v_low == "false":
            out[k] = False
            continue
        if (
            _BARE_IDENT_RE.match(v_clean)
            and v_low not in _PARAM_LITERAL_KEYWORDS
        ):
            out[k] = _ParamRef(v_clean)
            continue
        try:
            if "." in v_clean or "e" in v_clean.lower():
                out[k] = float(v_clean)
            else:
                out[k] = int(v_clean)
        except ValueError:
            out[k] = v_clean
    return out


def _parse_feature_endpoints(
    feature_name: str, raw: str
) -> List["FeatureEndpointIR"]:
    """Parse an ``endpoints: { ep1: {...}, ep2: {...} }`` block.

    Each inner block must have ``kind`` and ``output`` fields; ``inputs``
    defaults to ``[]`` and ``params`` defaults to ``{}``.

    Returns an empty list if ``raw`` is empty (endpoints are optional).
    """
    if not raw:
        return []
    raw = _strip_line_comments(raw)
    # Strip the outer braces of the endpoints map.
    inner = raw.lstrip("{").rstrip("}").strip()
    if not inner:
        return []

    out: List[FeatureEndpointIR] = []
    # Use _split_top_level so commas inside per-endpoint bodies don't
    # break us.
    for piece in _split_top_level(inner):
        if ":" not in piece:
            continue
        ep_name, body = piece.split(":", 1)
        ep_name = ep_name.strip().strip('"\'')
        body = body.strip()
        if not body.startswith("{") or not body.endswith("}"):
            raise NeuroMLError(
                f"feature {feature_name!r}: endpoint {ep_name!r} must be "
                f"a block ``{{ kind: ..., output: ..., ... }}``, got "
                f"{body!r}"
            )
        ep_body = body.lstrip("{").rstrip("}").strip()
        ep_props = _parse_properties(ep_body)
        if "kind" not in ep_props:
            raise NeuroMLError(
                f"feature {feature_name!r}: endpoint {ep_name!r} missing "
                f"required field `kind` (edge|modulator|transform)"
            )
        if "output" not in ep_props:
            raise NeuroMLError(
                f"feature {feature_name!r}: endpoint {ep_name!r} missing "
                f"required field `output`"
            )
        kind = ep_props["kind"].strip().strip('"\'')
        output = ep_props["output"].strip().strip('"\'')
        inputs_raw = ep_props.get("inputs", "[]").strip()
        inputs = [
            tok.strip().strip('"\'')
            for tok in inputs_raw.lstrip("[").rstrip("]").split(",")
            if tok.strip()
        ]
        ep_params = _parse_feature_params_dict(
            ep_props.get("params", "").strip()
        )
        out.append(FeatureEndpointIR(
            name=ep_name,
            id=ep_name,
            kind=kind,
            inputs=inputs,
            output=output,
            params=ep_params,
        ))
    return out


def _strip_line_comments(source: str) -> str:
    """Replace ``# ... <EOL>`` with spaces (preserve offsets + newlines).

    String-aware: hashes inside ``"..."`` / ``'...'`` are kept verbatim.
    Mirrors ``training_config._strip_comments`` so we don't take a
    cross-module dependency on a private helper.
    """
    out = []
    i, n = 0, len(source)
    in_str = None
    while i < n:
        ch = source[i]
        if in_str:
            out.append(ch)
            if ch == in_str and (i == 0 or source[i - 1] != "\\"):
                in_str = None
            i += 1
            continue
        if ch in ("'", '"'):
            in_str = ch
            out.append(ch)
            i += 1
            continue
        if ch == "#":
            while i < n and source[i] != "\n":
                out.append(" ")
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _extract_vesicles(source: str) -> List[VesicleIR]:
    """Extract all `vesicle Name { ... }` blocks."""
    out: List[VesicleIR] = []
    for name, body in _iter_named_blocks(source, "vesicle"):
        props = _parse_properties(body)
        trigger = props.get("trigger", "always").strip().strip('"\'')
        lifetime = int(props.get("lifetime", 16))
        content_dim = int(props.get("content_dim", 16))
        payload = props.get("payload", "structural_edit").strip().strip('"\'')

        out.append(VesicleIR(
            name=name,
            id=name,
            trigger=trigger,
            lifetime=lifetime,
            content_dim=content_dim,
            payload=payload
        ))
    return out


def _extract_sieves(source: str) -> List[SieveIR]:
    """Extract all `sieve Name { ... }` blocks."""
    out: List[SieveIR] = []
    for name, body in _iter_named_blocks(source, "sieve"):
        props = _parse_properties(body)
        kind = props.get("kind", "MotifRejection").strip().strip('"\'')
        gnorm_threshold = float(props.get("gnorm_threshold", 3.0))

        out.append(SieveIR(
            name=name,
            id=name,
            kind=kind,
            gnorm_threshold=gnorm_threshold
        ))
    return out


# ──────────────────────────────────────────────────────────────────────
# LanguageCortex DSL extractors (2026-06-15)
#
# Each ``_extract_X`` function pulls every ``X <name> { ... }`` block
# out of the source, validates required fields + enum membership, and
# returns the corresponding IR list. Cross-block reference validation
# (``funnel.inputs`` ⊂ declared experts, ``warmup.target`` ∈ declared
# funnels, ``funnel.method`` ∈ declared distillations) happens in
# :func:`_validate_languagecortex_refs` after every extractor has
# run, so the error messages can show the full known-name list.
# ──────────────────────────────────────────────────────────────────────


# Allowed enum tables — duplicated in neuroslm.dsl.warmup_rules for the
# rule engine; kept here for tight error reporting at parse time.
_EXPERT_POOL_MODES = {"last_token", "mean", "cls"}
_EXPERT_DTYPES = {"float32", "float16", "bfloat16"}
_DISTILL_METHODS = {"capacity_funneled", "vanilla_kd", "fitnet"}
_DISTILL_LOSSES = {"kl_div", "mse", "cosine"}
_FUNNEL_GATES = {"mean", "topk2", "softmax_router", "attention"}
_WARMUP_ACTIONS = {"detach", "anneal_alpha", "gate_to_zero"}
_WARMUP_COMBINATORS = {"any", "all"}
_WARMUP_OPS = {">=", ">", "<=", "<", "==", "!="}


def _extract_experts(source: str) -> List[ExpertIR]:
    """Extract all ``expert <name> { ... }`` blocks.

    Mandatory fields: ``model``, ``role``. Duplicate ``role`` across
    experts raises (each routing role must be unique — mirrors the
    ``_parse_experts_list`` ``duplicate domain`` check)."""
    source = _strip_line_comments(source)
    out: List[ExpertIR] = []
    seen_roles: Dict[str, str] = {}  # role → first expert name using it
    for name, body in _iter_named_blocks(source, "expert"):
        props = _parse_properties(body)
        if "model" not in props or not props["model"].strip().strip('"\''):
            raise NeuroMLError(
                f"expert {name!r}: missing required field `model` "
                "(HF model id or alias)"
            )
        if "role" not in props or not props["role"].strip().strip('"\''):
            raise NeuroMLError(
                f"expert {name!r}: missing required field `role` "
                "(routing key — must be unique across all experts)"
            )
        model = props["model"].strip().strip('"\'')
        role = props["role"].strip().strip('"\'')
        if role in seen_roles:
            raise NeuroMLError(
                f"expert {name!r}: duplicate role {role!r} (already "
                f"claimed by expert {seen_roles[role]!r}); each "
                "routing role must be unique"
            )
        seen_roles[role] = name

        # Optional fields with enum validation.
        pool = props.get("pool", "last_token").strip().strip('"\'')
        if pool not in _EXPERT_POOL_MODES:
            raise NeuroMLError(
                f"expert {name!r}: unknown pool {pool!r}; "
                f"must be one of {sorted(_EXPERT_POOL_MODES)}"
            )
        dtype = props.get("dtype", "float32").strip().strip('"\'')
        if dtype not in _EXPERT_DTYPES:
            raise NeuroMLError(
                f"expert {name!r}: unknown dtype {dtype!r}; "
                f"must be one of {sorted(_EXPERT_DTYPES)}"
            )
        frozen_raw = props.get("frozen", "true").strip().lower()
        frozen = frozen_raw != "false"

        out.append(ExpertIR(
            name=name,
            id=name,
            model=model,
            role=role,
            d_out=int(props.get("d_out", 0) or 0),
            frozen=frozen,
            dtype=dtype,
            device=props.get("device", "").strip().strip('"\''),
            pool=pool,
            cache=props.get("cache", "").strip().strip('"\''),
            auth_token=props.get("auth_token", "").strip().strip('"\''),
        ))
    return out


def _extract_distillations(source: str) -> List[DistillationIR]:
    """Extract all ``distillation <name> { ... }`` blocks."""
    source = _strip_line_comments(source)
    out: List[DistillationIR] = []
    for name, body in _iter_named_blocks(source, "distillation"):
        props = _parse_properties(body)
        method = props.get("method", "vanilla_kd").strip().strip('"\'')
        if method not in _DISTILL_METHODS:
            raise NeuroMLError(
                f"distillation {name!r}: unknown method {method!r}; "
                f"must be one of {sorted(_DISTILL_METHODS)}"
            )
        try:
            temperature = float(props.get("temperature", 4.0))
        except (TypeError, ValueError):
            raise NeuroMLError(
                f"distillation {name!r}: temperature must be a float, "
                f"got {props.get('temperature')!r}"
            )
        if temperature <= 0:
            raise NeuroMLError(
                f"distillation {name!r}: temperature must be > 0, "
                f"got {temperature}"
            )
        try:
            alpha = float(props.get("alpha", 0.5))
        except (TypeError, ValueError):
            raise NeuroMLError(
                f"distillation {name!r}: alpha must be a float, "
                f"got {props.get('alpha')!r}"
            )
        if not (0.0 <= alpha <= 1.0):
            raise NeuroMLError(
                f"distillation {name!r}: alpha must be in [0, 1], "
                f"got {alpha}"
            )
        loss = props.get("loss", "kl_div").strip().strip('"\'')
        if loss not in _DISTILL_LOSSES:
            raise NeuroMLError(
                f"distillation {name!r}: unknown loss {loss!r}; "
                f"must be one of {sorted(_DISTILL_LOSSES)}"
            )
        topk_start = int(props.get("topk_start", 4))
        topk_end = int(props.get("topk_end", 32))
        if topk_start > topk_end:
            raise NeuroMLError(
                f"distillation {name!r}: topk_start ({topk_start}) "
                f"must be <= topk_end ({topk_end})"
            )

        out.append(DistillationIR(
            name=name,
            id=name,
            method=method,
            temperature=temperature,
            alpha=alpha,
            bottleneck=int(props.get("bottleneck", 0)),
            loss=loss,
            topk_start=topk_start,
            topk_end=topk_end,
            topk_anneal_steps=int(props.get("topk_anneal_steps", 10000)),
            temperature_floor=float(props.get("temperature_floor", 1.0)),
        ))
    return out


def _parse_identifier_list(raw: str) -> List[str]:
    """Parse ``[A, B, C]`` into ``["A", "B", "C"]``.

    Whitespace-tolerant. Quote-stripping so quoted variants like
    ``["A", "B"]`` are accepted too (lib-template expansion may emit
    either form). Empty list returns ``[]``."""
    if raw is None:
        return []
    s = raw.strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    out: List[str] = []
    for item in s.split(","):
        item = item.strip().strip('"\'')
        if item:
            out.append(item)
    return out


def _extract_funnels(source: str) -> List[FunnelIR]:
    """Extract all ``funnel <name> { ... }`` blocks. Reference
    validation (inputs ⊂ experts, target ∈ pops, method ∈ distillations)
    runs separately in :func:`_validate_languagecortex_refs`."""
    source = _strip_line_comments(source)
    out: List[FunnelIR] = []
    for name, body in _iter_named_blocks(source, "funnel"):
        props = _parse_properties(body)
        if "inputs" not in props:
            raise NeuroMLError(
                f"funnel {name!r}: missing required field `inputs` "
                "(non-empty list of expert names)"
            )
        if "target" not in props or not props["target"].strip():
            raise NeuroMLError(
                f"funnel {name!r}: missing required field `target` "
                "(population name)"
            )
        inputs = _parse_identifier_list(props["inputs"])
        if not inputs:
            raise NeuroMLError(
                f"funnel {name!r}: inputs list must be non-empty"
            )
        gate = props.get("gate", "mean").strip().strip('"\'')
        if gate not in _FUNNEL_GATES:
            raise NeuroMLError(
                f"funnel {name!r}: unknown gate {gate!r}; "
                f"must be one of {sorted(_FUNNEL_GATES)}"
            )

        out.append(FunnelIR(
            name=name,
            id=name,
            inputs=inputs,
            target=props["target"].strip().strip('"\''),
            d_bottleneck=int(props.get("d_bottleneck", 0)),
            gate=gate,
            method_ref=props.get("method", "").strip().strip('"\''),
        ))
    return out


def _parse_warmup_rules(raw: str) -> List[WarmupRuleIR]:
    """Parse ``[ { metric: m, op: o, value: v, window: w }, ... ]``."""
    s = raw.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise NeuroMLError(
            f"warmup.rules must be a [...] list, got: {raw[:60]}"
        )
    s = s[1:-1].strip()
    if not s:
        return []
    # Split top-level `{...}` rows.
    rows: List[str] = []
    depth, in_str, start = 0, None, 0
    i = 0
    while i < len(s):
        ch = s[i]
        if in_str:
            if ch == in_str:
                in_str = None
        elif ch in ('"', "'"):
            in_str = ch
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                rows.append(s[start:i + 1])
        i += 1

    out: List[WarmupRuleIR] = []
    for row in rows:
        inner = row.strip().lstrip("{").rstrip("}").strip()
        kv = _parse_properties(inner)
        for required in ("metric", "op", "value"):
            if required not in kv:
                raise NeuroMLError(
                    f"warmup rule {row!r}: missing required field "
                    f"`{required}`"
                )
        op = kv["op"].strip().strip('"\'')
        if op not in _WARMUP_OPS:
            raise NeuroMLError(
                f"warmup rule {row!r}: unknown op {op!r}; "
                f"must be one of {sorted(_WARMUP_OPS)}"
            )
        try:
            value = float(kv["value"])
        except (TypeError, ValueError):
            raise NeuroMLError(
                f"warmup rule {row!r}: value must be numeric, "
                f"got {kv['value']!r}"
            )
        out.append(WarmupRuleIR(
            metric=kv["metric"].strip().strip('"\''),
            op=op,
            value=value,
            window=int(kv.get("window", 1)),
        ))
    return out


def _extract_warmups(source: str) -> List[WarmupIR]:
    """Extract all ``warmup <name> { ... }`` blocks."""
    source = _strip_line_comments(source)
    out: List[WarmupIR] = []
    for name, body in _iter_named_blocks(source, "warmup"):
        props = _parse_properties(body)
        if "target" not in props or not props["target"].strip():
            raise NeuroMLError(
                f"warmup {name!r}: missing required field `target` "
                "(funnel name)"
            )
        action = props.get("action", "detach").strip().strip('"\'')
        if action not in _WARMUP_ACTIONS:
            raise NeuroMLError(
                f"warmup {name!r}: unknown action {action!r}; "
                f"must be one of {sorted(_WARMUP_ACTIONS)}"
            )
        combinator = props.get("combinator", "any").strip().strip('"\'')
        if combinator not in _WARMUP_COMBINATORS:
            raise NeuroMLError(
                f"warmup {name!r}: unknown combinator {combinator!r}; "
                f"must be one of {sorted(_WARMUP_COMBINATORS)}"
            )
        if "rules" not in props:
            raise NeuroMLError(
                f"warmup {name!r}: missing required field `rules` "
                "(non-empty list of rule rows)"
            )
        rules = _parse_warmup_rules(props["rules"])
        if not rules:
            raise NeuroMLError(
                f"warmup {name!r}: rules list must be non-empty "
                "(the whole point of the block is to encode "
                "cutoff conditions)"
            )

        out.append(WarmupIR(
            name=name,
            id=name,
            target=props["target"].strip().strip('"\''),
            action=action,
            combinator=combinator,
            rules=rules,
        ))
    return out


def _extract_module_instances(source: str) -> List[ModuleInstanceIR]:
    """Extract all ``module <name> = <Lib> { ... }`` instantiations.

    The form is intentionally distinct from the other extractors —
    the parser must recognise the ``= <LibName>`` between the name
    and the brace. Validated lazily by :func:`compile_with_lib`,
    which knows about the lib search path.
    """
    source = _strip_line_comments(source)
    # ``module <name> = <Lib> {``
    pat = re.compile(r'\bmodule\s+(\w+)\s*=\s*(\w+)\s*\{')
    out: List[ModuleInstanceIR] = []
    for m in pat.finditer(source):
        open_idx = m.end() - 1
        try:
            body, _ = _slice_balanced_brace(source, open_idx)
        except NeuroMLError:
            continue
        name, lib = m.group(1), m.group(2)
        params = _parse_feature_params_dict("{" + body + "}")
        # `_parse_feature_params_dict` returns _ParamRef for bare
        # identifiers (so codegen can emit them unquoted). For module
        # instantiation we want literal names — convert them.
        cleaned: Dict = {}
        for k, v in params.items():
            if isinstance(v, _ParamRef):
                cleaned[k] = v.expr
            elif isinstance(v, str) and v.startswith("[") and v.endswith("]"):
                cleaned[k] = _parse_identifier_list(v)
            else:
                cleaned[k] = v
        # Also parse `experts: [A, B]` style list values that
        # _parse_feature_params_dict may have dropped on the floor
        # because it's not a dict-of-dicts parser. Re-extract from the
        # raw body for any key whose raw value starts with ``[``.
        list_pat = re.compile(r'(\w+)\s*:\s*(\[[^\]]*\])')
        for lm in list_pat.finditer(body):
            k = lm.group(1)
            if k not in cleaned or not isinstance(cleaned[k], list):
                cleaned[k] = _parse_identifier_list(lm.group(2))
        out.append(ModuleInstanceIR(
            name=name,
            id=name,
            lib=lib,
            params=cleaned,
        ))
    return out


def _validate_languagecortex_refs(prog: "ProgramIR") -> None:
    """Cross-block reference validation for the LanguageCortex DSL.

    Pinned by ``tests/dsl/test_funnel_block.py::TestReferenceResolution``
    and ``tests/dsl/test_warmup_block.py::TestValidation``.
    """
    expert_names = {e.name for e in prog.experts}
    pop_names = {p.name for p in prog.populations}
    distill_names = {d.name for d in prog.distillations}
    funnel_names = {f.name for f in prog.funnels}

    for f in prog.funnels:
        for inp in f.inputs:
            if inp not in expert_names:
                raise NeuroMLError(
                    f"funnel {f.name!r}: input {inp!r} is not a "
                    "declared `expert` block. Known experts: "
                    f"{sorted(expert_names) or '[]'}"
                )
        if f.target not in pop_names:
            raise NeuroMLError(
                f"funnel {f.name!r}: target population {f.target!r} "
                "is not declared. Known populations: "
                f"{sorted(pop_names) or '[]'}"
            )
        if f.method_ref and f.method_ref not in distill_names:
            raise NeuroMLError(
                f"funnel {f.name!r}: method {f.method_ref!r} is not "
                "a declared `distillation` block. Known: "
                f"{sorted(distill_names) or '[]'}"
            )

    for w in prog.warmups:
        if w.target not in funnel_names:
            raise NeuroMLError(
                f"warmup {w.name!r}: target funnel {w.target!r} is "
                "not declared. Known funnels: "
                f"{sorted(funnel_names) or '[]'}"
            )


# ── §11 LanguageCortex lib-module parser ──────────────────────────────
#
# A library module file (e.g. ``architectures/lib/LanguageCortex.neuro``)
# declares::
#
#     export module LanguageCortex {
#         params: {
#             teacher_warmup_steps: 10000,
#             d_bottleneck:         512,
#             cfd_temperature:      4.0,
#             cfd_alpha:            0.7,
#             experts:              []
#         },
#         body: """
#             distillation cfd { ... %cfd_temperature% ... }
#             funnel ensemble { inputs: [%experts%], ... }
#             warmup teacher_cutoff { ... value: %teacher_warmup_steps% }
#         """
#     }
#
# The parser is intentionally lenient about whitespace and accepts both
# ``"""...\"""`` triple-string bodies and ``"..."`` single-string bodies.

_LIB_MODULE_RE = re.compile(
    r"export\s+module\s+(\w+)\s*\{",
    re.DOTALL,
)
_TRIPLE_BODY_RE = re.compile(
    r'body\s*:\s*"""(.*?)"""',
    re.DOTALL,
)
_SINGLE_BODY_RE = re.compile(
    r'body\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)
_PARAMS_KEY_RE = re.compile(r"params\s*:\s*\{", re.DOTALL)


def _parse_lib_module(text: str, expected_name: str) -> Tuple[Dict[str, Any], str]:
    """Parse a lib-module file (``export module <Name> { params, body }``).

    Returns (defaults_dict, body_template_string). The body template is
    returned with leading/trailing whitespace stripped but ``%key%`` /
    ``$ENV`` markers preserved verbatim for downstream interpolation.

    Raises :class:`NeuroMLError` if the module block is missing, the
    expected name does not match, or the body section is absent.
    """
    text = _strip_line_comments(text)
    m = _LIB_MODULE_RE.search(text)
    if m is None:
        raise NeuroMLError(
            f"lib module file does not contain an `export module` block "
            f"(expected name: {expected_name!r})"
        )
    if m.group(1) != expected_name:
        raise NeuroMLError(
            f"lib module name mismatch: file declares "
            f"{m.group(1)!r} but caller requested {expected_name!r}"
        )

    open_idx = m.end() - 1
    body, _ = _slice_balanced_brace(text, open_idx)

    # ── defaults (params: {...}) — optional ──────────────────────
    defaults: Dict[str, Any] = {}
    pm = _PARAMS_KEY_RE.search(body)
    if pm is not None:
        params_open = pm.end() - 1
        params_body, _ = _slice_balanced_brace(body, params_open)
        defaults = _parse_lib_params_dict(params_body)

    # ── body template (body: """...""") ──────────────────────────
    tm = _TRIPLE_BODY_RE.search(body)
    if tm is not None:
        body_template = tm.group(1)
    else:
        sm = _SINGLE_BODY_RE.search(body)
        if sm is None:
            raise NeuroMLError(
                f"lib module {expected_name!r}: missing `body:` section"
            )
        body_template = sm.group(1).encode("utf-8").decode("unicode_escape")

    return defaults, body_template.strip("\n")


def _parse_lib_params_dict(raw: str) -> Dict[str, Any]:
    """Parse the ``params: {...}`` dict from a lib module.

    Values may be int / float / quoted-string / bare-identifier /
    list-of-identifiers. Identifiers and identifier-lists are returned
    as their string form so the interpolator emits them verbatim into
    the expanded template (where they will be re-parsed by the main
    compiler).
    """
    out: Dict[str, Any] = {}
    for piece in _split_top_level(raw):
        piece = piece.strip().rstrip(",").strip()
        if not piece or ":" not in piece:
            continue
        k, v = piece.split(":", 1)
        key = k.strip()
        val = v.strip().rstrip(",").strip()
        if not key:
            continue

        # Quoted string.
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            out[key] = val[1:-1]
            continue

        # List literal — keep brackets so interpolation emits a
        # comma-joined identifier sequence verbatim.
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            # Default: empty list expands to empty string (the
            # template author is expected to provide override).
            out[key] = inner
            continue

        # Numeric.
        try:
            if "." in val or "e" in val or "E" in val:
                out[key] = float(val)
            else:
                out[key] = int(val)
            continue
        except ValueError:
            pass

        # Bare identifier — store verbatim.
        out[key] = val

    return out
