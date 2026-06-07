# -*- coding: utf-8 -*-
"""Minimal NeuroML DSL compiler for circuit validation.

This is a stub implementation that validates basic DSL syntax.
The full compiler pipeline (lexer -> parser -> semantic analyzer -> IR gen)
is implemented separately in phase 1.
"""
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any


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
            synapses.append(SynapseIR(
                source=src, target=tgt, id=f"{src}_{tgt}",
                weight=weight, neurotransmitter=nt,
                equation=equation,
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
        )

        # Attach THSD blocks to IR
        ir.thsd_complexes = thsd_complexes
        ir.thsd_sheaves = thsd_sheaves

        return ir

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
