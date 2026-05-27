# -*- coding: utf-8 -*-
"""Minimal NeuroML DSL compiler for circuit validation.

This is a stub implementation that validates basic DSL syntax.
The full compiler pipeline (lexer -> parser -> semantic analyzer -> IR gen)
is implemented separately in phase 1.
"""
import re
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class NodeIR:
    """Minimal IR node for validation."""
    pass


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
    populations: List[PopulationIR] = None
    neurotransmitter_systems: List[NeurotransmitterSystemIR] = None
    synapses: List[SynapseIR] = None
    modulations: List[ModulationIR] = None
    circuits: List[CircuitIR] = None
    formal_specs: List[FormalSpecIR] = None
    sheaf_specs: List[SheetIR] = None

    def __post_init__(self):
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

    @property
    def nodes(self):
        return (self.populations + self.neurotransmitter_systems + self.synapses +
                self.modulations + self.formal_specs + self.sheaf_specs)


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

        # Basic validation: check for required keywords
        required = ['population']
        if not any(f'{k} ' in source for k in required):
            raise NeuroMLError("Missing required declarations")

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

        return ProgramIR(
            id="circuit",
            populations=pops,
            neurotransmitter_systems=nts,
            synapses=synapses,
            modulations=mods,
            formal_specs=formal_specs,
            sheaf_specs=sheaves,
        )

    @staticmethod
    def compile_file(filepath: str) -> ProgramIR:
        """Compile DSL from file."""
        with open(filepath, 'r') as f:
            source = f.read()
        return NeuroMLCompiler.compile(source)
