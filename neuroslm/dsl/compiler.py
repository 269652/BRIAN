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
class ConfigIR(NodeIR):
    """A `config { ... }` block — captures BrainConfig fields by name.

    All values are stored as raw strings; type coercion happens when
    materialized into a `BrainConfig` instance via `to_brain_config()`.
    This preserves DSL → IR isolation: the IR is JSON-serializable and
    knows nothing about the Python BrainConfig class.
    """
    id: str = "config"
    fields: Dict = None    # name -> raw string value

    def __post_init__(self):
        if self.fields is None:
            self.fields = {}


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
    config: ConfigIR = None    # Phase 1: the BrainConfig block

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


def _parse_properties(props_str: str) -> Dict:
    """Parse key: value, key: value style property strings (handles multi-line)."""
    if not props_str:
        return {}
    result = {}
    # Split on commas but be careful about line continuations
    pairs = re.split(r'[,\n]', props_str.strip())
    for pair in pairs:
        if ':' not in pair:
            continue
        key, value = pair.split(':', 1)
        key = key.strip()
        value = value.strip()
        if key and value:  # Only add non-empty key-value pairs
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
            pops.append(PopulationIR(
                name=name, count=count, id=name,
                dynamics=dynamics, timescale=timescale, capacity=capacity
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
            synapses.append(SynapseIR(
                source=src, target=tgt, id=f"{src}_{tgt}",
                weight=weight, neurotransmitter=nt
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
            mods.append(ModulationIR(
                source_nt=nt, target_population=pop, id=f"{nt}_{pop}",
                effect=effect, gain=gain
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

        # Extract the `config { ... }` block (Phase 1 of DSL refactor).
        # Single block at the top level holds all BrainConfig field assignments.
        # Multi-line tolerant: the `{ ... }` content may span many lines and
        # use either `field: value` or `field = value` style.
        config_ir = ConfigIR(id="config", fields={})
        cfg_match = re.search(r'config\s*\{([^}]*)\}', source, re.DOTALL)
        if cfg_match:
            body = cfg_match.group(1)
            # Strip line comments (# ...) so they don't get parsed as values
            body = re.sub(r'#[^\n]*', '', body)
            # Accept both `name: value` and `name = value`
            for line in body.split('\n'):
                m = re.match(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*(.+?)\s*,?\s*$', line)
                if m:
                    name, val = m.group(1), m.group(2).strip()
                    # Strip a trailing comma if any (already handled by regex but be safe)
                    val = val.rstrip(',').strip()
                    config_ir.fields[name] = val

        return ProgramIR(
            id="circuit",
            populations=pops,
            neurotransmitter_systems=nts,
            synapses=synapses,
            modulations=mods,
            formal_specs=formal_specs,
            sheaf_specs=sheaves,
            config=config_ir,
        )

    @staticmethod
    def compile_file(filepath: str) -> ProgramIR:
        """Compile DSL from file."""
        with open(filepath, 'r') as f:
            source = f.read()
        return NeuroMLCompiler.compile(source)


# ─────────────────────────────────────────────────────────────────────────
# Phase 1: DSL → BrainConfig materialization
# ─────────────────────────────────────────────────────────────────────────

def _coerce_value(raw: str, target_type):
    """Convert a DSL string value to the type expected by the BrainConfig
    dataclass field. Handles bool, int, float, str, Optional[int].

    Recognizes bare `true`/`false`/`True`/`False` for bool, `null`/`None` for
    None, numeric literals for int/float, and quoted strings for str.
    """
    raw = raw.strip()

    # None / null
    if raw in ('None', 'null', 'NULL'):
        return None

    # Strip outer quotes for string-style values
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
        raw_unquoted = raw[1:-1]
    else:
        raw_unquoted = raw

    # bool first — narrowest type
    if target_type is bool:
        if raw.lower() in ('true', '1', 'yes', 'on'):
            return True
        if raw.lower() in ('false', '0', 'no', 'off'):
            return False
        raise ValueError(f"can't coerce {raw!r} to bool")

    # numeric types
    if target_type in (int, float):
        # If the value has 'e' or '.' parse as float, else int
        if 'e' in raw.lower() or '.' in raw:
            return target_type(float(raw))
        return target_type(int(raw, 0))   # base=0 handles 0x, 0o, 0b too

    # str — strip quotes if present
    if target_type is str:
        return raw_unquoted

    # Fallback: try the type constructor directly
    try:
        return target_type(raw_unquoted)
    except Exception:
        return raw_unquoted


def to_brain_config(program: ProgramIR):
    """Materialize a `BrainConfig` instance from the DSL's `config { ... }` block.

    Reads `program.config.fields` (a dict[str, str] of raw DSL values),
    looks up each field's declared type on the BrainConfig dataclass, and
    coerces the value. Fields not present in the DSL keep their dataclass
    default. Fields present in the DSL but not on BrainConfig raise an
    error (catches typos in the DSL).

    Returns a fully-populated `BrainConfig`. Round-trip equivalent to any
    preset that was written in the DSL — see tests/dsl/test_config_compile.py.
    """
    # Import here to avoid circular dep at module load time.
    from neuroslm.config import BrainConfig
    import dataclasses

    if program.config is None or not program.config.fields:
        # Empty `config` block (or none) → just return defaults.
        return BrainConfig()

    # Build a type map from the dataclass fields
    type_map = {f.name: f.type for f in dataclasses.fields(BrainConfig)}

    # Unknown field names are typos in the DSL — fail loudly
    unknown = sorted(set(program.config.fields) - set(type_map))
    if unknown:
        raise NeuroMLError(
            f"DSL config has fields not present in BrainConfig: {unknown}. "
            f"Either remove them from the .neuro file or add them to "
            f"neuroslm/config.py:BrainConfig.")

    # Coerce each field. The dataclass `type` annotation can be a string
    # (PEP 563 / `from __future__ import annotations`) so we resolve via
    # the actual class attribute default's type as a fallback when the
    # annotation isn't directly callable.
    cfg = BrainConfig()
    for name, raw in program.config.fields.items():
        ann = type_map[name]
        # Resolve string annotations (e.g. 'int | None')
        if isinstance(ann, str):
            # Heuristic: split on '|', try each; coerce to first that works
            options = [t.strip() for t in ann.replace(' ', '').split('|')]
            value = None
            errors = []
            for opt in options:
                if opt in ('None', 'NoneType'):
                    if raw.strip() in ('None', 'null', 'NULL'):
                        value = None
                        break
                    continue
                target = {'int': int, 'float': float, 'bool': bool, 'str': str}.get(opt)
                if target is None:
                    continue
                try:
                    value = _coerce_value(raw, target)
                    break
                except Exception as e:
                    errors.append((target, str(e)))
            else:
                # No type matched cleanly
                raise NeuroMLError(
                    f"can't coerce field {name!r}={raw!r} to any of {options}: {errors}")
        else:
            value = _coerce_value(raw, ann)
        setattr(cfg, name, value)
    return cfg


def compile_to_brain_config(filepath: str):
    """One-shot: filepath -> BrainConfig. The intended entry point for
    `train.py --neuro <file>` once Phase 4 lands."""
    program = NeuroMLCompiler.compile_file(filepath)
    return to_brain_config(program)


def compile_to_brain(filepath: str, scale_overrides: Dict = None):
    """One-shot: filepath -> instantiated `neuroslm.brain.Brain`.

    Phase 1 implementation: the Python `Brain` class is the "interpreter"
    that consumes the DSL-compiled config. The Brain CLASS itself is still
    defined in Python (neuroslm/brain.py); Phase 2 of the refactor (see
    docs/DSL_REFACTOR.md) will introduce a codegen pass that emits a
    Python module from the DSL's `module { ... }` blocks, at which point
    this function will switch transparently to using the generated class.

    The contract is stable now: `compile_to_brain(path)` is the canonical
    way to materialize a model from a .neuro file. Future phases change
    the implementation, not the API.

    Parameters
    ----------
    filepath : path to a .neuro file containing a `config { ... }` block
    scale_overrides : optional dict of BrainConfig fields to override
        (e.g. for fast tests: `{'d_hidden': 64, 'lang_layers': 2}`).
        Applied AFTER the DSL config; the DSL stays read-only.
    """
    from neuroslm.brain import Brain
    cfg = compile_to_brain_config(filepath)
    if scale_overrides:
        for k, v in scale_overrides.items():
            setattr(cfg, k, v)
    return Brain(cfg)
