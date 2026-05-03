"""Structural & Projection Genomes — DNA encodes module architecture + NT wiring.

Three genome types per module:

1. ALGORITHMIC GENOME (existing in compiler.py)
   - Opcode program: what the module does (RECALL → GATE → WRITE_MEM ...)
   - Alleles = opcode logits + operands → decompiles to Lisp

2. STRUCTURAL GENOME (this file)
   - Architecture DNA: hidden_size, n_layers, n_heads, optimizer, activation
   - Receptor spec: which NTs bind, sign, sensitivity
   - NT modifier rules: "when DA > 0.7 → boost go_gate by 1.5×"

3. PROJECTION GENOME (this file)
   - NT wiring topology: which nucleus → which region, which NT, scale
   - Per-region reuptake rates, baseline levels
   - Encodes the entire neurochemical connectome

These genomes use a DIFFERENT encoding than the opcode genome:
  - Structural genome = named float/categorical alleles (like a config)
  - Projection genome = adjacency list of (src, dst, nt, scale) tuples

All three genome types are evolvable.
"""
from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from typing import Optional


# ════════════════════════════════════════════════════════════════════════
# NT Modifier — receptor-activated behavioral change
# ════════════════════════════════════════════════════════════════════════

@dataclass
class NTModifier:
    """A rule that modifies a module parameter based on NT levels.

    Example: NTModifier('5HT', '>', 0.6, 'select_gate', 0.5, 'scale')
    means: when 5HT > 0.6, multiply select_gate by 0.5 (safer selection).

    Biologically: high serotonin → PFC prefers cautious/prosocial actions.
    """
    nt: str                # which neurotransmitter ('DA', 'NE', '5HT', 'ACh', ...)
    comparator: str        # '>' or '<'
    threshold: float       # activation threshold
    target_param: str      # which module parameter to modify
    modifier_value: float  # how much to modify by
    modifier_type: str     # 'scale' (multiplicative) or 'bias' (additive)

    def applies(self, nt_level: float) -> bool:
        if self.comparator == '>':
            return nt_level > self.threshold
        return nt_level < self.threshold

    def apply(self, current_value: float, nt_level: float) -> float:
        if not self.applies(nt_level):
            return current_value
        if self.modifier_type == 'scale':
            return current_value * self.modifier_value
        return current_value + self.modifier_value

    def to_dict(self) -> dict:
        return {
            'nt': self.nt, 'comparator': self.comparator,
            'threshold': self.threshold, 'target_param': self.target_param,
            'modifier_value': self.modifier_value,
            'modifier_type': self.modifier_type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NTModifier":
        return cls(**d)


# ════════════════════════════════════════════════════════════════════════
# Receptor Spec — which NTs a module binds and how
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ReceptorSpec:
    """One receptor on a module: which NT binds, excitatory/inhibitory, weight."""
    nt: str
    sign: float       # +1 excitatory, -1 inhibitory
    weight: float     # initial sensitivity (learnable in ReceptorBank)
    receptor_type: str = ""  # e.g. "D1", "5HT2A", "M1", "CB1", "NMDA", "GABAA"

    def to_dict(self) -> dict:
        return {'nt': self.nt, 'sign': self.sign, 'weight': self.weight,
                'receptor_type': self.receptor_type}

    @classmethod
    def from_dict(cls, d: dict) -> "ReceptorSpec":
        return cls(**d)


# ════════════════════════════════════════════════════════════════════════
# Structural Genome — architecture DNA for each module
# ════════════════════════════════════════════════════════════════════════

# Valid choices for categorical alleles
OPTIMIZERS = ['adam', 'sgd', 'adamw', 'rmsprop']
ACTIVATIONS = ['gelu', 'relu', 'silu', 'tanh']
LEARNING_RULES = ['backprop', 'hebbian', 'reinforce', 'contrastive']


@dataclass
class StructuralGenome:
    """Per-module architecture DNA.

    Encodes the STRUCTURE of each module: how big it is, how it's trained,
    what receptors it has, what NT modifiers activate.

    This is a SEPARATE genome from the algorithmic opcode genome.
    The structural genome defines WHAT the module IS (architecture).
    The algorithmic genome defines WHAT the module DOES (behavior).
    """
    region: str

    # Architecture
    hidden_size: int = 384        # internal hidden dim
    n_layers: int = 2             # depth
    n_heads: int = 4              # attention heads (if applicable)
    activation: str = 'gelu'      # nonlinearity
    dropout: float = 0.0          # regularization
    layer_norm: bool = True       # use layer norm

    # Optimization
    optimizer: str = 'adamw'      # per-module optimizer type
    learning_rule: str = 'backprop'  # learning algorithm
    lr_scale: float = 1.0         # multiplier on global LR
    weight_decay_scale: float = 1.0  # multiplier on global WD

    # Receptors — which NTs bind to this module
    receptors: list[ReceptorSpec] = field(default_factory=list)

    # NT Modifiers — receptor-activated behavioral rules
    nt_modifiers: list[NTModifier] = field(default_factory=list)

    # Per-region NT dynamics
    reuptake_rates: dict[str, float] = field(default_factory=dict)
    baseline_levels: dict[str, float] = field(default_factory=dict)

    # Evolution
    generation: int = 0
    fitness: float = float('inf')
    id: str = field(default_factory=lambda: f"sg{random.randint(0, 1 << 32):08x}")

    def mutate(self, rate: float = 0.1) -> "StructuralGenome":
        child = StructuralGenome(
            region=self.region,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            activation=self.activation,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
            optimizer=self.optimizer,
            learning_rule=self.learning_rule,
            lr_scale=self.lr_scale,
            weight_decay_scale=self.weight_decay_scale,
            receptors=[ReceptorSpec(**r.to_dict()) for r in self.receptors],
            nt_modifiers=[NTModifier(**m.to_dict()) for m in self.nt_modifiers],
            reuptake_rates=dict(self.reuptake_rates),
            baseline_levels=dict(self.baseline_levels),
            generation=self.generation + 1,
        )
        # Mutate continuous values
        if random.random() < rate:
            child.lr_scale = max(0.1, child.lr_scale + random.gauss(0, 0.2))
        if random.random() < rate:
            child.weight_decay_scale = max(0.0, child.weight_decay_scale + random.gauss(0, 0.1))
        if random.random() < rate:
            child.dropout = max(0.0, min(0.5, child.dropout + random.gauss(0, 0.05)))
        # Mutate receptor weights
        for r in child.receptors:
            if random.random() < rate:
                r.weight = max(0.01, r.weight + random.gauss(0, 0.1))
        # Mutate modifier thresholds/values
        for m in child.nt_modifiers:
            if random.random() < rate:
                m.threshold = max(0.0, min(1.0, m.threshold + random.gauss(0, 0.1)))
            if random.random() < rate:
                m.modifier_value += random.gauss(0, 0.1)
        # Mutate reuptake rates
        for nt in child.reuptake_rates:
            if random.random() < rate:
                child.reuptake_rates[nt] = max(0.01, min(1.0,
                    child.reuptake_rates[nt] + random.gauss(0, 0.05)))
        return child

    def to_dict(self) -> dict:
        return {
            'region': self.region, 'id': self.id,
            'generation': self.generation, 'fitness': self.fitness,
            'hidden_size': self.hidden_size, 'n_layers': self.n_layers,
            'n_heads': self.n_heads, 'activation': self.activation,
            'dropout': self.dropout, 'layer_norm': self.layer_norm,
            'optimizer': self.optimizer, 'learning_rule': self.learning_rule,
            'lr_scale': self.lr_scale, 'weight_decay_scale': self.weight_decay_scale,
            'receptors': [r.to_dict() for r in self.receptors],
            'nt_modifiers': [m.to_dict() for m in self.nt_modifiers],
            'reuptake_rates': dict(self.reuptake_rates),
            'baseline_levels': dict(self.baseline_levels),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructuralGenome":
        g = cls(region=d['region'])
        for k in ('hidden_size', 'n_layers', 'n_heads', 'activation',
                  'dropout', 'layer_norm', 'optimizer', 'learning_rule',
                  'lr_scale', 'weight_decay_scale', 'generation', 'fitness', 'id'):
            if k in d:
                setattr(g, k, d[k])
        g.receptors = [ReceptorSpec.from_dict(r) for r in d.get('receptors', [])]
        g.nt_modifiers = [NTModifier.from_dict(m) for m in d.get('nt_modifiers', [])]
        g.reuptake_rates = d.get('reuptake_rates', {})
        g.baseline_levels = d.get('baseline_levels', {})
        return g


# ════════════════════════════════════════════════════════════════════════
# Per-region structural genome defaults — neuroscience-accurate
# ════════════════════════════════════════════════════════════════════════

def _struct_pfc() -> StructuralGenome:
    return StructuralGenome(
        region='pfc', hidden_size=384, n_layers=3, n_heads=4,
        activation='gelu', optimizer='adamw', learning_rule='backprop',
        lr_scale=1.0,
        receptors=[
            ReceptorSpec('DA',   +1, 0.6, 'D1'),      # D1 enhances working memory
            ReceptorSpec('5HT',  +1, 0.3, '5HT2A'),   # 5HT2A cortical gain
            ReceptorSpec('ACh',  +1, 0.4, 'M1'),      # M1 signal-to-noise
            ReceptorSpec('GABA', -1, 0.4, 'GABAA'),   # inhibition
            ReceptorSpec('NE',   +1, 0.5, 'alpha1'),   # arousal / urgency
        ],
        nt_modifiers=[
            # High 5HT → safer selection (risk aversion)
            NTModifier('5HT', '>', 0.6, 'select_gate', 0.5, 'scale'),
            # High NE → urgent/threatening tasks prioritized
            NTModifier('NE',  '>', 0.7, 'attend_gate', 1.5, 'scale'),
            # High DA → broader exploration (less inhibition)
            NTModifier('DA',  '>', 0.7, 'gate_threshold', 0.3, 'bias'),
            # Low DA → narrow focus (more inhibition)
            NTModifier('DA',  '<', 0.2, 'select_gate', 0.7, 'scale'),
        ],
        reuptake_rates={'DA': 0.15, 'NE': 0.10, '5HT': 0.05, 'GABA': 0.20},
        baseline_levels={'DA': 0.15, 'NE': 0.10, '5HT': 0.25, 'GABA': 0.30},
    )


def _struct_hippocampus() -> StructuralGenome:
    return StructuralGenome(
        region='hippocampus', hidden_size=384, n_layers=2, n_heads=4,
        activation='gelu', optimizer='adamw', learning_rule='hebbian',
        lr_scale=1.2,  # hippocampus learns faster (one-shot encoding)
        receptors=[
            ReceptorSpec('ACh',  +1, 0.5, 'M1'),      # ACh gates encoding
            ReceptorSpec('Glu',  +1, 0.4, 'NMDA'),    # NMDA gates plasticity
            ReceptorSpec('NE',   +1, 0.3, 'beta1'),    # emotional tagging
            ReceptorSpec('GABA', -1, 0.3, 'GABAA'),   # sparse coding inhibition
        ],
        nt_modifiers=[
            # High ACh → stronger encoding (attention enhances memory)
            NTModifier('ACh', '>', 0.5, 'store_gate', 1.5, 'scale'),
            # High NE → emotional memory prioritized (flashbulb memories)
            NTModifier('NE',  '>', 0.6, 'recall_gate', 1.3, 'scale'),
            # Low ACh → weaker encoding, more recall (mind-wandering state)
            NTModifier('ACh', '<', 0.2, 'recall_gate', 1.4, 'scale'),
        ],
        reuptake_rates={'ACh': 0.10, 'Glu': 0.30, 'NE': 0.15},
        baseline_levels={'ACh': 0.20, 'Glu': 0.40, 'NE': 0.10},
    )


def _struct_dmn() -> StructuralGenome:
    return StructuralGenome(
        region='dmn', hidden_size=384, n_layers=2, n_heads=4,
        activation='gelu', optimizer='adamw', learning_rule='backprop',
        lr_scale=0.8,  # DMN learns slower (default mode is stable)
        receptors=[
            ReceptorSpec('5HT',  -1, 0.4, '5HT1A'),   # 5HT suppresses DMN
            ReceptorSpec('ACh',  -1, 0.2, 'M1'),       # ACh suppresses wandering
            ReceptorSpec('DA',   +1, 0.3, 'D2'),       # D2 enhances creative thought
        ],
        nt_modifiers=[
            # High 5HT → DMN suppressed (focused task state)
            NTModifier('5HT', '>', 0.5, 'wander_gate', 0.3, 'scale'),
            # Low 5HT → DMN active (rumination, creativity)
            NTModifier('5HT', '<', 0.2, 'wander_gate', 1.5, 'scale'),
            # High ACh → task focus, less wandering
            NTModifier('ACh', '>', 0.5, 'recall_gate', 0.5, 'scale'),
        ],
        reuptake_rates={'5HT': 0.05, 'DA': 0.10},
        baseline_levels={'5HT': 0.30, 'DA': 0.10},
    )


def _struct_basal_ganglia() -> StructuralGenome:
    return StructuralGenome(
        region='basal_ganglia', hidden_size=256, n_layers=2, n_heads=4,
        activation='gelu', optimizer='adamw', learning_rule='reinforce',
        lr_scale=1.0,
        receptors=[
            ReceptorSpec('DA',   +1, 0.7, 'D1'),       # D1 direct pathway (Go)
            ReceptorSpec('DA',   -1, 0.5, 'D2'),       # D2 indirect pathway (NoGo)
            ReceptorSpec('GABA', -1, 0.5, 'GABAA'),    # tonic inhibition
            ReceptorSpec('ACh',  +1, 0.3, 'M4'),       # cholinergic interneurons
        ],
        nt_modifiers=[
            # High DA → Go pathway dominates (approach behavior)
            NTModifier('DA',  '>', 0.6, 'go_gate', 1.5, 'scale'),
            # Low DA → NoGo pathway dominates (avoidance)
            NTModifier('DA',  '<', 0.2, 'nogo_gate', 1.5, 'scale'),
            # High GABA → all action suppressed (freeze)
            NTModifier('GABA', '>', 0.7, 'go_gate', 0.3, 'scale'),
        ],
        reuptake_rates={'DA': 0.20, 'GABA': 0.25},
        baseline_levels={'DA': 0.15, 'GABA': 0.35},
    )


def _struct_critic() -> StructuralGenome:
    return StructuralGenome(
        region='critic', hidden_size=256, n_layers=2, n_heads=4,
        activation='gelu', optimizer='adamw', learning_rule='backprop',
        lr_scale=1.5,  # critic must learn fast (survival)
        receptors=[
            ReceptorSpec('NE',   +1, 0.7, 'alpha1'),   # NE = arousal/threat
            ReceptorSpec('5HT',  -1, 0.3, '5HT1A'),   # 5HT calms threat response
            ReceptorSpec('GABA', -1, 0.4, 'GABAA'),   # anxiolytic
        ],
        nt_modifiers=[
            # High NE → lower survival threshold (hypervigilant)
            NTModifier('NE',  '>', 0.7, 'select_gate', 0.5, 'scale'),
            # High 5HT → raise survival threshold (calmer)
            NTModifier('5HT', '>', 0.5, 'select_gate', 1.5, 'scale'),
        ],
        reuptake_rates={'NE': 0.15, '5HT': 0.05},
        baseline_levels={'NE': 0.15, '5HT': 0.25},
    )


def _struct_language() -> StructuralGenome:
    return StructuralGenome(
        region='language', hidden_size=512, n_layers=16, n_heads=8,
        activation='gelu', optimizer='adamw', learning_rule='backprop',
        lr_scale=1.0,
        receptors=[
            ReceptorSpec('ACh',  +1, 0.3, 'M1'),       # sharpens attention
            ReceptorSpec('eCB',  -1, 0.3, 'CB1'),      # retrograde suppression
            ReceptorSpec('DA',   +1, 0.2, 'D1'),       # semantic precision
        ],
        nt_modifiers=[
            # High ACh → sharper attention (better token prediction)
            NTModifier('ACh', '>', 0.5, 'attend_gate', 1.3, 'scale'),
            # High eCB → suppressed verbosity (less output)
            NTModifier('eCB', '>', 0.5, 'project_gain', 0.7, 'scale'),
        ],
        reuptake_rates={'ACh': 0.10, 'eCB': 0.20},
        baseline_levels={'ACh': 0.20},
    )


def _struct_thalamus() -> StructuralGenome:
    return StructuralGenome(
        region='thalamus', hidden_size=384, n_layers=2, n_heads=4,
        activation='gelu', optimizer='adamw', learning_rule='backprop',
        lr_scale=1.0,
        receptors=[
            ReceptorSpec('NE',   +1, 0.5, 'alpha1'),   # sharpens routing
            ReceptorSpec('GABA', -1, 0.3, 'GABAA'),   # reticular inhibition
            ReceptorSpec('ACh',  +1, 0.3, 'nAChR'),   # brainstem arousal
        ],
        nt_modifiers=[
            # High NE → sharper routing (emergency mode)
            NTModifier('NE',  '>', 0.6, 'sigmoid_gain', 1.5, 'scale'),
            # High GABA → dampened routing (sleep-like)
            NTModifier('GABA', '>', 0.6, 'gate_threshold', -0.3, 'bias'),
        ],
        reuptake_rates={'NE': 0.10, 'GABA': 0.25},
        baseline_levels={'NE': 0.15, 'GABA': 0.30},
    )


def _struct_world() -> StructuralGenome:
    return StructuralGenome(
        region='world', hidden_size=384, n_layers=2, n_heads=4,
        activation='gelu', optimizer='adamw', learning_rule='backprop',
        lr_scale=1.0,
        receptors=[
            ReceptorSpec('DA',  +1, 0.3, 'D1'),       # prediction confidence
            ReceptorSpec('NE',  +1, 0.3, 'beta1'),    # update urgency
        ],
        nt_modifiers=[
            NTModifier('NE', '>', 0.6, 'predict_gate', 1.3, 'scale'),
        ],
    )


def _struct_self_model() -> StructuralGenome:
    return StructuralGenome(
        region='self_model', hidden_size=256, n_layers=1, n_heads=4,
        activation='gelu', optimizer='adamw', learning_rule='backprop',
        lr_scale=0.5,  # slow self-model update (identity stability)
        receptors=[
            ReceptorSpec('5HT', +1, 0.3, '5HT2A'),   # self-awareness
        ],
        nt_modifiers=[],
    )


def _struct_sensory() -> StructuralGenome:
    return StructuralGenome(
        region='sensory', hidden_size=256, n_layers=1, n_heads=4,
        activation='relu', optimizer='adamw', learning_rule='backprop',
        receptors=[], nt_modifiers=[],
    )


def _struct_association() -> StructuralGenome:
    return StructuralGenome(
        region='association', hidden_size=384, n_layers=2, n_heads=4,
        activation='gelu', optimizer='adamw', learning_rule='backprop',
        receptors=[], nt_modifiers=[],
    )


def _struct_gws() -> StructuralGenome:
    return StructuralGenome(
        region='gws', hidden_size=384, n_layers=2, n_heads=8,
        activation='gelu', optimizer='adamw', learning_rule='backprop',
        receptors=[
            ReceptorSpec('NE',  +1, 0.5, 'alpha1'),  # competition sharpness
        ],
        nt_modifiers=[
            NTModifier('NE', '>', 0.6, 'attend_gate', 1.5, 'scale'),
        ],
    )


def _struct_cerebellum() -> StructuralGenome:
    return StructuralGenome(
        region='cerebellum', hidden_size=384, n_layers=2, n_heads=4,
        activation='gelu', optimizer='adamw', learning_rule='backprop',
        lr_scale=2.0,  # cerebellum: fast error-driven learning
        receptors=[
            ReceptorSpec('Glu', +1, 0.5, 'AMPA'),  # fast excitatory
        ],
        nt_modifiers=[],
    )


def _struct_motor() -> StructuralGenome:
    return StructuralGenome(
        region='motor', hidden_size=256, n_layers=2, n_heads=4,
        activation='gelu', optimizer='adamw', learning_rule='backprop',
        receptors=[
            ReceptorSpec('DA',   +1, 0.5, 'D1'),
            ReceptorSpec('GABA', -1, 0.4, 'GABAA'),
        ],
        nt_modifiers=[
            NTModifier('DA', '>', 0.6, 'project_gain', 1.3, 'scale'),
        ],
    )


def _struct_default(region: str) -> StructuralGenome:
    """Fallback for any region without explicit structural genome."""
    return StructuralGenome(
        region=region, hidden_size=256, n_layers=2, n_heads=4,
        activation='gelu', optimizer='adamw', learning_rule='backprop',
        receptors=[], nt_modifiers=[],
    )


STRUCTURAL_INIT_REGISTRY: dict[str, callable] = {
    'pfc':            _struct_pfc,
    'hippocampus':    _struct_hippocampus,
    'dmn':            _struct_dmn,
    'basal_ganglia':  _struct_basal_ganglia,
    'critic':         _struct_critic,
    'language':       _struct_language,
    'thalamus':       _struct_thalamus,
    'world':          _struct_world,
    'self_model':     _struct_self_model,
    'sensory':        _struct_sensory,
    'association':    _struct_association,
    'gws':            _struct_gws,
    'cerebellum':     _struct_cerebellum,
    'motor':          _struct_motor,
}


# ════════════════════════════════════════════════════════════════════════
# Projection Genome — DNA encodes the NT wiring topology
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ProjectionGene:
    """One axonal projection: src region → dst region via NT.

    The genome encodes the entire neurochemical connectome as a list
    of these genes. Evolution can add/remove/modify projections.
    """
    src: str               # source nucleus/region
    dst: str               # target region
    nt: str                # neurotransmitter carried
    release_scale: float   # how strongly src activity drives release
    carries_signal: bool = False  # whether to also carry embedding

    def to_dict(self) -> dict:
        return {'src': self.src, 'dst': self.dst, 'nt': self.nt,
                'release_scale': self.release_scale,
                'carries_signal': self.carries_signal}

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectionGene":
        return cls(**d)


@dataclass
class ProjectionGenome:
    """The NT wiring blueprint — encodes all projections between regions.

    This IS the connectome. Each ProjectionGene is an axonal bundle
    carrying a specific NT from one nucleus to another.

    The genome evolves: projections can be strengthened, weakened,
    added, or removed during training evolution.
    """
    projections: list[ProjectionGene] = field(default_factory=list)
    generation: int = 0
    fitness: float = float('inf')
    id: str = field(default_factory=lambda: f"pg{random.randint(0, 1 << 32):08x}")

    def mutate(self, rate: float = 0.1) -> "ProjectionGenome":
        child = ProjectionGenome(
            projections=[ProjectionGene(**p.to_dict()) for p in self.projections],
            generation=self.generation + 1,
        )
        # Mutate release scales
        for p in child.projections:
            if random.random() < rate:
                p.release_scale = max(0.0, min(2.0,
                    p.release_scale + random.gauss(0, 0.1)))
        return child

    def to_dict(self) -> dict:
        return {
            'projections': [p.to_dict() for p in self.projections],
            'generation': self.generation, 'fitness': self.fitness,
            'id': self.id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectionGenome":
        g = cls()
        g.projections = [ProjectionGene.from_dict(p) for p in d.get('projections', [])]
        g.generation = d.get('generation', 0)
        g.fitness = d.get('fitness', float('inf'))
        g.id = d.get('id', g.id)
        return g


def default_projection_genome() -> ProjectionGenome:
    """The initial NT wiring — neuroscience-accurate connectome.

    Based on known major NT pathways:
      - VTA → NAcc, PFC, BG  (dopamine — reward, motivation)
      - SNc → BG (nigrostriatal — motor control)
      - LC → PFC, Thalamus, Hippocampus (norepinephrine — arousal, attention)
      - Raphe → DMN, PFC, Hippocampus (serotonin — mood, impulse control)
      - NBM → Language, Hippocampus, PFC (acetylcholine — attention, memory)
      - PFC ↔ VTA (glutamate feedback — top-down control of DA)
      - Hippocampus → NAcc (contextual reward)
      - PFC → PFC (endocannabinoid retrograde — self-regulation)
      - BG → Thalamus (GABAergic — action gating)
      - PAG → LC (pain/threat → arousal)
    """
    return ProjectionGenome(projections=[
        # Dopamine pathways (mesolimbic + mesocortical)
        ProjectionGene('VTA',   'NAcc',         'DA',  1.0),
        ProjectionGene('VTA',   'pfc',          'DA',  0.8),
        ProjectionGene('VTA',   'basal_ganglia','DA',  1.0),
        ProjectionGene('VTA',   'hippocampus',  'DA',  0.4),
        # Nigrostriatal pathway
        ProjectionGene('SNc',   'basal_ganglia','DA',  0.9),
        ProjectionGene('SNc',   'motor',        'DA',  0.6),
        # Norepinephrine pathways (locus coeruleus)
        ProjectionGene('LC',    'pfc',          'NE',  0.7),
        ProjectionGene('LC',    'thalamus',     'NE',  0.7),
        ProjectionGene('LC',    'hippocampus',  'NE',  0.5),
        ProjectionGene('LC',    'language',     'NE',  0.3),
        ProjectionGene('LC',    'gws',          'NE',  0.5),
        # Serotonin pathways (raphe nuclei)
        ProjectionGene('Raphe', 'dmn',          '5HT', 0.6),
        ProjectionGene('Raphe', 'pfc',          '5HT', 0.5),
        ProjectionGene('Raphe', 'hippocampus',  '5HT', 0.3),
        ProjectionGene('Raphe', 'basal_ganglia','5HT', 0.4),
        ProjectionGene('Raphe', 'critic',       '5HT', 0.5),
        # Acetylcholine pathways (basal forebrain / NBM)
        ProjectionGene('NBM',   'language',     'ACh', 0.6),
        ProjectionGene('NBM',   'hippocampus',  'ACh', 0.6),
        ProjectionGene('NBM',   'pfc',          'ACh', 0.5),
        ProjectionGene('NBM',   'thalamus',     'ACh', 0.4),
        # Glutamate feedback (cortical → subcortical)
        ProjectionGene('pfc',   'VTA',          'Glu', 0.4),
        ProjectionGene('hippocampus', 'NAcc',   'Glu', 0.5),
        ProjectionGene('pfc',   'basal_ganglia','Glu', 0.3),
        # GABAergic (inhibitory)
        ProjectionGene('basal_ganglia', 'thalamus', 'GABA', 0.6),
        ProjectionGene('NAcc',  'VTA',          'GABA', 0.3),
        ProjectionGene('critic', 'pfc',         'GABA', 0.4),  # threat → inhibit PFC
        # Endocannabinoid (retrograde self-regulation)
        ProjectionGene('pfc',   'pfc',          'eCB', 0.3),
        ProjectionGene('language', 'language',  'eCB', 0.2),
        # Stress pathway
        ProjectionGene('PAG',   'LC',           'NE',  0.8),   # pain → arousal
        ProjectionGene('critic', 'LC',          'Glu', 0.5),   # threat → arousal
    ])


# ════════════════════════════════════════════════════════════════════════
# Unified DNA Bundle — all three genome types for the whole brain
# ════════════════════════════════════════════════════════════════════════

@dataclass
class BrainDNA:
    """Complete DNA for the brain: structural + projection genomes.

    The algorithmic genomes (opcode programs) live in ModuleGenomePool.
    This class holds the structural and projection genomes.
    """
    structural: dict[str, StructuralGenome] = field(default_factory=dict)
    projection: ProjectionGenome = field(default_factory=default_projection_genome)

    @classmethod
    def default(cls, regions: list[str] | None = None) -> "BrainDNA":
        """Create default DNA for all regions."""
        from .compiler import ALL_REGIONS
        regions = regions or ALL_REGIONS
        structural = {}
        for r in regions:
            init_fn = STRUCTURAL_INIT_REGISTRY.get(r)
            structural[r] = init_fn() if init_fn else _struct_default(r)
        return cls(structural=structural, projection=default_projection_genome())

    def get_structural(self, region: str) -> StructuralGenome:
        if region not in self.structural:
            self.structural[region] = _struct_default(region)
        return self.structural[region]

    def get_nt_modifiers(self, region: str) -> list[NTModifier]:
        return self.get_structural(region).nt_modifiers

    def get_receptors(self, region: str) -> list[ReceptorSpec]:
        return self.get_structural(region).receptors

    def apply_nt_modifiers(self, region: str, env: dict,
                           nt_levels: dict[str, float]) -> dict:
        """Apply NT modifier rules to a module's compiled env.

        This is called each forward tick: NT levels activate modifiers
        which shift module parameters in real-time.
        """
        modifiers = self.get_nt_modifiers(region)
        for mod in modifiers:
            level = nt_levels.get(mod.nt, 0.0)
            if mod.target_param in env:
                old_val = env[mod.target_param]
                if isinstance(old_val, (int, float)):
                    env[mod.target_param] = mod.apply(float(old_val), level)
        return env

    def to_dict(self) -> dict:
        return {
            'structural': {r: g.to_dict() for r, g in self.structural.items()},
            'projection': self.projection.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BrainDNA":
        structural = {r: StructuralGenome.from_dict(g)
                      for r, g in d.get('structural', {}).items()}
        proj = ProjectionGenome.from_dict(d['projection']) if 'projection' in d \
            else default_projection_genome()
        return cls(structural=structural, projection=proj)
