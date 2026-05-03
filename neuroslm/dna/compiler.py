"""Genome Compiler — each module's DNA encodes its algorithm directly.

The compilation pipeline:

    ModuleGenome (alleles = program steps as opcode logits + operands)
        │
        ▼  genome_to_tensors()  — genome IS the program, just reshaped
    Opcode Sequence (N_STEPS × [opcode_logits | operands])
        │
        ▼  decompile_to_lisp()  — at runtime, readable Lisp is generated
    Lisp Source Code (human-readable, inspectable)
        │
        ▼  execute_lisp()  — DSL interpreter runs the Lisp
    LispVM Environment (configures the module's behavior)

The genome IS the program. Each allele directly controls an opcode logit
or operand value. There are no static .lisp files — the genome is the
single source of truth.

At init, each module's genome encodes its correct initial algorithm:
  - Hippocampus: RECALL → GATE → WRITE_MEM → MODULATE
  - PFC: ATTEND → GATE → PROJECT → CMP_GT (working memory selection)
  - Critic: PREDICT → ERROR → CMP_GT → EMIT_NT (threat detection)
  - DMN: ATTEND → GATE → RECALL → OSCILLATE (mind-wandering)
  - Basal Ganglia: READ_NT → PROJECT → MODULATE → SIGMOID (action selection)
  - etc.

During training, backprop + evolution modify the alleles, changing the
algorithm. At any point you can decompile to see what the module does.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from dataclasses import dataclass, field
from typing import Optional

from .dsl import LispVM, LispError
from .latent_program import (
    LatentProgramDecoder, LatentProgramDecompiler,
    N_OPCODES, OPCODES, OPCODE_TO_IDX,
)


# ════════════════════════════════════════════════════════════════════════
# Program Step Builder — helper to construct genome alleles
# ════════════════════════════════════════════════════════════════════════

STEP_SIZE = N_OPCODES + 4  # opcode logits + 2 operands + 1 scalar + 1 gate
MAX_STEPS = 16


def _make_step(opcode: str, src: float = 0.0, dst: float = 0.0,
               val: float = 0.0, gate: float = 2.0) -> list[float]:
    """Build one program step as a flat list of STEP_SIZE floats.

    opcode: name from OPCODES (e.g. 'RECALL', 'GATE', 'PROJECT')
    src, dst: operand register indices (0-1 range, mapped to 0-7)
    val: scalar operand
    gate: pre-sigmoid gate value (2.0 ≈ 0.88 active, -2.0 ≈ 0.12 inactive)
    """
    logits = [-5.0] * N_OPCODES  # suppress all
    if opcode in OPCODE_TO_IDX:
        logits[OPCODE_TO_IDX[opcode]] = 5.0  # strongly activate this one
    return logits + [src, dst, val, gate]


def _make_nop() -> list[float]:
    """A no-op step (gate ≈ 0, so it's skipped)."""
    return _make_step('NOP', gate=-5.0)


def _pad_to(steps: list[list[float]], n: int = MAX_STEPS) -> list[float]:
    """Pad program to n steps with NOPs, then flatten."""
    while len(steps) < n:
        steps.append(_make_nop())
    flat = []
    for s in steps[:n]:
        flat.extend(s)
    return flat


# ════════════════════════════════════════════════════════════════════════
# Initial genome programs — what each module does at birth
# ════════════════════════════════════════════════════════════════════════

def _init_hippocampus() -> list[float]:
    """Hippocampus: recall associated memories, gate by ACh, encode new."""
    return _pad_to([
        _make_step('READ_NT',   src=0.5, val=0.5, gate=2.0),   # read ACh level
        _make_step('RECALL',    src=0.3, gate=2.0),              # query memory with dmn_query
        _make_step('GATE',      src=0.2, val=0.5, gate=2.0),    # gate recalls by novelty
        _make_step('MODULATE',  src=0.4, val=0.5, gate=2.0),    # ACh modulates recall strength
        _make_step('ATTEND',    src=0.3, dst=0.5, gate=1.5),    # attend over top-k recalls
        _make_step('WRITE_MEM', src=0.3, dst=0.5, gate=1.5),    # store new encoding
        _make_step('NORMALIZE', src=0.0, gate=1.0),              # normalize output
        _make_step('ERROR',     src=0.3, dst=0.5, gate=1.0),    # novelty = prediction error
    ])

def _init_pfc() -> list[float]:
    """PFC: attend to GWS slots + recalls, DA-gated selection of best."""
    return _pad_to([
        _make_step('READ_NT',   src=0.0, val=0.3, gate=2.0),   # read DA level
        _make_step('MODULATE',  src=0.0, val=0.6, gate=2.0),    # DA enhances working memory
        _make_step('ATTEND',    src=0.3, dst=0.5, gate=2.0),    # attend over GWS slots + recalls
        _make_step('GATE',      src=0.2, val=0.5, gate=2.0),    # GABA inhibition gate
        _make_step('PROJECT',   src=0.3, dst=0.5, gate=2.0),    # project to selection space
        _make_step('CMP_GT',    src=0.3, dst=0.2, gate=1.5),    # score candidates, pick best
        _make_step('NORMALIZE', src=0.0, gate=1.5),              # normalize selection
        _make_step('MODULATE',  src=0.3, val=0.3, gate=1.0),    # 5HT2A/M1 fine-tuning
    ])

def _init_dmn() -> list[float]:
    """DMN: generate query from GWS, mind-wander, produce thought."""
    return _pad_to([
        _make_step('ATTEND',    src=0.3, dst=0.5, gate=2.0),    # attend to GWS broadcast
        _make_step('GATE',      src=0.5, val=0.7, gate=2.0),    # blend with floating thought
        _make_step('PROJECT',   src=0.3, dst=0.3, gate=2.0),    # project to query space
        _make_step('RECALL',    src=0.3, gate=1.5),              # associative recall (wandering)
        _make_step('ADD',       src=0.3, dst=0.5, gate=1.5),    # creative noise injection
        _make_step('OSCILLATE', val=0.3, gate=1.0),              # default-mode oscillation
        _make_step('NORMALIZE', src=0.0, gate=1.5),              # normalize output query
        _make_step('MODULATE',  src=0.2, val=-0.4, gate=1.0),   # 5HT suppresses DMN
    ])

def _init_basal_ganglia() -> list[float]:
    """Basal Ganglia: DA-modulated action selection via direct/indirect."""
    return _pad_to([
        _make_step('READ_NT',   src=0.0, val=0.7, gate=2.0),   # read DA level
        _make_step('PROJECT',   src=0.3, dst=0.5, gate=2.0),    # project to action space
        _make_step('MODULATE',  src=0.0, val=0.7, gate=2.0),    # DA modulates selection
        _make_step('SIGMOID',   src=0.3, gate=2.0),              # action probabilities
        _make_step('CMP_GT',    src=0.3, dst=0.5, gate=1.5),    # direct vs indirect pathway
        _make_step('GATE',      src=0.3, val=0.5, gate=2.0),    # GABA inhibition
        _make_step('NORMALIZE', src=0.0, gate=1.5),              # normalize action
        _make_step('EMIT_NT',   val=0.7, gate=1.0),             # DA release on confidence
    ])

def _init_critic() -> list[float]:
    """Critic: evaluate world+self, predict threat, trigger NE surge."""
    return _pad_to([
        _make_step('PROJECT',   src=0.3, dst=0.5, gate=2.0),    # project world+self
        _make_step('PREDICT',   src=0.3, dst=0.5, gate=2.0),    # predict outcome
        _make_step('ERROR',     src=0.3, dst=0.5, gate=2.0),    # prediction error = threat
        _make_step('SIGMOID',   src=0.3, gate=2.0),              # squash to [0,1]
        _make_step('CMP_GT',    src=0.3, dst=0.7, gate=2.0),    # survival threshold check
        _make_step('EMIT_NT',   val=0.9, gate=1.5),             # NE surge if survival
        _make_step('GATE',      src=0.3, val=0.7, gate=1.0),    # caution bias
    ])

def _init_world_model() -> list[float]:
    """World Model: recurrent state update + forward prediction."""
    return _pad_to([
        _make_step('GATE',      src=0.3, val=0.5, gate=2.0),    # GRU-style update gate
        _make_step('TANH',      src=0.3, gate=2.0),              # candidate activation
        _make_step('ADD',       src=0.3, dst=0.5, gate=2.0),    # update hidden state
        _make_step('PREDICT',   src=0.3, dst=0.3, gate=2.0),    # forward prediction
        _make_step('NORMALIZE', src=0.0, gate=1.5),              # normalize state
        _make_step('ERROR',     src=0.3, dst=0.5, gate=1.0),    # surprise signal
    ])

def _init_self_model() -> list[float]:
    """Self Model: encode own action + NT state + thought → z_self."""
    return _pad_to([
        _make_step('READ_NT',   src=0.3, val=0.5, gate=2.0),   # read own NT state
        _make_step('GATE',      src=0.3, val=0.5, gate=2.0),    # GRU gate
        _make_step('TANH',      src=0.3, gate=2.0),              # candidate activation
        _make_step('ADD',       src=0.3, dst=0.5, gate=2.0),    # update self state
        _make_step('NORMALIZE', src=0.0, gate=1.5),              # normalize
    ])

def _init_language() -> list[float]:
    """Language Cortex: ACh/eCB-modulated causal transformer."""
    return _pad_to([
        _make_step('READ_NT',   src=0.5, val=0.3, gate=2.0),   # read ACh level
        _make_step('MODULATE',  src=0.0, val=0.3, gate=2.0),    # ACh sharpens attention
        _make_step('MODULATE',  src=0.1, val=-0.3, gate=1.5),   # eCB retrograde suppression
        _make_step('ADD',       src=0.3, dst=0.5, gate=2.0),    # inject DMN thought
        _make_step('ATTEND',    src=0.3, dst=0.5, gate=2.0),    # causal self-attention
        _make_step('NORMALIZE', src=0.0, gate=2.0),              # layer norm
        _make_step('PROJECT',   src=0.3, dst=0.5, gate=2.0),    # project to vocab
    ])

def _init_sensory() -> list[float]:
    """Sensory Cortex: extract salience from semantic stream."""
    return _pad_to([
        _make_step('RELU',      src=0.3, gate=2.0),
        _make_step('PROJECT',   src=0.3, dst=0.3, gate=2.0),
        _make_step('SIGMOID',   src=0.3, gate=2.0),              # salience [0,1]
        _make_step('GATE',      src=0.3, val=0.5, gate=1.5),    # habituation
    ])

def _init_association() -> list[float]:
    """Association Cortex: bind multi-modal streams."""
    return _pad_to([
        _make_step('BIND',      src=0.3, dst=0.5, gate=2.0),
        _make_step('PROJECT',   src=0.3, dst=0.3, gate=2.0),
        _make_step('NORMALIZE', src=0.0, gate=2.0),
    ])

def _init_thalamus() -> list[float]:
    """Thalamus: NE-sharpened content-aware MoE routing."""
    return _pad_to([
        _make_step('READ_NT',   src=0.3, val=0.5, gate=2.0),
        _make_step('PROJECT',   src=0.3, dst=0.5, gate=2.0),
        _make_step('MODULATE',  src=0.0, val=0.5, gate=2.0),    # NE sharpens
        _make_step('GATE',      src=0.3, val=-0.3, gate=1.5),   # GABA inhibits
        _make_step('SIGMOID',   src=0.3, gate=2.0),              # routing probs
    ])

def _init_gws() -> list[float]:
    """Global Workspace: NE-temperature competitive broadcast."""
    return _pad_to([
        _make_step('ATTEND',    src=0.3, dst=0.5, gate=2.0),    # competition
        _make_step('READ_NT',   src=0.3, val=0.5, gate=1.5),    # NE temperature
        _make_step('MODULATE',  src=0.0, val=0.5, gate=2.0),    # sharpen
        _make_step('GATE',      src=0.3, val=0.5, gate=2.0),    # broadcast threshold
        _make_step('NORMALIZE', src=0.0, gate=1.5),
    ])

def _init_forward_model() -> list[float]:
    """Forward Model: predict next world/self states from action."""
    return _pad_to([
        _make_step('PROJECT',   src=0.3, dst=0.5, gate=2.0),
        _make_step('PREDICT',   src=0.3, dst=0.5, gate=2.0),
        _make_step('PREDICT',   src=0.5, dst=0.3, gate=2.0),
        _make_step('NORMALIZE', src=0.0, gate=1.5),
    ])

def _init_evaluator() -> list[float]:
    """Evaluator: value estimate from predictions + NT state."""
    return _pad_to([
        _make_step('READ_NT',   src=0.3, val=0.5, gate=2.0),
        _make_step('PROJECT',   src=0.3, dst=0.5, gate=2.0),
        _make_step('ADD',       src=0.3, dst=0.5, gate=1.5),
        _make_step('SIGMOID',   src=0.3, gate=2.0),
    ])

def _init_motor() -> list[float]:
    """Motor Cortex: action → discrete command + language emission bias."""
    return _pad_to([
        _make_step('PROJECT',   src=0.3, dst=0.5, gate=2.0),
        _make_step('SIGMOID',   src=0.3, gate=2.0),
        _make_step('CMP_GT',    src=0.3, dst=0.5, gate=1.5),    # speak threshold
        _make_step('PROJECT',   src=0.3, dst=0.3, gate=2.0),    # language bias
        _make_step('GATE',      src=0.3, val=0.5, gate=1.5),    # inhibition
    ])

def _init_cerebellum() -> list[float]:
    """Cerebellum: fast error-driven forward model."""
    return _pad_to([
        _make_step('PREDICT',   src=0.3, dst=0.5, gate=2.0),
        _make_step('ERROR',     src=0.3, dst=0.5, gate=2.0),
        _make_step('MUL',       src=0.3, dst=0.5, gate=1.5),
        _make_step('ADD',       src=0.3, dst=0.5, gate=2.0),
        _make_step('NORMALIZE', src=0.0, gate=1.5),
    ])

def _init_entorhinal() -> list[float]:
    """Entorhinal Cortex: grid/place cells for conceptual navigation."""
    return _pad_to([
        _make_step('OSCILLATE', val=0.3, gate=2.0),
        _make_step('PROJECT',   src=0.3, dst=0.5, gate=2.0),
        _make_step('BIND',      src=0.3, dst=0.5, gate=1.5),
        _make_step('NORMALIZE', src=0.0, gate=1.5),
    ])

def _init_claustrum() -> list[float]:
    """Claustrum: cross-modal consciousness binding."""
    return _pad_to([
        _make_step('BIND',      src=0.3, dst=0.5, gate=2.0),
        _make_step('ATTEND',    src=0.3, dst=0.5, gate=2.0),
        _make_step('GATE',      src=0.3, val=0.5, gate=1.5),
        _make_step('NORMALIZE', src=0.0, gate=1.5),
    ])

def _init_cortical_sheet() -> list[float]:
    """Cortical Sheet: predictive processing columns."""
    return _pad_to([
        _make_step('PREDICT',   src=0.3, dst=0.5, gate=2.0),
        _make_step('ERROR',     src=0.3, dst=0.5, gate=2.0),
        _make_step('GATE',      src=0.3, val=0.5, gate=1.5),
        _make_step('ADD',       src=0.3, dst=0.5, gate=2.0),
        _make_step('NORMALIZE', src=0.0, gate=1.5),
    ])

def _init_neural_geometry() -> list[float]:
    """Neural Geometry: hyperdimensional VSA + fractal + manifold."""
    return _pad_to([
        _make_step('PROJECT',   src=0.3, dst=0.5, gate=2.0),
        _make_step('BIND',      src=0.3, dst=0.5, gate=2.0),
        _make_step('OSCILLATE', val=0.5, gate=1.5),
        _make_step('NORMALIZE', src=0.0, gate=1.5),
    ])

def _init_memory_consolidation() -> list[float]:
    """Memory Consolidation: episodic → consolidated (sleep replay)."""
    return _pad_to([
        _make_step('RECALL',    src=0.3, gate=2.0),              # replay episodes
        _make_step('GATE',      src=0.3, val=0.5, gate=2.0),    # importance gate
        _make_step('PREDICT',   src=0.3, dst=0.5, gate=1.5),    # generalize pattern
        _make_step('WRITE_MEM', src=0.3, dst=0.5, gate=2.0),    # write consolidated
        _make_step('NORMALIZE', src=0.0, gate=1.5),
    ])

def _init_comprehension_gate() -> list[float]:
    """Comprehension Gate: decides which observations become memories."""
    return _pad_to([
        _make_step('PROJECT',   src=0.3, dst=0.5, gate=2.0),    # embed observation
        _make_step('ERROR',     src=0.3, dst=0.5, gate=2.0),    # novelty vs existing
        _make_step('CMP_GT',    src=0.3, dst=0.1, gate=2.0),    # threshold check
        _make_step('GATE',      src=0.3, val=0.1, gate=2.0),    # sparse write gate (~10%)
    ])

def _init_narrative() -> list[float]:
    """Narrative System: temporal coherence and story-level reasoning."""
    return _pad_to([
        _make_step('ATTEND',    src=0.3, dst=0.5, gate=2.0),    # attend to recent events
        _make_step('RECALL',    src=0.3, gate=1.5),              # recall narrative context
        _make_step('BIND',      src=0.3, dst=0.5, gate=2.0),    # bind into coherent story
        _make_step('PREDICT',   src=0.3, dst=0.3, gate=1.5),    # predict next event
        _make_step('NORMALIZE', src=0.0, gate=1.5),
    ])


# Registry
_INIT_REGISTRY: dict[str, callable] = {
    'hippocampus':          _init_hippocampus,
    'pfc':                  _init_pfc,
    'dmn':                  _init_dmn,
    'basal_ganglia':        _init_basal_ganglia,
    'critic':               _init_critic,
    'world':                _init_world_model,
    'self_model':           _init_self_model,
    'language':             _init_language,
    'sensory':              _init_sensory,
    'association':          _init_association,
    'thalamus':             _init_thalamus,
    'gws':                  _init_gws,
    'forward_model':        _init_forward_model,
    'evaluator':            _init_evaluator,
    'motor':                _init_motor,
    'cerebellum':           _init_cerebellum,
    'entorhinal':           _init_entorhinal,
    'claustrum':            _init_claustrum,
    'cortical_sheet':       _init_cortical_sheet,
    'neural_geometry':      _init_neural_geometry,
    'memory_consolidation': _init_memory_consolidation,
    'comprehension_gate':   _init_comprehension_gate,
    'narrative':            _init_narrative,
}

ALL_REGIONS = list(_INIT_REGISTRY.keys())


# ════════════════════════════════════════════════════════════════════════
# ModuleGenome — the DNA IS the program
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ModuleGenome:
    """Per-module genome: alleles directly encode the opcode program.

    alleles[i * STEP_SIZE : (i+1) * STEP_SIZE] = one program step:
      - first N_OPCODES values = opcode logits (which operation)
      - next 4 values = [src_operand, dst_operand, scalar_val, gate]

    The genome IS the algorithm. Decompiling to Lisp is just formatting.
    """
    region: str
    alleles: list[float] = field(default_factory=list)
    generation: int = 0
    fitness: float = float("inf")
    parent_id: str | None = None
    id: str = field(default_factory=lambda: f"mg{random.randint(0, 1 << 32):08x}")

    @classmethod
    def from_init(cls, region: str) -> "ModuleGenome":
        """Create genome with the correct initial algorithm for this module."""
        init_fn = _INIT_REGISTRY.get(region)
        alleles = init_fn() if init_fn else _pad_to([])
        return cls(region=region, alleles=alleles)

    @classmethod
    def random(cls, region: str) -> "ModuleGenome":
        alleles = [random.gauss(0, 1) for _ in range(MAX_STEPS * STEP_SIZE)]
        return cls(region=region, alleles=alleles)

    def get_step(self, step: int) -> tuple[list[float], list[float]]:
        offset = step * STEP_SIZE
        chunk = self.alleles[offset:offset + STEP_SIZE]
        if len(chunk) < STEP_SIZE:
            chunk = chunk + [0.0] * (STEP_SIZE - len(chunk))
        return chunk[:N_OPCODES], chunk[N_OPCODES:]

    def dominant_opcode(self, step: int) -> str:
        logits, _ = self.get_step(step)
        idx = max(range(len(logits)), key=lambda i: logits[i])
        return OPCODES[idx] if idx < len(OPCODES) else 'NOP'

    def active_steps(self) -> list[int]:
        active = []
        for s in range(MAX_STEPS):
            _, ops = self.get_step(s)
            gate = 1.0 / (1.0 + math.exp(-ops[3]))
            if gate > 0.3:
                active.append(s)
        return active

    def summary(self) -> str:
        ops = [self.dominant_opcode(s) for s in self.active_steps()]
        return f"{self.region}: {' -> '.join(ops)}"

    def mutate(self, point_rate: float = 0.08, sigma: float = 0.3) -> "ModuleGenome":
        child = ModuleGenome(
            region=self.region, alleles=list(self.alleles),
            generation=self.generation + 1, parent_id=self.id)
        for i in range(len(child.alleles)):
            if random.random() < point_rate:
                child.alleles[i] += random.gauss(0.0, sigma)
        return child

    @staticmethod
    def crossover(a: "ModuleGenome", b: "ModuleGenome") -> "ModuleGenome":
        assert a.region == b.region
        alleles = []
        for s in range(MAX_STEPS):
            src = a if random.random() < 0.5 else b
            offset = s * STEP_SIZE
            alleles.extend(src.alleles[offset:offset + STEP_SIZE])
        return ModuleGenome(
            region=a.region, alleles=alleles,
            generation=max(a.generation, b.generation) + 1,
            parent_id=f"{a.id}+{b.id}")

    def to_dict(self) -> dict:
        return {"region": self.region, "id": self.id,
                "generation": self.generation, "fitness": self.fitness,
                "parent_id": self.parent_id, "alleles": self.alleles,
                "summary": self.summary()}

    @classmethod
    def from_dict(cls, d: dict) -> "ModuleGenome":
        return cls(region=d["region"], alleles=d.get("alleles", []),
                   generation=d.get("generation", 0),
                   fitness=d.get("fitness", float("inf")),
                   parent_id=d.get("parent_id"),
                   id=d.get("id", f"mg{random.randint(0, 1 << 32):08x}"))


# ════════════════════════════════════════════════════════════════════════
# Genome Compiler — genome → opcodes → Lisp → execute
# ════════════════════════════════════════════════════════════════════════

class GenomeCompiler(nn.Module):
    """Compiles ModuleGenomes: Genome alleles → Lisp → DSL Execute.

    No neural network needed — the genome alleles ARE the opcode logits.
    """

    def __init__(self, max_steps: int = MAX_STEPS):
        super().__init__()
        self.max_steps = max_steps
        self._compiled_lisp: dict[str, str] = {}
        self._compiled_envs: dict[str, dict] = {}

    def genome_to_tensors(self, genome: ModuleGenome
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.max_steps * STEP_SIZE
        alleles = genome.alleles[:n]
        if len(alleles) < n:
            alleles = alleles + [0.0] * (n - len(alleles))
        t = torch.tensor(alleles, dtype=torch.float32)
        t = t.view(self.max_steps, STEP_SIZE)
        return t[:, :N_OPCODES], t[:, N_OPCODES:]

    def decompile_to_lisp(self, genome: ModuleGenome, top_k: int = 2) -> str:
        opcode_logits, operands = self.genome_to_tensors(genome)
        lisp_src = LatentProgramDecompiler.decompile(
            opcode_logits, operands,
            region_name=genome.region, top_k=top_k)
        self._compiled_lisp[genome.region] = lisp_src
        return lisp_src

    def execute_lisp(self, region: str, lisp_src: str) -> dict:
        env = {"__region__": region, "layers": 2,
               "connections": "sequential", "learning_rule": "backprop",
               "projections": [], "nt_production": []}
        try:
            vm = LispVM()
            vm.run(lisp_src)
            for key in ("layers", "connections", "learning_rule",
                        "projections", "nt_production", "__region__"):
                if key in vm.env:
                    env[key] = vm.env[key]
            for key, val in vm.env.items():
                if callable(val) and not key.startswith("_") and key not in env:
                    env[key] = val
        except (LispError, Exception):
            pass
        self._compiled_envs[region] = env
        return env

    def compile(self, genome: ModuleGenome, top_k: int = 2) -> dict:
        lisp_src = self.decompile_to_lisp(genome, top_k=top_k)
        return self.execute_lisp(genome.region, lisp_src)

    def compile_batch(self, genomes: dict[str, ModuleGenome],
                      top_k: int = 2) -> dict[str, dict]:
        return {r: self.compile(g, top_k=top_k) for r, g in genomes.items()}

    def get_lisp(self, region: str) -> str:
        return self._compiled_lisp.get(region, f";; {region}: not compiled")

    def get_all_lisp(self) -> dict[str, str]:
        return dict(self._compiled_lisp)

    def get_env(self, region: str) -> dict:
        return self._compiled_envs.get(region, {})

    def save_all_lisp(self, directory: str, top_k: int = 2) -> str:
        import os
        os.makedirs(directory, exist_ok=True)
        for region, src in self._compiled_lisp.items():
            with open(os.path.join(directory, f"{region}_compiled.lisp"), "w") as f:
                f.write(src)
        return directory

    def compilation_report(self) -> str:
        lines = ["=" * 60, "  GENOME COMPILATION REPORT",
                 "  Genome → Opcode Sequence → Lisp → DSL Execute", "=" * 60]
        for region in sorted(self._compiled_lisp.keys()):
            lisp = self._compiled_lisp[region]
            n_lines = len(lisp.strip().splitlines())
            lines.append(f"\n  {region}:")
            lines.append(f"    Lisp lines: {n_lines}")
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# Module Genome Pool — per-region evolution
# ════════════════════════════════════════════════════════════════════════

class ModuleGenomePool:
    """Per-module genome pools with tournament evolution.

    Each module starts with from_init() = correct initial algorithm.
    Mutated variants compete; best algorithms survive.
    """

    def __init__(self, regions: list[str], pool_size: int = 4,
                 tournament_period: int = 200, mutation_rate: float = 0.08):
        self.pool_size = pool_size
        self.tournament_period = tournament_period
        self.mutation_rate = mutation_rate
        self.steps = 0
        self.pools: dict[str, list[ModuleGenome]] = {}
        self.active_idx: dict[str, int] = {}
        for region in regions:
            init_g = ModuleGenome.from_init(region)
            pool = [init_g]
            for _ in range(pool_size - 1):
                pool.append(init_g.mutate(point_rate=0.15, sigma=0.5))
            self.pools[region] = pool
            self.active_idx[region] = 0

    def active(self, region: str) -> ModuleGenome:
        return self.pools[region][self.active_idx[region]]

    def active_all(self) -> dict[str, ModuleGenome]:
        return {r: self.active(r) for r in self.pools}

    def report_all(self, loss: float):
        for region in self.pools:
            g = self.active(region)
            g.fitness = loss if math.isinf(g.fitness) else 0.97 * g.fitness + 0.03 * loss

    def step(self):
        self.steps += 1
        if self.steps % self.tournament_period == 0:
            for region in self.pools:
                self._evolve(region)
        if self.steps % max(1, self.tournament_period // self.pool_size) == 0:
            for region in self.pools:
                self.active_idx[region] = (
                    (self.active_idx[region] + 1) % self.pool_size)

    def _evolve(self, region: str):
        pool = self.pools[region]
        ranked = sorted(pool, key=lambda g: g.fitness)
        best = ranked[0]
        partner = ranked[1] if len(ranked) > 1 else best
        child = ModuleGenome.crossover(best, partner).mutate(self.mutation_rate)
        idx = pool.index(ranked[-1])
        pool[idx] = child

    def state(self) -> dict:
        return {"pools": {r: [g.to_dict() for g in gs]
                          for r, gs in self.pools.items()},
                "active_idx": dict(self.active_idx),
                "steps": self.steps}

    @classmethod
    def from_state(cls, state: dict) -> "ModuleGenomePool":
        regions = list(state["pools"].keys())
        pool_size = len(next(iter(state["pools"].values())))
        obj = cls.__new__(cls)
        obj.pool_size = pool_size
        obj.tournament_period = 200
        obj.mutation_rate = 0.08
        obj.pools = {r: [ModuleGenome.from_dict(d) for d in gs]
                     for r, gs in state["pools"].items()}
        obj.active_idx = state["active_idx"]
        obj.steps = state["steps"]
        return obj
