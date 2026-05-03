"""Genome Compiler — the central pipeline that makes every module inspectable.

The compilation pipeline for each brain module:

    ModuleGenome (alleles + opcodes)
        │
        ▼  compile()
    Latent Embedding (dense vector, differentiable)
        │
        ▼  decompile()
    Lisp Source Code (human-readable, inspectable)
        │
        ▼  execute()
    LispVM Environment (configures the module's behavior)

This is the key architectural insight: the genome IS the source of truth.
Training optimizes the latent embeddings via backprop, but at any point
we can decompile them back to readable Lisp and see exactly what each
module is doing. The Lisp is then executed by the DSL interpreter to
actually configure module behavior (projections, NT production, scoring
functions, etc.)

Scientific basis:
  - Genetic regulatory networks → gene expression → protein function
  - Here: genome → latent program → Lisp algorithm → module config
  - The latent space is the "protein folding" step: compact representation
    that unfolds into a functional algorithm
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from dataclasses import dataclass, field
from typing import Optional

from .dsl import LispVM, LispError, parse
from .latent_program import (
    LatentProgramDecoder, LatentProgramDecompiler,
    NeuralProgramInterpreter, N_OPCODES, OPCODES,
)


# ════════════════════════════════════════════════════════════════════════
# Per-module genome
# ════════════════════════════════════════════════════════════════════════

# Each module type has its own set of gene names that parameterize its
# behavior. These are the "alleles" — continuous values in [0, 1] that
# the genome compiler maps into the latent embedding.

MODULE_GENE_SPECS: dict[str, tuple[str, ...]] = {
    "language": (
        "attention_sharpness", "ach_sensitivity", "ecb_suppression",
        "thought_injection_gain", "layer_dropout_rate", "temperature",
        "repetition_penalty", "context_window_usage",
    ),
    "sensory": (
        "salience_threshold", "noise_gate", "contrast_gain",
        "habituation_rate", "novelty_boost",
    ),
    "association": (
        "binding_strength", "cross_modal_gain", "sparsity",
        "lateral_inhibition", "pattern_completion_threshold",
    ),
    "thalamus": (
        "routing_sharpness", "ne_sensitivity", "gaba_inhibition",
        "default_route_bias", "attention_gate",
    ),
    "world": (
        "prediction_horizon", "update_rate", "surprise_sensitivity",
        "model_confidence", "imagination_gain", "stability",
    ),
    "self_model": (
        "introspection_depth", "agency_threshold", "body_model_gain",
        "nt_sensitivity", "identity_stability",
    ),
    "pfc": (
        "working_mem_slots", "selection_threshold", "da_sensitivity",
        "inhibition_strength", "planning_depth", "cognitive_flexibility",
        "goal_persistence", "conflict_monitoring",
    ),
    "dmn": (
        "novelty_threshold", "thought_alpha", "wandering_rate",
        "creativity_gain", "association_depth", "rumination_brake",
        "insight_threshold", "default_mode_period",
    ),
    "hippocampus": (
        "recall_topk", "encoding_threshold", "consolidation_rate",
        "pattern_separation", "pattern_completion", "ach_gate",
        "replay_probability", "novelty_signal_gain",
    ),
    "basal_ganglia": (
        "action_threshold", "da_gain", "exploration_rate",
        "habit_strength", "gaba_inhibition", "direct_pathway_bias",
        "indirect_pathway_bias", "reward_prediction_lr",
    ),
    "critic": (
        "threat_sensitivity", "safety_threshold", "survival_urgency",
        "world_model_trust", "self_model_trust", "caution_bias",
    ),
    "forward_model": (
        "prediction_steps", "error_sensitivity", "learning_rate",
        "model_complexity", "uncertainty_estimate",
    ),
    "evaluator": (
        "value_discount", "nt_weight", "risk_sensitivity",
        "reward_horizon", "optimism_bias",
    ),
    "motor": (
        "speak_threshold", "action_confidence", "inhibition_gate",
        "fluency_bias", "emission_temperature",
    ),
    "cerebellum": (
        "timing_precision", "error_correction_rate", "forward_model_gain",
        "sequence_learning_rate", "adaptation_speed",
    ),
    "entorhinal": (
        "grid_resolution", "place_field_width", "path_integration_gain",
        "boundary_sensitivity", "conceptual_distance_scale",
    ),
    "claustrum": (
        "binding_threshold", "cross_modal_sync", "integration_window",
        "modality_weights_entropy", "consciousness_broadcast",
    ),
    "gws": (
        "competition_sharpness", "broadcast_threshold", "slot_capacity",
        "ne_temperature_gain", "ignition_threshold",
    ),
    "cortical_sheet": (
        "prediction_gain", "lateral_spread", "apical_weight",
        "basal_weight", "minicolumn_competition",
    ),
    "neural_geometry": (
        "fractal_depth", "manifold_curvature", "vsa_sparsity",
        "geodesic_step_size", "dimensionality",
    ),
}

# Opcode choices per module (categorical alternatives)
MODULE_OPCODE_SPECS: dict[str, tuple[tuple[str, int], ...]] = {
    "pfc": (
        ("selection_strategy", 4),      # argmax | softmax | tournament | threshold
        ("memory_update", 3),           # replace | gate | accumulate
        ("conflict_resolution", 3),     # inhibit_loser | blend | defer
    ),
    "dmn": (
        ("thought_update", 3),          # additive | replace | gated
        ("wandering_mode", 3),          # random | associative | goal_directed
        ("creativity_source", 3),       # noise | memory | recombination
    ),
    "hippocampus": (
        ("encoding_rule", 3),           # hebbian | surprise | gated
        ("recall_strategy", 3),         # nearest | spread_activation | pattern_complete
        ("consolidation_mode", 3),      # replay | interleave | compress
    ),
    "basal_ganglia": (
        ("action_selection", 4),        # winner_take_all | softmax | e_greedy | ucb
        ("pathway_balance", 3),         # direct_bias | indirect_bias | balanced
        ("learning_signal", 3),         # td_error | reward | curiosity
    ),
    "thalamus": (
        ("routing_mode", 4),            # content | salience | round_robin | learned
        ("gate_type", 3),               # hard | soft | adaptive
    ),
    "language": (
        ("attention_pattern", 3),       # causal | sliding_window | sparse
        ("head_specialization", 3),     # uniform | specialized | adaptive
    ),
}


@dataclass
class ModuleGenome:
    """Per-module genome: the DNA that compiles into a module's algorithm.

    Each module has:
      - region: which brain module this genome controls
      - alleles: float[0,1] vector, one per gene in MODULE_GENE_SPECS[region]
      - opcodes: int vector, categorical choices from MODULE_OPCODE_SPECS[region]
      - generation, fitness, lineage tracking
    """
    region: str
    alleles: list[float] = field(default_factory=list)
    opcodes: list[int] = field(default_factory=list)
    generation: int = 0
    fitness: float = float("inf")
    parent_id: str | None = None
    id: str = field(default_factory=lambda: f"mg{random.randint(0, 1 << 32):08x}")

    @classmethod
    def random(cls, region: str) -> "ModuleGenome":
        gene_names = MODULE_GENE_SPECS.get(region, ())
        opcode_specs = MODULE_OPCODE_SPECS.get(region, ())
        return cls(
            region=region,
            alleles=[random.random() for _ in gene_names],
            opcodes=[random.randrange(r) for _, r in opcode_specs],
        )

    def get(self, name: str) -> float:
        """Get allele value by gene name."""
        gene_names = MODULE_GENE_SPECS.get(self.region, ())
        if name in gene_names:
            idx = gene_names.index(name)
            return self.alleles[idx] if idx < len(self.alleles) else 0.5
        return 0.5

    def get_op(self, name: str) -> int:
        """Get opcode value by name."""
        opcode_specs = MODULE_OPCODE_SPECS.get(self.region, ())
        op_names = [n for n, _ in opcode_specs]
        if name in op_names:
            idx = op_names.index(name)
            return self.opcodes[idx] if idx < len(self.opcodes) else 0
        return 0

    def mutate(self, point_rate: float = 0.08, sigma: float = 0.15) -> "ModuleGenome":
        child = ModuleGenome(
            region=self.region,
            alleles=list(self.alleles),
            opcodes=list(self.opcodes),
            generation=self.generation + 1,
            parent_id=self.id,
        )
        for i in range(len(child.alleles)):
            if random.random() < point_rate:
                child.alleles[i] = max(0.0, min(1.0,
                    child.alleles[i] + random.gauss(0.0, sigma)))
        opcode_specs = MODULE_OPCODE_SPECS.get(self.region, ())
        for i, (_, r) in enumerate(opcode_specs):
            if i < len(child.opcodes) and random.random() < point_rate * 0.5:
                child.opcodes[i] = random.randrange(r)
        return child

    @staticmethod
    def crossover(a: "ModuleGenome", b: "ModuleGenome") -> "ModuleGenome":
        assert a.region == b.region
        return ModuleGenome(
            region=a.region,
            alleles=[ai if random.random() < 0.5 else bi
                     for ai, bi in zip(a.alleles, b.alleles)],
            opcodes=[ai if random.random() < 0.5 else bi
                     for ai, bi in zip(a.opcodes, b.opcodes)],
            generation=max(a.generation, b.generation) + 1,
            parent_id=f"{a.id}+{b.id}",
        )

    def to_dict(self) -> dict:
        gene_names = MODULE_GENE_SPECS.get(self.region, ())
        opcode_specs = MODULE_OPCODE_SPECS.get(self.region, ())
        return {
            "region": self.region,
            "id": self.id,
            "generation": self.generation,
            "fitness": self.fitness,
            "parent_id": self.parent_id,
            "alleles": dict(zip(gene_names, self.alleles)),
            "opcodes": dict(zip([n for n, _ in opcode_specs], self.opcodes)),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModuleGenome":
        region = d["region"]
        gene_names = MODULE_GENE_SPECS.get(region, ())
        opcode_specs = MODULE_OPCODE_SPECS.get(region, ())
        return cls(
            region=region,
            alleles=[d.get("alleles", {}).get(n, 0.5) for n in gene_names],
            opcodes=[d.get("opcodes", {}).get(n, 0) for n, _ in opcode_specs],
            generation=d.get("generation", 0),
            fitness=d.get("fitness", float("inf")),
            parent_id=d.get("parent_id"),
            id=d.get("id", f"mg{random.randint(0, 1 << 32):08x}"),
        )


# ════════════════════════════════════════════════════════════════════════
# Genome Compiler
# ════════════════════════════════════════════════════════════════════════

class GenomeCompiler(nn.Module):
    """Compiles ModuleGenomes through the full pipeline:

        Genome → Latent Embedding → Lisp Source → DSL Execution

    The compiler has three stages:

    1. **Genome → Latent** (compile_to_latent):
       A learned linear map from the genome's allele vector into the
       latent program space. This is the "gene expression" step — it
       determines how alleles fold into an algorithm.

    2. **Latent → Lisp** (decompile_to_lisp):
       Uses LatentProgramDecoder + LatentProgramDecompiler to produce
       human-readable Lisp from the latent vector. This is the
       "transcription" step — the algorithm becomes inspectable.

    3. **Lisp → Execute** (execute_lisp):
       Feeds the decompiled Lisp into the DSL interpreter (LispVM).
       The VM populates an environment with the module's config:
       layers, connections, projections, NT production rules, scoring
       functions, etc. This is the "translation" step — the algorithm
       becomes active.

    The full pipeline is differentiable through stages 1-2 (for
    backprop optimization of the latent space), and stage 3 provides
    the symbolic execution that makes it interpretable.
    """

    def __init__(self, d_latent: int = 128, max_steps: int = 16):
        super().__init__()
        self.d_latent = d_latent
        self.max_steps = max_steps

        # Per-region genome→latent projection (learned gene expression)
        self._region_projections: nn.ModuleDict = nn.ModuleDict()

        # Shared decoder: latent → opcode sequence
        self.decoder = LatentProgramDecoder(d_latent, max_steps)

        # Cache: region → last compiled Lisp source
        self._compiled_lisp: dict[str, str] = {}
        # Cache: region → last LispVM environment
        self._compiled_envs: dict[str, dict] = {}
        # Cache: region → latent vector
        self._compiled_latents: dict[str, torch.Tensor] = {}

    def _get_projection(self, region: str) -> nn.Linear:
        """Lazily create a genome→latent projection for a region."""
        if region not in self._region_projections:
            gene_names = MODULE_GENE_SPECS.get(region, ())
            opcode_specs = MODULE_OPCODE_SPECS.get(region, ())
            # Input dim = n_alleles + n_opcodes (opcodes are one-hot expanded)
            n_alleles = len(gene_names)
            n_opcodes_total = sum(r for _, r in opcode_specs)
            input_dim = n_alleles + n_opcodes_total
            # Ensure at least 1 input dim
            input_dim = max(input_dim, 1)
            proj = nn.Linear(input_dim, self.d_latent)
            # Initialize so initial genomes produce diverse latents
            nn.init.orthogonal_(proj.weight, gain=0.5)
            nn.init.zeros_(proj.bias)
            self._region_projections[region] = proj
        return self._region_projections[region]

    def genome_to_vector(self, genome: ModuleGenome) -> torch.Tensor:
        """Convert a ModuleGenome to a flat input vector.

        Alleles are passed through directly (continuous [0,1]).
        Opcodes are one-hot encoded and concatenated.
        """
        parts = list(genome.alleles)
        opcode_specs = MODULE_OPCODE_SPECS.get(genome.region, ())
        for i, (_, n_choices) in enumerate(opcode_specs):
            one_hot = [0.0] * n_choices
            if i < len(genome.opcodes):
                one_hot[genome.opcodes[i] % n_choices] = 1.0
            parts.extend(one_hot)
        if not parts:
            parts = [0.5]  # fallback
        return torch.tensor(parts, dtype=torch.float32)

    # ── Stage 1: Genome → Latent ──────────────────────────────────────

    def compile_to_latent(self, genome: ModuleGenome) -> torch.Tensor:
        """Gene expression: genome alleles → latent program embedding.

        This is a learned mapping — the projection weights are trained
        alongside the rest of the model, so the compiler learns to
        "express" genomes into useful algorithm embeddings.
        """
        proj = self._get_projection(genome.region)
        vec = self.genome_to_vector(genome).to(next(proj.parameters()).device)
        latent = proj(vec.unsqueeze(0)).squeeze(0)  # (d_latent,)
        latent = torch.tanh(latent)  # bound to [-1, 1]
        self._compiled_latents[genome.region] = latent.detach()
        return latent

    # ── Stage 2: Latent → Lisp ────────────────────────────────────────

    def decompile_to_lisp(self, region: str,
                          latent: torch.Tensor,
                          top_k: int = 2) -> str:
        """Transcription: latent embedding → human-readable Lisp source.

        The decoder maps the dense vector to an opcode sequence, which
        the decompiler renders as Lisp s-expressions.
        """
        with torch.no_grad():
            opcode_logits, operands = self.decoder(latent.unsqueeze(0))
        lisp_src = LatentProgramDecompiler.decompile(
            opcode_logits.squeeze(0), operands.squeeze(0),
            region_name=region, top_k=top_k,
        )
        self._compiled_lisp[region] = lisp_src
        return lisp_src

    # ── Stage 3: Lisp → Execute ───────────────────────────────────────

    def execute_lisp(self, region: str, lisp_src: str) -> dict:
        """Translation: Lisp source → executed DSL environment.

        Runs the decompiled Lisp through the DSL interpreter. The
        resulting environment contains the module's runtime config:
        projections, NT production rules, scoring functions, etc.

        If the decompiled Lisp fails to parse/execute (because the
        latent is still random early in training), returns a safe
        default environment.
        """
        env = {
            "__region__": region,
            "layers": 2,
            "connections": "sequential",
            "learning_rule": "backprop",
            "projections": [],
            "nt_production": [],
        }
        try:
            vm = LispVM()
            vm.run(lisp_src)
            # Extract configured values from VM environment
            for key in ("layers", "connections", "learning_rule",
                        "projections", "nt_production",
                        "__region__"):
                if key in vm.env:
                    env[key] = vm.env[key]
            # Also extract any defun'd functions
            for key, val in vm.env.items():
                if callable(val) and not key.startswith("_") and key not in env:
                    env[key] = val
        except (LispError, Exception):
            # Early in training, decompiled Lisp may be garbage.
            # That's fine — the defaults keep the module functional.
            pass

        self._compiled_envs[region] = env
        return env

    # ── Full pipeline ─────────────────────────────────────────────────

    def compile(self, genome: ModuleGenome, top_k: int = 2) -> dict:
        """Full compilation pipeline:
            Genome → Latent → Lisp → Execute → Environment

        Returns the DSL environment dict that configures the module.
        """
        # Stage 1: gene expression
        latent = self.compile_to_latent(genome)
        # Stage 2: transcription
        lisp_src = self.decompile_to_lisp(genome.region, latent, top_k=top_k)
        # Stage 3: translation
        env = self.execute_lisp(genome.region, lisp_src)
        return env

    def compile_batch(self, genomes: dict[str, ModuleGenome],
                      top_k: int = 2) -> dict[str, dict]:
        """Compile all module genomes. Returns {region: env_dict}."""
        return {region: self.compile(genome, top_k=top_k)
                for region, genome in genomes.items()}

    # ── Inspection ────────────────────────────────────────────────────

    def get_lisp(self, region: str) -> str:
        """Get the last compiled Lisp source for a region."""
        return self._compiled_lisp.get(region, f";; {region}: not yet compiled")

    def get_all_lisp(self) -> dict[str, str]:
        """Get all last-compiled Lisp sources."""
        return dict(self._compiled_lisp)

    def get_env(self, region: str) -> dict:
        """Get the last executed environment for a region."""
        return self._compiled_envs.get(region, {})

    def get_latent(self, region: str) -> Optional[torch.Tensor]:
        """Get the last compiled latent vector for a region."""
        return self._compiled_latents.get(region)

    def save_all_lisp(self, directory: str) -> str:
        """Write all compiled Lisp files to disk."""
        import os
        os.makedirs(directory, exist_ok=True)
        for region, src in self._compiled_lisp.items():
            path = os.path.join(directory, f"{region}_compiled.lisp")
            with open(path, "w") as f:
                f.write(src)
        return directory

    def compilation_report(self) -> str:
        """Human-readable report of all compiled modules."""
        lines = ["=" * 60, "  GENOME COMPILATION REPORT", "=" * 60]
        for region in sorted(self._compiled_lisp.keys()):
            lisp = self._compiled_lisp[region]
            env = self._compiled_envs.get(region, {})
            n_lines = len(lisp.strip().splitlines())
            n_proj = len(env.get("projections", []))
            n_nt = len(env.get("nt_production", []))
            n_fns = sum(1 for v in env.values() if callable(v))
            lines.append(f"\n  {region}:")
            lines.append(f"    Lisp lines:    {n_lines}")
            lines.append(f"    Projections:   {n_proj}")
            lines.append(f"    NT rules:      {n_nt}")
            lines.append(f"    Functions:     {n_fns}")
            lines.append(f"    Learning rule: {env.get('learning_rule', '?')}")
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# Module Genome Pool — evolution per region
# ════════════════════════════════════════════════════════════════════════

class ModuleGenomePool:
    """Maintains a pool of competing genomes per brain module.

    Like GenePool but tracks separate genomes for each region.
    Evolution is region-independent: each module evolves its own DNA.
    """

    def __init__(self, regions: list[str], pool_size: int = 4,
                 tournament_period: int = 200, mutation_rate: float = 0.08):
        self.pool_size = pool_size
        self.tournament_period = tournament_period
        self.mutation_rate = mutation_rate
        self.steps = 0

        # Per-region pools
        self.pools: dict[str, list[ModuleGenome]] = {}
        self.active_idx: dict[str, int] = {}
        for region in regions:
            self.pools[region] = [ModuleGenome.random(region)
                                  for _ in range(pool_size)]
            self.active_idx[region] = 0

    def active(self, region: str) -> ModuleGenome:
        """Get the currently active genome for a region."""
        return self.pools[region][self.active_idx[region]]

    def active_all(self) -> dict[str, ModuleGenome]:
        """Get active genomes for all regions."""
        return {r: self.active(r) for r in self.pools}

    def report(self, region: str, loss: float):
        """Record fitness for a region's active genome."""
        g = self.active(region)
        if math.isinf(g.fitness):
            g.fitness = loss
        else:
            g.fitness = 0.97 * g.fitness + 0.03 * loss

    def report_all(self, loss: float):
        """Record same loss for all regions (shared training signal)."""
        for region in self.pools:
            self.report(region, loss)

    def step(self):
        """Advance evolution clock. Run tournaments when due."""
        self.steps += 1
        if self.steps % self.tournament_period == 0:
            for region in self.pools:
                self._evolve(region)
        # Round-robin active genome
        if self.steps % max(1, self.tournament_period // self.pool_size) == 0:
            for region in self.pools:
                self.active_idx[region] = (
                    (self.active_idx[region] + 1) % self.pool_size
                )

    def _evolve(self, region: str):
        pool = self.pools[region]
        ranked = sorted(pool, key=lambda g: g.fitness)
        best, worst = ranked[0], ranked[-1]
        partner = ranked[1] if len(ranked) > 1 else best
        child = ModuleGenome.crossover(best, partner).mutate(self.mutation_rate)
        idx = pool.index(worst)
        pool[idx] = child

    def state(self) -> dict:
        return {
            "pools": {r: [g.to_dict() for g in gs]
                      for r, gs in self.pools.items()},
            "active_idx": dict(self.active_idx),
            "steps": self.steps,
        }

    @classmethod
    def from_state(cls, state: dict) -> "ModuleGenomePool":
        regions = list(state["pools"].keys())
        pool = cls(regions, pool_size=len(next(iter(state["pools"].values()))))
        pool.pools = {r: [ModuleGenome.from_dict(d) for d in gs]
                      for r, gs in state["pools"].items()}
        pool.active_idx = state["active_idx"]
        pool.steps = state["steps"]
        return pool
