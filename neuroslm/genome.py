"""DNA / genome for the DMN cognitive loop.

The DMN's behavior is parameterized by a fixed-length genome consisting of:
  - A vector of float "alleles"   (continuous hyperparameters in [0,1])
  - A vector of integer "opcodes" (discrete choices, e.g., which receptor to use)

The genome is *not* trained by SGD. Instead, it evolves by:
  - random point mutation (rare)
  - crossover with a sibling genome (when 'reproduce' is called)
  - selection pressure: a fitness score (lower LM loss + higher reward)
    is tracked, and after N steps the best genome out of a small pool wins.

The DMN reads the genome each tick to set:
  GENE_NOVELTY_THRESHOLD, GENE_THOUGHT_ALPHA, GENE_MAX_THINK_STEPS,
  GENE_REPLACE_BIAS,      GENE_HIPPO_TOPK_FRAC, GENE_DA_GAIN,
  GENE_NE_GAIN,           GENE_5HT_GAIN,        GENE_ACH_GAIN,
  GENE_THREAT_SENSITIVITY,GENE_WANDER_RATE,     GENE_DMN_PERIOD_FRAC,

Opcodes pick categorical alternatives:
  OP_PRIMARY_RECEPTOR (which NT receptor on PFC dominates)
  OP_ROUTE_BIAS       (which thalamic stream is preferred when undecided)
  OP_THOUGHT_UPDATE   (additive vs replace vs gated)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import random
from typing import Sequence

import torch


GENE_NAMES = (
    "novelty_threshold", "thought_alpha",     "max_think_steps",
    "replace_bias",      "hippo_topk_frac",   "da_gain",
    "ne_gain",           "ht5_gain",          "ach_gain",
    "threat_sensitivity","wander_rate",       "dmn_period_frac",
)
N_GENES = len(GENE_NAMES)
GENE_INDEX = {n: i for i, n in enumerate(GENE_NAMES)}

OPCODE_NAMES = ("primary_receptor", "route_bias", "thought_update")
N_OPS = len(OPCODE_NAMES)
OPCODE_RANGES = (4, 5, 3)   # number of alternatives per opcode


@dataclass
class Genome:
    alleles: list[float] = field(default_factory=list)
    opcodes: list[int]   = field(default_factory=list)
    # Track lineage / fitness
    generation: int = 0
    fitness: float = float("inf")     # lower = better (loss-based)
    parent_id: str | None = None
    id: str = field(default_factory=lambda: f"g{random.randint(0, 1<<32):08x}")

    @classmethod
    def random(cls) -> "Genome":
        g = cls(
            alleles=[random.random() for _ in range(N_GENES)],
            opcodes=[random.randrange(r) for r in OPCODE_RANGES],
        )
        return g

    def get(self, name: str) -> float:
        return self.alleles[GENE_INDEX[name]]

    def get_op(self, name: str) -> int:
        return self.opcodes[OPCODE_NAMES.index(name)]

    # ---- Evolution ops ----
    def mutate(self, point_rate: float = 0.05, sigma: float = 0.15) -> "Genome":
        child = Genome(
            alleles=list(self.alleles),
            opcodes=list(self.opcodes),
            generation=self.generation + 1,
            parent_id=self.id,
        )
        for i in range(N_GENES):
            if random.random() < point_rate:
                child.alleles[i] = max(0.0, min(1.0,
                    child.alleles[i] + random.gauss(0.0, sigma)))
        for i, r in enumerate(OPCODE_RANGES):
            if random.random() < point_rate * 0.5:
                child.opcodes[i] = random.randrange(r)
        return child

    @staticmethod
    def crossover(a: "Genome", b: "Genome") -> "Genome":
        child = Genome(
            alleles=[ai if random.random() < 0.5 else bi
                     for ai, bi in zip(a.alleles, b.alleles)],
            opcodes=[ai if random.random() < 0.5 else bi
                     for ai, bi in zip(a.opcodes, b.opcodes)],
            generation=max(a.generation, b.generation) + 1,
            parent_id=f"{a.id}+{b.id}",
        )
        return child

    def to_dict(self) -> dict:
        return {
            "id": self.id, "generation": self.generation,
            "fitness": self.fitness, "parent_id": self.parent_id,
            "alleles": dict(zip(GENE_NAMES, self.alleles)),
            "opcodes": dict(zip(OPCODE_NAMES, self.opcodes)),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(
            alleles=[d["alleles"][n] for n in GENE_NAMES],
            opcodes=[d["opcodes"][n] for n in OPCODE_NAMES],
            generation=d.get("generation", 0),
            fitness=d.get("fitness", float("inf")),
            parent_id=d.get("parent_id"),
            id=d.get("id", f"g{random.randint(0, 1<<32):08x}"),
        )


class GenePool:
    """Maintains a small pool of competing genomes; runs a tournament every
    `tournament_period` steps and replaces the worst with a mutated copy of
    the best. The active genome (used by the DMN) is always the current best.
    """
    def __init__(self, pool_size: int = 4, tournament_period: int = 200,
                 mutation_rate: float = 0.05, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
        self.pool: list[Genome] = [Genome.random() for _ in range(pool_size)]
        self.tournament_period = tournament_period
        self.mutation_rate = mutation_rate
        self.steps = 0
        self.active_idx = 0
        self.history: list[dict] = []

    def active(self) -> Genome:
        return self.pool[self.active_idx]

    def report(self, loss: float):
        """Record fitness for the active genome (running min)."""
        g = self.active()
        # EMA of loss for stability
        if math.isinf(g.fitness):
            g.fitness = loss
        else:
            g.fitness = 0.97 * g.fitness + 0.03 * loss
        self.steps += 1
        if self.steps % self.tournament_period == 0:
            self._evolve()
        # Round-robin: try a different genome next interval slice
        if self.steps % max(1, self.tournament_period // len(self.pool)) == 0:
            self.active_idx = (self.active_idx + 1) % len(self.pool)

    def _evolve(self):
        ranked = sorted(self.pool, key=lambda g: g.fitness)
        best, worst = ranked[0], ranked[-1]
        # Replace worst with crossover(best, runner-up) then mutate
        partner = ranked[1] if len(ranked) > 1 else best
        child = Genome.crossover(best, partner).mutate(self.mutation_rate)
        # Insert (replacing worst position)
        idx = self.pool.index(worst)
        self.pool[idx] = child
        self.history.append({
            "step": self.steps, "best": best.to_dict(),
            "child": child.to_dict(),
        })

    def state(self) -> dict:
        return {
            "pool": [g.to_dict() for g in self.pool],
            "steps": self.steps, "active_idx": self.active_idx,
        }

    @classmethod
    def from_state(cls, state: dict) -> "GenePool":
        gp = cls(pool_size=len(state["pool"]))
        gp.pool = [Genome.from_dict(d) for d in state["pool"]]
        gp.steps = state["steps"]
        gp.active_idx = state["active_idx"]
        return gp
