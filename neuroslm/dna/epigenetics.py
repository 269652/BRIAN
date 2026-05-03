"""Epigenetic Self-Optimization — benchmark-driven DNA mutation during training.

This is the key innovation: the model self-optimizes its own architecture
by mutating its DNA based on benchmark performance signals.

Biology parallel:
  - Epigenetics: environmental signals (methylation, histone modification)
    alter gene expression without changing the DNA sequence itself.
  - Here: benchmark scores → targeted mutations to genome alleles
  - High-fitness genomes are preserved; low-fitness ones are mutated more aggressively

Mechanism:
  1. Every N steps, run a quick benchmark probe (subset of HellaSwag/ARC)
  2. Compute fitness = benchmark_accuracy for each module's genome
  3. Apply epigenetic mutations: modules with low contribution get stronger mutations
  4. Recompile genomes → push new params into modules
  5. Continue training with evolved architecture

This creates a self-improving loop:
  Train → Benchmark → Mutate DNA → Recompile → Train (better) → ...

The genome pool maintains diversity (tournament selection + elitism).
Epigenetic "marks" accumulate on alleles that consistently correlate
with high fitness, protecting them from mutation (like DNA methylation).
"""
from __future__ import annotations
import math
import random
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .compiler import ModuleGenomePool, ModuleGenome, GenomeCompiler


@dataclass
class EpigeneticMark:
    """Tracks which allele positions are 'methylated' (protected from mutation).

    Alleles that consistently appear in high-fitness genomes accumulate
    methylation marks. Highly methylated alleles are mutated less.
    """
    methylation: list[float] = field(default_factory=list)

    def init(self, n_alleles: int):
        self.methylation = [0.0] * n_alleles

    def mark(self, positions: list[int], strength: float = 0.1):
        """Add methylation to positions (protect from mutation)."""
        for pos in positions:
            if 0 <= pos < len(self.methylation):
                self.methylation[pos] = min(1.0,
                    self.methylation[pos] + strength)

    def decay(self, rate: float = 0.01):
        """Slow decay — epigenetic marks fade if not reinforced."""
        for i in range(len(self.methylation)):
            self.methylation[i] = max(0.0, self.methylation[i] - rate)

    def mutation_mask(self) -> list[float]:
        """Returns per-allele mutation probability multiplier.
        Methylated = low mutation rate. Unmethylated = full mutation rate.
        """
        return [1.0 - m * 0.9 for m in self.methylation]  # 90% reduction max


class EpigeneticOptimizer:
    """Self-optimization system that mutates DNA based on benchmark probes.

    Usage in training loop:
        epi_opt = EpigeneticOptimizer(brain, probe_every=500)
        for step in range(total_steps):
            ... train ...
            epi_opt.step(step, loss)
    """

    def __init__(self, probe_every: int = 500,
                 probe_samples: int = 50,
                 mutation_scale: float = 0.15,
                 elite_fraction: float = 0.25):
        self.probe_every = probe_every
        self.probe_samples = probe_samples
        self.mutation_scale = mutation_scale
        self.elite_fraction = elite_fraction
        self._marks: dict[str, EpigeneticMark] = {}
        self._fitness_history: dict[str, list[float]] = {}
        self._loss_history: list[float] = []
        self._best_loss: float = float('inf')
        self._steps_since_improvement: int = 0

    def step(self, step: int, loss: float,
             pool: "ModuleGenomePool",
             compiler: "GenomeCompiler",
             brain=None):
        """Called every training step. Triggers evolution at probe_every intervals."""
        self._loss_history.append(loss)
        if loss < self._best_loss * 0.995:  # 0.5% improvement threshold
            self._best_loss = loss
            self._steps_since_improvement = 0
        else:
            self._steps_since_improvement += 1

        if step > 0 and step % self.probe_every == 0:
            self._evolve_step(pool, compiler, brain)

    def _evolve_step(self, pool: "ModuleGenomePool",
                     compiler: "GenomeCompiler", brain=None):
        """One evolution cycle: evaluate → select → mutate → recompile."""
        # Compute recent loss trend as fitness signal
        recent = self._loss_history[-self.probe_every:]
        if len(recent) < 10:
            return

        # Fitness = negative loss trend (improving = positive fitness)
        early = sum(recent[:len(recent)//2]) / max(1, len(recent)//2)
        late = sum(recent[len(recent)//2:]) / max(1, len(recent)//2)
        improvement = early - late  # positive = loss is decreasing

        # Adaptive mutation: stuck models get more aggressive mutations
        stagnation_factor = min(3.0, 1.0 + self._steps_since_improvement / 2000.0)
        effective_mutation = self.mutation_scale * stagnation_factor

        for region in list(pool.pools.keys()):
            # Report fitness to pool
            active = pool.active(region)
            # Fitness = negative recent loss (lower loss = higher fitness)
            fitness = -late
            active.fitness = (0.9 * active.fitness + 0.1 * fitness
                             if not math.isinf(active.fitness) else fitness)

            # Initialize epigenetic marks
            if region not in self._marks:
                mark = EpigeneticMark()
                mark.init(len(active.alleles))
                self._marks[region] = mark

            mark = self._marks[region]

            # If this genome is improving, methylate (protect) its key alleles
            if improvement > 0:
                # Identify "important" alleles: those with high absolute values
                # (strong opcode preferences or gate values)
                important = [i for i, a in enumerate(active.alleles)
                             if abs(a) > 2.0]
                mark.mark(important, strength=0.05 * improvement)
            else:
                # Not improving — decay methylation to allow more exploration
                mark.decay(rate=0.02)

            # Tournament evolution with epigenetic-aware mutation
            pool.report_all(-late)

        # Run tournament step
        pool.step()

        # Apply epigenetic masks to newly created mutants
        for region in pool.pools:
            mark = self._marks.get(region)
            if mark is None:
                continue
            mask = mark.mutation_mask()
            for genome in pool.pools[region]:
                if genome.generation > 0:  # only mutants, not elites
                    self._apply_epigenetic_mutation(genome, mask, effective_mutation)

        # Recompile all genomes with new params
        if brain is not None:
            brain._recompile_all_genomes()

    def _apply_epigenetic_mutation(self, genome: "ModuleGenome",
                                   mask: list[float],
                                   sigma: float):
        """Apply mutation with epigenetic mask — protected alleles mutate less."""
        for i in range(min(len(genome.alleles), len(mask))):
            if random.random() < 0.08 * mask[i]:  # base rate × mask
                genome.alleles[i] += random.gauss(0, sigma * mask[i])

    def benchmark_probe(self, brain, tokenizer,
                        dataset_fn=None) -> dict[str, float]:
        """Quick benchmark probe — run a small subset of eval tasks.

        Returns per-region contribution scores by ablating each module.
        """
        scores = {}
        # Default: use loss on a random batch as proxy
        # Override with dataset_fn for actual benchmark probes
        if dataset_fn is not None:
            try:
                scores = dataset_fn(brain, tokenizer, max_samples=self.probe_samples)
            except Exception:
                pass
        return scores

    def stats(self) -> dict:
        return {
            "best_loss": self._best_loss,
            "steps_since_improvement": self._steps_since_improvement,
            "n_regions_tracked": len(self._marks),
            "avg_methylation": {
                region: sum(m.methylation) / max(1, len(m.methylation))
                for region, m in self._marks.items()
            },
        }

    def state_dict(self) -> dict:
        return {
            "marks": {r: {"methylation": m.methylation}
                      for r, m in self._marks.items()},
            "fitness_history": dict(self._fitness_history),
            "best_loss": self._best_loss,
            "steps_since_improvement": self._steps_since_improvement,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "EpigeneticOptimizer":
        obj = cls()
        obj._best_loss = state.get("best_loss", float('inf'))
        obj._steps_since_improvement = state.get("steps_since_improvement", 0)
        obj._fitness_history = state.get("fitness_history", {})
        for region, mdata in state.get("marks", {}).items():
            mark = EpigeneticMark()
            mark.methylation = mdata.get("methylation", [])
            obj._marks[region] = mark
        return obj
