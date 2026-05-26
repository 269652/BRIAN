"""Evolutionary algorithm engine for circuit discovery."""
import random
import uuid
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any

from .compiler import NeuroMLCompiler
from .fitness import FitnessVector, compute_fitness, composite_fitness, pareto_frontier
from .mutations import MutationError, ALL_MUTATIONS


@dataclass
class CircuitGenotype:
    id: str
    source: str
    ir: Optional[Any] = None
    fitness_vector: Optional[FitnessVector] = None
    fitness_scalar: Optional[float] = None
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    is_valid: bool = True


@dataclass
class GenerationResult:
    generation: int
    population: List[CircuitGenotype]
    pareto_frontier: List[CircuitGenotype]
    best_fitness: float
    mean_fitness: float
    diversity: float
    n_invalid_mutations: int
    mutation_applied_counts: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (f"Gen {self.generation}: best={self.best_fitness:.3f} mean={self.mean_fitness:.3f} "
                f"diversity={self.diversity:.3f} frontier={len(self.pareto_frontier)} invalid={self.n_invalid_mutations}")


@dataclass
class EvolutionLog:
    generations: List[GenerationResult]
    best_ever: CircuitGenotype
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def summary(self) -> str:
        duration = (self.end_time or time.time()) - self.start_time
        best_fit = f"{self.best_ever.fitness_scalar:.3f}" if self.best_ever.fitness_scalar else "unevaluated"
        return f"Evolutionary Run: {len(self.generations)} generations in {duration:.1f}s, best={best_fit}"


class EvolutionaryEngine:
    def __init__(self, base_circuit: str, population_size: int = 20, fitness_weights: Dict = None,
                 n_elite: int = 4, mutation_rate: float = 0.8, crossover_rate: float = 0.2,
                 tournament_k: int = 3, stagnation_patience: int = 3, max_mutation_attempts: int = 10):
        try:
            NeuroMLCompiler.compile(base_circuit)
        except:
            raise ValueError("Base circuit does not compile")

        self.base_circuit = base_circuit
        self.population_size = population_size
        self.fitness_weights = fitness_weights or {}
        self.n_elite = min(n_elite, population_size // 2)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_k = tournament_k
        self.stagnation_patience = stagnation_patience
        self.max_mutation_attempts = max_mutation_attempts

        self.generation = 0
        self.population: List[CircuitGenotype] = []
        self.best_fitness_history: List[float] = []
        self.current_mutation_rate = mutation_rate
        self.stagnation_counter = 0

    def initialize_population(self) -> List[CircuitGenotype]:
        self.population = []
        base = CircuitGenotype(id=str(uuid.uuid4())[:8], source=self.base_circuit, generation=0)
        self.population.append(base)

        for _ in range(1, self.population_size):
            source = self.base_circuit
            for _ in range(random.randint(1, 3)):
                try:
                    op = random.choice(ALL_MUTATIONS)
                    if op.__name__ == 'add_modulation':
                        source = op(source, 'dopamine', 'sensory')
                    elif op.__name__ == 'add_feedback':
                        source = op(source, 'sensory', 'motor')
                    elif op.__name__ == 'add_gating':
                        source = op(source, 'dopamine', 'association')
                    elif op.__name__ == 'add_plasticity':
                        source = op(source, 'sensory', 'association')
                    elif op.__name__ == 'add_prediction_loop':
                        source = op(source, 'association')
                    elif op.__name__ == 'add_homeostatic_regulation':
                        source = op(source, 'association')
                    elif op.__name__ == 'add_sheaf_consistency':
                        source = op(source, 'association')
                    elif op.__name__ == 'mutate_numeric':
                        source = op(source)
                except:
                    pass

            g = CircuitGenotype(id=str(uuid.uuid4())[:8], source=source, generation=0)
            self.population.append(g)

        return self.population

    def evaluate(self, g: CircuitGenotype) -> CircuitGenotype:
        try:
            g.ir = NeuroMLCompiler.compile(g.source)
            g.fitness_vector = compute_fitness(g.ir)
            g.fitness_scalar = composite_fitness(g.fitness_vector, self.fitness_weights)
            g.is_valid = True
        except:
            g.is_valid = False
            g.fitness_scalar = 0.0
            g.fitness_vector = FitnessVector(0, 0, 0, 0, 0, 0)
        return g

    def evaluate_population(self, pop: List[CircuitGenotype]) -> List[CircuitGenotype]:
        for g in pop:
            if g.fitness_scalar is None:
                self.evaluate(g)
        return pop

    def select_parents(self, pop: List[CircuitGenotype]) -> Tuple[CircuitGenotype, CircuitGenotype]:
        valid = [g for g in pop if g.is_valid]
        if len(valid) < 2:
            valid = pop
        t1 = random.sample(valid, min(self.tournament_k, len(valid)))
        t2 = random.sample(valid, min(self.tournament_k, len(valid)))
        return max(t1, key=lambda g: g.fitness_scalar or 0.0), max(t2, key=lambda g: g.fitness_scalar or 0.0)

    def mutate(self, parent: CircuitGenotype) -> Optional[CircuitGenotype]:
        for _ in range(self.max_mutation_attempts):
            op = random.choice(ALL_MUTATIONS)
            try:
                if op.__name__ == 'add_modulation':
                    src = op(parent.source, 'dopamine', 'association')
                elif op.__name__ == 'add_feedback':
                    src = op(parent.source, 'sensory', 'motor')
                elif op.__name__ == 'add_gating':
                    src = op(parent.source, 'dopamine', 'association')
                elif op.__name__ == 'add_plasticity':
                    src = op(parent.source, 'sensory', 'association')
                elif op.__name__ == 'add_prediction_loop':
                    src = op(parent.source, 'association')
                elif op.__name__ == 'add_homeostatic_regulation':
                    src = op(parent.source, 'association')
                elif op.__name__ == 'add_sheaf_consistency':
                    src = op(parent.source, 'association')
                elif op.__name__ == 'mutate_numeric':
                    src = op(parent.source)
                elif op.__name__ == 'crossover':
                    continue
                else:
                    continue

                child = CircuitGenotype(id=str(uuid.uuid4())[:8], source=src, generation=self.generation,
                                      parent_ids=[parent.id], mutation_history=parent.mutation_history + [op.__name__])
                return child
            except:
                pass
        return None

    def crossover(self, a: CircuitGenotype, b: CircuitGenotype) -> Optional[CircuitGenotype]:
        try:
            from .mutations import crossover
            src = crossover(a.source, b.source)
            child = CircuitGenotype(id=str(uuid.uuid4())[:8], source=src, generation=self.generation,
                                  parent_ids=[a.id, b.id], mutation_history=['crossover'])
            return child
        except:
            return None

    def step(self) -> GenerationResult:
        self.population = self.evaluate_population(self.population)

        valid_fit = [g.fitness_scalar for g in self.population if g.is_valid]
        best_fit = max(valid_fit) if valid_fit else 0.0
        mean_fit = sum(valid_fit) / len(valid_fit) if valid_fit else 0.0

        unique = len(set(g.source for g in self.population))
        diversity = unique / len(self.population)

        valid_g = [(g, g.fitness_vector) for g in self.population if g.is_valid]
        frontier = pareto_frontier(valid_g) if valid_g else []

        if self.best_fitness_history and best_fit <= self.best_fitness_history[-1]:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        if self.stagnation_counter >= self.stagnation_patience:
            self.current_mutation_rate = min(1.0, self.mutation_rate * 1.5)
            self.stagnation_counter = 0
        else:
            self.current_mutation_rate = self.mutation_rate

        self.best_fitness_history.append(best_fit)

        next_gen = []
        elite = sorted([g for g in self.population if g.is_valid],
                      key=lambda g: g.fitness_scalar or 0.0, reverse=True)[:self.n_elite]
        next_gen.extend(elite)

        n_invalid = 0
        mut_counts = {}

        while len(next_gen) < self.population_size:
            if random.random() < self.crossover_rate and len(next_gen) > 1:
                a, b = self.select_parents(self.population)
                child = self.crossover(a, b)
                if child:
                    next_gen.append(child)
                    mut_counts['crossover'] = mut_counts.get('crossover', 0) + 1
                else:
                    n_invalid += 1
            else:
                parent = self.select_parents(self.population)[0]
                child = self.mutate(parent)
                if child:
                    next_gen.append(child)
                    op = child.mutation_history[-1] if child.mutation_history else 'unknown'
                    mut_counts[op] = mut_counts.get(op, 0) + 1
                else:
                    n_invalid += 1

        self.population = next_gen[:self.population_size]
        self.population = self.evaluate_population(self.population)
        self.generation += 1

        return GenerationResult(generation=self.generation-1, population=self.population, pareto_frontier=frontier,
                              best_fitness=best_fit, mean_fitness=mean_fit, diversity=diversity, n_invalid_mutations=n_invalid,
                              mutation_applied_counts=mut_counts)

    def run(self, n_generations: int) -> EvolutionLog:
        self.initialize_population()
        log = EvolutionLog(generations=[], best_ever=CircuitGenotype(id='init', source=self.base_circuit), start_time=time.time())

        for _ in range(n_generations):
            result = self.step()
            log.generations.append(result)
            if result.best_fitness > (log.best_ever.fitness_scalar or 0.0):
                best = max(self.population, key=lambda g: g.fitness_scalar or 0.0)
                log.best_ever = best

        log.end_time = time.time()
        return log
