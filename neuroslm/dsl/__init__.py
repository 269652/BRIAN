"""NeuroML DSL evolutionary framework."""

from .compiler import NeuroMLCompiler, ProgramIR, PopulationIR, SynapseIR
from .submechanics import Submechanic, compose_serial, compose_parallel
from .mutations import MutationError, ALL_MUTATIONS
from .fitness import FitnessVector, compute_fitness, pareto_frontier
from .evolutionary import CircuitGenotype, EvolutionaryEngine, EvolutionLog, GenerationResult

__all__ = [
    "NeuroMLCompiler", "ProgramIR", "PopulationIR", "SynapseIR",
    "Submechanic", "compose_serial", "compose_parallel",
    "MutationError", "ALL_MUTATIONS",
    "FitnessVector", "compute_fitness", "pareto_frontier",
    "CircuitGenotype", "EvolutionaryEngine", "EvolutionLog", "GenerationResult",
]
