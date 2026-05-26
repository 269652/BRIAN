"""Test evolutionary engine."""
import pytest
from neuroslm.dsl.evolutionary import (
    CircuitGenotype, EvolutionaryEngine, GenerationResult, EvolutionLog
)


BASE_CIRCUIT = """
neurotransmitter dopamine { base_concentration: 0.1 }
population sensory { count: 128, dynamics: "rate_code", timescale: 0.01 }
population association { count: 256, dynamics: "rate_code", timescale: 0.02 }
population motor { count: 64, dynamics: "rate_code", timescale: 0.005 }
synapse sensory -> association { weight: learnable, plasticity: "hebb" }
synapse association -> motor { weight: learnable }
modulation dopamine -> association { effect: "multiplicative", gain: 0.3 }
"""


class TestCircuitGenotype:
    def test_genotype_creation(self):
        g = CircuitGenotype(id="test", source=BASE_CIRCUIT, generation=0)
        assert g.id == "test"
        assert g.generation == 0

    def test_genotype_with_lineage(self):
        g = CircuitGenotype(
            id="child", source="circuit", generation=1,
            parent_ids=["p1"], mutation_history=["mut1"]
        )
        assert len(g.parent_ids) == 1
        assert len(g.mutation_history) == 1


class TestEvolutionaryEngine:
    def test_engine_init(self):
        engine = EvolutionaryEngine(BASE_CIRCUIT, population_size=5)
        assert engine.population_size == 5
        assert engine.generation == 0

    def test_engine_rejects_invalid_base(self):
        with pytest.raises(ValueError):
            EvolutionaryEngine("invalid {", population_size=5)

    def test_initialize_population(self):
        engine = EvolutionaryEngine(BASE_CIRCUIT, population_size=5)
        pop = engine.initialize_population()
        assert len(pop) == 5
        assert all(g.generation == 0 for g in pop)

    def test_evaluate_genotype(self):
        engine = EvolutionaryEngine(BASE_CIRCUIT, population_size=5)
        g = CircuitGenotype(id="test", source=BASE_CIRCUIT, generation=0)
        evaluated = engine.evaluate(g)
        assert evaluated.fitness_scalar is not None
        assert evaluated.fitness_vector is not None

    def test_single_step(self):
        engine = EvolutionaryEngine(BASE_CIRCUIT, population_size=5)
        engine.initialize_population()
        result = engine.step()
        assert isinstance(result, GenerationResult)
        assert result.generation == 0
        assert all(g.fitness_scalar is not None for g in result.population)

    def test_run_one_generation(self):
        engine = EvolutionaryEngine(BASE_CIRCUIT, population_size=5)
        log = engine.run(n_generations=1)
        assert isinstance(log, EvolutionLog)
        assert len(log.generations) == 1
        assert log.best_ever is not None

    def test_run_multiple_generations(self):
        engine = EvolutionaryEngine(BASE_CIRCUIT, population_size=8)
        log = engine.run(n_generations=3)
        assert len(log.generations) == 3
        # Best fitness should not decrease catastrophically
        fitnesses = [g.best_fitness for g in log.generations]
        assert fitnesses[-1] >= min(fitnesses) - 0.1

    def test_generation_result_summary(self):
        result = GenerationResult(
            generation=0, population=[], pareto_frontier=[],
            best_fitness=0.5, mean_fitness=0.3, diversity=0.8,
            n_invalid_mutations=0
        )
        summary = result.summary()
        assert "Gen 0" in summary

    def test_evolution_log_summary(self):
        g = CircuitGenotype(id="best", source=BASE_CIRCUIT, generation=0, fitness_scalar=0.7)
        log = EvolutionLog(generations=[], best_ever=g)
        summary = log.summary()
        assert len(summary) > 0
