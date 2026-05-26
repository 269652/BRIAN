"""Test mutations module."""
import pytest
from neuroslm.dsl.mutations import (
    MutationError, add_modulation, add_feedback, add_gating, mutate_numeric
)
from neuroslm.dsl.compiler import NeuroMLCompiler


BASE = """
neurotransmitter dopamine { base_concentration: 0.1 }
population sensory { count: 128 }
population motor { count: 64 }
synapse sensory -> motor { weight: learnable }
"""


class TestMutations:
    def test_add_modulation_valid(self):
        mutated = add_modulation(BASE, "dopamine", "sensory")
        assert "modulation dopamine -> sensory" in mutated
        ir = NeuroMLCompiler.compile(mutated)
        assert ir is not None

    def test_add_modulation_invalid_pop(self):
        with pytest.raises(MutationError):
            add_modulation(BASE, "dopamine", "undefined")

    def test_add_feedback(self):
        mutated = add_feedback(BASE, "sensory", "motor")
        ir = NeuroMLCompiler.compile(mutated)
        assert ir is not None

    def test_add_gating(self):
        mutated = add_gating(BASE, "dopamine", "motor")
        ir = NeuroMLCompiler.compile(mutated)
        assert ir is not None

    def test_mutate_numeric(self):
        mutated = mutate_numeric(BASE, sigma=0.1, rate=0.5)
        ir = NeuroMLCompiler.compile(mutated)
        assert ir is not None
