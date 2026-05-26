"""Test suite for NeuroML DSL submechanics (core neural building blocks).

Tests validate that submechanic functions generate compilable DSL fragments
and can be composed serially and in parallel.
"""
import pytest
from neuroslm.dsl.submechanics import Submechanic, compose_serial, compose_parallel
from neuroslm.dsl.compiler import NeuroMLCompiler


BASE_CIRCUIT = """
neurotransmitter dopamine {
    base_concentration: 0.1
    release_rate: 0.05
    reuptake_rate: 0.02
}

population sensory { count: 128, dynamics: "rate_code", timescale: 0.01 }
population association { count: 256, dynamics: "rate_code", timescale: 0.02 }
population motor { count: 64, dynamics: "rate_code", timescale: 0.005 }

synapse sensory -> association { weight: learnable, plasticity: "hebb" }
synapse association -> motor { weight: learnable }

modulation dopamine -> association {
    effect: "multiplicative"
    gain: 0.3
}
"""


class TestSubmechanicsGating:
    def test_gate_basic(self):
        gate_block = Submechanic.gate("dopamine", "association", gain=0.3)
        assert "modulation dopamine -> association" in gate_block
        assert 'effect: "multiplicative"' in gate_block

    def test_gate_compiles(self):
        gate_block = Submechanic.gate("dopamine", "motor", gain=0.5)
        combined = BASE_CIRCUIT + "\n" + gate_block
        ir = NeuroMLCompiler.compile(combined)
        assert ir is not None


class TestSubmechanicsSelection:
    def test_selection_basic(self):
        sel_block = Submechanic.selection("sensory", "association", k=1)
        assert "synapse association -> association" in sel_block
        assert "weight: -0.5" in sel_block

    def test_selection_compiles(self):
        sel_block = Submechanic.selection("sensory", "motor", k=1)
        combined = BASE_CIRCUIT + "\n" + sel_block
        ir = NeuroMLCompiler.compile(combined)
        assert ir is not None


class TestComposition:
    def test_compose_serial(self):
        gate = Submechanic.gate("dopamine", "motor")
        attr = Submechanic.attractor("association")
        composed = compose_serial(gate, attr)
        assert "modulation dopamine -> motor" in composed
        assert "synapse association -> association" in composed

    def test_compose_serial_compiles(self):
        gate = Submechanic.gate("dopamine", "motor")
        home = Submechanic.homeostasis("sensory")
        composed = compose_serial(gate, home)
        combined = BASE_CIRCUIT + "\n" + composed
        ir = NeuroMLCompiler.compile(combined)
        assert ir is not None
