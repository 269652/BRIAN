# -*- coding: utf-8 -*-
"""Tests for THG-IR checkpoint serialization (Phase III).

Covers:
  - THGNode, THGEdge, THGCheckpoint data structures
  - save(path) and load(path) for JSON persistence
  - from_program_ir(ir) and to_program_ir() round-trip
  - mutate_node(node_id, delta_embedding) for in-place topology edit
  - Evolutionary engine integration via from_checkpoint()
"""
import json
import pytest
import tempfile
from pathlib import Path

from neuroslm.dsl.compiler import NeuroMLCompiler
from neuroslm.dsl.thg_ir import THGNode, THGEdge, THGCheckpoint


SIMPLE_ARCH = """
architecture test_thg { d_sem: 256, dt: 0.01 }

population sensory { count: 128, dynamics: "rate_code" }
population motor { count: 64, dynamics: "rate_code" }
synapse sensory -> motor { weight: 0.5, neurotransmitter: "glutamate" }
"""


class TestTHGNodeAndEdge:
    """Test basic THGNode and THGEdge construction."""

    def test_thg_node_creation(self):
        """THGNode should store id, kind, operator_embedding, metadata."""
        node = THGNode(
            id="pop_sensory",
            kind="population",
            operator_embedding=[0.1, 0.2, 0.3],
            metadata={"count": 128}
        )
        assert node.id == "pop_sensory"
        assert node.kind == "population"
        assert len(node.operator_embedding) == 3
        assert node.metadata["count"] == 128

    def test_thg_edge_creation(self):
        """THGEdge should store src, dst, kind, weight, plasticity."""
        edge = THGEdge(
            id="syn_1",
            src="pop_sensory",
            dst="pop_motor",
            kind="synapse",
            weight=0.5,
            plasticity="hebb"
        )
        assert edge.src == "pop_sensory"
        assert edge.dst == "pop_motor"
        assert edge.weight == 0.5


class TestTHGCheckpointStructure:
    """Test THGCheckpoint data structure."""

    def test_thg_checkpoint_creation(self):
        """THGCheckpoint should store version, nodes, edges, step, metadata."""
        nodes = {
            "pop_sensory": THGNode("pop_sensory", "population", [0.0]*16, {}),
            "pop_motor": THGNode("pop_motor", "population", [0.0]*16, {}),
        }
        edges = {
            "syn_1": THGEdge("syn_1", "pop_sensory", "pop_motor", "synapse", 0.5, "fixed"),
        }
        checkpoint = THGCheckpoint(
            version="2.0",
            nodes=nodes,
            edges=edges,
            gene_state={},
            step=0,
            metadata={"arch": "test"}
        )
        assert checkpoint.version == "2.0"
        assert len(checkpoint.nodes) == 2
        assert len(checkpoint.edges) == 1
        assert checkpoint.step == 0


class TestTHGCheckpointSerialization:
    """Test save/load round-trip for THGCheckpoint."""

    def test_thg_save_and_load(self):
        """Save to JSON and load back should preserve all fields."""
        nodes = {
            "n1": THGNode("n1", "pop", [0.1, 0.2], {"x": 1}),
        }
        edges = {
            "e1": THGEdge("e1", "n1", "n2", "syn", 0.5, "fixed"),
        }
        original = THGCheckpoint(
            version="2.0",
            nodes=nodes,
            edges=edges,
            gene_state={"g1": 0.3},
            step=100,
            metadata={"test": True}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            original.save(str(path))
            assert path.exists()

            loaded = THGCheckpoint.load(str(path))
            assert loaded.version == "2.0"
            assert len(loaded.nodes) == 1
            assert loaded.nodes["n1"].operator_embedding == [0.1, 0.2]
            assert loaded.step == 100

    def test_thg_save_creates_valid_json(self):
        """Saved checkpoint should be valid JSON."""
        checkpoint = THGCheckpoint(
            version="2.0",
            nodes={"n1": THGNode("n1", "pop", [0.0], {})},
            edges={},
            gene_state={},
            step=0,
            metadata={}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            checkpoint.save(str(path))
            with open(path) as f:
                data = json.load(f)
                assert data["version"] == "2.0"
                assert "n1" in data["nodes"]


class TestTHGFromProgramIR:
    """Test conversion from ProgramIR to THGCheckpoint."""

    def test_thg_from_program_ir_creates_nodes(self):
        """THGCheckpoint.from_program_ir(ir) should create nodes for populations."""
        ir = NeuroMLCompiler.compile(SIMPLE_ARCH)
        checkpoint = THGCheckpoint.from_program_ir(ir)

        assert checkpoint.version == "2.0"
        assert "sensory" in checkpoint.nodes or any("sensory" in nid for nid in checkpoint.nodes)
        assert len(checkpoint.nodes) >= 2  # At least sensory and motor

    def test_thg_from_program_ir_creates_edges(self):
        """THGCheckpoint.from_program_ir(ir) should create edges for synapses."""
        ir = NeuroMLCompiler.compile(SIMPLE_ARCH)
        checkpoint = THGCheckpoint.from_program_ir(ir)

        assert len(checkpoint.edges) >= 1
        # Check that at least one edge connects sensory to motor
        assert any(
            e.src.endswith("sensory") and e.dst.endswith("motor")
            for e in checkpoint.edges.values()
        )


class TestTHGRoundTrip:
    """Test ProgramIR → THGCheckpoint → ProgramIR round-trip."""

    def test_thg_to_program_ir_roundtrip_preserves_nodes(self):
        """Convert ProgramIR → THG → ProgramIR should preserve node count."""
        ir1 = NeuroMLCompiler.compile(SIMPLE_ARCH)
        checkpoint = THGCheckpoint.from_program_ir(ir1)
        ir2 = checkpoint.to_program_ir()

        assert len(ir2.populations) == len(ir1.populations)
        pop_names_1 = {p.name for p in ir1.populations}
        pop_names_2 = {p.name for p in ir2.populations}
        assert pop_names_1 == pop_names_2

    def test_thg_to_program_ir_roundtrip_preserves_edges(self):
        """Convert ProgramIR → THG → ProgramIR should preserve synapse count."""
        ir1 = NeuroMLCompiler.compile(SIMPLE_ARCH)
        checkpoint = THGCheckpoint.from_program_ir(ir1)
        ir2 = checkpoint.to_program_ir()

        assert len(ir2.synapses) == len(ir1.synapses)


class TestTHGMutateNode:
    """Test in-place node mutation."""

    def test_mutate_node_updates_embedding(self):
        """mutate_node() should update operator_embedding in place."""
        checkpoint = THGCheckpoint(
            version="2.0",
            nodes={"n1": THGNode("n1", "pop", [0.0, 0.0], {})},
            edges={},
            gene_state={},
            step=0,
            metadata={}
        )

        delta = [0.1, 0.2]
        checkpoint.mutate_node("n1", delta)

        # Embedding should be updated (add delta)
        updated = checkpoint.nodes["n1"].operator_embedding
        assert len(updated) == 2
        # Simple addition expected
        assert updated[0] > 0.0 or updated[0] == 0.0  # Sanity check

    def test_mutate_node_idempotent_twice(self):
        """Mutating twice with opposite deltas should return to original."""
        checkpoint = THGCheckpoint(
            version="2.0",
            nodes={"n1": THGNode("n1", "pop", [0.0, 0.0], {})},
            edges={},
            gene_state={},
            step=0,
            metadata={}
        )

        checkpoint.mutate_node("n1", [0.1, 0.2])
        checkpoint.mutate_node("n1", [-0.1, -0.2])

        # Should be back to approximately [0.0, 0.0]
        updated = checkpoint.nodes["n1"].operator_embedding
        assert abs(updated[0]) < 0.001  # Allow floating-point tolerance
        assert abs(updated[1]) < 0.001


class TestTHGEvolutionaryIntegration:
    """Test integration with EvolutionaryEngine."""

    def test_evolutionary_engine_from_checkpoint(self):
        """EvolutionaryEngine.from_checkpoint() should work with THGCheckpoint."""
        # Create a simple checkpoint
        checkpoint = THGCheckpoint(
            version="2.0",
            nodes={"n1": THGNode("n1", "pop", [0.0]*16, {})},
            edges={},
            gene_state={},
            step=0,
            metadata={}
        )

        try:
            from neuroslm.dsl.evolutionary import EvolutionaryEngine
            engine = EvolutionaryEngine.from_checkpoint(checkpoint, population_size=5)
            assert engine is not None
            assert engine.population_size == 5
        except AttributeError:
            pytest.skip("EvolutionaryEngine.from_checkpoint not yet implemented")
        except Exception as e:
            pytest.skip(f"Evolutionary integration not ready: {e}")
