# -*- coding: utf-8 -*-
"""TDD: HypergraphExecutor — runtime tensor routing through the hypergraph IR.

The HypergraphExecutor replaces the exec()-based codegen path. Instead of
lifting DSL → Python source → exec() → static nn.Module, the IR itself
persists at runtime: each HyperNode (population) becomes a learnable
nn.Linear, each HyperEdge (synapse) becomes a differentiable projection,
and the forward pass does topological traversal so gradients flow through
the graph topology naturally.

Contracts pinned here:
  1. Constructor registers nn.Module parameters for every population node
     and every synapse edge.
  2. Forward output shape matches d_model per population.
  3. Gradients reach ALL node and edge parameters (no gradient blocking).
  4. Topological order: later nodes see earlier nodes' outputs.
  5. Multi-input convergence: a node receiving from two sources gets both.
  6. Single-node (no-edge) graph: output = node_layer(x).
  7. Roundtrip: executor.to_ir() reproduces the same node/edge structure.
"""
import pytest
import torch
import torch.nn as nn

from neuroslm.compiler.hypergraph_ir import (
    HypergraphIR,
    HyperNode,
    HyperEdge,
    SourceMap,
)
from neuroslm.compiler.hypergraph_executor import HypergraphExecutor


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_ir(*pop_names: str, synapses: list[tuple[str, str]] = None) -> HypergraphIR:
    """Build a minimal HypergraphIR with the given populations and synapses."""
    nodes = [
        HyperNode(id=f"population:{n}", kind="population", name=n)
        for n in pop_names
    ]
    edges = []
    for i, (src, dst) in enumerate(synapses or []):
        edges.append(HyperEdge(
            id=f"synapse:{src}->{dst}",
            kind="synapse",
            members=[src, dst],
        ))
    return HypergraphIR(nodes=nodes, hyperedges=edges, source_map=SourceMap(""))


D = 32  # small d_model for tests
B = 4   # batch size


# ── Contract 1: constructor wires parameters for every node and edge ──────────

class TestConstructor:
    def test_node_layers_created_for_each_population(self):
        ir = _make_ir("a", "b", "c")
        ex = HypergraphExecutor(ir, d_model=D)
        # One layer per population
        assert len(ex.node_layers) == 3

    def test_edge_projections_created_for_each_synapse(self):
        ir = _make_ir("a", "b", synapses=[("a", "b")])
        ex = HypergraphExecutor(ir, d_model=D)
        assert len(ex.edge_projections) == 1

    def test_no_edges_means_no_projections(self):
        ir = _make_ir("x")
        ex = HypergraphExecutor(ir, d_model=D)
        assert len(ex.edge_projections) == 0

    def test_non_synapse_edges_not_projected(self):
        """Modulation edges are not routing projections."""
        ir = _make_ir("a", "b")
        ir.hyperedges.append(HyperEdge(
            id="modulation:nt->a", kind="modulation", members=["nt", "a"]
        ))
        ex = HypergraphExecutor(ir, d_model=D)
        assert len(ex.edge_projections) == 0  # modulation ≠ synapse

    def test_is_nn_module(self):
        ir = _make_ir("a")
        ex = HypergraphExecutor(ir, d_model=D)
        assert isinstance(ex, nn.Module)


# ── Contract 2: forward output shape ─────────────────────────────────────────

class TestForwardShape:
    def test_output_is_dict_keyed_by_population_name(self):
        ir = _make_ir("cortex", "striatum", synapses=[("cortex", "striatum")])
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D)
        out = ex(x)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"cortex", "striatum"}

    def test_each_output_has_correct_shape(self):
        ir = _make_ir("a", "b", "c", synapses=[("a", "b"), ("b", "c")])
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D)
        out = ex(x)
        for name, tensor in out.items():
            assert tensor.shape == (B, D), f"{name}: expected ({B},{D}), got {tensor.shape}"

    def test_single_population_output_shape(self):
        ir = _make_ir("solo")
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D)
        out = ex(x)
        assert out["solo"].shape == (B, D)


# ── Contract 3: gradient flow through all node layers ────────────────────────

class TestGradientFlowNodes:
    def test_all_node_layer_params_receive_gradients(self):
        ir = _make_ir("a", "b", "c", synapses=[("a", "b"), ("b", "c")])
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D, requires_grad=True)
        out = ex(x)
        # Use the last node in topo order as the loss root
        loss = out["c"].sum()
        loss.backward()
        for name, layer in ex.node_layers.items():
            for param_name, p in layer.named_parameters():
                assert p.grad is not None, (
                    f"node_layers[{name}].{param_name} has no gradient"
                )

    def test_all_edge_projection_params_receive_gradients(self):
        ir = _make_ir("a", "b", "c", synapses=[("a", "b"), ("b", "c")])
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D)
        out = ex(x)
        loss = out["c"].sum()
        loss.backward()
        for name, proj in ex.edge_projections.items():
            for param_name, p in proj.named_parameters():
                assert p.grad is not None, (
                    f"edge_projections[{name}].{param_name} has no gradient"
                )

    def test_input_tensor_receives_gradient(self):
        ir = _make_ir("a", "b", synapses=[("a", "b")])
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D, requires_grad=True)
        out = ex(x)
        out["b"].sum().backward()
        assert x.grad is not None


# ── Contract 4: topological ordering ─────────────────────────────────────────

class TestTopologicalOrder:
    def test_source_node_output_is_not_raw_input(self):
        """Source node (no in-edges) applies its own transformation to x."""
        ir = _make_ir("src", "dst", synapses=[("src", "dst")])
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.zeros(B, D)
        # With zero input, node output is bias-only — check it's not zero
        # (nn.Linear bias defaults to non-zero Kaiming uniform)
        out = ex(x)
        # src has no incoming edges — it transforms x through its own layer
        # If the layer had non-zero bias, output != x
        src_layer = ex.node_layers[ex._safe_key("src")]
        expected_src = torch.relu(src_layer(x))
        assert torch.allclose(out["src"], expected_src)

    def test_downstream_node_receives_upstream_output(self):
        """dst's output must functionally depend on src's layer parameters."""
        ir = _make_ir("src", "dst", synapses=[("src", "dst")])
        ex = HypergraphExecutor(ir, d_model=D)

        x = torch.randn(B, D)
        out1 = ex(x)

        # Perturb src's layer weight — dst's output must change
        with torch.no_grad():
            src_layer = ex.node_layers[ex._safe_key("src")]
            src_layer.weight.add_(torch.randn_like(src_layer.weight) * 0.5)

        out2 = ex(x)
        # dst output must differ because it depends on src via the synapse
        assert not torch.allclose(out1["dst"], out2["dst"]), (
            "dst's output did not change when src's parameters changed — "
            "synapse routing is broken"
        )

    def test_three_hop_chain_end_depends_on_start(self):
        """A → B → C: perturbing A's layer changes C's output."""
        ir = _make_ir("a", "b", "c", synapses=[("a", "b"), ("b", "c")])
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D)
        out1 = ex(x)

        with torch.no_grad():
            a_layer = ex.node_layers[ex._safe_key("a")]
            a_layer.weight.add_(torch.randn_like(a_layer.weight) * 0.5)

        out2 = ex(x)
        assert not torch.allclose(out1["c"], out2["c"])


# ── Contract 5: multi-input convergence ──────────────────────────────────────

class TestMultiInputConvergence:
    def test_node_receiving_two_sources_depends_on_both(self):
        """A→C and B→C: C's output must depend on both A and B."""
        ir = _make_ir("a", "b", "c",
                      synapses=[("a", "c"), ("b", "c")])
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D)
        out_base = ex(x)

        with torch.no_grad():
            a_layer = ex.node_layers[ex._safe_key("a")]
            a_layer.weight.add_(torch.randn_like(a_layer.weight) * 0.5)
        out_a_changed = ex(x)
        assert not torch.allclose(out_base["c"], out_a_changed["c"]), \
            "C did not change when A's parameters changed"

        # Reset and perturb B
        ex2 = HypergraphExecutor(ir, d_model=D)
        out_base2 = ex2(x)
        with torch.no_grad():
            b_layer = ex2.node_layers[ex2._safe_key("b")]
            b_layer.weight.add_(torch.randn_like(b_layer.weight) * 0.5)
        out_b_changed = ex2(x)
        assert not torch.allclose(out_base2["c"], out_b_changed["c"]), \
            "C did not change when B's parameters changed"


# ── Contract 6: single-node (no-edge) graph ──────────────────────────────────

class TestSingleNode:
    def test_single_node_output_equals_relu_of_layer(self):
        ir = _make_ir("solo")
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D)
        out = ex(x)
        solo_layer = ex.node_layers[ex._safe_key("solo")]
        expected = torch.relu(solo_layer(x))
        assert torch.allclose(out["solo"], expected)

    def test_single_node_gradient_flows(self):
        ir = _make_ir("solo")
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D, requires_grad=True)
        out = ex(x)
        out["solo"].sum().backward()
        assert x.grad is not None
        solo_layer = ex.node_layers[ex._safe_key("solo")]
        assert solo_layer.weight.grad is not None


# ── Contract 7: roundtrip to_ir() ────────────────────────────────────────────

class TestRoundtrip:
    def test_to_ir_has_same_node_count(self):
        ir = _make_ir("a", "b", "c", synapses=[("a", "b"), ("b", "c")])
        ex = HypergraphExecutor(ir, d_model=D)
        recovered = ex.to_ir()
        assert len(recovered.nodes) == len(ir.nodes)

    def test_to_ir_has_same_edge_count(self):
        ir = _make_ir("a", "b", synapses=[("a", "b")])
        ex = HypergraphExecutor(ir, d_model=D)
        recovered = ex.to_ir()
        assert len(recovered.hyperedges) == len(ir.hyperedges)

    def test_to_ir_population_names_preserved(self):
        ir = _make_ir("cortex", "striatum", synapses=[("cortex", "striatum")])
        ex = HypergraphExecutor(ir, d_model=D)
        recovered = ex.to_ir()
        orig_names = {n.name for n in ir.nodes if n.kind == "population"}
        recovered_names = {n.name for n in recovered.nodes if n.kind == "population"}
        assert orig_names == recovered_names

    def test_to_ir_edge_members_preserved(self):
        ir = _make_ir("a", "b", "c",
                      synapses=[("a", "b"), ("b", "c")])
        ex = HypergraphExecutor(ir, d_model=D)
        recovered = ex.to_ir()
        orig_members = [tuple(e.members) for e in ir.hyperedges]
        recovered_members = [tuple(e.members) for e in recovered.hyperedges]
        assert orig_members == recovered_members


# ── Contract 8: build_harness integration ────────────────────────────────────

class TestBuildHarnessIntegration:
    """build_harness(..., use_hypergraph_executor=True) wires HypergraphExecutor
    as the circuit and the harness forward pass produces valid logits."""

    @pytest.fixture()
    def arch_root(self, tmp_path):
        root = tmp_path / "test_arch"
        root.mkdir()
        (root / "arch.neuro").write_text(
            "architecture demo { d_sem: 32 }\n"
            'population sensory { count: 32, dynamics: "rate_code" }\n'
            'population motor { count: 32, dynamics: "rate_code" }\n'
            "synapse sensory -> motor { weight: 0.5 }\n",
            encoding="utf-8",
        )
        return root

    def test_circuit_is_hypergraph_executor(self, arch_root):
        from neuroslm.train_dsl import build_harness
        harness = build_harness(
            arch_root=arch_root,
            vocab_size=256,
            d_sem=32,
            use_hypergraph_executor=True,
        )
        assert isinstance(harness.circuit, HypergraphExecutor)

    def test_harness_forward_produces_correct_logit_shape(self, arch_root):
        from neuroslm.train_dsl import build_harness
        harness = build_harness(
            arch_root=arch_root,
            vocab_size=256,
            d_sem=32,
            use_hypergraph_executor=True,
        )
        ids = torch.randint(0, 256, (2, 8))  # (batch=2, seq_len=8)
        logits = harness(ids)
        assert logits.shape == (2, 8, 256)

    def test_harness_gradients_flow_through_executor(self, arch_root):
        from neuroslm.train_dsl import build_harness
        harness = build_harness(
            arch_root=arch_root,
            vocab_size=256,
            d_sem=32,
            use_hypergraph_executor=True,
        )
        ids = torch.randint(0, 256, (2, 8))
        logits = harness(ids)
        loss = logits.sum()
        loss.backward()
        # Every parameter in the executor must have a gradient
        for name, p in harness.circuit.named_parameters():
            assert p.grad is not None, f"circuit.{name} has no gradient"
