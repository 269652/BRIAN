# -*- coding: utf-8 -*-
"""TDD: Activation-based heat for the HypergraphExecutor.

Grad norms measure training pressure (how the loss pulls each parameter).
Activation norms measure information throughput (how much signal actually
flows through each node / edge during the forward pass). This file pins
the second kind — what the heatmap overlay must show.

Contracts:
  A) HypergraphExecutor records per-node and per-edge activation norms
     during forward() without breaking the gradient graph.
  B) executor_activation_norms() reads those norms and returns a dict
     keyed by IR element IDs — same schema as executor_grad_norms.
  C) Nodes with larger output tensors report higher activation norms.
  D) Edges with larger projection contributions report higher norms.
  E) HeatmapPublisher renders a PNG via render_hypergraph when a
     png_renderer callback is supplied.
  F) PNG path is added to the git commit alongside the JSON.
"""
import math
import pytest
import torch
import torch.nn as nn

from neuroslm.compiler.hypergraph_ir import HypergraphIR, HyperNode, HyperEdge, SourceMap
from neuroslm.compiler.hypergraph_executor import HypergraphExecutor, executor_activation_norms


def _make_ir(*pop_names: str, synapses=None) -> HypergraphIR:
    nodes = [HyperNode(id=f"population:{n}", kind="population", name=n)
             for n in pop_names]
    edges = [
        HyperEdge(id=f"synapse:{s}->{d}", kind="synapse", members=[s, d])
        for s, d in (synapses or [])
    ]
    return HypergraphIR(nodes=nodes, hyperedges=edges, source_map=SourceMap(""))


D = 32
B = 4


# ── Contract A: norms recorded during forward ─────────────────────────────

class TestActivationNormsCollected:
    def test_norms_empty_before_any_forward(self):
        ir = _make_ir("a", "b", synapses=[("a", "b")])
        ex = HypergraphExecutor(ir, d_model=D)
        result = executor_activation_norms(ex)
        assert result == {}

    def test_all_population_ids_present_after_forward(self):
        ir = _make_ir("sensory", "motor", synapses=[("sensory", "motor")])
        ex = HypergraphExecutor(ir, d_model=D)
        ex(torch.randn(B, D))
        result = executor_activation_norms(ex)
        assert "population:sensory" in result
        assert "population:motor" in result

    def test_all_synapse_ids_present_after_forward(self):
        ir = _make_ir("a", "b", "c", synapses=[("a", "b"), ("b", "c")])
        ex = HypergraphExecutor(ir, d_model=D)
        ex(torch.randn(B, D))
        result = executor_activation_norms(ex)
        assert "synapse:a->b" in result
        assert "synapse:b->c" in result

    def test_values_are_non_negative(self):
        ir = _make_ir("a", "b", synapses=[("a", "b")])
        ex = HypergraphExecutor(ir, d_model=D)
        ex(torch.randn(B, D))
        for key, val in executor_activation_norms(ex).items():
            assert val >= 0.0, f"{key} has negative norm {val}"

    def test_forward_does_not_break_gradient_graph(self):
        ir = _make_ir("a", "b", synapses=[("a", "b")])
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D, requires_grad=True)
        out = ex(x)
        out["b"].sum().backward()
        assert x.grad is not None, "forward with norm tracking broke the gradient"


# ── Contract B: executor_activation_norms reads the recorded norms ─────────

class TestExecutorActivationNorms:
    def test_returns_dict(self):
        ir = _make_ir("a")
        ex = HypergraphExecutor(ir, d_model=D)
        ex(torch.randn(B, D))
        assert isinstance(executor_activation_norms(ex), dict)

    def test_updates_on_each_forward(self):
        ir = _make_ir("solo")
        ex = HypergraphExecutor(ir, d_model=D)
        ex(torch.zeros(B, D))   # first forward — bias-only output
        norms_1 = dict(executor_activation_norms(ex))
        ex(torch.randn(B, D) * 10)  # large input
        norms_2 = dict(executor_activation_norms(ex))
        # Second forward with large input should give different norms
        assert norms_1 != norms_2, "norms did not update on second forward"


# ── Contract C: larger output → higher node norm ──────────────────────────

class TestNodeNormScales:
    def test_nonzero_activation_gives_positive_norm(self):
        ir = _make_ir("solo")
        ex = HypergraphExecutor(ir, d_model=D)
        # With a large random input the bias + linear will produce non-zero output
        ex(torch.randn(B, D) * 5)
        result = executor_activation_norms(ex)
        assert result["population:solo"] > 0.0

    def test_node_norm_matches_relu_output_rms(self):
        """population norm = RMS of the node's ReLU output tensor."""
        ir = _make_ir("solo")
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D)
        out = ex(x)
        # RMS = ‖tensor‖_F / sqrt(numel)
        expected_rms = float(
            out["solo"].detach().norm().item() / math.sqrt(out["solo"].numel())
        )
        result = executor_activation_norms(ex)
        assert math.isclose(result["population:solo"], expected_rms, rel_tol=1e-4), (
            f"expected RMS {expected_rms:.6f}, got {result['population:solo']:.6f}"
        )


# ── Contract D: edge norm measures projection contribution ────────────────

class TestEdgeNormScales:
    def test_edge_norm_positive_after_forward(self):
        ir = _make_ir("a", "b", synapses=[("a", "b")])
        ex = HypergraphExecutor(ir, d_model=D)
        ex(torch.randn(B, D) * 3)
        result = executor_activation_norms(ex)
        assert result["synapse:a->b"] > 0.0


# ── Contract E+F: PNG publisher ───────────────────────────────────────────

class TestPNGPublisher:
    def test_publisher_accepts_png_renderer_and_png_path(self, tmp_path):
        from neuroslm.evolution.publisher import HeatmapPublisher
        pub = HeatmapPublisher(
            heatmap_path=str(tmp_path / "hm.json"),
            commit_every=0,
            png_renderer=lambda hm, path: None,
            png_path=str(tmp_path / "hm.png"),
        )
        assert pub.png_renderer is not None
        assert pub.png_path is not None

    def test_publish_calls_png_renderer(self, tmp_path):
        from neuroslm.evolution.publisher import HeatmapPublisher
        from neuroslm.evolution.heatmap import TrainingHeatmap

        called_with = []
        def fake_renderer(hm, path):
            called_with.append((hm, path))
            # Write a dummy PNG so git-add doesn't fail on missing file
            import pathlib
            pathlib.Path(path).write_bytes(b"FAKEPNG")

        pub = HeatmapPublisher(
            heatmap_path=str(tmp_path / "hm.json"),
            commit_every=0,
            runner=lambda args, cwd=None: 0,
            png_renderer=fake_renderer,
            png_path=str(tmp_path / "hm.png"),
        )
        hm = TrainingHeatmap()
        hm.update({"population:a": 1.0}, step=1)
        pub.publish(hm, step=1)
        assert len(called_with) == 1

    def test_publish_adds_png_to_git(self, tmp_path):
        from neuroslm.evolution.publisher import HeatmapPublisher
        from neuroslm.evolution.heatmap import TrainingHeatmap
        import pathlib

        png_path = tmp_path / "hm.png"
        added = []
        def fake_runner(args, cwd=None):
            if args[0] == "add":
                added.extend(args[1:])
            return 0

        pub = HeatmapPublisher(
            heatmap_path=str(tmp_path / "hm.json"),
            commit_every=0,
            runner=fake_runner,
            png_renderer=lambda hm, path: pathlib.Path(path).write_bytes(b"PNG"),
            png_path=str(png_path),
        )
        hm = TrainingHeatmap()
        hm.update({"population:a": 0.5}, step=1)
        pub.publish(hm, step=1)
        assert str(png_path) in added, f"PNG not in git add: {added}"

    def test_dot_and_png_both_committed_when_both_set(self, tmp_path):
        from neuroslm.evolution.publisher import HeatmapPublisher
        from neuroslm.evolution.heatmap import TrainingHeatmap
        import pathlib

        dot_path = tmp_path / "hm.dot"
        png_path = tmp_path / "hm.png"
        added = []
        def fake_runner(args, cwd=None):
            if args[0] == "add":
                added.extend(args[1:])
            return 0

        pub = HeatmapPublisher(
            heatmap_path=str(tmp_path / "hm.json"),
            commit_every=0,
            runner=fake_runner,
            dot_renderer=lambda hm: "digraph G {}",
            dot_path=str(dot_path),
            png_renderer=lambda hm, path: pathlib.Path(path).write_bytes(b"PNG"),
            png_path=str(png_path),
        )
        hm = TrainingHeatmap()
        hm.update({"population:a": 1.0}, step=1)
        pub.publish(hm, step=1)
        assert str(dot_path) in added
        assert str(png_path) in added
