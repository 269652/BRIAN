# -*- coding: utf-8 -*-
"""TDD: HypergraphExecutor heat pipeline.

Three contracts:

  A) executor_grad_norms(executor) -> IR-keyed grad norms
     The executor's params are named node_layers.{key}.* — the generic
     alias path can't find them. This function directly maps each
     node layer / edge projection to its IR element ID.

  B) HeatmapHook.grad_norm_fn — custom extractor override
     When a callable is supplied, step() calls it instead of
     parameter_grad_norms so the executor's signals reach the heatmap.

  C) HeatmapPublisher.dot_renderer — DOT file committed alongside JSON
     When a dot_renderer callable is supplied, publish() writes a .dot
     file and adds it to the same git commit as the heatmap JSON.
"""
import math
import pytest
import torch

from neuroslm.compiler.hypergraph_ir import HypergraphIR, HyperNode, HyperEdge, SourceMap
from neuroslm.compiler.hypergraph_executor import HypergraphExecutor, executor_grad_norms
from neuroslm.evolution.heatmap import TrainingHeatmap
from neuroslm.evolution.harness_hook import HeatmapHook


# ── helpers ────────────────────────────────────────────────────────────────

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


def _backward(executor, x=None):
    if x is None:
        x = torch.randn(B, D)
    out = executor(x)
    sink = list(out.values())[-1]
    sink.sum().backward()


# ── Contract A: executor_grad_norms ────────────────────────────────────────

class TestExecutorGradNorms:
    def test_returns_dict(self):
        ir = _make_ir("a", "b", synapses=[("a", "b")])
        ex = HypergraphExecutor(ir, d_model=D)
        _backward(ex)
        result = executor_grad_norms(ex)
        assert isinstance(result, dict)

    def test_all_population_ids_present_after_backward(self):
        ir = _make_ir("sensory", "motor", synapses=[("sensory", "motor")])
        ex = HypergraphExecutor(ir, d_model=D)
        _backward(ex)
        result = executor_grad_norms(ex)
        assert "population:sensory" in result
        assert "population:motor" in result

    def test_all_synapse_ids_present_after_backward(self):
        ir = _make_ir("a", "b", "c",
                      synapses=[("a", "b"), ("b", "c")])
        ex = HypergraphExecutor(ir, d_model=D)
        _backward(ex)
        result = executor_grad_norms(ex)
        assert "synapse:a->b" in result
        assert "synapse:b->c" in result

    def test_values_are_non_negative(self):
        ir = _make_ir("a", "b", synapses=[("a", "b")])
        ex = HypergraphExecutor(ir, d_model=D)
        _backward(ex)
        result = executor_grad_norms(ex)
        for key, val in result.items():
            assert val >= 0.0, f"{key} has negative norm {val}"

    def test_empty_when_no_backward(self):
        ir = _make_ir("a", "b", synapses=[("a", "b")])
        ex = HypergraphExecutor(ir, d_model=D)
        # No backward — no grads anywhere
        result = executor_grad_norms(ex)
        assert result == {}

    def test_values_are_positive_after_backward(self):
        ir = _make_ir("a", "b", synapses=[("a", "b")])
        ex = HypergraphExecutor(ir, d_model=D)
        _backward(ex)
        result = executor_grad_norms(ex)
        for key, val in result.items():
            assert val > 0.0, f"{key} has zero grad norm — gradient not flowing"

    def test_node_norms_are_l2_aggregates(self):
        """Each population's norm = sqrt(sum of squared grad norms of its params)."""
        ir = _make_ir("solo")
        ex = HypergraphExecutor(ir, d_model=D)
        _backward(ex)
        result = executor_grad_norms(ex)
        layer = ex.node_layers[ex._safe_key("solo")]
        expected = math.sqrt(
            sum(p.grad.norm(2).item() ** 2
                for p in layer.parameters() if p.grad is not None)
        )
        assert math.isclose(result["population:solo"], expected, rel_tol=1e-5)


# ── Contract B: HeatmapHook.grad_norm_fn ───────────────────────────────────

class TestHeatmapHookGradNormFn:
    def test_hook_accepts_grad_norm_fn_kwarg(self):
        ir = _make_ir("a")
        ex = HypergraphExecutor(ir, d_model=D)
        hook = HeatmapHook(
            model=ex,
            ir=ir,
            every_n=1,
            grad_norm_fn=lambda: {},
        )
        assert hook.grad_norm_fn is not None

    def test_hook_uses_grad_norm_fn_instead_of_default(self):
        """When grad_norm_fn is set, step() must call it and NOT named_parameters()."""
        ir = _make_ir("a", "b", synapses=[("a", "b")])
        ex = HypergraphExecutor(ir, d_model=D)
        _backward(ex)

        calls = []
        def fake_extractor():
            calls.append(1)
            # Return signals keyed by IR element IDs
            return {
                "population:a": 2.5,
                "population:b": 0.5,
                "synapse:a->b": 1.5,
            }

        heatmap = TrainingHeatmap()
        hook = HeatmapHook(
            model=ex,
            ir=ir,
            heatmap=heatmap,
            every_n=1,
            grad_norm_fn=fake_extractor,
        )
        hook.step(1)
        assert len(calls) == 1, "grad_norm_fn was not called"

    def test_hook_step_updates_heatmap_via_fn(self):
        ir = _make_ir("a", "b", synapses=[("a", "b")])
        ex = HypergraphExecutor(ir, d_model=D)
        _backward(ex)

        heatmap = TrainingHeatmap()
        hook = HeatmapHook(
            model=ex,
            ir=ir,
            heatmap=heatmap,
            every_n=1,
            grad_norm_fn=lambda: executor_grad_norms(ex),
        )
        hook.step(1)
        # After one step some nodes should have heat > 0
        assert any(e.heat > 0.0 for e in heatmap.entries.values()), (
            "heatmap has no non-zero entries after step()"
        )

    def test_hook_step_skips_on_wrong_cadence(self):
        ir = _make_ir("a")
        ex = HypergraphExecutor(ir, d_model=D)
        calls = []
        hook = HeatmapHook(
            model=ex,
            ir=ir,
            every_n=500,
            grad_norm_fn=lambda: calls.append(1) or {},
        )
        hook.step(1)   # not on cadence
        hook.step(499) # not on cadence
        assert len(calls) == 0

        hook.step(500) # on cadence
        assert len(calls) == 1


# ── Contract C: HeatmapPublisher.dot_renderer ──────────────────────────────

class TestHeatmapPublisherDotRenderer:
    def test_publisher_accepts_dot_renderer_and_dot_path(self, tmp_path):
        from neuroslm.evolution.publisher import HeatmapPublisher
        dot_path = tmp_path / "heatmap.dot"
        pub = HeatmapPublisher(
            heatmap_path=str(tmp_path / "heatmap.json"),
            commit_every=0,  # no git
            dot_renderer=lambda hm: "digraph G {}",
            dot_path=str(dot_path),
        )
        assert pub.dot_renderer is not None
        assert pub.dot_path is not None

    def test_publish_writes_dot_file_when_renderer_set(self, tmp_path):
        from neuroslm.evolution.publisher import HeatmapPublisher

        heatmap_path = tmp_path / "heatmap.json"
        dot_path = tmp_path / "heatmap.dot"

        git_calls = []
        def fake_runner(args, cwd=None):
            git_calls.append(args)
            return 0

        pub = HeatmapPublisher(
            heatmap_path=str(heatmap_path),
            commit_every=0,
            runner=fake_runner,
            dot_renderer=lambda hm: f"digraph G {{ label=\"step {hm.step}\"; }}",
            dot_path=str(dot_path),
        )

        heatmap = TrainingHeatmap()
        heatmap.update({"population:a": 1.0}, step=42)
        pub.publish(heatmap, step=42)

        assert dot_path.exists(), "dot file not written"
        content = dot_path.read_text()
        assert "digraph" in content

    def test_publish_includes_dot_in_git_add(self, tmp_path):
        from neuroslm.evolution.publisher import HeatmapPublisher

        heatmap_path = tmp_path / "heatmap.json"
        dot_path = tmp_path / "heatmap.dot"

        added = []
        def fake_runner(args, cwd=None):
            if args[0] == "add":
                added.extend(args[1:])
            return 0

        pub = HeatmapPublisher(
            heatmap_path=str(heatmap_path),
            commit_every=0,
            runner=fake_runner,
            dot_renderer=lambda hm: "digraph G {}",
            dot_path=str(dot_path),
        )

        heatmap = TrainingHeatmap()
        heatmap.update({"population:a": 0.5}, step=1)
        pub.publish(heatmap, step=1)

        assert str(dot_path) in added, (
            f"dot_path not in git add args: {added}"
        )
