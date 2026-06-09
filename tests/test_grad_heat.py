# -*- coding: utf-8 -*-
"""TDD: gradient heat collector (L2).

Maps model parameter gradients to HypergraphIR element ids:
  - node signal  = L2 aggregate of grad-norms of params whose top-level
                   name token matches the node's name (alias-able)
  - edge signal  = mean of its endpoint nodes' signals
and folds them into a TrainingHeatmap.
"""
import math

import pytest

from neuroslm.compiler.hypergraph_ir import lift_dsl_to_hypergraph
from neuroslm.evolution.grad_heat import (
    parameter_grad_norms, signals_from_grad_norms, update_heatmap,
)
from neuroslm.evolution.heatmap import TrainingHeatmap


SAMPLE = (
    "architecture a { d_sem: 256 }\n"
    "neurotransmitter dopamine { base_concentration: 0.5 }\n"
    'population cortex { count: 512, dynamics: "rate_code" }\n'
    'population striatum { count: 256, dynamics: "rate_code" }\n'
    "synapse cortex -> striatum { weight: 0.5 }\n"
    "modulation dopamine -> striatum { gain: 1.2 }\n"
)


class TestParameterGradNorms:
    def test_grad_norm_matches_torch(self):
        torch = pytest.importorskip("torch")
        p = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        p.grad = torch.tensor([3.0, 4.0])  # ||.||_2 = 5
        norms = parameter_grad_norms([("cortex.weight", p)])
        assert norms["cortex.weight"] == pytest.approx(5.0)

    def test_params_without_grad_are_skipped(self):
        torch = pytest.importorskip("torch")
        p = torch.nn.Parameter(torch.tensor([1.0]))
        p.grad = None
        norms = parameter_grad_norms([("pfc.weight", p)])
        assert "pfc.weight" not in norms


class TestSignalsFromGradNorms:
    def test_node_signal_is_l2_aggregate_of_param_norms(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        grad_norms = {"cortex.w": 3.0, "cortex.b": 4.0, "striatum.w": 1.0,
                      "dopamine.x": 2.0}
        signals, kinds = signals_from_grad_norms(grad_norms, ir)
        assert signals["population:cortex"] == pytest.approx(5.0)  # sqrt(9+16)
        assert signals["population:striatum"] == pytest.approx(1.0)
        assert signals["neurotransmitter:dopamine"] == pytest.approx(2.0)
        assert kinds["population:cortex"] == "node"

    def test_edge_signal_is_mean_of_endpoints(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        grad_norms = {"cortex.w": 3.0, "cortex.b": 4.0, "striatum.w": 1.0,
                      "dopamine.x": 2.0}
        signals, kinds = signals_from_grad_norms(grad_norms, ir)
        # synapse cortex->striatum: mean(5, 1) = 3
        assert signals["synapse:cortex->striatum"] == pytest.approx(3.0)
        # modulation dopamine->striatum: mean(2, 1) = 1.5
        assert signals["modulation:dopamine->striatum"] == pytest.approx(1.5)
        assert kinds["synapse:cortex->striatum"] == "edge"

    def test_alias_maps_model_name_to_ir_node(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        # model calls it "ctx" but the IR node is "cortex"
        grad_norms = {"ctx.weight": 2.0}
        signals, _ = signals_from_grad_norms(grad_norms, ir, alias={"ctx": "cortex"})
        assert signals["population:cortex"] == pytest.approx(2.0)

    def test_unmatched_params_are_ignored(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        grad_norms = {"unknown_module.w": 9.0}
        signals, _ = signals_from_grad_norms(grad_norms, ir)
        assert "population:cortex" not in signals
        # unmatched contributes to no node
        assert all(not k.endswith("unknown_module") for k in signals)


class TestUpdateHeatmap:
    def test_update_heatmap_folds_signals(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        grad_norms = {"cortex.w": 3.0, "cortex.b": 4.0, "striatum.w": 1.0}
        hm = TrainingHeatmap()
        update_heatmap(hm, grad_norms, ir, step=100)
        assert hm.step == 100
        assert hm.heat("population:cortex") == pytest.approx(5.0)
        # hottest node is cortex
        assert hm.hot_paths(threshold=0.7, kind="node")[0] == "population:cortex"

    def test_update_heatmap_invokes_publisher(self):
        ir = lift_dsl_to_hypergraph(SAMPLE)
        grad_norms = {"cortex.w": 3.0}
        hm = TrainingHeatmap()

        published = []

        class _Pub:
            def maybe_publish(self, heatmap, step):
                published.append(step)
                return True

        update_heatmap(hm, grad_norms, ir, step=500, publisher=_Pub())
        assert published == [500]
