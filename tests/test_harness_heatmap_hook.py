# -*- coding: utf-8 -*-
"""TDD: L2-wire harness hook.

The harness exposes a way to plug in the L1+L2 heatmap collector so it
updates incrementally during training without coupling the harness to
the evolution subsystem. The contract:

  HeatmapHook(model, ir, *, every_n=100, publisher=None, alias=None,
              enabled=True) — pure data carrier + .step(step_idx).
  .step(step_idx) is the one method the harness calls; it:
    1. respects `enabled` (no-op when False)
    2. fires only every `every_n` steps
    3. computes grad norms from the model's named_parameters()
    4. folds them into the heatmap via update_heatmap(...)
    5. wraps everything in try/except so a heatmap failure never
       crashes training
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── tiny torch-free fake "model" + "ir" so the test runs anywhere ─


class _FakeParam:
    """Stands in for a torch.Tensor with a .grad attribute."""
    def __init__(self, grad_norm):
        # We construct a 1-element tensor whose L2 norm equals grad_norm.
        import torch
        if grad_norm is None:
            self.grad = None
        else:
            self.grad = torch.tensor([float(grad_norm)])


class _FakeModel:
    """Minimal stand-in: only need .named_parameters()."""
    def __init__(self, grad_by_name):
        self._params = {name: _FakeParam(g) for name, g in grad_by_name.items()}

    def named_parameters(self):
        return list(self._params.items())


class _FakeNode:
    def __init__(self, id_, name, kind="population"):
        self.id = id_
        self.name = name
        self.kind = kind


class _FakeEdge:
    def __init__(self, id_, members, kind="synapse"):
        self.id = id_
        self.members = list(members)
        self.kind = kind


class _FakeIR:
    def __init__(self, nodes, edges):
        self.nodes = list(nodes)
        self.hyperedges = list(edges)


@pytest.fixture
def tiny_model():
    return _FakeModel({
        "cortex.weight":   1.0,
        "cortex.bias":     0.5,
        "striatum.weight": 0.1,
    })


@pytest.fixture
def tiny_ir():
    return _FakeIR(
        nodes=[
            _FakeNode("population:cortex",   "cortex"),
            _FakeNode("population:striatum", "striatum"),
        ],
        edges=[
            _FakeEdge("synapse:cortex->striatum", ["cortex", "striatum"]),
        ],
    )


# ── tests ──────────────────────────────────────────────────────────


class TestHeatmapHookBasics:
    def test_step_at_cadence_writes_to_heatmap(self, tiny_model, tiny_ir):
        from neuroslm.evolution.harness_hook import HeatmapHook
        from neuroslm.evolution import TrainingHeatmap
        hm = TrainingHeatmap()
        hook = HeatmapHook(tiny_model, tiny_ir, heatmap=hm, every_n=1)
        hook.step(step_idx=1)
        # cortex got the largest grads -> highest signal
        norm = hm.normalized()
        assert norm["population:cortex"] > norm["population:striatum"]

    def test_step_below_cadence_is_noop(self, tiny_model, tiny_ir):
        from neuroslm.evolution.harness_hook import HeatmapHook
        from neuroslm.evolution import TrainingHeatmap
        hm = TrainingHeatmap()
        hook = HeatmapHook(tiny_model, tiny_ir, heatmap=hm, every_n=10)
        hook.step(step_idx=5)                     # 5 % 10 != 0
        assert hm.entries == {}                   # nothing written

    def test_step_disabled_is_noop(self, tiny_model, tiny_ir):
        from neuroslm.evolution.harness_hook import HeatmapHook
        from neuroslm.evolution import TrainingHeatmap
        hm = TrainingHeatmap()
        hook = HeatmapHook(tiny_model, tiny_ir, heatmap=hm,
                           every_n=1, enabled=False)
        hook.step(step_idx=1)
        assert hm.entries == {}

    def test_failure_swallowed_does_not_crash(self, tiny_ir):
        """A broken model (no named_parameters) must NOT bubble an
        exception into the training loop."""
        from neuroslm.evolution.harness_hook import HeatmapHook
        from neuroslm.evolution import TrainingHeatmap
        hm = TrainingHeatmap()

        class Broken:
            def named_parameters(self):
                raise RuntimeError("simulated kernel crash")

        hook = HeatmapHook(Broken(), tiny_ir, heatmap=hm, every_n=1)
        # Must NOT raise.
        hook.step(step_idx=1)
        assert hm.entries == {}                   # nothing written

    def test_alias_map_is_honored(self, tiny_ir):
        from neuroslm.evolution.harness_hook import HeatmapHook
        from neuroslm.evolution import TrainingHeatmap
        # The model uses 'hippo' for the hippocampus node.
        model = _FakeModel({"hippo.weight": 2.0, "cortex.weight": 1.0})
        ir = _FakeIR(
            nodes=[
                _FakeNode("population:hippocampus", "hippocampus"),
                _FakeNode("population:cortex", "cortex"),
            ],
            edges=[],
        )
        hm = TrainingHeatmap()
        hook = HeatmapHook(
            model, ir, heatmap=hm, every_n=1,
            alias={"hippo": "hippocampus"},
        )
        hook.step(step_idx=1)
        norm = hm.normalized()
        # hippo aliased -> the hippocampus node got the bigger signal
        assert norm["population:hippocampus"] > norm["population:cortex"]


class TestHeatmapHookPublisher:
    def test_publisher_invoked_at_cadence(self, tiny_model, tiny_ir):
        from neuroslm.evolution.harness_hook import HeatmapHook
        from neuroslm.evolution import TrainingHeatmap
        hm = TrainingHeatmap()
        pub = MagicMock()
        hook = HeatmapHook(tiny_model, tiny_ir, heatmap=hm,
                           every_n=1, publisher=pub)
        hook.step(step_idx=42)
        pub.maybe_publish.assert_called_once()
        # Step idx is forwarded to publisher so its own cadence works.
        call_args = pub.maybe_publish.call_args
        assert call_args is not None
        forwarded_step = (
            call_args.kwargs.get("step")
            if call_args.kwargs and "step" in call_args.kwargs
            else call_args.args[1]
        )
        assert forwarded_step == 42


class TestHeatmapHookFactory:
    def test_from_arch_root_resolves_ir_and_heatmap(self, tmp_path, tiny_model):
        """The factory builds the IR from arch.neuro and creates a
        results-directory heatmap automatically."""
        from neuroslm.evolution.harness_hook import HeatmapHook

        # Minimal arch.neuro on disk
        arch_root = tmp_path / "tiny_arch"
        arch_root.mkdir()
        (arch_root / "arch.neuro").write_text(
            "architecture a { d_sem: 256 }\n"
            'population cortex { count: 64, dynamics: "rate_code" }\n'
            'population striatum { count: 32, dynamics: "rate_code" }\n'
            "synapse cortex -> striatum { weight: 0.5 }\n",
            encoding="utf-8",
        )
        out_path = tmp_path / "heatmap.json"
        hook = HeatmapHook.from_arch_root(
            tiny_model, arch_root,
            every_n=1, heatmap_path=out_path,
        )
        # The IR was lifted -> cortex/striatum nodes present
        ids = {n.id for n in hook.ir.nodes}
        assert "population:cortex" in ids
        assert "population:striatum" in ids

    def test_from_arch_root_disabled_when_no_arch_neuro(self, tmp_path, tiny_model):
        from neuroslm.evolution.harness_hook import HeatmapHook
        empty_root = tmp_path / "empty_arch"
        empty_root.mkdir()                          # no arch.neuro inside
        hook = HeatmapHook.from_arch_root(
            tiny_model, empty_root, every_n=1,
        )
        assert hook.enabled is False                # graceful no-op
