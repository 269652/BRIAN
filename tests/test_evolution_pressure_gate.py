# -*- coding: utf-8 -*-
"""TDD: ``EvolutionLoop`` must gate the mutation cycle on heatmap pressure
in addition to clock cadence.

Problem
-------
Today ``tick(step, loss)`` fires the propose → gate → persist cycle on
pure cadence (``step % mutate_every == 0`` with ``len(loss_window) ≥ 8``).
That means a steady, well-converged run mutates on every
``mutate_every``-th step even when *nothing* in the hypergraph is
actually under gradient pressure — wasting compute on no-op cycles and,
worse, slowly drifting the DNA on noise.

Desired behaviour
-----------------
The mutation cycle should fire only when BOTH conditions hold:

  (a) clock cadence is on  (existing behaviour), AND
  (b) the heatmap shows at least one element whose *normalised* heat
      exceeds ``pressure_threshold`` — i.e. some part of the graph is
      genuinely hotter than the rest, not just uniformly warm noise.

Contract
--------
* New parameter ``pressure_threshold: float = 0.0`` on
  ``EvolutionLoop``.  Default ``0.0`` keeps today's behaviour exactly
  (any positive max heat satisfies ``> 0.0``).
* When ``> 0.0``, ``tick`` returns ``None`` on cadence steps whose
  normalised heatmap has ``max < pressure_threshold``.
* When the gate fires successfully, the returned cycle dict gains a
  ``"max_pressure"`` key (float in ``[0, 1]``) reporting the gate value
  so operators can tune the threshold from log output.
* The skip path increments a new ``stats["n_cycles_skipped_lowpressure"]``
  counter — every gate decision is recorded.
* Failure mode: when the heatmap is empty (no parameters have produced
  signals yet), pressure = 0 → cycle is skipped (not crashed).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch


# ── shared fixtures — kept in sync with tests/test_evolution_training_loop.py ──
# tests/ isn't a Python package, so we can't import the sibling fixtures.
# Duplicating them here keeps both test files independent of each other
# while still locking the same contract.


SAMPLE_DSL = (
    "architecture pressure_gate_test { d_sem: 256 }\n"
    "neurotransmitter dopamine { base_concentration: 0.5 }\n"
    'population cortex { count: 64, dynamics: "rate_code" }\n'
    'population striatum { count: 32, dynamics: "rate_code" }\n'
    "synapse cortex -> striatum { weight: 0.6 }\n"
)


@pytest.fixture
def tiny_arch_root(tmp_path):
    arch_root = tmp_path / "tiny_arch"
    arch_root.mkdir()
    (arch_root / "arch.neuro").write_text(SAMPLE_DSL, encoding="utf-8")
    return arch_root


@pytest.fixture
def tiny_dna(tmp_path):
    dna_path = tmp_path / "tiny.dna"
    payload = {
        "version": "1.0",
        "length": 256,
        "data": [0.1] * 256,
        "invariants": {},
    }
    dna_path.write_text(json.dumps(payload), encoding="utf-8")
    return dna_path


def _model_with_hot_cortex_grads(seed: int = 0):
    """MagicMock model with one big grad on a cortex.* param and a tiny
    one on a striatum.* param — drives the heatmap to a known shape
    (cortex node HOT, striatum node COLD)."""
    torch.manual_seed(seed)
    p_hot = torch.zeros(8, requires_grad=True)
    p_hot.grad = torch.full_like(p_hot, 5.0)        # L2 norm ≈ 14.14
    p_cold = torch.zeros(8, requires_grad=True)
    p_cold.grad = torch.full_like(p_cold, 0.001)    # L2 norm ≈ 0.0028
    model = MagicMock()
    model.named_parameters = lambda: iter([
        ("cortex.weight",   p_hot),
        ("striatum.weight", p_cold),
    ])
    model._attached_hook = None
    def _attach(hook):
        model._attached_hook = hook
    model.attach_heatmap_hook = _attach
    return model


def _inject_hot_cold(loop):
    """Push a known hot/cold signal pattern into the heatmap so tests
    don't depend on the harness actually running training steps.

    Mirrors the helper in tests/test_evolution_training_loop.py:
    cortex node = 1.0 (HOT), striatum node = 0.01 (cold), all edges
    = 0.005 (cold). The EMA is settled by 3 successive updates so the
    cold values lock in.
    """
    hm = loop._hook.heatmap
    signals, kinds = {}, {}
    for node in loop._hook.ir.nodes:
        if node.name == "cortex":
            signals[node.id] = 1.0
        else:
            signals[node.id] = 0.01
        kinds[node.id] = "node"
    for edge in loop._hook.ir.hyperedges:
        signals[edge.id] = 0.005
        kinds[edge.id] = "edge"
    for _ in range(3):
        hm.update(signals, kinds=kinds)


# ── 1. Backward compatibility — default pressure_threshold = 0.0 ─────


class TestPressureThresholdDefault:
    """Default pressure_threshold=0.0 must keep today's behaviour."""

    def test_default_pressure_threshold_is_zero(
            self, tiny_arch_root, tiny_dna, tmp_path):
        from neuroslm.evolution.training_loop import EvolutionLoop
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert loop.pressure_threshold == 0.0

    def test_default_fires_on_cadence_with_any_pressure(
            self, tiny_arch_root, tiny_dna, tmp_path):
        """At default threshold=0, any positive max heat → cycle fires."""
        from neuroslm.evolution.training_loop import EvolutionLoop
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=tmp_path / "ckpt",
            heatmap_every=1, mutate_every=10,
            save_heatmap_every=0,
        )
        # Warm the loss window (Welch-t needs ≥8 samples) + populate the heatmap.
        for s in range(1, 10):
            loop.tick(step=s, loss=4.5 - 0.01 * s)
        # Cadence step at 10 → with the hot-cortex model, max_pressure > 0
        # → cycle MUST fire (cumulative counter advances).
        loop.tick(step=10, loss=4.0)
        assert loop.stats["n_cycles_fired"] >= 1


# ── 2. Strict-pressure mode — high threshold skips low-pressure cycles ─


class TestPressureGateSkipsLowPressure:
    """When pressure_threshold > observed max, cadence cycles are
    skipped and the skip counter advances."""

    def test_high_threshold_skips_cycle(
            self, tiny_arch_root, tiny_dna, tmp_path):
        from neuroslm.evolution.training_loop import EvolutionLoop
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=tmp_path / "ckpt",
            heatmap_every=1, mutate_every=10,
            save_heatmap_every=0,
            pressure_threshold=1.5,   # > max possible normalised heat (1.0)
        )
        for s in range(1, 10):
            loop.tick(step=s, loss=4.5 - 0.01 * s)
        result = loop.tick(step=10, loss=4.0)
        # The propose/gate cycle MUST NOT have run.
        assert result is None or result.get("skipped") == "low_pressure"
        assert loop.stats["n_cycles_fired"] == 0
        assert loop.stats["n_cycles_skipped_lowpressure"] >= 1

    def test_low_threshold_fires_cycle(
            self, tiny_arch_root, tiny_dna, tmp_path):
        """Counterexample: with threshold=0.01 the same setup must fire."""
        from neuroslm.evolution.training_loop import EvolutionLoop
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=tmp_path / "ckpt",
            heatmap_every=1, mutate_every=10,
            save_heatmap_every=0,
            pressure_threshold=0.01,
        )
        _inject_hot_cold(loop)
        for s in range(1, 10):
            loop.tick(step=s, loss=4.5 - 0.01 * s)
        loop.tick(step=10, loss=4.0)
        assert loop.stats["n_cycles_fired"] >= 1
        assert loop.stats["n_cycles_skipped_lowpressure"] == 0


# ── 3. Telemetry — max_pressure surfaces in the cycle result ─────────


class TestPressureTelemetry:
    """When the cycle fires, the result dict reports max_pressure."""

    def test_max_pressure_present_when_cycle_fires(
            self, tiny_arch_root, tiny_dna, tmp_path):
        from neuroslm.evolution.training_loop import EvolutionLoop
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=tmp_path / "ckpt",
            heatmap_every=1, mutate_every=10,
            save_heatmap_every=0,
            pressure_threshold=0.0,
        )
        for s in range(1, 10):
            loop.tick(step=s, loss=4.5 - 0.01 * s)
        result = loop.tick(step=10, loss=4.0)
        assert result is not None
        assert "max_pressure" in result
        # Normalised heat is by definition in [0, 1].
        assert 0.0 <= result["max_pressure"] <= 1.0

    def test_max_pressure_reflects_hot_cortex(
            self, tiny_arch_root, tiny_dna, tmp_path):
        """With the hot-cortex mock the max-normalised heat is 1.0
        (cortex dominates striatum by ~5000×)."""
        from neuroslm.evolution.training_loop import EvolutionLoop
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=tmp_path / "ckpt",
            heatmap_every=1, mutate_every=10,
            save_heatmap_every=0,
        )
        _inject_hot_cold(loop)
        for s in range(1, 10):
            loop.tick(step=s, loss=4.5 - 0.01 * s)
        result = loop.tick(step=10, loss=4.0)
        assert result is not None
        assert result["max_pressure"] == pytest.approx(1.0, abs=1e-6)


# ── 4. Empty heatmap → safe skip, no crash ───────────────────────────


class TestEmptyHeatmapSafeSkip:
    """When the heatmap has produced no signals yet (e.g. heatmap_every
    misaligned), pressure = 0 → cycle is skipped, not crashed."""

    def test_empty_heatmap_skips_cycle_when_threshold_positive(
            self, tiny_arch_root, tiny_dna, tmp_path):
        from neuroslm.evolution.training_loop import EvolutionLoop
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=tmp_path / "ckpt",
            heatmap_every=10_000,    # never rolls up in this test
            mutate_every=10,
            save_heatmap_every=0,
            pressure_threshold=0.1,
        )
        # Warm the loss window without ever updating the heatmap.
        for s in range(1, 11):
            loop.tick(step=s, loss=4.5 - 0.01 * s)
        # Step 10 is on cadence; heatmap is empty → must skip safely.
        assert loop.stats["n_cycles_fired"] == 0
        assert loop.stats["n_cycles_skipped_lowpressure"] >= 1


# ── 5. Counter never goes negative / always present ──────────────────


class TestStatsContract:
    """The new stats key must exist from construction (even when 0)."""

    def test_skipped_counter_initialised_to_zero(
            self, tiny_arch_root, tiny_dna, tmp_path):
        from neuroslm.evolution.training_loop import EvolutionLoop
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert "n_cycles_skipped_lowpressure" in loop.stats
        assert loop.stats["n_cycles_skipped_lowpressure"] == 0
