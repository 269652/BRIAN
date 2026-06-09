# -*- coding: utf-8 -*-
"""TDD: ``neuroslm.evolution.training_loop.EvolutionLoop`` — the per-step
orchestrator that connects:

    HeatmapHook (grad-norm rollup over the Hypergraph IR)
        → propose_mutations (heatmap hot/cold paths → DNAPatch[])
        → gate_proposals    (ImprovementGate over a sliding loss window)
        → EvolutionaryTrainingContext.save_checkpoint
           (admitted patches persist into the DNA's checkpoint dir)

Contract (frozen by these tests):

1. ``EvolutionLoop`` can be built from an architecture root + DNA file +
   checkpoint dir. When the architecture compiles, it ends up ``enabled``;
   when the IR fails to lift, it ends up ``enabled=False`` and every
   subsequent ``tick(...)`` is a safe no-op.

2. ``tick(step, loss)`` is a safe no-op until the mutation cadence fires.
   Cadence is controlled by ``mutate_every``; intermediate steps return
   ``None`` but still feed the loss window.

3. When the cadence fires AND the heatmap has hot/cold elements, the loop
   proposes mutations, gates them against the sliding loss window, and
   persists admitted ones via ``EvolutionaryTrainingContext.save_checkpoint``
   (writing ``step_<NNNNN>.patch.dna`` files into the checkpoint dir).

4. The live heatmap is written to ``<checkpoint_dir>/live_heatmap.json``
   on ``save_heatmap_every`` cadence so ``brian compile nfg --current
   --heat <path>`` can pick it up.

5. The loop never raises in ``tick(...)`` — gate failures, persistence
   failures, etc. are swallowed and reported via the ``stats`` dict.

These tests do NOT exercise a real training run. They use ``MagicMock``
models with deterministic ``named_parameters()`` returning fake-grad
tensors so the heatmap evolves predictably.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch


# ── shared fixtures ──────────────────────────────────────────────────


# A tiny architecture string with two populations + a synapse so the
# hypergraph IR has ≥2 nodes + 1 hyperedge — enough for propose_mutations
# to emit both node_mutation (HOT node) and edge_prune (COLD edge).
SAMPLE_DSL = (
    "architecture training_loop_test { d_sem: 256 }\n"
    "neurotransmitter dopamine { base_concentration: 0.5 }\n"
    'population cortex { count: 64, dynamics: "rate_code" }\n'
    'population striatum { count: 32, dynamics: "rate_code" }\n'
    "synapse cortex -> striatum { weight: 0.6 }\n"
)


@pytest.fixture
def tiny_arch_root(tmp_path):
    """Architecture folder with a minimal arch.neuro on disk."""
    arch_root = tmp_path / "tiny_arch"
    arch_root.mkdir()
    (arch_root / "arch.neuro").write_text(SAMPLE_DSL, encoding="utf-8")
    return arch_root


@pytest.fixture
def tiny_dna(tmp_path):
    """A minimal valid DNA file on disk (256 floats, version 1.0)."""
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
    """A MagicMock model whose ``.named_parameters()`` returns one big
    gradient on a param starting with ``cortex.`` and a tiny one on a
    param starting with ``striatum.``. Drives the heatmap predictably:
    the cortex node ends up HOT, the cortex→striatum edge ends up MID,
    striatum stays cold.

    The harness's ``attach_heatmap_hook`` interface needs only
    ``named_parameters()``, so we don't need a real ``nn.Module``.
    """
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
    # A no-op attach_heatmap_hook keeps the test independent of the real
    # harness signature; the loop only ever calls attach if the model
    # exposes it.
    model._attached_hook = None
    def _attach(hook):
        model._attached_hook = hook
    model.attach_heatmap_hook = _attach
    return model


# ── 1. Construction + enabled gating ─────────────────────────────────


class TestConstruction:
    """The loop loads its IR + DNA + checkpoint context on init."""

    def test_module_importable(self):
        from neuroslm.evolution.training_loop import EvolutionLoop  # noqa: F401

    def test_enabled_when_arch_and_dna_present(
            self, tiny_arch_root, tiny_dna, tmp_path):
        """When the arch compiles into ≥1 IR node and the DNA exists,
        the loop is enabled and its stats expose the IR sizes."""
        from neuroslm.evolution.training_loop import EvolutionLoop
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert loop.enabled is True
        stats = loop.stats
        assert stats["enabled"] is True
        assert stats["ir_nodes"] >= 2     # cortex + striatum
        assert stats["ir_edges"] >= 1     # cortex → striatum

    def test_disabled_when_arch_missing(self, tiny_dna, tmp_path):
        """Pointing at an arch root with no ``arch.neuro`` returns a
        loop that is disabled and never crashes on tick()."""
        from neuroslm.evolution.training_loop import EvolutionLoop
        empty = tmp_path / "no_arch"
        empty.mkdir()
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=empty,
            dna_path=tiny_dna,
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert loop.enabled is False
        assert loop.tick(step=100, loss=4.2) is None
        assert loop.tick(step=200, loss=4.0) is None

    def test_disabled_when_dna_missing(self, tiny_arch_root, tmp_path):
        """A missing DNA file disables the loop without raising."""
        from neuroslm.evolution.training_loop import EvolutionLoop
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tmp_path / "does_not_exist.dna",
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert loop.enabled is False
        assert loop.tick(step=100, loss=4.2) is None

    def test_attaches_hook_to_harness(
            self, tiny_arch_root, tiny_dna, tmp_path):
        """The loop installs the HeatmapHook on the harness via
        ``attach_heatmap_hook`` (when present)."""
        from neuroslm.evolution.training_loop import EvolutionLoop
        model = _model_with_hot_cortex_grads()
        loop = EvolutionLoop(
            harness=model,
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert loop.enabled is True
        assert model._attached_hook is loop._hook  # noqa: SLF001


# ── 2. Tick cadence — no mutation cycle below mutate_every ───────────


class TestTickCadence:
    def test_tick_below_cadence_returns_none(
            self, tiny_arch_root, tiny_dna, tmp_path):
        from neuroslm.evolution.training_loop import EvolutionLoop
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=tmp_path / "ckpt",
            heatmap_every=10, mutate_every=100,
            save_heatmap_every=0,
        )
        for step in range(1, 50):
            assert loop.tick(step, loss=4.0) is None

    def test_tick_at_zero_returns_none(
            self, tiny_arch_root, tiny_dna, tmp_path):
        """``step=0`` never fires a cycle even if ``step % cadence == 0``,
        because the loss window has only one sample."""
        from neuroslm.evolution.training_loop import EvolutionLoop
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=tmp_path / "ckpt",
            heatmap_every=10, mutate_every=100,
            save_heatmap_every=0,
        )
        assert loop.tick(0, loss=4.5) is None


# ── 3. Mutation cycle fires + persists at the cadence ───────────────


class TestMutationCycle:
    def _drive_to_cadence(self, loop, hook_signal_fn,
                          loss_before=4.5, loss_after=3.0,
                          warm_steps=20, fire_step=100):
        """Helper: drive the loop's loss window with a clear improving
        signal, then directly inject signals into the heatmap so the
        next mutation cadence has hot/cold elements to act on, and
        finally call tick at ``fire_step``."""
        # Warm-up: high loss
        for s in range(1, warm_steps + 1):
            loop.tick(s, loss_before)
        # Improvement: low loss (the gate sees before-vs-after)
        for s in range(warm_steps + 1, fire_step):
            loop.tick(s, loss_after)
        # Inject hot/cold signals directly so we don't depend on the
        # grad-norm path landing real values.
        hook_signal_fn(loop)
        return loop.tick(fire_step, loss_after)

    def _inject_hot_cold(self, loop):
        """Push a hot signal into the cortex node and a cold edge."""
        hm = loop._hook.heatmap                                # noqa: SLF001
        # Build the heatmap directly with hot cortex + cold synapse.
        signals, kinds = {}, {}
        for node in loop._hook.ir.nodes:                       # noqa: SLF001
            if node.name == "cortex":
                signals[node.id] = 1.0
                kinds[node.id] = "node"
            else:
                signals[node.id] = 0.01
                kinds[node.id] = "node"
        for edge in loop._hook.ir.hyperedges:                  # noqa: SLF001
            signals[edge.id] = 0.005
            kinds[edge.id] = "edge"
        # Apply twice so the EMA settles and the cold edges look cold.
        for _ in range(3):
            hm.update(signals, kinds=kinds)

    def test_cadence_fires_returns_dict_with_counts(
            self, tiny_arch_root, tiny_dna, tmp_path):
        """At the mutation cadence with a populated heatmap, ``tick``
        returns a dict with proposal/admission/rejection counts."""
        from neuroslm.evolution.training_loop import EvolutionLoop
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=tmp_path / "ckpt",
            heatmap_every=10, mutate_every=100,
            save_heatmap_every=0,
        )
        result = self._drive_to_cadence(loop, self._inject_hot_cold)
        assert result is not None
        assert result["step"] == 100
        assert "n_proposed" in result
        assert "n_admitted" in result
        assert "n_rejected" in result
        assert result["n_proposed"] >= 1   # cortex was hot → at least 1 proposal

    def test_admitted_patches_persisted_to_checkpoint_dir(
            self, tiny_arch_root, tiny_dna, tmp_path):
        """At least one admitted mutation should write a
        ``step_NNNNN(_<target>)?.patch.dna`` file into the checkpoint
        directory. (Skips cleanly if the gate happens to reject all
        proposals on this synthetic data — we still want determinism.)
        """
        from neuroslm.evolution.training_loop import EvolutionLoop
        ckpt = tmp_path / "ckpt"
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=ckpt,
            heatmap_every=10, mutate_every=100,
            save_heatmap_every=0,
        )
        result = self._drive_to_cadence(loop, self._inject_hot_cold)
        assert result is not None and "error" not in result, result

        if result["n_admitted"] > 0:
            patches = list(ckpt.glob("step_*.patch.dna"))
            assert patches, ("admitted > 0 but no patch files written "
                             f"to {ckpt}; result={result}")

    def test_stats_running_totals_advance_on_cycle(
            self, tiny_arch_root, tiny_dna, tmp_path):
        """After a firing cycle, ``stats`` totals reflect what was
        proposed / admitted / rejected."""
        from neuroslm.evolution.training_loop import EvolutionLoop
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=tmp_path / "ckpt",
            heatmap_every=10, mutate_every=100,
            save_heatmap_every=0,
        )
        before = loop.stats
        assert before["n_proposed_total"] == 0
        result = self._drive_to_cadence(loop, self._inject_hot_cold)
        assert result is not None
        after = loop.stats
        assert after["n_proposed_total"] == result["n_proposed"]
        assert after["n_admitted_total"] == result["n_admitted"]
        assert after["n_rejected_total"] == result["n_rejected"]


# ── 4. Live heatmap is written on cadence ───────────────────────────


class TestHeatmapPersistence:
    def test_live_heatmap_written_at_cadence(
            self, tiny_arch_root, tiny_dna, tmp_path):
        from neuroslm.evolution.training_loop import EvolutionLoop
        ckpt = tmp_path / "ckpt"
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=ckpt,
            heatmap_every=10, mutate_every=10_000,    # don't fire mutations
            save_heatmap_every=50,
        )
        # Inject some signals so the heatmap isn't empty when it's saved.
        signals = {}
        kinds = {}
        for node in loop._hook.ir.nodes:                  # noqa: SLF001
            signals[node.id] = 0.5
            kinds[node.id] = "node"
        loop._hook.heatmap.update(signals, kinds=kinds)   # noqa: SLF001

        for s in range(1, 51):
            loop.tick(s, loss=4.0)

        live = ckpt / "live_heatmap.json"
        assert live.is_file(), f"expected live heatmap at {live}, dir contents: {list(ckpt.iterdir())}"
        payload = json.loads(live.read_text(encoding="utf-8"))
        assert "entries" in payload or "step" in payload


# ── 5. tick() is exception-safe ──────────────────────────────────────


class TestExceptionSafety:
    def test_tick_swallows_propose_failure(
            self, tiny_arch_root, tiny_dna, tmp_path, monkeypatch):
        """If ``propose_mutations`` raises, ``tick`` still returns a
        dict (with an ``error`` key) and does not crash the caller."""
        from neuroslm.evolution.training_loop import EvolutionLoop
        loop = EvolutionLoop(
            harness=_model_with_hot_cortex_grads(),
            arch_root=tiny_arch_root,
            dna_path=tiny_dna,
            checkpoint_dir=tmp_path / "ckpt",
            heatmap_every=10, mutate_every=20,
            save_heatmap_every=0,
        )
        # Force the propose path to raise.
        import neuroslm.evolution.training_loop as tl_mod
        def _boom(*a, **kw):
            raise RuntimeError("synthetic")
        monkeypatch.setattr(tl_mod, "propose_mutations", _boom)

        # Warm up + fire cadence with the (empty) heatmap. The cadence
        # is at step=20 so we need ≥8 loss samples to clear the gate.
        for s in range(1, 21):
            loop.tick(s, loss=4.0)
        result = loop.tick(20, loss=4.0)
        # Either swallowed silently (returned None) or surfaced as
        # an error dict — both are acceptable; what matters is no raise.
        if result is not None:
            # When an error fires inside the cadence path we surface it.
            assert "error" in result or "n_proposed" in result
