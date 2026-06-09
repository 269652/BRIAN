# -*- coding: utf-8 -*-
"""Per-step evolutionary-training orchestrator.

``EvolutionLoop`` wires the five evolution-pipeline pieces into the
training loop:

    HeatmapHook          — fold per-step grad-norms into a hypergraph-IR
                            heatmap (cadence: ``heatmap_every``)
    propose_mutations    — HOT nodes → node_mutation patches,
                           HOT edges  → edge_strengthen patches,
                           COLD edges → edge_prune patches
    gate_proposals       — Welch's t-test admission over a sliding loss
                            window (the "improvement" gate); optional
                            structural / Lean gates plug in if attached
    save_checkpoint      — admitted patches are persisted into the
                            evolutionary checkpoint dir as
                            ``step_<NNNNN>(_<target>)?.patch.dna``
    live_heatmap.json    — written on ``save_heatmap_every`` so
                            ``brian compile nfg --current --heat`` can
                            colour the rendered NFG with current activity

The class is deliberately conservative: anything that goes wrong inside
``tick(...)`` is caught and surfaced via the return value or the
``stats`` dict — a malfunctioning evolution mechanism must never crash a
training run.

Usage from ``neuroslm/train_dsl.py`` (DNA mode)::

    from neuroslm.evolution.training_loop import EvolutionLoop

    loop = EvolutionLoop(
        harness=harness,
        arch_root=arch_root,           # the architectures/<name>/ folder
        dna_path=Path(args.dna),
        checkpoint_dir=Path(args.ckpt_dir) / "evolution",
        heatmap_every=args.heatmap_every,
        mutate_every=args.mutate_every,
        save_heatmap_every=args.save_heatmap_every,
    )
    print(f"[train_dsl] evolution loop: {loop.stats}")

    for step in range(args.steps):
        loss = harness.train_step(...)
        cycle = loop.tick(step, loss)
        if cycle is not None and cycle.get("n_admitted", 0) > 0:
            print(f"[train_dsl] @ step {step}: evolution cycle {cycle}")

The contract is locked by ``tests/test_evolution_training_loop.py``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from neuroslm.evolution.harness_hook import HeatmapHook
from neuroslm.evolution.mutator import propose_mutations
from neuroslm.evolution.gate import gate_proposals, ImprovementEvidence
from neuroslm.utils.colab import EvolutionaryTrainingContext
from neuroslm.verification.improvement_gate import ImprovementGate


__all__ = ["EvolutionLoop"]


@dataclass
class EvolutionLoop:
    """Per-step evolutionary-training orchestrator.

    Parameters
    ----------
    harness
        The training harness (must expose ``named_parameters()`` and
        optionally ``attach_heatmap_hook(hook)``; the real
        :class:`BRIANHarness` satisfies both).
    arch_root
        Path to the ``architectures/<name>/`` folder used to lift the
        hypergraph IR. When ``arch.neuro`` is missing or the lift
        fails, the loop becomes a permanent no-op (``enabled=False``).
    dna_path
        Path to the ``.dna`` file the patch stack accumulates against.
        When missing, the loop becomes a permanent no-op.
    checkpoint_dir
        Directory where admitted patches are written
        (``step_<NNNNN>.patch.dna``) and the live heatmap is saved
        (``live_heatmap.json``).
    heatmap_every
        Cadence for grad-norm rollups into the heatmap. Forwarded to
        the underlying :class:`HeatmapHook`.
    mutate_every
        Cadence for the propose → gate → persist cycle. ``0`` disables
        the cycle entirely (heatmap still updates on its own cadence).
    save_heatmap_every
        Cadence for writing the live heatmap to
        ``<checkpoint_dir>/live_heatmap.json``. ``0`` disables disk
        writes.
    hot_threshold, cold_threshold
        Forwarded to :func:`propose_mutations`.
    loss_window_size
        Number of most-recent training losses kept in the sliding
        window that feeds before/after evidence to ``gate_proposals``.
    """

    harness: Any
    arch_root: Path
    dna_path: Path
    checkpoint_dir: Path
    heatmap_every: int = 50
    mutate_every: int = 500
    save_heatmap_every: int = 500
    hot_threshold: float = 0.7
    cold_threshold: float = 0.1
    loss_window_size: int = 64

    # ── internal state — populated by __post_init__ ──────────────────
    _hook: Optional[HeatmapHook] = field(init=False, default=None, repr=False)
    _evo_ctx: Optional[EvolutionaryTrainingContext] = field(
        init=False, default=None, repr=False)
    _gate: Optional[ImprovementGate] = field(init=False, default=None, repr=False)
    _loss_window: Deque[float] = field(init=False, default_factory=deque, repr=False)
    _stats: Dict[str, Any] = field(init=False, default_factory=dict, repr=False)
    _last_cycle_step: int = field(init=False, default=-1, repr=False)

    # ── construction ────────────────────────────────────────────────

    def __post_init__(self) -> None:
        # Normalise paths.
        self.arch_root = Path(self.arch_root)
        self.dna_path = Path(self.dna_path)
        self.checkpoint_dir = Path(self.checkpoint_dir)

        # Stats default to disabled; replaced below if everything wires.
        self._stats = {
            "enabled": False,
            "reason": "uninitialised",
            "ir_nodes": 0,
            "ir_edges": 0,
            "resume_step": 0,
            "patches_loaded": 0,
            "n_proposed_total": 0,
            "n_admitted_total": 0,
            "n_rejected_total": 0,
            "n_cycles_fired": 0,
            "n_heatmaps_saved": 0,
        }
        self._loss_window = deque(maxlen=self.loss_window_size)

        # Refuse to enable when the DNA is missing — the patch stack
        # has nowhere to anchor and ``EvolutionaryTrainingContext``
        # would raise ``FileNotFoundError`` on __enter__.
        if not self.dna_path.is_file():
            self._stats["reason"] = f"dna not found: {self.dna_path}"
            return

        # Refuse to enable when the architecture cannot be lifted into
        # a hypergraph IR. ``HeatmapHook.from_arch_root`` already
        # returns an ``enabled=False`` hook in that case; we mirror
        # that flag at the loop level.
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            self._stats["reason"] = f"cannot create checkpoint_dir: {exc!r}"
            return

        self._hook = HeatmapHook.from_arch_root(
            self.harness,
            arch_root=self.arch_root,
            every_n=self.heatmap_every,
            heatmap_path=None,   # we save the live heatmap ourselves
            verbose=False,
        )
        if not self._hook.enabled:
            self._stats["reason"] = (
                f"hypergraph IR could not be lifted from {self.arch_root}")
            return

        # Attach the hook to the harness when it supports it. Falls
        # back gracefully — the loop can still drive the cycle as long
        # as the harness exposes ``named_parameters()``.
        attach = getattr(self.harness, "attach_heatmap_hook", None)
        if callable(attach):
            try:
                attach(self._hook)
            except Exception:  # noqa: BLE001 — never crash on attach
                pass

        # Load the patch stack from the checkpoint dir so subsequent
        # runs resume into the right step / DNA state. Failures here
        # are fatal-soft: we keep the heatmap working but disable the
        # mutation cycle so we never overwrite a corrupt patch stack.
        try:
            self._evo_ctx = EvolutionaryTrainingContext(
                str(self.dna_path), str(self.checkpoint_dir))
            self._evo_ctx.__enter__()
            self._stats["resume_step"] = self._evo_ctx.resume_step
            self._stats["patches_loaded"] = len(self._evo_ctx.patches or [])
        except Exception as exc:  # noqa: BLE001
            self._stats["reason"] = f"evolution context: {exc!r}"
            return

        self._gate = ImprovementGate()
        self._stats.update({
            "enabled": True,
            "reason": "ok",
            "ir_nodes": len(self._hook.ir.nodes),
            "ir_edges": len(self._hook.ir.hyperedges),
        })

    # ── public API ──────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return bool(self._stats.get("enabled"))

    @property
    def stats(self) -> Dict[str, Any]:
        """Read-only snapshot of cumulative loop telemetry."""
        return dict(self._stats)

    def tick(self, step: int, loss: float) -> Optional[Dict[str, Any]]:
        """Per-step entry point.

        Returns ``None`` when no mutation cycle fired this step.
        Returns a result dict on a firing step (containing ``step``,
        ``n_proposed``, ``n_admitted``, ``n_rejected``, ``kinds_admitted``
        on success, or an ``error`` key on a swallowed failure).
        """
        if not self.enabled:
            return None

        # Feed the sliding loss window unconditionally.
        try:
            self._loss_window.append(float(loss))
        except (TypeError, ValueError):
            return None

        # Live-heatmap save cadence — independent of mutation cadence.
        self._maybe_save_heatmap(step)

        # Mutation cadence: skip step 0, skip when not on cadence, and
        # require at least 8 loss samples so the Welch t-test has a
        # non-trivial before/after split.
        if self.mutate_every <= 0:
            return None
        if step <= 0 or step % self.mutate_every != 0:
            return None
        if len(self._loss_window) < 8:
            return None

        return self._run_cycle(step)

    # ── internals ───────────────────────────────────────────────────

    def _maybe_save_heatmap(self, step: int) -> None:
        if self.save_heatmap_every <= 0 or step <= 0:
            return
        if step % self.save_heatmap_every != 0:
            return
        assert self._hook is not None  # for type-checker
        try:
            out = self.checkpoint_dir / "live_heatmap.json"
            self._hook.heatmap.save(str(out))
            self._stats["n_heatmaps_saved"] += 1
        except Exception:  # noqa: BLE001 — heatmap save never blocks training
            pass

    def _run_cycle(self, step: int) -> Dict[str, Any]:
        assert self._hook is not None and self._evo_ctx is not None
        try:
            proposals = propose_mutations(
                self._hook.heatmap, self._hook.ir, step=step,
                hot_threshold=self.hot_threshold,
                cold_threshold=self.cold_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            return {"step": step, "error": f"propose_mutations: {exc!r}"}

        self._stats["n_cycles_fired"] += 1

        if not proposals:
            return {"step": step, "n_proposed": 0,
                    "n_admitted": 0, "n_rejected": 0,
                    "kinds_admitted": []}

        # Evidence: split the loss window in half. The Welch t-test
        # expects ≥2 samples per side; the n>=8 gate above guarantees
        # ≥4 per side.
        window = list(self._loss_window)
        mid = max(1, len(window) // 2)
        before, after = window[:mid], window[mid:]
        evidence_by_target = {
            p.target: ImprovementEvidence(before=before, after=after)
            for p in proposals
        }

        try:
            admitted, rejected = gate_proposals(
                proposals, evidence_by_target,
                improvement_gate=self._gate,
            )
        except Exception as exc:  # noqa: BLE001
            return {"step": step, "error": f"gate_proposals: {exc!r}"}

        if admitted:
            mutations = [
                {
                    "kind": p.kind,
                    "target": p.target,
                    "delta": list(p.delta),
                    "metadata": dict(p.metadata),
                }
                for p in admitted
            ]
            try:
                self._evo_ctx.save_checkpoint(step, mutations)
            except Exception as exc:  # noqa: BLE001
                return {
                    "step": step,
                    "n_proposed": len(proposals),
                    "n_admitted": len(admitted),
                    "n_rejected": len(rejected),
                    "error": f"save_checkpoint: {exc!r}",
                }

        self._stats["n_proposed_total"] += len(proposals)
        self._stats["n_admitted_total"] += len(admitted)
        self._stats["n_rejected_total"] += len(rejected)
        self._last_cycle_step = step

        return {
            "step": step,
            "n_proposed": len(proposals),
            "n_admitted": len(admitted),
            "n_rejected": len(rejected),
            "kinds_admitted": sorted({p.kind for p in admitted}),
        }
