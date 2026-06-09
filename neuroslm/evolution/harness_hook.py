# -*- coding: utf-8 -*-
"""L2-wire — harness hook for incremental heatmap updates.

A thin wrapper around L2's ``update_heatmap`` that respects an
``every_n`` cadence, supports an optional ``HeatmapPublisher`` for
git-commit/push side-effects, and **swallows every exception** so a
heatmap-collection failure can never break a training run.

Typical usage in the harness:

    from neuroslm.evolution.harness_hook import HeatmapHook

    hook = HeatmapHook.from_arch_root(
        model, arch_root="architectures/rcc_bowtie",
        every_n=100,
        heatmap_path="results/heatmaps/rcc_bowtie.heatmap.json",
    )

    for step in range(total_steps):
        train_step(...)
        hook.step(step)            # no-op when disabled / outside cadence
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from neuroslm.evolution.grad_heat import (
    parameter_grad_norms, update_heatmap,
)
from neuroslm.evolution.heatmap import TrainingHeatmap


__all__ = ["HeatmapHook"]


@dataclass
class HeatmapHook:
    """Incremental heatmap collector with publisher cadence + safety net.

    Attributes
    ----------
    model
        Any object exposing ``.named_parameters()`` returning
        ``Iterable[(name, tensor-with-.grad)]`` — typically a
        ``torch.nn.Module`` (or our ``BRIANHarness``).
    ir
        :class:`HypergraphIR` (or duck-typed equivalent with
        ``.nodes`` + ``.hyperedges``). Built once per training run.
    heatmap
        The :class:`TrainingHeatmap` to fold updates into. Created
        empty if not supplied.
    every_n
        Cadence: ``step(step_idx)`` is a no-op unless
        ``step_idx % every_n == 0``. Default ``100``.
    publisher
        Optional publisher with a ``maybe_publish(heatmap, step)``
        method (see :class:`HeatmapPublisher`). Called on every
        fired step; the publisher manages its own commit cadence.
    alias
        Optional ``{model_token: ir_node_name}`` map fed to
        :func:`update_heatmap` for region-name mismatches
        (e.g. ``{"hippo": "hippocampus"}``).
    enabled
        Master switch. ``False`` -> every ``step()`` call is a no-op.
        The factory method auto-sets this to ``False`` when the arch
        root is missing, so callers don't need to special-case absent
        DSL files.
    verbose
        If ``True``, print a single line to stderr every cadence-fire.
    """
    model: Any
    ir: Any
    heatmap: TrainingHeatmap = field(default_factory=TrainingHeatmap)
    every_n: int = 100
    publisher: Optional[Any] = None
    alias: Optional[Dict[str, str]] = None
    enabled: bool = True
    verbose: bool = False

    # ── factory ────────────────────────────────────────────────────

    @classmethod
    def from_arch_root(
        cls,
        model: Any,
        arch_root: Union[str, Path],
        *,
        every_n: int = 100,
        heatmap_path: Optional[Union[str, Path]] = None,
        commit_every: int = 0,
        push: bool = False,
        remote: str = "origin",
        branch: Optional[str] = None,
        alias: Optional[Dict[str, str]] = None,
        verbose: bool = False,
    ) -> "HeatmapHook":
        """Build a hook from an ``architectures/<arch>/`` root.

        Looks for ``<arch_root>/arch.neuro`` and lifts it through
        :func:`lift_arch_to_hypergraph`. When the file is absent (or
        anything else goes wrong during lifting), returns a
        ``HeatmapHook(enabled=False)`` so the training loop can call
        ``hook.step(...)`` blindly without a guard.
        """
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph

        arch_root = Path(arch_root)
        try:
            ir = lift_arch_to_hypergraph(arch_root)
        except Exception:
            ir = None

        if ir is None or not getattr(ir, "nodes", None):
            return cls(
                model=model, ir=_EmptyIR(), enabled=False,
                every_n=every_n, alias=alias, verbose=verbose,
            )

        publisher = None
        if heatmap_path is not None and commit_every > 0:
            from neuroslm.evolution.publisher import HeatmapPublisher
            publisher = HeatmapPublisher(
                heatmap_path=str(heatmap_path),
                commit_every=commit_every,
                push=push, remote=remote, branch=branch,
            )

        return cls(
            model=model, ir=ir, every_n=every_n,
            publisher=publisher, alias=alias, verbose=verbose,
        )

    # ── the one method the harness calls ───────────────────────────

    def step(self, step_idx: int) -> None:
        """Update the heatmap iff ``enabled`` and ``step_idx`` is on cadence.

        Never raises — failures are caught and (when ``verbose``)
        logged to stderr.
        """
        if not self.enabled:
            return
        if self.every_n <= 0:
            return
        if step_idx % self.every_n != 0:
            return

        try:
            grad_norms = parameter_grad_norms(self.model.named_parameters())
            update_heatmap(
                self.heatmap, grad_norms, self.ir,
                step=step_idx,
                publisher=self.publisher,
                alias=self.alias or {},
            )
            if self.verbose:
                hottest = self.heatmap.rank(top=3)
                print(
                    f"[heatmap @ step {step_idx}] hottest: {hottest}",
                    file=sys.stderr,
                )
        except Exception as exc:
            if self.verbose:
                print(
                    f"[heatmap @ step {step_idx}] failure swallowed: {exc!r}",
                    file=sys.stderr,
                )
            # Swallow — never crash training.
            return


# ── internals ──────────────────────────────────────────────────────


class _EmptyIR:
    """Stand-in IR for the disabled-hook case. Exists so callers can
    still introspect ``hook.ir.nodes`` without a None-check."""
    nodes = ()
    hyperedges = ()
