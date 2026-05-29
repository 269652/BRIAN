# -*- coding: utf-8 -*-
"""DSL adapter for NeuralOrchestrator's aux-loss contributions.

NeuralOrchestrator is a *routing* layer (17M params across ~10 nn.Module
sub-cortices: cerebellum, entorhinal, claustrum, ACC, BG, thalamus,
amygdala, …). It does not add new *mathematical content* — it composes
existing modules and emits routing metrics (`identity_drift`, `neural_calm`,
plus per-stage forward calls). Re-implementing each sub-cortex in DSL is
out of scope for a port: those modules are atoms in the same sense that
`nn.Linear` is an atom in DSL.

What this adapter does instead is define the **interface contract** the
DSL Brain aggregator needs to match Brain's trunk-gradient picture:

    Brain at brain.py:1823-1824:
        total += aux_w * (0.01 * id_drift + 0.01 * (1 - calm))

So the adapter expects: a callable that produces the two scalars
`(id_drift, calm)` per forward, and the aggregator applies the same
weighted-sum formula bit-for-bit. The actual Brain orchestrator (composed
of all its sub-modules) is the production source for those scalars;
the parity guarantee is on the *aggregation*, not on a DSL re-build
of cerebellum-as-DSL.

For tests we also provide a `MockOrchestrator` that returns fixed scalars,
so the aggregator's formula can be parity-checked against Brain's
formula without instantiating 17M params.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Tuple

import torch
import torch.nn as nn


@dataclass
class OrchestratorMetrics:
    """The two scalars the orchestrator contributes to the trunk gradient.

    Both default to "neutral" (no aux contribution) so omitting an
    orchestrator is equivalent to id_drift=0, calm=1 — i.e. `0.01 * 0 +
    0.01 * (1 - 1) = 0`. Same null behavior Brain has when the
    orchestrator branch is skipped.
    """
    identity_drift: torch.Tensor
    neural_calm: torch.Tensor

    @staticmethod
    def neutral(device=None, dtype=None) -> "OrchestratorMetrics":
        return OrchestratorMetrics(
            identity_drift=torch.zeros((), device=device, dtype=dtype),
            neural_calm=torch.ones((), device=device, dtype=dtype),
        )


class MockOrchestrator(nn.Module):
    """Test-only orchestrator returning fixed metrics.

    Used by the DSL Brain aggregator parity tests to assert that
    `total += aux_w * (0.01 * id_drift + 0.01 * (1-calm))` is computed
    bit-identically without spinning up the real 17M-param routing graph.
    """
    def __init__(self, id_drift: float = 0.0, calm: float = 1.0):
        super().__init__()
        self.register_buffer("_idrift", torch.tensor(float(id_drift)))
        self.register_buffer("_calm",   torch.tensor(float(calm)))

    def metrics(self) -> OrchestratorMetrics:
        return OrchestratorMetrics(self._idrift, self._calm)


def orchestrator_aux_contribution(metrics: OrchestratorMetrics,
                                   aux_w: float = 1.0) -> torch.Tensor:
    """`aux_w * (0.01 * id_drift + 0.01 * (1 - calm))` — Brain's formula.

    Single source of truth for this aggregation. Used by the DSL Brain
    aggregator AND by Brain (via a small refactor at brain.py:1823 if you
    want the same constants in one place) — so the constants `0.01, 0.01`
    can't drift between the two sides.
    """
    return aux_w * (0.01 * metrics.identity_drift
                    + 0.01 * (1.0 - metrics.neural_calm))
