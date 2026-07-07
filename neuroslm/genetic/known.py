# -*- coding: utf-8 -*-
"""Prior-art gate — a registry of known algorithms so the search seeks novelty.

Exploration that keeps rediscovering SGD, Adam, or plain backprop wastes compute.
This registry holds known algorithms as NGL programs and judges whether a
candidate is "the same algorithm" by **semantic-space structure** (the op
histogram + structural features), which is hyperparameter-invariant: SGD@0.01 and
SGD@0.5 have the same signature. The discovery loop can then penalize rediscovery
and reward genuine novelty.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from neuroslm.genetic.language import Instruction, Program
from neuroslm.genetic.optimizer import (
    sgd_program, momentum_program, rmsprop_program, adam_program, lion_program,
)


def _gradient_program() -> Program:
    """The trivial rule: update ∝ raw gradient (i.e. plain backprop-driven SGD)."""
    return Program([Instruction("cscale", "t5", ("t0",), const=-1.0)],
                   n_scalar=8, n_tensor=16, out_reg="t5", meta={"name": "gradient"})


class KnownAlgorithms:
    """A set of known algorithms keyed by name, compared in semantic space."""

    def __init__(self, programs: Dict[str, Program] = None, threshold: float = 0.75):
        self._progs: Dict[str, Program] = dict(programs or {})
        self.threshold = threshold
        self._vecs: Dict[str, np.ndarray] = {
            n: p.semantic_vector() for n, p in self._progs.items()
        }

    def add(self, name: str, program: Program) -> None:
        self._progs[name] = program
        self._vecs[name] = program.semantic_vector()

    def names(self) -> List[str]:
        return sorted(self._progs)

    def nearest(self, program: Program):
        """Return (name, distance) of the closest known algorithm in semantic space."""
        v = program.semantic_vector()
        best_name, best_d = None, float("inf")
        for n, kv in self._vecs.items():
            d = float(np.linalg.norm(v - kv))
            if d < best_d:
                best_name, best_d = n, d
        return best_name, best_d

    def distance(self, program: Program) -> float:
        _, d = self.nearest(program)
        return d

    def is_known(self, program: Program) -> bool:
        return self.distance(program) <= self.threshold


def default_known_algorithms(threshold: float = 0.75) -> KnownAlgorithms:
    return KnownAlgorithms(
        {
            "gradient": _gradient_program(),
            "sgd": sgd_program(),
            "momentum": momentum_program(),
            "rmsprop": rmsprop_program(),
            "adam": adam_program(),
            "lion": lion_program(),
        },
        threshold=threshold,
    )


def novelty_vs_known(program: Program, known: KnownAlgorithms) -> float:
    """Distance to the nearest known algorithm — larger means more novel."""
    return known.distance(program)


def known_programs() -> Dict[str, Program]:
    """Every known ML algorithm/mechanic as an NGL program (for prior-art seeding).

    The standard optimizers, the reusable macro building blocks (divisive
    normalization, rms scaling, sign-momentum, bounded gain), and the identity
    modulation — the spaces the search should NOT waste budget rediscovering.
    """
    from neuroslm.genetic.macros import default_macro_library, expand_macros
    from neuroslm.genetic.language import Instruction

    out: Dict[str, Program] = dict(default_known_algorithms()._progs)

    # macro building blocks — record their expanded computation
    lib = default_macro_library()
    for m in lib.macros():
        call = Program(
            [Instruction("call", "t2", tuple(f"t{i}" for i in range(m.n_inputs)),
                         macro=m.name)],
            n_scalar=6, n_tensor=16, out_reg="t2")
        out[f"macro:{m.name}"] = expand_macros(call, lib)

    # canonical modulation motifs the trunk explorer would otherwise rediscover
    out["identity_gain"] = Program(
        [Instruction("const", "t5", (), const=1.0)], 4, 8, "t5")
    out["tanh_gain"] = Program(
        [Instruction("tanh", "t2", ("t0",))], 4, 8, "t2")
    out["sigmoid_gain"] = Program(
        [Instruction("sigmoid", "t2", ("t0",))], 4, 8, "t2")
    return out


def seed_ledger_with_known(ledger, run_id: str = "prior-art") -> int:
    """Record every known algorithm in the ledger so the explorer skips them.

    Recorded with ``outcome="known"`` (delta 0) → ``SearchLedger.is_dud`` returns
    True, so the training explorer / discovery search treats these as already-
    explored dead space and spends its budget only on novel mechanics. Idempotent
    (the ledger dedups by semantic signature).
    """
    n_before = ledger.stats()["total"]
    for name, prog in known_programs().items():
        ledger.record(prog, outcome="known", delta=0.0, run_id=run_id,
                      kind="known", step=0)
    return ledger.stats()["total"] if n_before == 0 else ledger.stats()["total"]
