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
