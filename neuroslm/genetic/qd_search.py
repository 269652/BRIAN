# -*- coding: utf-8 -*-
"""Quality-diversity (MAP-Elites) search over the NGL semantic manifold.

Ordinary GA collapses a population to one winner. MAP-Elites (Mouret & Clune 2015)
instead *illuminates* the search space: it projects each program to a low-dim
**behavioural descriptor** (its structural "shape"), keeps the single best
performer per descriptor cell, and iterates — mutating elites and re-placing them.
The archive that results is a diverse zoo of high-performing algorithms spread
across the geometry of the language: cheap-and-simple in one region, deep-and-
adaptive in another. This is the concrete, well-founded realization of "let novel
algorithms emerge from the mathematical shapes of the semantic manifold" — the
descriptor axes *are* the manifold coordinates, and the archive maps which shapes
yield which capabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

from neuroslm.genetic.language import Program, REGISTRY
from neuroslm.genetic.evolve import random_program, mutate


def descriptor(program: Program) -> Tuple[int, int]:
    """Low-dim structural coordinates = a point on the shape manifold.

    Axis 0: program length (compute depth), bucketed.
    Axis 1: number of distinct op-families used (structural diversity).
    """
    n = len(program.instructions)
    length_bin = min(n // 2, 6)
    fams = set()
    for ins in program.instructions:
        fams.add(REGISTRY[ins.op].family if ins.op in REGISTRY else "macro")
    return (length_bin, min(len(fams), 5))


@dataclass
class Archive:
    cells: Dict[Tuple, Tuple[Program, float]] = field(default_factory=dict)

    def add(self, program: Program, fitness: float, cell: Tuple) -> bool:
        cur = self.cells.get(cell)
        if cur is None or fitness > cur[1]:
            self.cells[cell] = (program.copy(), float(fitness))
            return True
        return False

    def coverage(self) -> int:
        return len(self.cells)

    def elites(self) -> List[Tuple[Tuple, Program, float]]:
        return [(c, p, f) for c, (p, f) in self.cells.items()]

    def best(self) -> Tuple[Program, float]:
        return max(self.cells.values(), key=lambda pf: pf[1])

    def random_elite(self, rng) -> Program:
        if not self.cells:
            return None
        keys = list(self.cells)
        return self.cells[keys[rng.integers(len(keys))]][0]

    def to_dict(self) -> dict:
        return {
            "coverage": self.coverage(),
            "cells": [
                {"cell": list(c), "fitness": f, "program": p.to_source()}
                for c, (p, f) in sorted(self.cells.items())
            ],
        }


def map_elites(evaluate: Callable[[Program], float], rng,
               n_iters: int = 300, init_size: int = 48,
               length: int = 6, n_scalar: int = 4, n_tensor: int = 8,
               seeds: List[Program] = None,
               descriptor_fn: Callable[[Program], Tuple] = descriptor,
               macro_library=None,
               on_iter: Callable[[int, int, "Archive"], None] = None) -> Archive:
    """Illuminate the manifold: keep the best program per shape-cell."""
    arch = Archive()

    pop: List[Program] = list(seeds or [])
    while len(pop) < init_size:
        pop.append(random_program(rng, length, n_scalar, n_tensor))
    for p in pop:
        if macro_library is not None:
            p.library = macro_library
        arch.add(p, evaluate(p), descriptor_fn(p))

    for it in range(n_iters):
        parent = arch.random_elite(rng)
        if parent is None:
            break
        child = mutate(parent, rng, library=macro_library)
        arch.add(child, evaluate(child), descriptor_fn(child))
        if on_iter is not None and (it + 1) % max(1, n_iters // 10) == 0:
            on_iter(it + 1, n_iters, arch)

    return arch
