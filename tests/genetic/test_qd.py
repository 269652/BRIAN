# -*- coding: utf-8 -*-
"""Quality-diversity (MAP-Elites) search over the semantic manifold.

Instead of collapsing to one winner, MAP-Elites *illuminates* the space: it bins
programs by their structural "shape" (a low-dim descriptor) and keeps the best
performer per cell. The result is a diverse zoo of high-performing algorithms
spread across the geometry of the language — the honest, computable version of
"let novel algorithms emerge from mathematical shapes in the semantic manifold".
"""
import numpy as np
import torch

from neuroslm.genetic.language import Instruction, Memory, Program
from neuroslm.genetic.qd_search import descriptor, map_elites, Archive


def _score_to_target(prog, target, x0):
    mem = Memory(prog.n_scalar, prog.n_tensor)
    mem.write("t0", x0)
    prog.execute(mem)
    out = mem.read(prog.out_reg)
    try:
        out = out.reshape(target.shape)
    except RuntimeError:
        return -10.0
    return float(-torch.mean((out - target) ** 2))   # higher = better


class TestDescriptor:
    def test_descriptor_is_deterministic_and_low_dim(self):
        p = Program([Instruction("tanh", "t2", ("t0",)),
                     Instruction("neg", "t3", ("t2",))], 2, 6, "t3")
        d1 = descriptor(p)
        d2 = descriptor(p)
        assert d1 == d2
        assert isinstance(d1, tuple) and 2 <= len(d1) <= 4

    def test_different_shapes_get_different_cells(self):
        short = Program([Instruction("tanh", "t2", ("t0",))], 2, 6, "t2")
        long = Program([Instruction("tanh", "t2", ("t0",)),
                        Instruction("sigmoid", "t3", ("t2",)),
                        Instruction("neg", "t4", ("t3",)),
                        Instruction("abs", "t5", ("t4",))], 2, 8, "t5")
        assert descriptor(short) != descriptor(long)


class TestArchive:
    def test_keeps_best_per_cell(self):
        arch = Archive()
        p = Program([Instruction("tanh", "t2", ("t0",))], 2, 6, "t2")
        arch.add(p, 1.0, (0, 1))
        arch.add(p, 2.0, (0, 1))   # better → replaces
        arch.add(p, 0.5, (0, 1))   # worse → ignored
        assert arch.cells[(0, 1)][1] == 2.0
        assert arch.coverage() == 1


class TestMapElites:
    def test_illuminates_multiple_cells(self):
        target = torch.tanh(torch.tensor([1.0, -1.0, 0.5, 0.2]))
        x0 = torch.tensor([1.0, -1.0, 0.5, 0.2])
        rng = np.random.default_rng(0)
        arch = map_elites(
            lambda p: _score_to_target(p, target, x0),
            rng, n_iters=150, init_size=40, length=5, n_scalar=4, n_tensor=6,
        )
        # the search filled several distinct shape-cells → it illuminated the space
        assert arch.coverage() >= 3
        # elites are genuinely diverse (distinct descriptors)
        descs = {d for d in arch.cells}
        assert len(descs) == arch.coverage()

    def test_quality_improves_over_random_init(self):
        target = torch.tanh(torch.tensor([2.0, -2.0, 1.0, 0.3])) * 0.5
        x0 = torch.tensor([2.0, -2.0, 1.0, 0.3])
        rng = np.random.default_rng(1)
        score = lambda p: _score_to_target(p, target, x0)
        # baseline: best of a random batch
        base = max(score(_rand(rng)) for _ in range(40))
        arch = map_elites(score, rng, n_iters=250, init_size=40,
                          length=5, n_scalar=4, n_tensor=6)
        assert arch.best()[1] >= base   # QD finds at least as good, plus diversity


def _rand(rng):
    from neuroslm.genetic.evolve import random_program
    return random_program(rng, 5, 4, 6)
