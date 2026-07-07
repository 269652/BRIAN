# -*- coding: utf-8 -*-
"""Contracts for NGL simplification discovery (neuroslm/genetic/simplify.py).

Two layers:
- ``dead_code_eliminate`` — exact liveness: drop instructions whose result never
  reaches ``out_reg``. Behaviour is provably preserved (no probes needed).
- ``simplify`` — a peephole + shrink superoptimizer that finds a shorter program
  behaviourally equivalent to the original on random probes (algebraic identities
  like add-0 / mul-1 / neg-neg / transpose-transpose, plus try-delete).
"""
import numpy as np
import torch

from neuroslm.genetic.language import Instruction, Memory, Program
from neuroslm.genetic.optimizer import adam_program
from neuroslm.genetic.simplify import (
    dead_code_eliminate,
    programs_equivalent,
    simplify,
)


def _run(prog, inputs):
    mem = Memory(prog.n_scalar, prog.n_tensor)
    for reg, val in inputs.items():
        mem.write(reg, val)
    prog.execute(mem)
    return mem.read(prog.out_reg)


class TestEquivalence:
    def test_equivalent_detects_same_and_different(self):
        a = Program([Instruction("mul", "t2", ("t0",), const=None) if False else
                     Instruction("cscale", "t2", ("t0",), const=2.0)], 2, 4, "t2")
        b = Program([Instruction("add", "t2", ("t0", "t0"))], 2, 4, "t2")  # x+x == 2x
        c = Program([Instruction("neg", "t2", ("t0",))], 2, 4, "t2")
        assert programs_equivalent(a, b, n_probes=8, seed=0)
        assert not programs_equivalent(a, c, n_probes=8, seed=0)


class TestDeadCodeElimination:
    def test_dce_removes_unused_instructions(self):
        # t9 is computed but never feeds t2 (the out_reg) → dead
        prog = Program(
            [
                Instruction("add", "t2", ("t0", "t1")),
                Instruction("mul", "t9", ("t0", "t0")),   # dead
                Instruction("sign", "t8", ("t9",)),       # dead (feeds only t8)
            ],
            n_scalar=2, n_tensor=10, out_reg="t2",
        )
        slim = dead_code_eliminate(prog)
        assert len(slim.instructions) == 1
        x = {"t0": torch.randn(4), "t1": torch.randn(4)}
        assert torch.allclose(_run(prog, x), _run(slim, x))

    def test_dce_keeps_chain_feeding_output(self):
        prog = Program(
            [
                Instruction("add", "t2", ("t0", "t1")),
                Instruction("mul", "t3", ("t2", "t2")),   # feeds out
                Instruction("neg", "t9", ("t0",)),        # dead
            ],
            n_scalar=2, n_tensor=10, out_reg="t3",
        )
        slim = dead_code_eliminate(prog)
        assert len(slim.instructions) == 2
        assert "t9" not in {i.out for i in slim.instructions}

    def test_dce_handles_register_overwrite(self):
        # t2 is written twice; only the last write (and its inputs) is live
        prog = Program(
            [
                Instruction("mul", "t2", ("t0", "t0")),   # overwritten before use → dead
                Instruction("add", "t2", ("t0", "t1")),   # live
            ],
            n_scalar=2, n_tensor=10, out_reg="t2",
        )
        slim = dead_code_eliminate(prog)
        assert len(slim.instructions) == 1
        x = {"t0": torch.randn(3), "t1": torch.randn(3)}
        assert torch.allclose(_run(prog, x), _run(slim, x))


class TestSimplify:
    def test_simplify_removes_algebraic_noise(self):
        # (((x + 0) * 1) then neg neg) == x ; simplify should collapse it
        prog = Program(
            [
                Instruction("const", "t1", (), const=0.0),
                Instruction("add", "t2", ("t0", "t1")),   # x + 0
                Instruction("cscale", "t3", ("t2",), const=1.0),  # * 1
                Instruction("neg", "t4", ("t3",)),
                Instruction("neg", "t5", ("t4",)),         # neg neg
            ],
            n_scalar=2, n_tensor=8, out_reg="t5",
        )
        slim = simplify(prog, n_probes=10, seed=0)
        assert len(slim.instructions) < len(prog.instructions)
        assert programs_equivalent(prog, slim, n_probes=16, seed=1)

    def test_simplify_preserves_a_real_optimizer(self):
        # Adam has no dead code; simplify must not change its behaviour
        prog = adam_program(lr=0.01)
        slim = simplify(prog, n_probes=8, seed=0)
        assert programs_equivalent(prog, slim, n_probes=12, seed=2)
        assert len(slim.instructions) <= len(prog.instructions)

    def test_simplify_reports_reduction(self):
        prog = Program(
            [
                Instruction("add", "t2", ("t0", "t1")),
                Instruction("mul", "t9", ("t0", "t0")),   # dead
            ],
            n_scalar=2, n_tensor=8, out_reg="t2",
        )
        slim, stats = simplify(prog, n_probes=8, seed=0, return_stats=True)
        assert stats["before"] == 2
        assert stats["after"] == 1
        assert stats["removed"] == 1
