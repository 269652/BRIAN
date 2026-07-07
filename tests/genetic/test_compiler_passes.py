# -*- coding: utf-8 -*-
"""Standard compiler passes over the NGL expression DAG.

CSE (common-subexpression elimination), constant folding, and a unified
``optimize`` pipeline — the textbook passes, made explicit and individually
tested, on top of the DCE + algebraic rewriter already present.
"""
import torch

from neuroslm.genetic.language import Instruction, Memory, Program
from neuroslm.genetic.rewrite import cse, constant_fold, optimize
from neuroslm.genetic.simplify import programs_equivalent


def _run(prog, inputs):
    mem = Memory(prog.n_scalar, prog.n_tensor)
    for reg, val in inputs.items():
        mem.write(reg, val)
    prog.execute(mem)
    return mem.read(prog.out_reg)


class TestCSE:
    def test_duplicate_computation_deduplicated(self):
        # a = t0+t1 ; b = t0+t1 ; out = a*b  → the add is computed once
        prog = Program(
            [
                Instruction("add", "t2", ("t0", "t1")),
                Instruction("add", "t3", ("t0", "t1")),
                Instruction("mul", "t4", ("t2", "t3")),
            ],
            n_scalar=2, n_tensor=6, out_reg="t4",
        )
        slim = cse(prog)
        adds = [i for i in slim.instructions if i.op == "add"]
        assert len(adds) == 1
        x = {"t0": torch.randn(4), "t1": torch.randn(4)}
        assert torch.allclose(_run(prog, x), _run(slim, x), atol=1e-6)

    def test_cse_respects_register_overwrite(self):
        # the second add reads a different t1, so it is NOT a common subexpression
        prog = Program(
            [
                Instruction("add", "t2", ("t0", "t1")),
                Instruction("neg", "t1", ("t1",)),        # t1 changes
                Instruction("add", "t3", ("t0", "t1")),   # different value now
                Instruction("mul", "t4", ("t2", "t3")),
            ],
            n_scalar=2, n_tensor=6, out_reg="t4",
        )
        slim = cse(prog)
        x = {"t0": torch.randn(3), "t1": torch.randn(3)}
        assert torch.allclose(_run(prog, x), _run(slim, x), atol=1e-6)
        assert len([i for i in slim.instructions if i.op == "add"]) == 2


class TestConstantFolding:
    def test_folds_pure_constant_subexpression(self):
        # add(const 2, const 3) → const 5 ; out = mul(t0, 5)
        prog = Program(
            [
                Instruction("const", "t1", (), const=2.0),
                Instruction("const", "t2", (), const=3.0),
                Instruction("add", "t3", ("t1", "t2")),
                Instruction("mul", "t4", ("t0", "t3")),
            ],
            n_scalar=2, n_tensor=6, out_reg="t4",
        )
        slim = constant_fold(prog)
        # no add remains; a const 5 (or folded) feeds the mul
        assert not any(i.op == "add" for i in slim.instructions)
        x = {"t0": torch.randn(4)}
        assert torch.allclose(_run(slim, x), x["t0"] * 5.0, atol=1e-6)

    def test_folding_preserves_behaviour(self):
        prog = Program(
            [
                Instruction("const", "t1", (), const=4.0),
                Instruction("sqrt", "t2", ("t1",)),       # sqrt(4)=2
                Instruction("cscale", "t3", ("t0",), const=1.0),
                Instruction("mul", "t4", ("t3", "t2")),
            ],
            n_scalar=2, n_tensor=6, out_reg="t4",
        )
        slim = constant_fold(prog)
        x = {"t0": torch.randn(3)}
        assert torch.allclose(_run(prog, x), _run(slim, x), atol=1e-6)


class TestOptimizePipeline:
    def test_optimize_combines_passes(self):
        # dead code + duplicate add + const fold + algebraic, all at once
        prog = Program(
            [
                Instruction("const", "t1", (), const=0.0),
                Instruction("add", "t2", ("t0", "t1")),   # x + 0 → x
                Instruction("add", "t3", ("t0", "t1")),   # dup + identity
                Instruction("mul", "t9", ("t0", "t0")),   # dead
                Instruction("mul", "t4", ("t2", "t3")),   # x*x
            ],
            n_scalar=2, n_tensor=10, out_reg="t4",
        )
        slim = optimize(prog)
        assert len(slim.instructions) < len(prog.instructions)
        assert programs_equivalent(prog, slim, n_probes=12, seed=0)

    def test_optimize_is_idempotent(self):
        prog = Program(
            [
                Instruction("add", "t2", ("t0", "t1")),
                Instruction("add", "t3", ("t0", "t1")),
                Instruction("mul", "t4", ("t2", "t3")),
            ],
            n_scalar=2, n_tensor=6, out_reg="t4",
        )
        once = optimize(prog)
        twice = optimize(once)
        assert len(twice.instructions) == len(once.instructions)
