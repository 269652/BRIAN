# -*- coding: utf-8 -*-
"""Contracts for the NGL algebraic rewrite engine (neuroslm/genetic/rewrite.py).

The engine turns a program into an expression DAG, applies value-preserving
algebraic identities to a fixpoint (each accepted rewrite globally verified on
random probes), and lowers back with common-subexpression elimination. This is
how the simplifier discovers reductions a peephole pass can't, e.g. (x+x)−x = x.
"""
import numpy as np
import torch

from neuroslm.genetic.language import Instruction, Memory, Program
from neuroslm.genetic.optimizer import adam_program, lion_program
from neuroslm.genetic.evolve import random_program
from neuroslm.genetic.simplify import programs_equivalent
from neuroslm.genetic.rewrite import (
    to_expr,
    from_expr,
    algebraic_simplify,
    Leaf,
    Const,
    Node,
)


def _run(prog, inputs):
    mem = Memory(prog.n_scalar, prog.n_tensor)
    for reg, val in inputs.items():
        mem.write(reg, val)
    prog.execute(mem)
    return mem.read(prog.out_reg)


class TestExprRoundTrip:
    def test_roundtrip_preserves_behaviour_adam(self):
        prog = adam_program(lr=0.01)
        rebuilt = from_expr(to_expr(prog), prog)
        assert programs_equivalent(prog, rebuilt, n_probes=8, seed=0)

    def test_roundtrip_preserves_behaviour_random(self):
        rng = np.random.default_rng(3)
        for _ in range(15):
            prog = random_program(rng, length=6, n_scalar=4, n_tensor=6)
            rebuilt = from_expr(to_expr(prog), prog)
            assert programs_equivalent(prog, rebuilt, n_probes=6, seed=1)

    def test_to_expr_builds_tree(self):
        prog = Program([Instruction("add", "t2", ("t0", "t1"))], 2, 4, "t2")
        e = to_expr(prog)
        assert isinstance(e, Node) and e.op == "add"
        assert e.args == (Leaf("t0"), Leaf("t1"))


class TestAlgebraicIdentities:
    def test_x_plus_x_minus_x_reduces_to_x(self):
        # (t0 + t0) - t0  ==  t0
        prog = Program(
            [
                Instruction("add", "t2", ("t0", "t0")),
                Instruction("sub", "t3", ("t2", "t0")),
            ],
            n_scalar=2, n_tensor=6, out_reg="t3",
        )
        slim = algebraic_simplify(prog, n_probes=10, seed=0)
        assert len(slim.instructions) < len(prog.instructions)
        # equivalent to the identity on t0
        x = {"t0": torch.randn(4), "t1": torch.randn(4)}
        assert torch.allclose(_run(prog, x), _run(slim, x), atol=1e-6)
        assert torch.allclose(_run(slim, x), x["t0"], atol=1e-6)

    def test_add_zero_removed(self):
        prog = Program(
            [
                Instruction("const", "t1", (), const=0.0),
                Instruction("add", "t2", ("t0", "t1")),
            ],
            n_scalar=2, n_tensor=6, out_reg="t2",
        )
        slim = algebraic_simplify(prog, n_probes=8, seed=0)
        assert len(slim.instructions) < len(prog.instructions)
        x = {"t0": torch.randn(3)}
        assert torch.allclose(_run(slim, x), x["t0"], atol=1e-6)

    def test_neg_neg_removed(self):
        prog = Program(
            [
                Instruction("neg", "t2", ("t0",)),
                Instruction("neg", "t3", ("t2",)),
            ],
            n_scalar=2, n_tensor=6, out_reg="t3",
        )
        slim = algebraic_simplify(prog, n_probes=8, seed=0)
        assert len(slim.instructions) < len(prog.instructions)
        x = {"t0": torch.randn(3)}
        assert torch.allclose(_run(slim, x), x["t0"], atol=1e-6)

    def test_cscale_constant_folding(self):
        # cscale(cscale(t0, 2), 3) == cscale(t0, 6)
        prog = Program(
            [
                Instruction("cscale", "t2", ("t0",), const=2.0),
                Instruction("cscale", "t3", ("t2",), const=3.0),
            ],
            n_scalar=2, n_tensor=6, out_reg="t3",
        )
        slim = algebraic_simplify(prog, n_probes=8, seed=0)
        assert len(slim.instructions) == 1
        folded = slim.instructions[0]
        assert folded.op == "cscale"
        assert abs(folded.const - 6.0) < 1e-9
        x = {"t0": torch.randn(3)}
        assert torch.allclose(_run(slim, x), x["t0"] * 6.0, atol=1e-6)

    def test_like_terms_combined(self):
        # cscale(t0, 2) + t0 == cscale(t0, 3)
        prog = Program(
            [
                Instruction("cscale", "t2", ("t0",), const=2.0),
                Instruction("add", "t3", ("t2", "t0")),
            ],
            n_scalar=2, n_tensor=6, out_reg="t3",
        )
        slim = algebraic_simplify(prog, n_probes=8, seed=0)
        x = {"t0": torch.randn(4)}
        assert torch.allclose(_run(slim, x), x["t0"] * 3.0, atol=1e-6)
        assert len(slim.instructions) <= len(prog.instructions)


class TestSoundness:
    def test_never_changes_behaviour_random(self):
        rng = np.random.default_rng(7)
        for _ in range(25):
            prog = random_program(rng, length=8, n_scalar=4, n_tensor=6)
            slim = algebraic_simplify(prog, n_probes=8, seed=2)
            assert programs_equivalent(prog, slim, n_probes=12, seed=3)
            assert len(slim.instructions) <= len(prog.instructions)

    def test_preserves_real_optimizer(self):
        for prog in (adam_program(lr=0.01), lion_program()):
            slim = algebraic_simplify(prog, n_probes=8, seed=0)
            assert programs_equivalent(prog, slim, n_probes=12, seed=1)

    def test_idempotent(self):
        rng = np.random.default_rng(11)
        prog = random_program(rng, length=8, n_scalar=4, n_tensor=6)
        once = algebraic_simplify(prog, n_probes=8, seed=0)
        twice = algebraic_simplify(once, n_probes=8, seed=0)
        assert len(twice.instructions) == len(once.instructions)


class TestSimplifyIntegration:
    def test_simplify_uses_algebra_on_compiled_arch(self):
        # the (h+h)-h redundancy inside a compiled FFN must now be removed
        from neuroslm.dsl.nn_lang import compile_layer
        from neuroslm.genetic.compile_arch import compile_layer_to_ngl
        from neuroslm.genetic.simplify import simplify

        dsl = """
        layer BloatedFFN(D, H) {
            param gamma: (D,) init=ones
            param w1: (H, D) init=xavier
            param w2: (H, D) init=xavier
            param w3: (D, H) init=xavier
            forward(x) {
                h = rmsnorm(x, gamma)
                z = h + h
                h2 = z - h
                m = swiglu(h2, w1, w2, w3)
                return x + m
            }
        }
        """
        compiled = compile_layer_to_ngl(dsl)
        n0 = len(compiled.program.instructions)
        slim, stats = simplify(compiled.program, n_probes=12, seed=0, return_stats=True)
        assert stats["after"] < n0
        assert programs_equivalent(compiled.program, slim, n_probes=16, seed=1)
        # the add(h,h) and sub(_,h) pair should be gone
        ops = [i.op for i in slim.instructions]
        assert ops.count("add") <= 1  # only the final residual add remains
