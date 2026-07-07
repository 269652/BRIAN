# -*- coding: utf-8 -*-
"""Contracts for the NGL register-machine core (neuroslm/genetic/language.py).

These pin the *semantics* of the language, not just its shape: every op has a
closed-form meaning and the machine executes a program deterministically. A key
invariant for evolvability is that execution is **total** — a random/ill-typed
program must never raise; it produces a value (usually a safe fallback).
"""
import math

import numpy as np
import torch

from neuroslm.genetic.language import (
    Instruction,
    Memory,
    Program,
    REGISTRY,
)


class TestOpSemantics:
    def test_add_mul_sub_neg(self):
        g = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(REGISTRY["add"].fn(g, g), g * 2)
        assert torch.allclose(REGISTRY["mul"].fn(g, g), g * g)
        assert torch.allclose(REGISTRY["sub"].fn(g, g), torch.zeros_like(g))
        assert torch.allclose(REGISTRY["neg"].fn(g), -g)

    def test_div_is_total_on_zero(self):
        g = torch.tensor([1.0, 2.0])
        z = torch.zeros(2)
        out = REGISTRY["div"].fn(g, z)
        assert torch.isfinite(out).all()  # eps-guarded, never inf/nan

    def test_sqrt_and_log_total_on_negatives(self):
        x = torch.tensor([-4.0, 4.0])
        assert torch.isfinite(REGISTRY["sqrt"].fn(x)).all()
        assert torch.isfinite(REGISTRY["log"].fn(x)).all()

    def test_sign_and_abs(self):
        x = torch.tensor([-2.0, 0.0, 3.0])
        assert torch.allclose(REGISTRY["sign"].fn(x), torch.tensor([-1.0, 0.0, 1.0]))
        assert torch.allclose(REGISTRY["abs"].fn(x), torch.tensor([2.0, 0.0, 3.0]))

    def test_select_is_conditional(self):
        cond = torch.tensor([1.0, -1.0])  # >0 picks a, else b
        a = torch.tensor([10.0, 10.0])
        b = torch.tensor([20.0, 20.0])
        out = REGISTRY["select"].fn(cond, a, b)
        assert torch.allclose(out, torch.tensor([10.0, 20.0]))

    def test_norm_and_rms_are_scalars(self):
        x = torch.tensor([3.0, 4.0])
        assert math.isclose(float(REGISTRY["norm"].fn(x)), 5.0, rel_tol=1e-5)
        rms = float(REGISTRY["rms"].fn(x))
        assert math.isclose(rms, math.sqrt((9 + 16) / 2), rel_tol=1e-5)

    def test_matmul_falls_back_when_shapes_mismatch(self):
        a = torch.randn(3, 4)
        b = torch.randn(7, 2)  # incompatible
        out = REGISTRY["matmul"].fn(a, b)  # must not raise
        assert torch.isfinite(out).all()

    def test_matmul_correct_when_compatible(self):
        a = torch.randn(3, 4)
        b = torch.randn(4, 2)
        assert torch.allclose(REGISTRY["matmul"].fn(a, b), a @ b, atol=1e-5)


class TestRegisterMachine:
    def test_unwritten_read_returns_zero(self):
        mem = Memory(n_scalar=2, n_tensor=2)
        assert float(mem.read("t0").sum()) == 0.0
        assert float(mem.read("s1")) == 0.0

    def test_write_then_read(self):
        mem = Memory(n_scalar=2, n_tensor=2)
        v = torch.tensor([5.0, 6.0])
        mem.write("t1", v)
        assert torch.allclose(mem.read("t1"), v)

    def test_program_executes_in_order(self):
        # t2 = t0 + t1 ; t3 = t2 * t2
        prog = Program(
            instructions=[
                Instruction("add", "t2", ("t0", "t1")),
                Instruction("mul", "t3", ("t2", "t2")),
            ],
            n_scalar=1,
            n_tensor=4,
            out_reg="t3",
        )
        mem = Memory(n_scalar=1, n_tensor=4)
        mem.write("t0", torch.tensor([1.0, 2.0]))
        mem.write("t1", torch.tensor([3.0, 4.0]))
        prog.execute(mem)
        assert torch.allclose(mem.read("t3"), torch.tensor([16.0, 36.0]))

    def test_const_instruction_loads_constant(self):
        prog = Program(
            instructions=[Instruction("const", "s0", (), const=2.5)],
            n_scalar=1,
            n_tensor=1,
            out_reg="s0",
        )
        mem = Memory(n_scalar=1, n_tensor=1)
        prog.execute(mem)
        assert math.isclose(float(mem.read("s0")), 2.5)

    def test_execution_is_total_on_garbage_program(self):
        # random ops/registers must never raise
        rng = np.random.default_rng(0)
        names = list(REGISTRY.keys())
        for _ in range(50):
            instrs = []
            for _ in range(8):
                op = names[rng.integers(len(names))]
                ins = tuple(
                    f"t{rng.integers(4)}" if rng.random() < 0.7 else f"s{rng.integers(2)}"
                    for _ in range(REGISTRY[op].n_in)
                )
                instrs.append(Instruction(op, f"t{rng.integers(4)}", ins, const=float(rng.normal())))
            prog = Program(instrs, n_scalar=2, n_tensor=4, out_reg="t0")
            mem = Memory(n_scalar=2, n_tensor=4)
            mem.write("t0", torch.randn(3, 3))
            mem.write("t1", torch.randn(3, 3))
            prog.execute(mem)  # must not raise
            out = mem.read(prog.out_reg)
            assert torch.isfinite(out).all()


class TestSemanticSpace:
    def test_semantic_vector_is_deterministic_and_fixed_length(self):
        prog = Program(
            [Instruction("add", "t2", ("t0", "t1")), Instruction("sign", "t2", ("t2",))],
            n_scalar=1,
            n_tensor=3,
            out_reg="t2",
        )
        v1 = prog.semantic_vector()
        v2 = prog.semantic_vector()
        assert v1.shape == v2.shape
        assert np.allclose(v1, v2)

    def test_semantic_vector_distinguishes_programs(self):
        p_sgd = Program([Instruction("mul", "t2", ("t0",), const=-0.1)], 1, 3, "t2")
        p_sign = Program([Instruction("sign", "t2", ("t0",))], 1, 3, "t2")
        assert not np.allclose(p_sgd.semantic_vector(), p_sign.semantic_vector())
