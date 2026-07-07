# -*- coding: utf-8 -*-
"""Macro / ADF abstraction — reusable sub-programs the GA can compose.

A macro is a named NGL sub-program with formal input registers and one output.
A parent program calls it as a single `call` instruction; execution inlines the
body (fresh temps, inputs remapped). This lets the search build complex
algorithms from chunks instead of re-deriving primitives every generation.
"""
import numpy as np
import torch

from neuroslm.genetic.language import Instruction, Memory, Program
from neuroslm.genetic.macros import (
    Macro,
    MacroLibrary,
    expand_macros,
    default_macro_library,
)


def _run(prog, inputs, library=None):
    prog = expand_macros(prog, library) if library else prog
    mem = Memory(prog.n_scalar, prog.n_tensor)
    for reg, val in inputs.items():
        mem.write(reg, val)
    prog.execute(mem)
    return mem.read(prog.out_reg)


def _square_gain_macro():
    # macro(a) = sigmoid(a * a)   with formal input i0, output o0
    body = Program(
        [
            Instruction("mul", "i0", ("i0", "i0")),
            Instruction("sigmoid", "o0", ("i0",)),
        ],
        n_scalar=2, n_tensor=6, out_reg="o0",
    )
    return Macro(name="sqgain", body=body, n_inputs=1)


class TestExpansion:
    def test_call_inlines_and_matches_manual(self):
        lib = MacroLibrary([_square_gain_macro()])
        prog = Program(
            [Instruction("call", "t2", ("t0",), macro="sqgain")],
            n_scalar=2, n_tensor=6, out_reg="t2",
        )
        flat = expand_macros(prog, lib)
        assert not any(i.op == "call" for i in flat.instructions)
        h = torch.randn(3, 4)
        got = _run(prog, {"t0": h}, library=lib)
        expected = torch.sigmoid(h * h)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_call_uses_actual_input_register(self):
        lib = MacroLibrary([_square_gain_macro()])
        # feed t1 into the macro, not t0
        prog = Program(
            [Instruction("call", "t2", ("t1",), macro="sqgain")],
            n_scalar=2, n_tensor=6, out_reg="t2",
        )
        h = torch.randn(5)
        got = _run(prog, {"t0": torch.zeros(5), "t1": h}, library=lib)
        assert torch.allclose(got, torch.sigmoid(h * h), atol=1e-6)

    def test_two_calls_do_not_collide(self):
        lib = MacroLibrary([_square_gain_macro()])
        prog = Program(
            [
                Instruction("call", "t2", ("t0",), macro="sqgain"),
                Instruction("call", "t3", ("t1",), macro="sqgain"),
                Instruction("add", "t4", ("t2", "t3")),
            ],
            n_scalar=2, n_tensor=8, out_reg="t4",
        )
        a, b = torch.randn(4), torch.randn(4)
        got = _run(prog, {"t0": a, "t1": b}, library=lib)
        expected = torch.sigmoid(a * a) + torch.sigmoid(b * b)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_nested_macro_expands(self):
        inner = _square_gain_macro()
        # outer(a) = sqgain(a) + a  (calls inner)
        outer_body = Program(
            [
                Instruction("call", "t5", ("i0",), macro="sqgain"),
                Instruction("add", "o0", ("t5", "i0")),
            ],
            n_scalar=2, n_tensor=8, out_reg="o0",
        )
        outer = Macro("outer", outer_body, n_inputs=1)
        lib = MacroLibrary([inner, outer])
        prog = Program([Instruction("call", "t2", ("t0",), macro="outer")],
                       n_scalar=2, n_tensor=8, out_reg="t2")
        h = torch.randn(3)
        got = _run(prog, {"t0": h}, library=lib)
        assert torch.allclose(got, torch.sigmoid(h * h) + h, atol=1e-6)

    def test_cycle_is_guarded(self):
        # a macro that calls itself must not infinite-loop; expansion is bounded
        rec_body = Program([Instruction("call", "o0", ("i0",), macro="rec")],
                           n_scalar=2, n_tensor=4, out_reg="o0")
        lib = MacroLibrary([Macro("rec", rec_body, n_inputs=1)])
        prog = Program([Instruction("call", "t2", ("t0",), macro="rec")],
                       n_scalar=2, n_tensor=4, out_reg="t2")
        import pytest
        with pytest.raises(ValueError):
            expand_macros(prog, lib)


class TestTransparentExecution:
    def test_execute_auto_flattens_with_attached_library(self):
        lib = MacroLibrary([_square_gain_macro()])
        prog = Program(
            [Instruction("call", "t2", ("t0",), macro="sqgain")],
            n_scalar=2, n_tensor=6, out_reg="t2", library=lib,
        )
        mem = Memory(prog.n_scalar, prog.n_tensor)
        h = torch.randn(4)
        mem.write("t0", h)
        prog.execute(mem)   # no explicit expand — library on the Program
        assert torch.allclose(mem.read("t2"), torch.sigmoid(h * h), atol=1e-6)


class TestDefaultLibrary:
    def test_default_library_has_reusable_macros(self):
        lib = default_macro_library()
        assert len(lib) >= 2
        # every macro body executes without error on a probe
        for m in lib.macros():
            flat = expand_macros(
                Program([Instruction("call", "t2", tuple(f"t{i}" for i in range(m.n_inputs)),
                                     macro=m.name)], 4, 12, "t2"),
                lib,
            )
            mem = Memory(4, 12)
            for i in range(m.n_inputs):
                mem.write(f"t{i}", torch.randn(3, 3))
            flat.execute(mem)
            assert torch.isfinite(mem.read(flat.out_reg)).all()


class TestGAIntegration:
    def test_mutate_can_graft_a_macro_call(self):
        import numpy as np
        from neuroslm.genetic.evolve import mutate
        from neuroslm.genetic.optimizer import sgd_program
        rng = np.random.default_rng(0)
        lib = default_macro_library()
        base = sgd_program(lr=0.1)
        grafted = 0
        for _ in range(60):
            child = mutate(base, rng, kind="insert_call", library=lib)
            if any(i.op == "call" for i in child.instructions):
                grafted += 1
                # the child carries the library and executes (flattens) cleanly
                mem = Memory(child.n_scalar, child.n_tensor)
                mem.write("t0", torch.randn(4))
                mem.write("t1", torch.randn(4))
                child.execute(mem)
                assert torch.isfinite(mem.read(child.out_reg)).all()
        assert grafted >= 55  # insert_call reliably grafts a macro
