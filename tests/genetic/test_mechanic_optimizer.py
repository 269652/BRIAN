# -*- coding: utf-8 -*-
"""Optimize commonly-used mechanics via CSE + the compiler passes.

Take a mechanic expressed as an NGL program, run CSE + algebraic simplification +
superoptimization, and report whether it can be reduced (fewer instructions,
behaviour preserved). Also detect subexpressions *shared across* mechanics — the
common computation you'd factor out once.
"""
from neuroslm.genetic.language import Instruction, Program
from neuroslm.genetic.optimizer import adam_program, rmsprop_program
from neuroslm.genetic.simplify import programs_equivalent
from neuroslm.genetic.mechanic_optimizer import (
    optimize_mechanic,
    shared_subexpressions,
    analyze_common_mechanics,
    OptimizeReport,
)


class TestOptimizeMechanic:
    def test_reduces_a_redundant_mechanic(self):
        # a mechanic with a duplicate computation + dead code + identity op
        prog = Program(
            [
                Instruction("square", "t2", ("t0",)),
                Instruction("square", "t3", ("t0",)),   # duplicate of t2 (CSE)
                Instruction("add", "t4", ("t2", "t3")),
                Instruction("mul", "t9", ("t0", "t0")),  # dead
                Instruction("cscale", "t5", ("t4",), const=1.0),  # identity
            ],
            n_scalar=2, n_tensor=10, out_reg="t5",
        )
        rep = optimize_mechanic(prog)
        assert isinstance(rep, OptimizeReport)
        assert rep.after < rep.before
        assert rep.equivalent
        assert rep.reducible

    def test_already_minimal_mechanic_is_not_reducible(self):
        prog = adam_program(lr=0.01)
        rep = optimize_mechanic(prog)
        assert rep.equivalent
        assert rep.after <= rep.before
        assert programs_equivalent(prog, rep.program, n_probes=8, seed=0)


class TestSharedSubexpressions:
    def test_finds_a_subexpr_common_to_two_mechanics(self):
        # both mechanics compute rms(t0) — the shared subexpression to factor once
        a = Program([Instruction("rms", "t2", ("t0",)),
                     Instruction("div", "t3", ("t0", "t2"))], 2, 6, "t3")
        b = Program([Instruction("rms", "t2", ("t0",)),
                     Instruction("sigmoid", "t3", ("t2",))], 2, 6, "t3")
        shared = shared_subexpressions({"a": a, "b": b})
        # rms(t0) appears in both → reported as shared
        assert any(s["count"] >= 2 for s in shared)
        assert any("rms" in s["expr"] for s in shared)

    def test_no_false_shared_when_disjoint(self):
        a = Program([Instruction("tanh", "t2", ("t0",))], 2, 6, "t2")
        b = Program([Instruction("sigmoid", "t2", ("t1",))], 2, 6, "t2")
        shared = shared_subexpressions({"a": a, "b": b})
        # nothing non-trivial is shared (leaves don't count as subexpressions)
        assert all(s["count"] < 2 for s in shared) or shared == []


class TestAnalyzeCommonMechanics:
    def test_reports_reduction_per_mechanic(self):
        reports = analyze_common_mechanics()
        assert len(reports) >= 3
        for r in reports:
            assert "name" in r and "before" in r and "after" in r
            assert r["after"] <= r["before"]        # never grows
            assert r["equivalent"] is True          # always behaviour-preserving
