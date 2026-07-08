# -*- coding: utf-8 -*-
"""Contracts for NGL genetic operators + Pareto GA (neuroslm/genetic/evolve.py).

The operators must keep programs *executable* (totality already guarantees no
crash, but structural validity — valid ops/registers/out_reg — must hold), must
actually change behaviour, and the GA must move a population toward better
fitness on a toy multi-objective problem.
"""
import numpy as np
import torch

from neuroslm.genetic.language import Program, Memory, Instruction
from neuroslm.genetic.optimizer import sgd_program, lion_program
from neuroslm.genetic.evolve import (
    random_program,
    mutate,
    crossover,
    Objective,
    pareto_front,
    auto_evolve,
)


def _executes(prog: Program) -> bool:
    mem = Memory(prog.n_scalar, prog.n_tensor)
    mem.write("t0", torch.randn(4, 4))
    mem.write("t1", torch.randn(4, 4))
    prog.execute(mem)
    out = mem.read(prog.out_reg)
    return torch.isfinite(out).all().item()


class TestRandomProgram:
    def test_random_program_is_valid_and_executes(self):
        rng = np.random.default_rng(1)
        for _ in range(30):
            prog = random_program(rng, length=6, n_scalar=4, n_tensor=6)
            assert isinstance(prog, Program)
            assert _executes(prog)
            # out_reg must be a real register slot
            assert prog.out_reg[0] in ("s", "t")


class TestMutation:
    def test_mutation_preserves_executability(self):
        rng = np.random.default_rng(2)
        base = lion_program()
        for _ in range(50):
            child = mutate(base, rng)
            assert _executes(child)

    def test_mutation_does_not_alter_parent(self):
        rng = np.random.default_rng(3)
        base = sgd_program(lr=0.1)
        n_before = len(base.instructions)
        _ = mutate(base, rng)
        assert len(base.instructions) == n_before

    def test_mutation_changes_something(self):
        rng = np.random.default_rng(4)
        base = lion_program()
        changed = 0
        for _ in range(20):
            child = mutate(base, rng)
            if child.to_source() != base.to_source():
                changed += 1
        assert changed >= 15  # the vast majority of mutations are effective


class TestMutationOfCallInstructions:
    """`call` instructions (macro invocations) have no REGISTRY entry by
    design — `point_reg`/`point_const` mutations must not blindly do
    `REGISTRY[old.op]` on them (KeyError: 'call' seen on a live Colab
    `discover optimizer --macros` run)."""

    def _program_with_call(self):
        from neuroslm.genetic.language import Instruction
        return Program(
            [Instruction("call", "t2", ("t0", "t1"), macro="sqgain")],
            n_scalar=2, n_tensor=4, out_reg="t2",
        )

    def test_point_reg_on_call_instruction_does_not_crash(self):
        rng = np.random.default_rng(6)
        base = self._program_with_call()
        for _ in range(50):
            child = mutate(base, rng, kind="point_reg")
            assert _executes_with_library(child)

    def test_point_const_on_call_instruction_does_not_crash(self):
        rng = np.random.default_rng(7)
        base = self._program_with_call()
        for _ in range(50):
            child = mutate(base, rng, kind="point_const")
            assert _executes_with_library(child)

    def test_auto_evolve_with_macros_and_point_mutations_does_not_crash(self):
        # reproduces the exact failure mode: a macro_library present so
        # `call` instructions can be grafted, then enough generations that
        # point_reg/point_const eventually lands on one.
        from neuroslm.genetic.macros import Macro, MacroLibrary
        from neuroslm.genetic.language import Instruction

        body = Program(
            [Instruction("mul", "i0", ("i0", "i0")), Instruction("sigmoid", "o0", ("i0",))],
            n_scalar=2, n_tensor=6, out_reg="o0",
        )
        lib = MacroLibrary([Macro(name="sqgain", body=body, n_inputs=1)])
        rng = np.random.default_rng(8)
        base = sgd_program(lr=0.1)

        def _obj(prog):
            return Objective((-float(len(prog.instructions)),))

        # library present -> insert_call is reachable; many generations ->
        # point_reg/point_const will eventually hit a call instruction.
        result = auto_evolve(_obj, rng, seeds=[base], pop_size=12,
                             generations=15, macro_library=lib)
        assert result.best_program is not None


def _executes_with_library(prog: Program) -> bool:
    mem = Memory(prog.n_scalar, prog.n_tensor)
    mem.write("t0", torch.randn(4))
    mem.write("t1", torch.randn(4))
    from neuroslm.genetic.macros import Macro, MacroLibrary
    body = Program(
        [Instruction("mul", "i0", ("i0", "i0")), Instruction("sigmoid", "o0", ("i0",))],
        n_scalar=2, n_tensor=6, out_reg="o0",
    )
    lib = MacroLibrary([Macro(name="sqgain", body=body, n_inputs=1)])
    prog = prog.copy()
    prog.library = lib
    prog.execute(mem)
    out = mem.read(prog.out_reg)
    return torch.isfinite(out).all().item()


class TestCrossover:
    def test_crossover_produces_valid_executable_child(self):
        rng = np.random.default_rng(5)
        a, b = lion_program(), sgd_program(lr=0.05)
        for _ in range(30):
            child = crossover(a, b, rng)
            assert _executes(child)


class TestPareto:
    def test_pareto_front_selects_nondominated(self):
        # Objectives are all-MAXIMISED. A=(4,1), B=(1,4) are nondominated;
        # C=(1,1) is dominated by both.
        pop = ["A", "B", "C"]
        objs = [
            Objective((4.0, 1.0)),
            Objective((1.0, 4.0)),
            Objective((1.0, 1.0)),
        ]
        front = pareto_front(list(zip(pop, objs)))
        assert set(front) == {"A", "B"}


class TestAutoEvolve:
    def test_auto_evolve_improves_fitness(self):
        # Objective: find a program whose output on a fixed input approaches a
        # target that needs *composition* (tanh then scale), so gen0 rarely nails
        # it and the GA has to work. Fitness = -MSE (maximise). Pure language
        # search, no torch training — fast and deterministic.
        x0 = torch.tensor([2.0, -2.0, 1.0, 0.3])
        target = torch.tanh(x0) * 0.5  # optimum ≈ cscale(tanh(t0), 0.5)

        def evaluate(prog: Program):
            mem = Memory(prog.n_scalar, prog.n_tensor)
            mem.write("t0", x0)
            mem.write("t1", torch.tensor([0.5, 0.5, 0.5, 0.5]))
            prog.execute(mem)
            out = mem.read(prog.out_reg)
            try:
                out = out.reshape(4)
            except RuntimeError:
                out = out.flatten()[:4]
                if out.numel() < 4:
                    out = torch.zeros(4)
            mse = torch.mean((out - target) ** 2).item()
            return Objective((-mse,))

        rng = np.random.default_rng(0)
        result = auto_evolve(
            evaluate,
            rng,
            pop_size=40,
            generations=30,
            length=6,
            n_scalar=4,
            n_tensor=6,
        )
        # elitism: the running best never regresses
        assert result.history[-1] >= result.history[0]
        # the GA reaches a good solution on this small problem
        assert result.best_objective.values[0] > -0.1
        # improvement happened (unless gen0 was already essentially optimal)
        assert result.history[-1] > result.history[0] or result.history[0] > -0.02

    def test_history_is_monotonic_even_with_novelty_pressure(self):
        # With novelty_weight > 0, selection favours semantic distance from the
        # rest of the (ever-changing) population — a bonus that is population-
        # relative and NOT comparable across generations. The *tracked best* must
        # still be judged on the caller's real objective (loss/cost), never on a
        # stale novelty snapshot, or "best_loss" can appear to regress generation
        # to generation (seen on a live run: 0.4014 -> 0.7035) even though the
        # true best-found-so-far never got worse.
        target = torch.tensor([1.0, -1.0, 0.5, 0.5])

        def evaluate(prog: Program):
            mem = Memory(prog.n_scalar, prog.n_tensor)
            mem.write("t0", torch.tensor([2.0, -2.0, 1.0, 1.0]))
            prog.execute(mem)
            out = mem.read(prog.out_reg)
            try:
                out = out.reshape(4)
            except RuntimeError:
                out = out.flatten()[:4]
                if out.numel() < 4:
                    out = torch.zeros(4)
            mse = torch.mean((out - target) ** 2).item()
            return Objective((-mse, -0.01 * len(prog.instructions)))

        rng = np.random.default_rng(1)
        result = auto_evolve(
            evaluate, rng, pop_size=30, generations=25, length=6,
            n_scalar=4, n_tensor=6, weights=[1.0, 1.0, 0.5],
            novelty_weight=0.4,
        )
        assert all(a <= b + 1e-9 for a, b in zip(result.history, result.history[1:]))
        # the reported best objective must be the *raw* (loss, cost) pair —
        # never a novelty-inflated value that doesn't correspond to real quality.
        assert len(result.best_objective.values) == 2

    def test_primary_metric_is_monotonic_even_when_cost_trades_off(self):
        # `best_objective` (values[0], values[1]=cost) is a genuine multi-
        # objective champion: a much cheaper program can legitimately out-score
        # a lower-loss one on the *combined* scalar, so values[0] alone (the
        # thing progress logs print as "best_loss") is not guaranteed monotonic
        # by that tracker (seen live: gen0=0.5602 -> gen1=0.6430, both cuda and
        # cpu). `primary_objective` must track values[0] in isolation — the
        # single best "primary metric" (loss) ever evaluated, cost be damned —
        # so it can never look like it regressed.
        target = torch.tensor([1.0, -1.0, 0.5, 0.5])
        seen_values0 = []

        def evaluate(prog: Program):
            mem = Memory(prog.n_scalar, prog.n_tensor)
            mem.write("t0", torch.tensor([2.0, -2.0, 1.0, 1.0]))
            prog.execute(mem)
            out = mem.read(prog.out_reg)
            try:
                out = out.reshape(4)
            except RuntimeError:
                out = out.flatten()[:4]
                if out.numel() < 4:
                    out = torch.zeros(4)
            mse = torch.mean((out - target) ** 2).item()
            v0 = -mse
            seen_values0.append(v0)
            return Objective((v0, -0.05 * len(prog.instructions)))

        rng = np.random.default_rng(2)
        result = auto_evolve(
            evaluate, rng, pop_size=30, generations=20, length=6,
            n_scalar=4, n_tensor=6, weights=[1.0, 1.0],
        )
        assert result.primary_objective.values[0] == max(seen_values0)
