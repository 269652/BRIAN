# -*- coding: utf-8 -*-
"""Contracts for NGL genetic operators + Pareto GA (neuroslm/genetic/evolve.py).

The operators must keep programs *executable* (totality already guarantees no
crash, but structural validity — valid ops/registers/out_reg — must hold), must
actually change behaviour, and the GA must move a population toward better
fitness on a toy multi-objective problem.
"""
import numpy as np
import torch

from neuroslm.genetic.language import Program, Memory
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
