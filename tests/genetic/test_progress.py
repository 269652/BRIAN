# -*- coding: utf-8 -*-
"""Discovery emits per-generation progress so a long GPU run isn't a silent gap."""
import numpy as np

from neuroslm.genetic.evolve import auto_evolve, Objective
from neuroslm.genetic.language import Program, Instruction, Memory


def _trivial_eval(prog):
    # score = -number of instructions (deterministic, cheap)
    return Objective((-float(len(prog.instructions)),))


class TestAutoEvolveCallback:
    def test_on_generation_called_each_generation_plus_gen0(self):
        calls = []

        def on_gen(gen, total, best_obj):
            calls.append((gen, total, best_obj))

        rng = np.random.default_rng(0)
        auto_evolve(_trivial_eval, rng, pop_size=6, generations=4,
                    length=4, n_scalar=2, n_tensor=4, on_generation=on_gen)
        # gen 0 (initial) + one per generation
        assert [c[0] for c in calls] == [0, 1, 2, 3, 4]
        assert all(c[1] == 4 for c in calls)
        assert all(isinstance(c[2], Objective) for c in calls)

    def test_no_callback_still_runs(self):
        rng = np.random.default_rng(0)
        res = auto_evolve(_trivial_eval, rng, pop_size=6, generations=2,
                          length=4, n_scalar=2, n_tensor=4)
        assert res.best_program is not None


class TestDiscoveryProgress:
    def test_optimizer_progress_prints_lines(self, capsys):
        from neuroslm.genetic.discovery import run_optimizer_discovery
        run_optimizer_discovery(seed=0, pop_size=8, generations=3, steps=15,
                                progress=True)
        out = capsys.readouterr().out
        # one progress line per generation, mentioning gen index and a metric
        assert out.count("gen ") >= 3
        assert "loss" in out.lower()

    def test_trunk_progress_prints_lines(self, capsys):
        from neuroslm.genetic.neuro_evolve import run_trunk_evolution
        run_trunk_evolution(seed=0, pop_size=6, generations=2, steps=12,
                            progress=True)
        out = capsys.readouterr().out
        assert out.count("gen ") >= 2
        assert "ppl" in out.lower()

    def test_progress_off_is_silent(self, capsys):
        from neuroslm.genetic.discovery import run_optimizer_discovery
        run_optimizer_discovery(seed=0, pop_size=8, generations=2, steps=15)
        out = capsys.readouterr().out
        assert "gen " not in out
