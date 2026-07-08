# -*- coding: utf-8 -*-
"""Contracts for the CPU discovery harness (neuroslm/genetic/discovery.py).

The harness must (1) benchmark an NGL update-rule by actually training a tiny
model on CPU and differentiate a good optimizer from a bad one, and (2) run a
seeded Pareto search that returns an optimizer at least as good as the SGD
baseline on the same task. Everything here runs in seconds on CPU with tiny
models — the whole point of the vision's "discovery on CPU" requirement.
"""
import math

import torch

from neuroslm.genetic.optimizer import sgd_program, adam_program
from neuroslm.genetic.discovery import (
    benchmark_optimizer,
    run_optimizer_discovery,
    run_flow_modulation_discovery,
)


class TestBenchmark:
    def test_benchmark_returns_finite_curve(self):
        res = benchmark_optimizer(sgd_program(lr=0.05), steps=25, seed=0)
        assert len(res.curve) > 0
        assert math.isfinite(res.final_loss)
        assert res.cost == 1  # sgd is a single instruction

    def test_adam_beats_sgd_on_regression(self):
        # a well-tuned Adam should reach lower loss than plain SGD in few steps
        sgd = benchmark_optimizer(sgd_program(lr=0.02), steps=40, seed=0)
        adam = benchmark_optimizer(adam_program(lr=0.05), steps=40, seed=0)
        assert adam.final_loss < sgd.final_loss

    def test_divergent_program_is_penalised_not_crashing(self):
        # an absurd lr makes SGD diverge; the harness must return a finite, large
        # loss rather than raise or return inf
        res = benchmark_optimizer(sgd_program(lr=1e6), steps=20, seed=0)
        assert math.isfinite(res.final_loss)
        assert res.final_loss >= benchmark_optimizer(sgd_program(lr=0.02), steps=20, seed=0).final_loss


class TestOptimizerDiscovery:
    def test_discovery_matches_or_beats_sgd_baseline(self):
        outcome = run_optimizer_discovery(
            seed=0,
            pop_size=16,
            generations=5,
            steps=25,
            include_sota_seeds=True,
        )
        assert outcome.best_program is not None
        assert math.isfinite(outcome.best_final_loss)
        # discovered rule is at least as good as plain SGD on the held task
        assert outcome.best_final_loss <= outcome.sgd_baseline_loss + 1e-6
        # a Pareto front (loss vs update-rule cost) is exposed
        assert len(outcome.front) >= 1

    def test_discovery_is_deterministic_under_seed(self):
        a = run_optimizer_discovery(seed=3, pop_size=12, generations=3, steps=20)
        b = run_optimizer_discovery(seed=3, pop_size=12, generations=3, steps=20)
        assert a.best_final_loss == b.best_final_loss


class TestProgressReporting:
    def test_progress_line_shows_cost_alongside_loss(self):
        # The tracked "best" is a (loss, cost) trade-off (per the module's own
        # documented objective), so a bare "best_loss=" line can look like it
        # regressed generation-to-generation when a cheaper-but-slightly-worse
        # program becomes the new combined-objective champion. The progress
        # line must show the cost term too, so that isn't mistaken for a bug.
        lines = []
        run_optimizer_discovery(seed=0, pop_size=12, generations=3, steps=20,
                                novelty_weight=0.3, progress=lines.append)
        gen_lines = [l for l in lines if l.strip().startswith("[optimizer] gen")]
        assert gen_lines
        for line in gen_lines:
            assert "best_loss=" in line
            assert "cost=" in line


class TestFlowModulationDiscovery:
    def test_flow_modulation_runs_and_scores_ei(self):
        outcome = run_flow_modulation_discovery(
            seed=0,
            pop_size=12,
            generations=3,
            steps=20,
        )
        assert outcome.best_program is not None
        assert math.isfinite(outcome.best_final_loss)
        assert math.isfinite(outcome.best_ei)
