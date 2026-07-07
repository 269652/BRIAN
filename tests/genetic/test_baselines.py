# -*- coding: utf-8 -*-
"""Baseline algorithms with tradeoffs — seed the search from the arch's optimizer.

The current SmolLM trunk trains with Adam(W). Rather than search from scratch, the
discovery should start from that baseline (and optionally several, with their
tradeoffs) and search *outward* from a known-good algorithm.
"""
import math

from neuroslm.genetic.baselines import Baseline, default_baselines, seeds_for
from neuroslm.genetic.discovery import run_optimizer_discovery


class TestBaselineRegistry:
    def test_has_the_optimizers_with_tradeoffs(self):
        b = default_baselines()
        for name in ("sgd", "momentum", "rmsprop", "adam", "lion"):
            assert name in b
            bl = b[name]
            assert isinstance(bl, Baseline)
            assert bl.cost >= 1
            assert bl.memory >= 0
            assert bl.stability in ("low", "medium", "high")

    def test_tradeoffs_are_sensible(self):
        b = default_baselines()
        # SGD is stateless; Adam keeps two moments; Lion one
        assert b["sgd"].memory == 0
        assert b["adam"].memory >= 2
        assert b["lion"].memory >= 1
        # Adam costs more per step than SGD
        assert b["adam"].cost > b["sgd"].cost


class TestSeedsFor:
    def test_seeds_for_returns_programs(self):
        progs = seeds_for(["adam", "lion"])
        assert len(progs) == 2
        assert all(hasattr(p, "instructions") for p in progs)

    def test_unknown_baseline_is_ignored_or_errors_clearly(self):
        import pytest
        with pytest.raises(KeyError):
            seeds_for(["not_an_optimizer"])


class TestSeededDiscovery:
    def test_discovery_can_start_from_adam(self):
        outcome = run_optimizer_discovery(
            seed=0, pop_size=12, generations=3, steps=20,
            seed_from=["adam"],
        )
        assert math.isfinite(outcome.best_final_loss)
        # seeding from a strong baseline should not be worse than the SGD baseline
        assert outcome.best_final_loss <= outcome.sgd_baseline_loss + 1e-6

    def test_multiple_baselines_seed_the_population(self):
        outcome = run_optimizer_discovery(
            seed=0, pop_size=12, generations=2, steps=20,
            seed_from=["adam", "lion", "rmsprop"],
        )
        assert math.isfinite(outcome.best_final_loss)
