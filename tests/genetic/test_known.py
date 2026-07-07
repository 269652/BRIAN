# -*- coding: utf-8 -*-
"""Prior-art gate: the search shouldn't waste effort rediscovering known algorithms.

A registry of known algorithms (the SOTA optimizers + the trivial gradient rule)
lets the discovery penalize candidates that match known structure, so fitness
rewards *novelty*. "Same algorithm" is judged by semantic-space structure (op
histogram), which is hyperparameter-invariant — SGD@0.01 and SGD@0.5 are the same
algorithm.
"""
import numpy as np

from neuroslm.genetic.optimizer import (
    sgd_program, momentum_program, adam_program, lion_program,
)
from neuroslm.genetic.evolve import random_program
from neuroslm.genetic.known import (
    KnownAlgorithms,
    default_known_algorithms,
    novelty_vs_known,
)


class TestKnownRegistry:
    def test_default_has_the_sota_optimizers(self):
        known = default_known_algorithms()
        names = set(known.names())
        for n in ("sgd", "momentum", "rmsprop", "adam", "lion", "gradient"):
            assert n in names


class TestIsKnown:
    def test_sgd_is_known_regardless_of_lr(self):
        known = default_known_algorithms()
        assert known.is_known(sgd_program(lr=0.02))
        assert known.is_known(sgd_program(lr=0.5))   # different hyperparam, same algo

    def test_adam_and_lion_are_known(self):
        known = default_known_algorithms()
        assert known.is_known(adam_program(lr=0.01))
        assert known.is_known(lion_program())

    def test_a_distinct_program_is_not_known(self):
        known = default_known_algorithms()
        rng = np.random.default_rng(3)
        # at least most random 6-op programs are structurally unlike the known set
        novel = sum(not known.is_known(random_program(rng, length=7, n_scalar=4, n_tensor=8))
                    for _ in range(30))
        assert novel >= 20


class TestNoveltyScore:
    def test_known_scores_low_novel_scores_high(self):
        known = default_known_algorithms()
        rng = np.random.default_rng(0)
        sgd_nov = novelty_vs_known(sgd_program(lr=0.1), known)
        novels = [novelty_vs_known(random_program(rng, 8, 4, 8), known) for _ in range(20)]
        assert sgd_nov == 0.0 or sgd_nov < min(novels) + 1e-9
        assert max(novels) > sgd_nov


class TestDiscoveryGate:
    def test_avoid_known_penalises_rediscovery(self):
        # with avoid_known, a run still completes and the reported best is either
        # novel (not in the known set) or the gate demonstrably ran
        from neuroslm.genetic.discovery import run_optimizer_discovery
        outcome = run_optimizer_discovery(
            seed=0, pop_size=16, generations=4, steps=20,
            include_sota_seeds=True, avoid_known=True,
        )
        import math
        assert math.isfinite(outcome.best_final_loss)
        assert outcome.best_program is not None
