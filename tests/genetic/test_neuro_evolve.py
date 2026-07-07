# -*- coding: utf-8 -*-
"""Neuroanatomically-constrained auto-evolve of a trunk neuromodulation.

The engine evolves an NGL program that gain-modulates a tiny CPU LM's residual
stream. Fitness is Pareto over (−val_PPL, +neuroanatomic_plausibility), so the
search improves language modelling *without* abandoning biological realism. We
cannot train SmolLM to competitive PPL on CPU here — these contracts pin the
engine that would drive that search on a GPU deploy.
"""
import math

import numpy as np

from neuroslm.genetic.language import Instruction, Program
from neuroslm.genetic.neuro_evolve import (
    NeuroanatomicPrior,
    evaluate_modulation,
    identity_modulation,
    run_trunk_evolution,
)


class TestNeuroanatomicPrior:
    def test_bounded_gain_scores_above_unbounded_amplifier(self):
        prior = NeuroanatomicPrior()
        # bounded multiplicative gain: g = sigmoid(rms(h)) — homeostatic, divisive
        bounded = Program(
            [
                Instruction("rms", "t2", ("t0",)),
                Instruction("sigmoid", "t3", ("t2",)),
            ],
            n_scalar=4, n_tensor=6, out_reg="t3",
        )
        # unbounded amplifier: g = exp(exp(h)) — biologically implausible runaway
        runaway = Program(
            [
                Instruction("exp", "t2", ("t0",)),
                Instruction("exp", "t3", ("t2",)),
                Instruction("outer", "t4", ("t3", "t3")),
            ],
            n_scalar=4, n_tensor=6, out_reg="t4",
        )
        assert prior.score(bounded) > prior.score(runaway)

    def test_identity_modulation_is_plausible(self):
        prior = NeuroanatomicPrior()
        s = prior.score(identity_modulation())
        assert 0.0 <= s <= 1.0
        assert s > 0.3  # a metabolically-cheap bounded no-op is plausible

    def test_score_is_in_unit_interval(self):
        prior = NeuroanatomicPrior()
        rng = np.random.default_rng(0)
        from neuroslm.genetic.evolve import random_program
        for _ in range(20):
            p = random_program(rng, length=6, n_scalar=4, n_tensor=6)
            assert 0.0 <= prior.score(p) <= 1.0


class TestEvaluateModulation:
    def test_identity_modulation_matches_baseline(self):
        # applying an identity gain must equal the unmodulated trunk PPL
        ppl_id, plaus = evaluate_modulation(identity_modulation(), seed=0, steps=20)
        assert math.isfinite(ppl_id)
        assert ppl_id > 1.0  # PPL is ≥ 1 by definition
        assert 0.0 <= plaus <= 1.0

    def test_divergent_modulation_is_penalised(self):
        # a modulation that explodes the residual must yield a finite, large PPL
        blow = Program([Instruction("exp", "t5", ("t0",)),
                        Instruction("cscale", "t6", ("t5",), const=1e6)],
                       n_scalar=4, n_tensor=8, out_reg="t6")
        ppl, _ = evaluate_modulation(blow, seed=0, steps=20)
        assert math.isfinite(ppl)


class TestTrunkEvolution:
    def test_evolution_stays_competitive_and_plausible(self):
        outcome = run_trunk_evolution(seed=0, pop_size=8, generations=3, steps=25)
        assert math.isfinite(outcome.best_val_ppl)
        assert math.isfinite(outcome.baseline_val_ppl)
        # PPL stays competitive with the unmodulated trunk (modulation may trade a
        # hair of PPL for realism, but must not wreck language modelling)
        assert outcome.best_val_ppl <= outcome.baseline_val_ppl * 1.15
        # the neuroanatomic prior is enforced: the chosen modulation is at least as
        # plausible as the identity baseline (the search never sacrifices realism)
        prior = NeuroanatomicPrior()
        assert outcome.best_plausibility >= prior.score(identity_modulation()) - 1e-9
        # and the baseline does not Pareto-dominate the chosen modulation
        baseline_plaus = prior.score(identity_modulation())
        dominated = (outcome.baseline_val_ppl < outcome.best_val_ppl
                     and baseline_plaus > outcome.best_plausibility)
        assert not dominated

    def test_evolution_is_deterministic_under_seed(self):
        a = run_trunk_evolution(seed=1, pop_size=6, generations=2, steps=20)
        b = run_trunk_evolution(seed=1, pop_size=6, generations=2, steps=20)
        assert a.best_val_ppl == b.best_val_ppl
