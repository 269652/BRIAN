# -*- coding: utf-8 -*-
"""Evolve the KL-distillation λ-schedule (gap_nats → λ) as an NGL program.

`BRIANHarness._distillation_lambda` hardcodes a piecewise-linear ramp
(floor/ceiling/lambda_max). This module evolves that ramp as an NGL
program instead, scored by a proxy simulation of the actual physical
tension the ramp exists to resolve: a schedule that stays high once the
trunk has already surpassed the cortex (gap < 0) keeps pulling the trunk
toward a now-worse teacher, which HURTS; a schedule that collapses to 0
too early leaves catch-up help on the table. `distill_linear_program()`
is an exact NGL reconstruction of the current piecewise-linear formula
(via two `relu`s — see the `relu(gap-floor) - relu(gap-ceiling)` identity),
used as both the GA's elite seed and the baseline to beat.
"""
from __future__ import annotations

import math

from neuroslm.genetic.language import Instruction, Program


class TestDistillLinearProgram:
    """distill_linear_program() must be an EXACT NGL reconstruction of the
    piecewise-linear ramp in BRIANHarness._distillation_lambda — numeric
    match at every regime, not just "some value comes out"."""

    def _lambda_fn(self, program):
        from neuroslm.genetic.distill_evolve import _make_lambda_fn
        return _make_lambda_fn(program)

    def test_zero_below_floor(self):
        from neuroslm.genetic.distill_evolve import distill_linear_program
        prog = distill_linear_program(floor=0.1, ceiling=2.0, lambda_max=1.0)
        fn = self._lambda_fn(prog)
        assert fn(-1.0) == 0.0
        assert fn(0.0) == 0.0

    def test_saturates_at_lambda_max_above_ceiling(self):
        from neuroslm.genetic.distill_evolve import distill_linear_program
        prog = distill_linear_program(floor=0.1, ceiling=2.0, lambda_max=1.0)
        fn = self._lambda_fn(prog)
        assert abs(fn(5.0) - 1.0) < 1e-5
        assert abs(fn(2.0) - 1.0) < 1e-5

    def test_midpoint_interpolation_matches_harness_formula(self):
        # Same fixture values as tests/training/test_cortex_distillation_and_gating.py
        floor, ceiling, lambda_max = 0.1, 2.0, 1.0
        from neuroslm.genetic.distill_evolve import distill_linear_program
        prog = distill_linear_program(floor=floor, ceiling=ceiling, lambda_max=lambda_max)
        fn = self._lambda_fn(prog)
        midpoint = (floor + ceiling) / 2
        expected = lambda_max / 2
        assert abs(fn(midpoint) - expected) < 1e-5

    def test_matches_harness_lambda_across_gaps(self):
        """Cross-check against the ACTUAL BRIANHarness._distillation_lambda,
        not a re-derivation of the formula — catches drift if either side
        changes independently."""
        import torch
        import torch.nn as nn
        from neuroslm.dsl.training_config import TrainingConfig
        from neuroslm.harness import BRIANHarness

        class _FakeLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Parameter(torch.randn(16, 8) * 0.02)
                self.lm_head = nn.Parameter(torch.randn(16, 8) * 0.02)

            def forward(self, ids):
                return torch.nn.functional.linear(self.embed[ids], self.lm_head)

        cfg = TrainingConfig()
        cfg.multi_cortex.distillation_enabled = True
        cfg.multi_cortex.distillation_lambda_max = 1.0
        cfg.multi_cortex.distillation_gap_floor = 0.1
        cfg.multi_cortex.distillation_gap_ceiling = 2.0
        h = BRIANHarness.from_language_model(
            language_model=_FakeLM(), vocab_size=16, d_sem=8, training_config=cfg)

        from neuroslm.genetic.distill_evolve import distill_linear_program
        prog = distill_linear_program(floor=0.1, ceiling=2.0, lambda_max=1.0)
        fn = self._lambda_fn(prog)

        for gap in (-1.0, 0.0, 0.1, 0.5, 1.0, 1.05, 2.0, 3.0):
            expected = h._distillation_lambda(gap_nats=gap)
            got = fn(gap)
            assert abs(got - expected) < 1e-5, (
                f"gap={gap}: NGL program λ={got} != harness λ={expected}")


class TestSimulateDistillation:
    def test_finite_final_loss_for_linear_program(self):
        from neuroslm.genetic.distill_evolve import distill_linear_program, simulate_distillation
        final_loss, trajectory, invalid = simulate_distillation(
            distill_linear_program(), steps=40, seed=0)
        assert math.isfinite(final_loss)
        assert not invalid
        assert len(trajectory) == 40

    def test_deterministic_under_seed(self):
        from neuroslm.genetic.distill_evolve import distill_linear_program, simulate_distillation
        a, _, _ = simulate_distillation(distill_linear_program(), steps=30, seed=7)
        b, _, _ = simulate_distillation(distill_linear_program(), steps=30, seed=7)
        assert a == b

    def test_always_full_distillation_is_worse_than_gap_gated_ramp(self):
        """A schedule that never decays (constant λ=lambda_max regardless of
        gap) keeps pulling the trunk toward the cortex even after the trunk
        has already surpassed it (gap < 0) — physically harmful. The
        gap-gated ramp must reach a strictly lower (better) final loss."""
        from neuroslm.genetic.distill_evolve import distill_linear_program, simulate_distillation
        always_full = Program(
            [Instruction("const", "t1", (), const=1.0)],
            n_scalar=4, n_tensor=10, out_reg="t1", meta={"name": "always_full"})
        gated_loss, _, gated_invalid = simulate_distillation(
            distill_linear_program(), steps=60, seed=3)
        full_loss, _, full_invalid = simulate_distillation(always_full, steps=60, seed=3)
        assert not gated_invalid and not full_invalid
        assert gated_loss < full_loss, (
            f"gap-gated ramp (final_loss={gated_loss}) should beat "
            f"always-on distillation (final_loss={full_loss})")

    def test_never_distilling_leaves_catchup_help_on_the_table(self):
        """A schedule that is always λ=0 forfeits the catch-up benefit —
        the gap-gated ramp must reach a strictly lower final loss than
        never distilling at all."""
        from neuroslm.genetic.distill_evolve import distill_linear_program, simulate_distillation
        never = Program(
            [Instruction("const", "t1", (), const=0.0)],
            n_scalar=4, n_tensor=10, out_reg="t1", meta={"name": "never"})
        gated_loss, _, _ = simulate_distillation(distill_linear_program(), steps=60, seed=5)
        never_loss, _, _ = simulate_distillation(never, steps=60, seed=5)
        assert gated_loss < never_loss


class TestRunDistillEvolution:
    def test_evolution_beats_or_matches_baseline(self):
        from neuroslm.genetic.distill_evolve import run_distill_evolution
        outcome = run_distill_evolution(seed=0, pop_size=8, generations=3, steps=30)
        assert math.isfinite(outcome.best_final_loss)
        assert math.isfinite(outcome.baseline_final_loss)
        # distill_linear is seeded + protected by elitism, so evolved must
        # never be worse than the current formula.
        assert outcome.best_final_loss <= outcome.baseline_final_loss + 1e-6

    def test_deterministic_under_seed(self):
        from neuroslm.genetic.distill_evolve import run_distill_evolution
        a = run_distill_evolution(seed=1, pop_size=6, generations=2, steps=20)
        b = run_distill_evolution(seed=1, pop_size=6, generations=2, steps=20)
        assert a.best_final_loss == b.best_final_loss

    def test_outcome_has_front_stats(self):
        from neuroslm.genetic.distill_evolve import run_distill_evolution
        outcome = run_distill_evolution(seed=2, pop_size=6, generations=2, steps=20)
        assert isinstance(outcome.front_stats, list)
        assert len(outcome.front_stats) >= 1
        for s in outcome.front_stats:
            assert "final_loss" in s and "plausibility" in s


class TestInstallDistillationScheduleFromStore:
    def test_loads_and_wires_a_real_callable(self, tmp_path):
        from neuroslm.genetic.distill_evolve import (
            distill_linear_program, install_distillation_schedule_from_store)
        from neuroslm.genetic.modulation_store import ModulationStore, ModulationRecord

        store = ModulationStore(tmp_path)
        store.save(ModulationRecord(name="my_distill", program=distill_linear_program(),
                                    metrics={"final_loss": 1.0}))

        class _FakeHarness:
            def install_distillation_schedule(self, fn):
                self.installed_fn = fn

        h = _FakeHarness()
        report = install_distillation_schedule_from_store(h, "my_distill", store_dir=tmp_path)
        assert report["installed"] == "my_distill"
        assert callable(h.installed_fn)
        # midpoint of distill_linear_program()'s defaults (floor=0.1, ceiling=2.0,
        # lambda_max=1.0) -> lambda_max/2 = 0.5 — a real numeric round-trip
        # through the saved-and-reloaded .neuro file, not a mock.
        assert abs(h.installed_fn(1.05) - 0.5) < 1e-5

    def test_missing_name_raises(self, tmp_path):
        import pytest
        from neuroslm.genetic.distill_evolve import install_distillation_schedule_from_store
        with pytest.raises(KeyError):
            install_distillation_schedule_from_store(object(), "nope", store_dir=tmp_path)
