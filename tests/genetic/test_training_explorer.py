# -*- coding: utf-8 -*-
"""Exploration wired into training: search every N steps, keep-if-better, ledger.

The explorer fires on a step cadence, searches an NGL modulation (skipping ledger
duds), A/B-tests it against the current model via a caller-supplied score fn,
keeps it only if the metric improves, and records every attempt to the persistent
ledger so future runs don't re-search the same space.
"""
import numpy as np

from neuroslm.genetic.language import Instruction, Program, Memory
from neuroslm.genetic.ledger import SearchLedger
from neuroslm.genetic.neuro_evolve import identity_modulation
from neuroslm.genetic.training_explorer import (
    TrainingExplorer,
    ExploreConfig,
    run_training_with_exploration,
)


class TestCadence:
    def test_fires_only_on_the_interval(self):
        led = SearchLedger(":memory:")
        fired = []

        def score_fn(prog):
            # deterministic: reward programs that output a large-norm gain
            mem = Memory(prog.n_scalar, prog.n_tensor)
            import torch
            mem.write("t0", torch.ones(4))
            prog.execute(mem)
            out = mem.read(prog.out_reg)
            return float((out - 2.0).abs().mean())   # target gain ≈ 2

        exp = TrainingExplorer(led, ExploreConfig(explore_every=500, pop_size=8,
                                                  generations=3), run_id="t")
        for step in range(0, 1600, 100):
            r = exp.maybe_explore(step, score_fn)
            if r is not None:
                fired.append(step)
        assert fired == [500, 1000, 1500]


class TestKeepIfBetter:
    def test_records_outcome_and_improves_or_reverts(self):
        led = SearchLedger(":memory:")

        def score_fn(prog):
            import torch
            mem = Memory(prog.n_scalar, prog.n_tensor)
            mem.write("t0", torch.ones(4))
            prog.execute(mem)
            out = mem.read(prog.out_reg).reshape(-1)[:1]
            return float((out - 3.0).abs().mean())   # want output ≈ 3

        exp = TrainingExplorer(led, ExploreConfig(explore_every=500, pop_size=16,
                                                  generations=6), run_id="t")
        res = exp.explore(500, score_fn)
        assert res.step == 500
        assert res.baseline == score_fn(identity_modulation())
        # the ledger recorded the attempt(s)
        assert led.stats()["total"] >= 1
        # if it improved, best_score < baseline; the outcome is consistent
        if res.improved:
            assert res.best_score < res.baseline
            assert led.outcome_of(res.best_program) == "kept"


class TestProgress:
    def test_explore_emits_per_generation_progress(self):
        led = SearchLedger(":memory:")

        def score_fn(prog):
            import torch
            mem = Memory(prog.n_scalar, prog.n_tensor)
            mem.write("t0", torch.ones(4))
            prog.execute(mem)
            return float(mem.read(prog.out_reg).reshape(-1)[:1].abs().mean())

        msgs = []
        exp = TrainingExplorer(led, ExploreConfig(explore_every=500, pop_size=8,
                                                  generations=4), run_id="t")
        exp.explore(500, score_fn, progress=msgs.append)
        assert msgs, "expected progress messages"
        assert any("gen" in m for m in msgs)          # per-generation lines
        assert any("500" in m for m in msgs)          # tagged with the step


class TestLedgerDedup:
    def test_second_run_skips_known_duds(self, tmp_path):
        path = tmp_path / "ledger.json"

        # a score fn where NOTHING beats identity → everything is a dud
        def score_fn(prog):
            return 0.0 if prog.to_source() == identity_modulation().to_source() else 5.0

        led1 = SearchLedger(path)
        exp1 = TrainingExplorer(led1, ExploreConfig(explore_every=500, pop_size=10,
                                                    generations=3), run_id="run1")
        exp1.explore(500, score_fn)
        led1.save()
        searched_after_run1 = led1.stats()["total"]
        assert searched_after_run1 >= 1

        # run 2 loads the ledger; it should recognise duds it already tried
        led2 = SearchLedger(path)
        exp2 = TrainingExplorer(led2, ExploreConfig(explore_every=500, pop_size=10,
                                                    generations=3), run_id="run2")
        res2 = exp2.explore(500, score_fn)
        assert res2.n_skipped_duds >= 1   # the gate consulted prior history


class TestStability:
    def test_baseline_does_not_diverge(self):
        # a destabilizing modulation installed into training must not run the
        # baseline away — the guard reverts it, so the reported ppls stay bounded.
        import math
        res = run_training_with_exploration(
            total_steps=1500, explore_every=500, seed=0,
            pop_size=10, generations=4, inner_steps=15)
        expl = res["explorations"]
        assert len(expl) >= 2
        base0 = expl[0]["baseline_ppl"]
        assert math.isfinite(res["final_val_ppl"])
        # the guard bounds *runaway* divergence (unguarded this hits ~190x); it
        # can't bound a single bad window to <5x, so allow generous slack while
        # still catching a true blow-up.
        for e in expl:
            assert e["baseline_ppl"] <= base0 * 25.0, e
        assert res["final_val_ppl"] <= base0 * 25.0
        assert "reverts" in res                          # guard is observable


class TestPersistence:
    def test_installed_winners_persist_with_delta(self, tmp_path):
        from neuroslm.genetic.modulation_store import ModulationStore
        store = ModulationStore(tmp_path / "mods")
        res = run_training_with_exploration(
            total_steps=500, explore_every=500, seed=0,
            pop_size=12, generations=4, inner_steps=15, store=store)
        saved = store.list_all()
        assert res["persisted"] == len(saved)
        assert len(saved) >= 1                      # the step-500 winner is saved
        for rec in saved:
            assert rec.metrics["delta_ppl"] > 0     # kept ⇒ genuine improvement
            assert "baseline_ppl" in rec.metrics and "step" in rec.metrics
            assert len(rec.program.instructions) >= 1   # round-trips to a program

    def test_persisted_programs_are_minimal(self, tmp_path):
        # the saved .neuro must be the minimal mechanism, not op-salad with dead code
        from neuroslm.genetic.modulation_store import ModulationStore
        from neuroslm.genetic.simplify import dead_code_eliminate
        store = ModulationStore(tmp_path / "mods")
        run_training_with_exploration(
            total_steps=500, explore_every=500, seed=0,
            pop_size=16, generations=6, inner_steps=20, store=store)
        saved = store.list_all()
        assert len(saved) >= 1
        for rec in saved:
            # already DCE-minimal → a further DCE pass removes nothing
            assert len(dead_code_eliminate(rec.program).instructions) == \
                   len(rec.program.instructions), rec.program.to_source()

    def test_persisted_count_never_exceeds_installs(self, tmp_path):
        # reverted installs are dropped, so survivors ≤ installs
        from neuroslm.genetic.modulation_store import ModulationStore
        store = ModulationStore(tmp_path / "mods")
        res = run_training_with_exploration(
            total_steps=2000, explore_every=500, seed=1,
            pop_size=10, generations=4, inner_steps=15, store=store)
        installs = sum(1 for e in res["explorations"] if e["improved"])
        assert res["persisted"] <= installs
        assert res["persisted"] == len(store.list_all())


class TestEndToEnd:
    def test_training_with_exploration_runs_and_logs(self, tmp_path):
        led = SearchLedger(tmp_path / "l.json")
        result = run_training_with_exploration(
            total_steps=1200, explore_every=500, seed=0,
            ledger=led, pop_size=8, generations=3, inner_steps=15,
        )
        # explorations happened at 500 and 1000
        assert len(result["explorations"]) == 2
        assert "final_val_ppl" in result
        assert led.stats()["total"] >= 1
