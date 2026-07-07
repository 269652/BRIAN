# -*- coding: utf-8 -*-
"""Persistent search ledger — don't rediscover the same patch on every run.

Each searched program gets a stable *semantic signature* (structure, not
hyperparameters). The ledger persists to disk so a later run can see what was
already tried and its outcome, and skip known duds instead of re-searching them.
"""
from neuroslm.genetic.language import Instruction, Program
from neuroslm.genetic.optimizer import sgd_program, adam_program
from neuroslm.genetic.ledger import SearchLedger, SearchRecord


def _p(op):
    return Program([Instruction(op, "t2", ("t0",))], 2, 4, "t2")


class TestSignature:
    def test_signature_is_stable_and_structural(self):
        led = SearchLedger(":memory:")
        # same structure, different hyperparameter → same signature
        assert led.signature(sgd_program(lr=0.01)) == led.signature(sgd_program(lr=0.9))
        # different structure → different signature
        assert led.signature(sgd_program()) != led.signature(adam_program())


class TestRecordAndQuery:
    def test_has_searched_after_record(self):
        led = SearchLedger(":memory:")
        p = _p("tanh")
        assert not led.has_searched(p)
        led.record(p, outcome="rejected", delta=0.1)
        assert led.has_searched(p)

    def test_is_dud_when_rejected_or_not_improving(self):
        led = SearchLedger(":memory:")
        rej = _p("tanh")
        kept = _p("sigmoid")
        led.record(rej, outcome="rejected", delta=0.2)     # worse
        led.record(kept, outcome="kept", delta=-0.3)       # improved (loss down)
        assert led.is_dud(rej)
        assert not led.is_dud(kept)

    def test_stats_counts_outcomes(self):
        led = SearchLedger(":memory:")
        led.record(_p("tanh"), outcome="rejected", delta=0.1)
        led.record(_p("sigmoid"), outcome="kept", delta=-0.1)
        led.record(_p("relu"), outcome="searched", delta=0.0)
        s = led.stats()
        assert s["total"] == 3
        assert s["kept"] == 1 and s["rejected"] == 1


class TestPersistence:
    def test_survives_reload_across_runs(self, tmp_path):
        path = tmp_path / "ledger.json"
        led = SearchLedger(path)
        p = _p("tanh")
        led.record(p, outcome="rejected", delta=0.5, run_id="run-1")
        led.save()

        # a fresh process / new run loads the same ledger
        led2 = SearchLedger(path)
        assert led2.has_searched(p)
        assert led2.is_dud(p)
        assert led2.stats()["total"] == 1

    def test_record_dedups_by_signature(self, tmp_path):
        path = tmp_path / "ledger.json"
        led = SearchLedger(path)
        # same structure recorded twice → one ledger entry, latest outcome wins
        led.record(sgd_program(lr=0.1), outcome="searched", delta=0.0)
        led.record(sgd_program(lr=0.2), outcome="kept", delta=-0.4)
        assert led.stats()["total"] == 1
        assert not led.is_dud(sgd_program(lr=0.3))  # latest outcome was "kept"
