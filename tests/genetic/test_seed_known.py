# -*- coding: utf-8 -*-
"""Seed the ledger with every known algorithm so the explorer skips those spaces."""
from neuroslm.genetic.ledger import SearchLedger
from neuroslm.genetic.optimizer import sgd_program, adam_program, lion_program
from neuroslm.genetic.evolve import random_program
from neuroslm.genetic.known import seed_ledger_with_known
import numpy as np


class TestSeedKnown:
    def test_seeding_records_the_known_algorithms(self):
        led = SearchLedger(":memory:")
        n = seed_ledger_with_known(led)
        assert n >= 6                     # the optimizers + macro blocks + identity
        assert led.stats()["total"] == n

    def test_known_algorithms_become_duds(self):
        led = SearchLedger(":memory:")
        seed_ledger_with_known(led)
        # the explorer's is_dud gate now skips these known spaces
        assert led.is_dud(sgd_program())
        assert led.is_dud(adam_program())
        assert led.is_dud(lion_program())

    def test_novel_program_is_not_skipped(self):
        led = SearchLedger(":memory:")
        seed_ledger_with_known(led)
        rng = np.random.default_rng(3)
        novel = sum(not led.is_dud(random_program(rng, 7, 4, 8)) for _ in range(30))
        assert novel >= 20               # genuinely new mechanics still pass

    def test_seeding_persists(self, tmp_path):
        path = tmp_path / "led.json"
        led = SearchLedger(path)
        seed_ledger_with_known(led)
        led.save()
        led2 = SearchLedger(path)         # a fresh run inherits the prior art
        assert led2.is_dud(adam_program())

    def test_idempotent_seeding(self):
        led = SearchLedger(":memory:")
        a = seed_ledger_with_known(led)
        b_total = led.stats()["total"]
        seed_ledger_with_known(led)        # re-seed → dedup by signature
        assert led.stats()["total"] == b_total
