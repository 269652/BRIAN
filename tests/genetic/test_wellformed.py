# -*- coding: utf-8 -*-
"""Search-quality: penalize programs that read uninitialized registers.

NGL reads of an unwritten register return zeros, so the GA happily emits programs
that read undefined registers (implicit-0 noise) — the discovered `run_0_step4000`
winner did exactly this, which is also what trips the stateful mislabel. Counting
those reads and penalizing them in the search fitness steers discovery toward
clean, well-formed mechanics without changing the reported ppl.
"""
from neuroslm.genetic.language import Instruction, Program
from neuroslm.genetic.evolve import undefined_reads
from neuroslm.genetic.ledger import SearchLedger
from neuroslm.genetic.training_explorer import TrainingExplorer, ExploreConfig


def _p(instrs, out):
    return Program(list(instrs), n_scalar=8, n_tensor=16, out_reg=out)


class TestUndefinedReads:
    def test_reading_only_the_input_is_clean(self):
        p = _p([Instruction("neg", "t2", ("t0",))], "t2")
        assert undefined_reads(p, inputs=("t0",)) == 0

    def test_reading_an_unwritten_register_counts(self):
        p = _p([Instruction("abs", "t2", ("t5",))], "t2")   # t5 never written
        assert undefined_reads(p, inputs=("t0",)) == 1

    def test_write_before_read_is_clean(self):
        p = _p([Instruction("neg", "t2", ("t0",)),
                Instruction("abs", "t3", ("t2",))], "t3")     # t2 written then read
        assert undefined_reads(p, inputs=("t0",)) == 0

    def test_each_undefined_read_counts(self):
        p = _p([Instruction("add", "t2", ("t5", "t6"))], "t2")  # t5, t6 undefined
        assert undefined_reads(p, inputs=("t0",)) == 2

    def test_const_has_no_reads(self):
        p = _p([Instruction("const", "s0", (), const=1.0),
                Instruction("cscale", "t2", ("t0",), const=2.0)], "t2")
        assert undefined_reads(p, inputs=("t0",)) == 0

    def test_extra_inputs_are_allowed(self):
        p = _p([Instruction("add", "t2", ("t0", "t1"))], "t2")
        assert undefined_reads(p, inputs=("t0", "t1")) == 0
        assert undefined_reads(p, inputs=("t0",)) == 1        # t1 undefined here


class TestFitnessPenalty:
    def test_clean_program_has_no_penalty(self):
        exp = TrainingExplorer(SearchLedger(":memory:"),
                               ExploreConfig(wellformed_penalty=0.05, inputs=("t0",)))
        clean = _p([Instruction("tanh", "t2", ("t0",))], "t2")
        assert exp._fitness_penalty(clean) == 1.0

    def test_ill_formed_program_is_penalized(self):
        exp = TrainingExplorer(SearchLedger(":memory:"),
                               ExploreConfig(wellformed_penalty=0.05, inputs=("t0",)))
        messy = _p([Instruction("add", "t2", ("t5", "t6"))], "t2")   # 2 undefined
        assert exp._fitness_penalty(messy) == 1.0 + 0.05 * 2

    def test_penalty_off_is_neutral(self):
        exp = TrainingExplorer(SearchLedger(":memory:"),
                               ExploreConfig(wellformed_penalty=0.0, inputs=("t0",)))
        messy = _p([Instruction("abs", "t2", ("t9",))], "t2")
        assert exp._fitness_penalty(messy) == 1.0
