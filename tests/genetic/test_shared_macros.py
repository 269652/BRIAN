# -*- coding: utf-8 -*-
"""Extract shared subexpressions into reusable macros + promote references.

When the search finds an improved (or simply shared) computation, it should be
factored **once** into a macro and reused by every mechanic that computes it —
not left duplicated in the one program it was found in. `extract_shared_as_macros`
does exactly that: it finds multi-op subexpressions common to ≥2 mechanics,
lifts each into a `Macro`, and rewrites every mechanic to `call` it — verifying
(expand → probe-equivalence) that behaviour is preserved.

`promote_modulation` is the second half of the user's ask: a modulation that has
been validated through training/ablation can be marked as the new *reference*
implementation.
"""
from neuroslm.genetic.language import Instruction, Program
from neuroslm.genetic.macros import expand_macros
from neuroslm.genetic.simplify import programs_equivalent
from neuroslm.genetic.optimizer import momentum_program
from neuroslm.genetic.modulation_store import ModulationStore, ModulationRecord
from neuroslm.genetic.shared_macros import (
    extract_shared_as_macros, ExtractionResult, promote_modulation, is_reference,
)


def _mech_a():
    # sqrt(square(t0)) is the shared 2-op subexpression
    return Program([
        Instruction("square", "t2", ("t0",)),
        Instruction("sqrt", "t3", ("t2",)),
        Instruction("add", "t4", ("t3", "t1")),
    ], n_scalar=8, n_tensor=16, out_reg="t4")


def _mech_b():
    return Program([
        Instruction("square", "t2", ("t0",)),
        Instruction("sqrt", "t3", ("t2",)),
        Instruction("mul", "t5", ("t3", "t3")),
    ], n_scalar=8, n_tensor=16, out_reg="t5")


class TestExtractSharedMacros:
    def test_shared_subexpr_becomes_a_macro(self):
        res = extract_shared_as_macros({"a": _mech_a(), "b": _mech_b()})
        assert isinstance(res, ExtractionResult)
        assert len(res.library) >= 1
        assert len(res.extracted) >= 1
        assert res.extracted[0]["ops"] >= 2

    def test_both_mechanics_call_the_macro(self):
        res = extract_shared_as_macros({"a": _mech_a(), "b": _mech_b()})
        macro_names = set(res.library.names())
        for name in ("a", "b"):
            calls = [i.macro for i in res.mechanics[name].instructions if i.op == "call"]
            assert any(c in macro_names for c in calls), name

    def test_rewrite_preserves_behaviour(self):
        res = extract_shared_as_macros({"a": _mech_a(), "b": _mech_b()})
        for name, orig in (("a", _mech_a()), ("b", _mech_b())):
            rewritten = res.mechanics[name]
            rewritten.library = res.library
            flat = expand_macros(rewritten, res.library)
            assert programs_equivalent(orig, flat, n_probes=12, seed=1)

    def test_reused_everywhere_not_just_where_found(self):
        # a third mechanic that also computes sqrt(square(t0)) picks up the macro
        c = Program([
            Instruction("square", "t2", ("t0",)),
            Instruction("sqrt", "t3", ("t2",)),
            Instruction("neg", "t6", ("t3",)),
        ], n_scalar=8, n_tensor=16, out_reg="t6")
        res = extract_shared_as_macros({"a": _mech_a(), "b": _mech_b(), "c": c})
        assert any(i.op == "call" for i in res.mechanics["c"].instructions)


class TestNoFalseExtraction:
    def test_disjoint_mechanics_yield_nothing(self):
        a = Program([Instruction("tanh", "t2", ("t0",))], 8, 16, "t2")
        b = Program([Instruction("sigmoid", "t2", ("t1",))], 8, 16, "t2")
        res = extract_shared_as_macros({"a": a, "b": b})
        assert len(res.library) == 0
        assert len(res.extracted) == 0

    def test_stateful_mechanic_is_left_untouched(self):
        # momentum re-writes its buffer register → not safe to factor; must not crash
        res = extract_shared_as_macros({"m": momentum_program(), "a": _mech_a()})
        assert "m" in res.mechanics    # returned, unchanged, no exception


class TestPromoteModulation:
    def test_promote_marks_reference(self, tmp_path):
        store = ModulationStore(tmp_path)
        rec = ModulationRecord(name="good_gain",
                               program=Program([Instruction("tanh", "t2", ("t0",))], 4, 8, "t2"),
                               metrics={"ppl": 40.0})
        store.save(rec)
        promote_modulation(store, "good_gain")
        reloaded = store.get("good_gain")
        assert is_reference(reloaded)

    def test_unpromoted_is_not_reference(self, tmp_path):
        store = ModulationStore(tmp_path)
        rec = ModulationRecord(name="plain",
                               program=Program([Instruction("tanh", "t2", ("t0",))], 4, 8, "t2"))
        store.save(rec)
        assert is_reference(store.get("plain")) is False
