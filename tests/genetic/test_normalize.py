# -*- coding: utf-8 -*-
"""Semantic normalization — collapse equivalent expressions to a canonical form.

Syntactically different but semantically identical NGL programs (``neg(neg(x))``,
``x + 0``, ``x · 1``) are reduced to one canonical representative and substituted
everywhere, so the exploration search never wastes budget on syntactic variants
of something it has already seen.

Two layers of equality:
  * **rewrite-equal** — both reduce to the *same* canonical form under the
    convergent rewrite system (a decidable proof of equality);
  * **probe-equal** — canonical forms differ but agree on many random probes
    (e.g. ``x·x`` vs ``square(x)``, which no rule connects) → merged, with the
    lowest-complexity form chosen as canonical.

The canonical representative is the most-used member (``prefer="frequency"``) or
the lowest-complexity one (``prefer="simplest"``).
"""
from neuroslm.genetic.language import Instruction, Program
from neuroslm.genetic.simplify import programs_equivalent
from neuroslm.genetic.normalize import (
    canonical_form, semantic_signature, complexity, cyclomatic,
    normalize_semantics, NormalizeResult, SemanticClass,
)


def _p(instrs, out, ns=8, nt=16):
    return Program(list(instrs), n_scalar=ns, n_tensor=nt, out_reg=out)


# three syntactically distinct identities: all compute t0
def _neg_neg():
    return _p([Instruction("neg", "t2", ("t0",)),
               Instruction("neg", "t3", ("t2",))], "t3")


def _add_zero():
    return _p([Instruction("const", "t2", (), const=0.0),
               Instruction("add", "t3", ("t0", "t2"))], "t3")


def _scale_one():
    return _p([Instruction("cscale", "t2", ("t0",), const=1.0)], "t2")


class TestCanonicalForm:
    def test_reduces_and_preserves_behaviour(self):
        orig = _neg_neg()
        canon = canonical_form(orig)
        assert len(canon.instructions) < len(orig.instructions)
        assert programs_equivalent(orig, canon, n_probes=12, seed=1)

    def test_equivalent_programs_share_a_signature(self):
        s1 = semantic_signature(_neg_neg())
        s2 = semantic_signature(_add_zero())
        s3 = semantic_signature(_scale_one())
        assert s1 == s2 == s3

    def test_inequivalent_programs_differ(self):
        a = semantic_signature(_neg_neg())
        b = semantic_signature(_p([Instruction("tanh", "t2", ("t0",))], "t2"))
        assert a != b


class TestComplexity:
    def test_cyclomatic_counts_decisions(self):
        straight = _p([Instruction("add", "t2", ("t0", "t1"))], "t2")
        branchy = _p([Instruction("gt", "t2", ("t0", "t1")),
                      Instruction("select", "t3", ("t2", "t0", "t1"))], "t3")
        assert cyclomatic(straight) == 1
        assert cyclomatic(branchy) >= 2

    def test_complexity_orders_by_size(self):
        small = _p([Instruction("square", "t2", ("t0",))], "t2")
        big = _p([Instruction("abs", "t2", ("t0",)),
                  Instruction("mul", "t3", ("t2", "t2"))], "t3")
        assert complexity(small) < complexity(big)


class TestNormalizeByFrequency:
    def test_canonical_is_most_used(self):
        progs = {"nn": _neg_neg(), "az": _add_zero(), "so": _scale_one()}
        counts = {"nn": 1, "az": 5, "so": 1}
        res = normalize_semantics(progs, counts=counts, prefer="frequency")
        assert isinstance(res, NormalizeResult)
        # all three collapse to the single most-used member
        assert res.canonical_of["nn"] == "az"
        assert res.canonical_of["az"] == "az"
        assert res.canonical_of["so"] == "az"

    def test_one_class_for_the_equivalent_set(self):
        progs = {"nn": _neg_neg(), "az": _add_zero(), "so": _scale_one()}
        res = normalize_semantics(progs)
        assert len(res.classes) == 1
        assert set(res.classes[0].members) == {"nn", "az", "so"}

    def test_substitution_preserves_behaviour(self):
        progs = {"nn": _neg_neg(), "az": _add_zero(), "so": _scale_one()}
        res = normalize_semantics(progs, counts={"nn": 1, "az": 5, "so": 1})
        for name, orig in progs.items():
            assert programs_equivalent(orig, res.programs[name], n_probes=12, seed=2)


class TestNormalizeBySimplest:
    def test_probe_equal_forms_merge_and_pick_simplest(self):
        # square(x) and abs(x)*abs(x) are probe-equal but not rewrite-equal
        square = _p([Instruction("square", "t2", ("t0",))], "t2")
        absmul = _p([Instruction("abs", "t2", ("t0",)),
                     Instruction("mul", "t3", ("t2", "t2"))], "t3")
        res = normalize_semantics({"sq": square, "am": absmul}, prefer="simplest")
        assert len(res.classes) == 1
        # the 1-op square is simpler than the 2-op abs·abs → canonical
        assert res.canonical_of["am"] == "sq"
        assert res.canonical_of["sq"] == "sq"


class TestExplorerIntegration:
    def test_explorer_canonicalizes_before_ledger(self):
        # syntactic variants collapse to one ledger signature → not re-searched
        from neuroslm.genetic.ledger import SearchLedger
        from neuroslm.genetic.training_explorer import TrainingExplorer, ExploreConfig
        led = SearchLedger(":memory:")
        ex = TrainingExplorer(led, ExploreConfig(normalize=True))
        assert ex._canonical_sig(_neg_neg()) == ex._canonical_sig(_add_zero())

    def test_normalize_off_leaves_variants_distinct(self):
        from neuroslm.genetic.ledger import SearchLedger
        from neuroslm.genetic.training_explorer import TrainingExplorer, ExploreConfig
        led = SearchLedger(":memory:")
        ex = TrainingExplorer(led, ExploreConfig(normalize=False))
        # raw signatures of two different syntaxes differ without normalization
        assert ex._canonical_sig(_neg_neg()) != ex._canonical_sig(_add_zero())


class TestStatefulSoundness:
    def test_stateful_programs_are_not_probe_merged(self):
        # sgd and momentum agree on a single zeroed-state probe but diverge across
        # steps — normalization must NOT collapse them (state ≠ stateless).
        from neuroslm.genetic.optimizer import sgd_program, momentum_program
        res = normalize_semantics({"sgd": sgd_program(),
                                   "momentum": momentum_program()})
        assert res.canonical_of["sgd"] == "sgd"
        assert res.canonical_of["momentum"] == "momentum"
        assert len(res.classes) == 2


class TestNormalizeSeparatesDistinct:
    def test_distinct_semantics_stay_separate(self):
        progs = {
            "id": _neg_neg(),
            "tanh": _p([Instruction("tanh", "t2", ("t0",))], "t2"),
            "relu": _p([Instruction("relu", "t2", ("t0",))], "t2"),
        }
        res = normalize_semantics(progs)
        # three genuinely different behaviours → three classes
        assert len(res.classes) == 3
        for name in progs:
            assert res.canonical_of[name] == name
