# -*- coding: utf-8 -*-
"""Static semantic analysis / abstract interpretation of NGL programs.

`analyze(program)` runs an abstract interpreter over the NGL register file,
propagating a small lattice of value facts (bounded / non-negative / normalized
/ sign-only / mixes-across-elements) through each op's transfer function. The
resulting `SemanticSummary` says *what a mechanic does* (role, boundedness,
whether it normalizes, whether it carries state) in a form both a human and the
CSE / mechanic-reuse search can read.

The payoff is `interchangeable(a, b)`: two mechanics that occupy the same
semantic role with compatible abstract outputs are substitution candidates, so
the discovery loop can propose swapping one for a cheaper equivalent.
"""
from neuroslm.genetic.language import Instruction, Program
from neuroslm.genetic.optimizer import (
    sgd_program, momentum_program, adam_program, lion_program,
)
from neuroslm.genetic.attention_primitives import single_head_attention_program
from neuroslm.genetic.semantics import (
    analyze, describe, interchangeable, SemanticSummary, AbstractValue,
)


def _prog(instrs, out, ns=8, nt=16):
    return Program(list(instrs), n_scalar=ns, n_tensor=nt, out_reg=out)


class TestAbstractDomain:
    def test_tanh_output_is_bounded_and_elementwise(self):
        p = _prog([Instruction("tanh", "t2", ("t0",))], "t2")
        s = analyze(p)
        assert s.bounded is True
        assert s.elementwise is True
        assert s.output.bounded is True

    def test_sigmoid_output_is_bounded_nonneg(self):
        p = _prog([Instruction("sigmoid", "t2", ("t0",))], "t2")
        s = analyze(p)
        assert s.output.bounded is True
        assert s.output.nonneg is True

    def test_matmul_mixes_across_elements(self):
        p = _prog([Instruction("matmul", "t2", ("t0", "t1"))], "t2")
        s = analyze(p)
        assert s.elementwise is False
        assert s.output.mixes is True
        assert s.bounded is False

    def test_relu_is_nonneg_but_unbounded(self):
        p = _prog([Instruction("relu", "t2", ("t0",))], "t2")
        s = analyze(p)
        assert s.output.nonneg is True
        assert s.output.bounded is False

    def test_clip_bounds_an_unbounded_input(self):
        p = _prog([Instruction("clip", "t2", ("t0",), const=3.0)], "t2")
        s = analyze(p)
        assert s.output.bounded is True


class TestNormalizing:
    def test_l2norm_last_is_normalizing(self):
        p = _prog([Instruction("l2norm_last", "t2", ("t0",))], "t2")
        s = analyze(p)
        assert s.normalizing is True
        assert s.output.normalized is True

    def test_rmsnorm_is_normalizing(self):
        p = _prog([Instruction("rmsnorm", "t2", ("t0", "t1"))], "t2")
        s = analyze(p)
        assert s.normalizing is True

    def test_plain_add_is_not_normalizing(self):
        p = _prog([Instruction("add", "t2", ("t0", "t1"))], "t2")
        s = analyze(p)
        assert s.normalizing is False


class TestSignAndState:
    def test_lion_is_sign_based(self):
        s = analyze(lion_program())
        assert s.sign_based is True

    def test_sgd_is_stateless(self):
        s = analyze(sgd_program())
        assert s.stateful is False

    def test_momentum_carries_state(self):
        # reads buf=t2 before writing it, then writes it → persistent buffer
        s = analyze(momentum_program())
        assert s.stateful is True

    def test_adam_carries_state(self):
        s = analyze(adam_program())
        assert s.stateful is True


class TestRole:
    def test_single_nonlinearity_is_an_activation(self):
        p = _prog([Instruction("gelu", "t2", ("t0",))], "t2")
        assert analyze(p).role == "activation"

    def test_norm_program_role_is_normalization(self):
        p = _prog([Instruction("l2norm_last", "t2", ("t0",))], "t2")
        assert analyze(p).role == "normalization"

    def test_attention_program_role_is_attention(self):
        s = analyze(single_head_attention_program())
        assert s.role == "attention"
        assert s.elementwise is False

    def test_optimizer_update_role(self):
        # convention: reads grad t0 + param t1, carries state → optimizer
        assert analyze(adam_program()).role == "optimizer_update"


class TestDescribe:
    def test_describe_is_human_readable(self):
        text = describe(analyze(single_head_attention_program()))
        assert isinstance(text, str) and len(text) > 20
        assert "attention" in text.lower()

    def test_describe_accepts_a_program_directly(self):
        text = describe(sgd_program())
        assert isinstance(text, str) and len(text) > 0

    def test_describe_mentions_boundedness(self):
        text = describe(_prog([Instruction("tanh", "t2", ("t0",))], "t2"))
        assert "bounded" in text.lower()


class TestInterchangeable:
    def test_two_bounded_activations_are_interchangeable(self):
        a = _prog([Instruction("tanh", "t2", ("t0",))], "t2")
        b = _prog([Instruction("sigmoid", "t2", ("t0",))], "t2")
        assert interchangeable(a, b) is True

    def test_activation_and_attention_are_not_interchangeable(self):
        a = _prog([Instruction("tanh", "t2", ("t0",))], "t2")
        b = single_head_attention_program()
        assert interchangeable(a, b) is False

    def test_bounded_and_unbounded_activation_not_interchangeable(self):
        a = _prog([Instruction("tanh", "t2", ("t0",))], "t2")     # bounded
        b = _prog([Instruction("relu", "t2", ("t0",))], "t2")     # unbounded
        assert interchangeable(a, b) is False

    def test_summary_is_serializable(self):
        s = analyze(sgd_program())
        assert isinstance(s, SemanticSummary)
        d = s.to_dict()
        assert d["role"] == "optimizer_update"
        assert "output" in d
