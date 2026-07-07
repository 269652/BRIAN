# -*- coding: utf-8 -*-
"""softpick — a rectified, not-sum-to-one softmax replacement, as an NGL primitive.

softpick(x)_i = ReLU(e^{x_i} - 1) / (Σ_j |e^{x_j} - 1| + eps)   (over the last axis)

Verified against Zuhri et al., 2025 (arXiv:2504.20966). Adding it as an axis-aware
NGL op makes it an *evolvable* attention normalizer: the search can now mutate a
``softmax_last`` into a ``softpick_last`` and discover attention that permits true
zeros (no attention sink), instead of the mechanism being an opaque black box.
"""
import numpy as np
import torch

from neuroslm.genetic.language import REGISTRY, Instruction, Program, Memory
from neuroslm.genetic.evolve import _OP_NAMES
from neuroslm.genetic.attention_primitives import (
    single_head_attention_program, single_head_attention_softpick,
)
from neuroslm.genetic.semantics import analyze


def _softpick(x):
    return REGISTRY["softpick_last"].fn(x)


class TestSoftpickOp:
    def test_registered_over_last_axis(self):
        assert "softpick_last" in REGISTRY
        assert REGISTRY["softpick_last"].family == "nonlin"

    def test_shape_is_preserved(self):
        x = torch.randn(2, 3, 5)
        assert _softpick(x).shape == x.shape

    def test_non_negative(self):
        x = torch.randn(4, 7)
        assert (_softpick(x) >= 0).all()

    def test_negative_logits_produce_true_zeros(self):
        # x_i <= 0  ⇒  e^{x_i}-1 <= 0  ⇒  ReLU numerator = 0  (real sparsity)
        x = torch.tensor([[-1.0, -2.0, 3.0]])
        out = _softpick(x)
        assert out[0, 0].item() == 0.0
        assert out[0, 1].item() == 0.0
        assert out[0, 2].item() > 0.0

    def test_all_positive_row_sums_to_one(self):
        x = torch.tensor([[1.0, 2.0, 0.5, 3.0]])       # all > 0
        s = _softpick(x).sum(dim=-1).item()
        assert abs(s - 1.0) < 1e-4

    def test_row_with_negatives_sums_below_one(self):
        x = torch.tensor([[3.0, -2.0, -1.0, 0.5]])     # some <= 0
        s = _softpick(x).sum(dim=-1).item()
        assert s < 1.0                                  # not sum-to-one

    def test_finite_on_extreme_logits(self):
        x = torch.tensor([[100.0, -100.0, 0.0]])
        out = _softpick(x)
        assert torch.isfinite(out).all()

    def test_gradient_flows_including_negative_entries(self):
        x = torch.tensor([[2.0, -1.0, 0.5]], requires_grad=True)
        _softpick(x).sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        # the abs() denominator lets gradient reach even the negative entry
        assert x.grad[0, 1].item() != 0.0


class TestSearchable:
    def test_in_the_mutation_vocabulary(self):
        # a nonlin op → the GA can graft it; the mechanism is now evolvable
        assert "softpick_last" in _OP_NAMES

    def test_runs_inside_a_program(self):
        p = Program([Instruction("softpick_last", "t2", ("t0",))], 4, 8, "t2")
        mem = Memory(n_scalar=4, n_tensor=8)
        mem.write("t0", torch.randn(3, 6))
        out = p.execute(mem).read("t2")
        assert out.shape == (3, 6) and torch.isfinite(out).all()


class TestSemantics:
    def test_softpick_is_a_bounded_normalizing_mixer(self):
        p = Program([Instruction("softpick_last", "t2", ("t0",))], 4, 8, "t2")
        s = analyze(p)
        assert s.output.bounded is True
        assert s.output.nonneg is True
        assert s.normalizing is True
        assert s.elementwise is False        # mixes across the last axis


class TestSoftpickAttention:
    def test_variant_runs_and_matches_shape(self):
        B, T, D = 2, 5, 8
        prog = single_head_attention_softpick()
        mem = Memory(n_scalar=4, n_tensor=24)
        mem.write("t0", torch.randn(B, T, D))
        for r, W in (("t1", None), ("t2", None), ("t3", None), ("t4", None)):
            mem.write(r, torch.randn(D, D))
        out = prog.execute(mem).read(prog.out_reg)
        assert out.shape == (B, T, D)
        assert torch.isfinite(out).all()

    def test_variant_is_an_attention_mechanism(self):
        assert analyze(single_head_attention_softpick()).role == "attention"

    def test_differs_from_softmax_variant(self):
        B, T, D = 1, 4, 8
        x = torch.randn(B, T, D)
        ws = [torch.randn(D, D) for _ in range(4)]

        def run(prog):
            mem = Memory(n_scalar=4, n_tensor=24)
            mem.write("t0", x)
            for i, W in enumerate(ws):
                mem.write(f"t{i+1}", W)
            return prog.execute(mem).read(prog.out_reg)

        a = run(single_head_attention_program())
        b = run(single_head_attention_softpick())
        assert not torch.allclose(a, b, atol=1e-4)
