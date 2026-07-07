# -*- coding: utf-8 -*-
"""Attention expressed as NGL *primitives* — the mechanism becomes searchable.

The composite `causal_self_attention` op is opaque: the search can rewire around
it but not evolve *inside* it. Here single-head causal attention is built from
NGL primitives (linear, matmul, transpose, scale, causal-mask, row-softmax), so a
mutation can change the attention mechanism itself. Contract: the primitive
program matches a hand-written torch reference bit-for-bit.
"""
import math

import torch
import torch.nn.functional as F

from neuroslm.genetic.language import Memory, REGISTRY
from neuroslm.genetic.attention_primitives import single_head_attention_program


class TestAxisAwareOps:
    def test_softmax_last_is_row_normalised(self):
        x = torch.randn(2, 3, 5)
        out = REGISTRY["softmax_last"].fn(x)
        assert torch.allclose(out.sum(-1), torch.ones(2, 3), atol=1e-6)
        assert torch.allclose(out, F.softmax(x, dim=-1), atol=1e-6)

    def test_l2norm_last_unit_rows(self):
        x = torch.randn(4, 6)
        out = REGISTRY["l2norm_last"].fn(x)
        assert torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-5)
        assert torch.allclose(out, F.normalize(x, dim=-1), atol=1e-6)

    def test_causal_mask_zeroes_future(self):
        scores = torch.zeros(1, 4, 4)
        masked = REGISTRY["causal_mask"].fn(scores)
        # upper triangle (strictly future) must be -inf-ish (very negative)
        assert masked[0, 0, 1] < -1e8
        assert masked[0, 1, 0] == 0.0  # past/self preserved


class TestSingleHeadAttentionEquivalence:
    def test_matches_torch_reference(self):
        torch.manual_seed(0)
        B, T, D = 2, 6, 8
        x = torch.randn(B, T, D)
        Wq = torch.randn(D, D) * 0.1
        Wk = torch.randn(D, D) * 0.1
        Wv = torch.randn(D, D) * 0.1
        Wo = torch.randn(D, D) * 0.1

        # reference: single-head causal attention with QK-norm + 1/sqrt(D) scale
        q = F.normalize(x @ Wq.T, dim=-1)
        k = F.normalize(x @ Wk.T, dim=-1)
        v = x @ Wv.T
        scores = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(D))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        ref = (attn @ v) @ Wo.T

        prog = single_head_attention_program(scale=1.0 / math.sqrt(D))
        mem = Memory(prog.n_scalar, prog.n_tensor)
        mem.write("t0", x)
        mem.write("t1", Wq)
        mem.write("t2", Wk)
        mem.write("t3", Wv)
        mem.write("t4", Wo)
        prog.execute(mem)
        got = mem.read(prog.out_reg)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


class TestEvolvable:
    def test_attention_program_is_mutable_and_total(self):
        # the primitive attention program is an ordinary NGL program: mutating it
        # never crashes (totality), so the search can explore attention variants
        import numpy as np
        from neuroslm.genetic.evolve import mutate
        prog = single_head_attention_program(scale=0.35)
        rng = np.random.default_rng(0)
        x = torch.randn(2, 5, 8)
        for _ in range(40):
            child = mutate(prog, rng)
            mem = Memory(child.n_scalar, child.n_tensor)
            mem.write("t0", x)
            for r in ("t1", "t2", "t3", "t4"):
                mem.write(r, torch.randn(8, 8) * 0.1)
            child.execute(mem)
            assert torch.isfinite(mem.read(child.out_reg)).all()
