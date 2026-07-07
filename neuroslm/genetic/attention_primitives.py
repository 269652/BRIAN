# -*- coding: utf-8 -*-
"""Single-head causal self-attention expressed as NGL primitives.

The composite ``causal_self_attention`` op is opaque — the search can rewire
*around* it but never evolve the mechanism itself. Building attention from
primitives (linear, l2-norm, matmul, transpose, scale, causal-mask, row-softmax)
turns the attention computation into an ordinary NGL program, so a mutation can
change *how attention works* (drop QK-norm, swap the score function, add a gate),
not just where it sits. Contract (tests): this program equals a hand-written
torch reference bit-for-bit.

Calling convention (registers pre-bound by the harness):
    t0 = x   (B, T, D)
    t1 = Wq, t2 = Wk, t3 = Wv, t4 = Wo   (each D×D, nn.Linear layout: out×in)
    output = attention(x)   (B, T, D)
"""
from __future__ import annotations

from neuroslm.genetic.language import Instruction, Program


def single_head_attention_program(scale: float = 0.125) -> Program:
    """QK-normalised single-head causal attention as a primitive NGL program."""
    instrs = [
        Instruction("linear", "t5", ("t0", "t1")),        # q = x Wq^T
        Instruction("l2norm_last", "t6", ("t5",)),         # QK-norm
        Instruction("linear", "t7", ("t0", "t2")),         # k = x Wk^T
        Instruction("l2norm_last", "t8", ("t7",)),
        Instruction("linear", "t9", ("t0", "t3")),         # v = x Wv^T
        Instruction("transpose", "t10", ("t8",)),          # k^T
        Instruction("matmul", "t11", ("t6", "t10")),       # scores = q k^T
        Instruction("cscale", "t12", ("t11",), const=scale),  # / sqrt(d)
        Instruction("causal_mask", "t13", ("t12",)),       # mask future
        Instruction("softmax_last", "t14", ("t13",)),      # row softmax
        Instruction("matmul", "t15", ("t14", "t9")),       # context = attn v
        Instruction("linear", "t16", ("t15", "t4")),       # out = ctx Wo^T
    ]
    return Program(instrs, n_scalar=4, n_tensor=24, out_reg="t16",
                   meta={"name": "single_head_attention", "scale": scale})
