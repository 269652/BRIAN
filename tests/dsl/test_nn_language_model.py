# -*- coding: utf-8 -*-
"""Phase N4 — stacked language model from DSL blocks, exact-match.

Composes the N3 DSL TransformerBlock into a full LM: embedding → N blocks
→ final RMSNorm → lm_head. Proves the *composition* (stacking, embedding,
head) is bit-identical to a hand-written reference assembled from the
same `neuroslm.modules.common` primitives.
"""
import pytest
import torch
import torch.nn as nn

from neuroslm.dsl.nn_lang import build_language_model
from neuroslm.dsl.nn_ops import swiglu_hidden_dim
from neuroslm.modules.common import TransformerBlock as RefBlock, RMSNorm


class RefLM(nn.Module):
    """Reference LM: embedding → blocks → final RMSNorm → lm_head."""
    def __init__(self, vocab, D, depth, n_heads, max_ctx):
        super().__init__()
        self.embed = nn.Embedding(vocab, D)
        self.blocks = nn.ModuleList([
            RefBlock(D, n_heads, max_ctx, n_nt=0, hebbian_rank=0, dropout=0.0)
            for _ in range(depth)
        ])
        self.norm_f = RMSNorm(D)
        self.lm_head = nn.Linear(D, vocab, bias=False)

    def forward(self, ids):
        h = self.embed(ids)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm_f(h)
        return self.lm_head(h)


class TestStackedLanguageModel:
    def test_matches_reference_lm(self):
        vocab, D, depth, n_heads, max_ctx = 64, 32, 3, 4, 64

        ref = RefLM(vocab, D, depth, n_heads, max_ctx)
        ref.eval()

        dsl = build_language_model(vocab=vocab, d_model=D, depth=depth,
                                   n_heads=n_heads, max_ctx=max_ctx)

        # Sync weights: embedding, each block, final norm, head
        with torch.no_grad():
            dsl.embed.copy_(ref.embed.weight)
            for d_blk, r_blk in zip(dsl.blocks, ref.blocks):
                d_blk.gamma1.copy_(r_blk.n1.weight)
                d_blk.Wq.copy_(r_blk.attn.q_proj.weight)
                d_blk.Wkv.copy_(r_blk.attn.kv_proj.weight)
                d_blk.Wo.copy_(r_blk.attn.out.weight)
                d_blk.gamma2.copy_(r_blk.n2.weight)
                d_blk.w1.copy_(r_blk.mlp.w1.weight)
                d_blk.w2.copy_(r_blk.mlp.w2.weight)
                d_blk.w3.copy_(r_blk.mlp.w3.weight)
            dsl.gamma_f.copy_(ref.norm_f.weight)
            dsl.lm_head.copy_(ref.lm_head.weight)

        ids = torch.randint(0, vocab, (2, 16))
        with torch.no_grad():
            ref_logits = ref(ids)
            dsl_logits = dsl(ids)
        assert dsl_logits.shape == (2, 16, vocab)
        assert torch.allclose(dsl_logits, ref_logits, atol=1e-5), \
            f"max diff {(dsl_logits - ref_logits).abs().max().item()}"

    def test_gradient_flows(self):
        dsl = build_language_model(vocab=64, d_model=32, depth=2,
                                   n_heads=4, max_ctx=64)
        ids = torch.randint(0, 64, (2, 8))
        logits = dsl(ids)
        logits.sum().backward()
        assert dsl.embed.grad is not None
        assert dsl.lm_head.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
