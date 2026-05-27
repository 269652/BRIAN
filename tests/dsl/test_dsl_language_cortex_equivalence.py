# -*- coding: utf-8 -*-
"""N8 keystone — full DSL LanguageCortex bit-identical to Brain's.

Brain's LanguageCortex(baseline=False) interleaves [Standard, Diff,
MoD+Diff] blocks with a NeuralGeometryAdapter after each, then a final
RMSNorm + lm_head. All four block types are pure-DSL and bit-identical
to their Python references; this test assembles them in the same order
and asserts LM-logits parity step-for-step.
"""
import pytest
import torch
import torch.nn as nn

from neuroslm.dsl.nn_lang import build_dsl_language_cortex
from neuroslm.modules.language import LanguageCortex as RefLC


class TestDSLLanguageCortexExactMatch:
    def test_dsl_logits_match_reference(self):
        vocab, d_model, depth, n_heads, max_ctx = 64, 64, 4, 8, 64

        ref = RefLC(
            vocab_size=vocab, d_hidden=d_model, d_sem=d_model // 2,
            n_layers=depth, n_heads=n_heads, max_ctx=max_ctx,
            n_kv_heads=None, n_nt=0, hebbian_rank=0, dropout=0.0,
            geometry_expansion=2.0, mod_capacity=0.5,
            use_attention_pool=True,
        )
        # Training mode disables CALM early-exit (which can perturb the
        # forward path in eval even with zero-init head). dropout=0 so
        # train/eval are otherwise identical for logits.
        ref.train()

        dsl = build_dsl_language_cortex(
            vocab=vocab, d_model=d_model, depth=depth,
            n_heads=n_heads, max_ctx=max_ctx, n_kv_heads=n_heads,
            geometry_expansion=2.0, mod_capacity=0.5,
        )
        dsl.train()

        # Sync params: embed, head, final norm, then per-block + per-adapter.
        with torch.no_grad():
            dsl.embed.copy_(ref.tok_emb.weight)
            dsl.gamma_f.copy_(ref.norm_f.weight)
            dsl.lm_head.copy_(ref.lm_head.weight)
            for i in range(depth):
                self._sync_block(dsl.blocks[i], ref.blocks[i], i % 3, ref)
                self._sync_adapter(dsl.adapters[i], ref.adapters[i])

        ids = torch.randint(0, vocab, (2, 16))
        with torch.no_grad():
            ref_out = ref(ids)
            ref_logits = ref_out[0]
            dsl_logits = dsl(ids)
        diff = (ref_logits - dsl_logits).abs().max().item()
        assert diff < 1e-4, f"DSL LanguageCortex logits diverged (max diff {diff})"

    @staticmethod
    def _sync_block(dsl_blk, ref_blk, pattern: int, ref):
        with torch.no_grad():
            if pattern == 0:
                # Standard
                dsl_blk.gamma1.copy_(ref_blk.n1.weight)
                dsl_blk.Wq.copy_(ref_blk.attn.q_proj.weight)
                dsl_blk.Wkv.copy_(ref_blk.attn.kv_proj.weight)
                dsl_blk.Wo.copy_(ref_blk.attn.out.weight)
                dsl_blk.gamma2.copy_(ref_blk.n2.weight)
                dsl_blk.w1.copy_(ref_blk.mlp.w1.weight)
                dsl_blk.w2.copy_(ref_blk.mlp.w2.weight)
                dsl_blk.w3.copy_(ref_blk.mlp.w3.weight)
            elif pattern == 1:
                # Diff
                dsl_blk.gamma1.copy_(ref_blk.n1.weight)
                dsl_blk.Wq.copy_(ref_blk.attn.q_proj.weight)
                dsl_blk.Wkv.copy_(ref_blk.attn.kv_proj.weight)
                dsl_blk.Wo.copy_(ref_blk.attn.out.weight)
                dsl_blk.lambda_init.copy_(ref_blk.attn.lambda_init)
                dsl_blk.sub_norm.copy_(ref_blk.attn.sub_norm.weight)
                dsl_blk.gamma2.copy_(ref_blk.n2.weight)
                dsl_blk.w1.copy_(ref_blk.mlp.w1.weight)
                dsl_blk.w2.copy_(ref_blk.mlp.w2.weight)
                dsl_blk.w3.copy_(ref_blk.mlp.w3.weight)
            else:
                # MoD
                dsl_blk.router_w1.copy_(ref_blk.router.router[0].weight)
                dsl_blk.router_b1.copy_(ref_blk.router.router[0].bias)
                dsl_blk.router_w2.copy_(ref_blk.router.router[2].weight)
                dsl_blk.router_b2.copy_(ref_blk.router.router[2].bias)
                dsl_blk.gamma1.copy_(ref_blk.n1.weight)
                dsl_blk.Wq.copy_(ref_blk.attn.q_proj.weight)
                dsl_blk.Wkv.copy_(ref_blk.attn.kv_proj.weight)
                dsl_blk.Wo.copy_(ref_blk.attn.out.weight)
                dsl_blk.lambda_init.copy_(ref_blk.attn.lambda_init)
                dsl_blk.sub_norm.copy_(ref_blk.attn.sub_norm.weight)
                dsl_blk.gamma2.copy_(ref_blk.n2.weight)
                dsl_blk.w1.copy_(ref_blk.mlp.w1.weight)
                dsl_blk.w2.copy_(ref_blk.mlp.w2.weight)
                dsl_blk.w3.copy_(ref_blk.mlp.w3.weight)

    @staticmethod
    def _sync_adapter(dsl_adp, ref_adp):
        with torch.no_grad():
            dsl_adp.gamma.copy_(ref_adp.norm.weight)
            dsl_adp.Wup.copy_(ref_adp.up.weight)
            dsl_adp.kern_a.copy_(ref_adp.kern_a)
            dsl_adp.kern_b.copy_(ref_adp.kern_b)
            dsl_adp.Wgate.copy_(ref_adp.gate.weight)
            dsl_adp.bgate.copy_(ref_adp.gate.bias)
            dsl_adp.Wdown.copy_(ref_adp.down.weight)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
