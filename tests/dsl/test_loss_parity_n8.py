# -*- coding: utf-8 -*-
"""N8 acceptance criterion — loss parity DSL ↔ Brain.

Once LM-logits match (test_dsl_language_cortex_equivalence.py), the
training loss and the per-step gradient update must also match exactly
on the same data + synced weights. This is the activated form of the
previously-skipped scaffold in test_training_equivalence.py.
"""
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.dsl.nn_lang import build_dsl_language_cortex
from neuroslm.modules.language import LanguageCortex as RefLC


def _build_pair(vocab=64, d_model=64, depth=4, n_heads=8, max_ctx=64):
    """Build matched DSL cortex + reference LanguageCortex with synced weights."""
    ref = RefLC(
        vocab_size=vocab, d_hidden=d_model, d_sem=d_model // 2,
        n_layers=depth, n_heads=n_heads, max_ctx=max_ctx,
        n_kv_heads=None, n_nt=0, hebbian_rank=0, dropout=0.0,
        geometry_expansion=2.0, mod_capacity=0.5,
        use_attention_pool=True,
    )
    ref.train()    # disable CALM (eval-only), keep dropout=0

    dsl = build_dsl_language_cortex(
        vocab=vocab, d_model=d_model, depth=depth,
        n_heads=n_heads, max_ctx=max_ctx, n_kv_heads=n_heads,
        geometry_expansion=2.0, mod_capacity=0.5,
    )
    dsl.train()

    with torch.no_grad():
        dsl.embed.copy_(ref.tok_emb.weight)
        dsl.gamma_f.copy_(ref.norm_f.weight)
        dsl.lm_head.copy_(ref.lm_head.weight)
        for i in range(depth):
            _sync_block(dsl.blocks[i], ref.blocks[i], i % 3)
            _sync_adapter(dsl.adapters[i], ref.adapters[i])
    return dsl, ref, vocab


def _sync_block(dsl_blk, ref_blk, pattern):
    with torch.no_grad():
        if pattern == 0:
            for n in ("gamma1", "Wq", "Wkv", "Wo", "gamma2", "w1", "w2", "w3"):
                getattr(dsl_blk, n).copy_(_ref_param(ref_blk, n))
        elif pattern == 1:
            for n in ("gamma1", "Wq", "Wkv", "Wo", "lambda_init", "sub_norm",
                      "gamma2", "w1", "w2", "w3"):
                getattr(dsl_blk, n).copy_(_ref_param(ref_blk, n))
        else:
            for n in ("router_w1", "router_b1", "router_w2", "router_b2",
                      "gamma1", "Wq", "Wkv", "Wo", "lambda_init", "sub_norm",
                      "gamma2", "w1", "w2", "w3"):
                getattr(dsl_blk, n).copy_(_ref_param(ref_blk, n))


def _ref_param(ref_blk, dsl_name):
    """Map DSL param name → reference tensor."""
    table = {
        "gamma1": ref_blk.n1.weight,
        "gamma2": ref_blk.n2.weight,
        "Wq": getattr(ref_blk.attn, "q_proj").weight,
        "Wkv": getattr(ref_blk.attn, "kv_proj").weight,
        "Wo": getattr(ref_blk.attn, "out").weight,
        "w1": ref_blk.mlp.w1.weight,
        "w2": ref_blk.mlp.w2.weight,
        "w3": ref_blk.mlp.w3.weight,
    }
    if hasattr(ref_blk.attn, "lambda_init"):
        table["lambda_init"] = ref_blk.attn.lambda_init
    if hasattr(ref_blk.attn, "sub_norm"):
        table["sub_norm"] = ref_blk.attn.sub_norm.weight
    if hasattr(ref_blk, "router"):
        table["router_w1"] = ref_blk.router.router[0].weight
        table["router_b1"] = ref_blk.router.router[0].bias
        table["router_w2"] = ref_blk.router.router[2].weight
        table["router_b2"] = ref_blk.router.router[2].bias
    return table[dsl_name]


def _sync_adapter(dsl_adp, ref_adp):
    with torch.no_grad():
        dsl_adp.gamma.copy_(ref_adp.norm.weight)
        dsl_adp.Wup.copy_(ref_adp.up.weight)
        dsl_adp.kern_a.copy_(ref_adp.kern_a)
        dsl_adp.kern_b.copy_(ref_adp.kern_b)
        dsl_adp.Wgate.copy_(ref_adp.gate.weight)
        dsl_adp.bgate.copy_(ref_adp.gate.bias)
        dsl_adp.Wdown.copy_(ref_adp.down.weight)


# ── 1. LM-loss parity at step 0 ────────────────────────────────────────

class TestLossValueParity:
    def test_lm_loss_matches_at_step0(self):
        dsl, ref, vocab = _build_pair()
        ids = torch.randint(0, vocab, (2, 16))
        targets = torch.randint(0, vocab, (2, 16))

        with torch.no_grad():
            ref_out = ref(ids)
            ref_logits = ref_out[0]
            dsl_logits = dsl(ids)
            ref_loss = F.cross_entropy(ref_logits.reshape(-1, vocab),
                                        targets.reshape(-1))
            dsl_loss = F.cross_entropy(dsl_logits.reshape(-1, vocab),
                                        targets.reshape(-1))
        diff = abs(ref_loss.item() - dsl_loss.item())
        assert diff < 1e-5, \
            f"step-0 LM loss diverged: ref={ref_loss.item()} dsl={dsl_loss.item()}"


# ── 2. Step-for-step parity through a few optimizer updates ───────────

class TestStepForStepParity:
    def test_three_steps_identical(self):
        """Same init + same data + same optimizer/LR → same losses for N steps."""
        torch.manual_seed(0)
        dsl, ref, vocab = _build_pair(vocab=32, d_model=32, depth=2,
                                       n_heads=4, max_ctx=32)
        # Plain SGD with fixed LR for both — minimal optimizer state to diverge
        ref_opt = torch.optim.SGD(ref.parameters(), lr=1e-3)
        dsl_opt = torch.optim.SGD(dsl.parameters(), lr=1e-3)

        g = torch.Generator().manual_seed(42)
        ref_losses, dsl_losses = [], []
        for step in range(3):
            ids = torch.randint(0, vocab, (2, 8), generator=g)
            targets = torch.randint(0, vocab, (2, 8), generator=g)

            ref_logits = ref(ids)[0]
            ref_loss = F.cross_entropy(ref_logits.reshape(-1, vocab),
                                        targets.reshape(-1))
            ref_opt.zero_grad(); ref_loss.backward(); ref_opt.step()
            ref_losses.append(ref_loss.item())

            dsl_logits = dsl(ids)
            dsl_loss = F.cross_entropy(dsl_logits.reshape(-1, vocab),
                                        targets.reshape(-1))
            dsl_opt.zero_grad(); dsl_loss.backward(); dsl_opt.step()
            dsl_losses.append(dsl_loss.item())

        for i, (r, d) in enumerate(zip(ref_losses, dsl_losses)):
            diff = abs(r - d)
            # Tolerance grows after the first step because of accumulated
            # float-precision differences in the synced gradients, but should
            # still be small (≪ 1e-2).
            assert diff < 1e-3, \
                f"step {i+1}: ref_loss={r} dsl_loss={d} (diff {diff})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
