# -*- coding: utf-8 -*-
"""Phase N2 — causal self-attention, exact-match vs common.CausalSelfAttention.

This is the keystone op: the thing that makes a model a *language* model.
The DSL `causal_self_attention` must reproduce Brain's attention bit-for-
bit on the base path (RoPE + QK-norm + GQA + causal SDPA), so a DSL-built
transformer is identical to the reference.

NT-modulated temperature and Hebbian traces are conditional branches
(n_nt>0 / hebbian_rank>0) handled in a later phase; here we pin the base
path the trunk uses by default (n_nt=0, hebbian_rank=0, dropout=0).
"""
import pytest
import torch

from neuroslm.dsl import nn_ops
from neuroslm.modules.common import CausalSelfAttention


def _sync_attention_weights(ref: CausalSelfAttention):
    """Return the weights of a reference attention block for the DSL op."""
    return {
        "q_weight": ref.q_proj.weight,
        "kv_weight": ref.kv_proj.weight,
        "out_weight": ref.out.weight,
    }


class TestCausalSelfAttentionMHA:
    """Plain multi-head attention (n_kv_heads == n_heads)."""

    def test_matches_reference(self):
        dim, n_heads, max_ctx = 64, 8, 128
        ref = CausalSelfAttention(dim, n_heads, max_ctx,
                                  n_nt=0, hebbian_rank=0, dropout=0.0)
        ref.eval()

        x = torch.randn(2, 16, dim)
        with torch.no_grad():
            ref_out = ref(x)
            dsl_out = nn_ops.causal_self_attention(
                x,
                q_weight=ref.q_proj.weight,
                kv_weight=ref.kv_proj.weight,
                out_weight=ref.out.weight,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                max_ctx=max_ctx,
            )
        assert torch.allclose(dsl_out, ref_out, atol=1e-5), \
            f"max diff {(dsl_out - ref_out).abs().max().item()}"


class TestCausalSelfAttentionGQA:
    """Grouped-query attention (n_kv_heads < n_heads)."""

    def test_matches_reference_gqa(self):
        dim, n_heads, n_kv_heads, max_ctx = 64, 8, 2, 128
        ref = CausalSelfAttention(dim, n_heads, max_ctx,
                                  n_kv_heads=n_kv_heads,
                                  n_nt=0, hebbian_rank=0, dropout=0.0)
        ref.eval()

        x = torch.randn(2, 16, dim)
        with torch.no_grad():
            ref_out = ref(x)
            dsl_out = nn_ops.causal_self_attention(
                x,
                q_weight=ref.q_proj.weight,
                kv_weight=ref.kv_proj.weight,
                out_weight=ref.out.weight,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                max_ctx=max_ctx,
            )
        assert torch.allclose(dsl_out, ref_out, atol=1e-5), \
            f"max diff {(dsl_out - ref_out).abs().max().item()}"


class TestDifferentialAttention:
    """N8 keystone — bit-identical to DiffTransformerBlock's attention."""

    def test_matches_reference_no_nt(self):
        from neuroslm.modules.differential_attention import DifferentialAttention
        dim, n_heads, max_ctx = 64, 8, 64
        ref = DifferentialAttention(dim, n_heads, max_ctx, n_nt=0)
        ref.eval()

        x = torch.randn(2, 16, dim)
        with torch.no_grad():
            ref_out = ref(x)
            dsl_out = nn_ops.differential_attention(
                x,
                q_weight=ref.q_proj.weight,
                kv_weight=ref.kv_proj.weight,
                out_weight=ref.out.weight,
                lambda_init=ref.lambda_init,
                sub_norm_weight=ref.sub_norm.weight,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                max_ctx=max_ctx,
            )
        diff = (dsl_out - ref_out).abs().max().item()
        assert diff < 1e-5, f"diff-attn diverged (max diff {diff})"


class TestCausality:
    """A future token must not influence an earlier one's output."""

    def test_causal_mask_respected(self):
        dim, n_heads, max_ctx = 32, 4, 64
        ref = CausalSelfAttention(dim, n_heads, max_ctx,
                                  n_nt=0, hebbian_rank=0, dropout=0.0)
        ref.eval()
        x = torch.randn(1, 10, dim)
        with torch.no_grad():
            out1 = nn_ops.causal_self_attention(
                x, ref.q_proj.weight, ref.kv_proj.weight, ref.out.weight,
                n_heads=n_heads, n_kv_heads=n_heads, max_ctx=max_ctx)
            # Perturb the LAST token; outputs for earlier tokens must be unchanged
            x2 = x.clone()
            x2[:, -1] += 5.0
            out2 = nn_ops.causal_self_attention(
                x2, ref.q_proj.weight, ref.kv_proj.weight, ref.out.weight,
                n_heads=n_heads, n_kv_heads=n_heads, max_ctx=max_ctx)
        assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
