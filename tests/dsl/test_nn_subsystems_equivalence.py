# -*- coding: utf-8 -*-
"""Phase N7 — cognitive attention subsystems, exact-match vs Brain.

The attention-level subsystems that make Brain's attention novel:
  * NeuromodulatedScale — NT vector → per-head attention temperature
  * HebbianTrace       — low-rank fast-weight relational memory bias

Each DSL op must be bit-identical to its `neuroslm.modules.neuro_attention`
reference so the full NT-modulated / Hebbian attention path can be
assembled in the DSL with no behavioral drift.
"""
import pytest
import torch

from neuroslm.dsl import nn_ops
from neuroslm.modules.neuro_attention import NeuromodulatedScale, HebbianTrace


class TestNeuromodulatedScale:
    def test_matches_reference(self):
        n_nt, n_heads = 7, 6
        ref = NeuromodulatedScale(n_nt, n_heads)
        with torch.no_grad():
            ref.proj.weight.copy_(torch.randn(n_heads, n_nt))
            ref.proj.bias.copy_(torch.randn(n_heads))

        nt = torch.randn(4, n_nt)
        ref_out = ref(nt)
        dsl_out = nn_ops.neuromod_scale(nt, ref.proj.weight, ref.proj.bias)
        assert dsl_out.shape == (4, n_heads, 1, 1)
        assert torch.allclose(dsl_out, ref_out, atol=1e-6)

    def test_identity_at_zero_init(self):
        # Zero-init proj → scale ≈ 1.0 (attention unmodified). Matches the
        # reference's deliberate identity initialisation.
        ref = NeuromodulatedScale(7, 6)   # zero-init by default
        nt = torch.randn(2, 7)
        out = nn_ops.neuromod_scale(nt, ref.proj.weight, ref.proj.bias)
        assert torch.allclose(out, torch.ones_like(out), atol=1e-3)


class TestHebbianTrace:
    def test_matches_reference(self):
        head_dim, rank = 16, 8
        ref = HebbianTrace(head_dim, rank=rank)
        ref.eval()

        B, H, T = 2, 4, 12
        q = torch.randn(B, H, T, head_dim)
        k = torch.randn(B, H, T, head_dim)

        with torch.no_grad():
            ref_out = ref(q, k)
            dsl_out = nn_ops.hebbian_trace(
                q, k,
                query_proj_w=ref.query_proj.weight,
                key_proj_w=ref.key_proj.weight,
                log_decay=ref.log_decay,
                scale=ref.scale,
            )
        assert dsl_out.shape == (B, H, T, T)
        assert torch.allclose(dsl_out, ref_out, atol=1e-5), \
            f"max diff {(dsl_out - ref_out).abs().max().item()}"

    def test_causal_zero_above_diagonal(self):
        head_dim, rank = 16, 4
        ref = HebbianTrace(head_dim, rank=rank)
        with torch.no_grad():
            ref.scale.copy_(torch.tensor(1.0))   # make the bias non-zero
        B, H, T = 1, 2, 8
        q = torch.randn(B, H, T, head_dim)
        k = torch.randn(B, H, T, head_dim)
        out = nn_ops.hebbian_trace(q, k, ref.query_proj.weight,
                                   ref.key_proj.weight, ref.log_decay, ref.scale)
        # Strictly-upper triangle (future) must be exactly zero
        upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        assert out[..., upper].abs().max() == 0.0


class TestPredictiveCodingHead:
    """N8 aux-loss term — exact-match vs neuro_attention.PredictiveCodingHead."""

    def test_matches_reference(self):
        from neuroslm.modules.neuro_attention import PredictiveCodingHead
        dim = 64
        ref = PredictiveCodingHead(dim)
        # The reference zero-inits both Linear weights → loss starts at the
        # zero-delta baseline. Randomise so we exercise a meaningful path.
        with torch.no_grad():
            ref.pred[0].weight.copy_(torch.randn_like(ref.pred[0].weight) * 0.01)
            ref.pred[2].weight.copy_(torch.randn_like(ref.pred[2].weight) * 0.01)
        h_cur = torch.randn(2, 16, dim)
        h_next = torch.randn(2, 16, dim)
        with torch.no_grad():
            ref_loss = ref(h_cur, h_next)
            dsl_loss = nn_ops.predictive_coding_head(
                h_cur, h_next,
                w1=ref.pred[0].weight, b1=ref.pred[0].bias,
                w2=ref.pred[2].weight,
            )
        assert torch.allclose(dsl_loss, ref_loss, atol=1e-6), \
            f"PCH diverged (ref={ref_loss.item()} dsl={dsl_loss.item()})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
