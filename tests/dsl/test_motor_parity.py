# -*- coding: utf-8 -*-
"""DSLMotorCortex ↔ Brain MotorCortex parity tests.

Asserts:
  1. Output shapes match for both no-survival and with-survival paths.
  2. Forward outputs are torch.allclose at atol 1e-6 on every return tensor.
  3. Parameter gradients are torch.allclose at atol 1e-6 — guarding against
     the silent forward-only divergence pattern we hit on loss clipping.
"""
import pytest
import torch
import torch.nn as nn

from neuroslm.modules.motor import MotorCortex
from neuroslm.dsl.subsystems.motor import (
    DSLMotorCortex, sync_from_brain, ACTION_INDEX, N_ACTIONS,
)


def _build_pair(d_action=64, d_sem=128, d_hidden=192):
    torch.manual_seed(0)
    brain = MotorCortex(d_action, d_sem, d_hidden)
    dsl = DSLMotorCortex(d_action, d_sem, d_hidden)
    sync_from_brain(dsl, brain)
    return brain, dsl


class TestForwardParity:
    def test_outputs_match_no_survival(self):
        brain, dsl = _build_pair()
        torch.manual_seed(1)
        x = torch.randn(8, 64)
        b_thought, b_bias, b_idx, b_logits, b_probs = brain(x)
        d_thought, d_bias, d_idx, d_logits, d_probs = dsl(x)
        assert torch.allclose(b_thought, d_thought, atol=1e-6)
        assert torch.allclose(b_bias,    d_bias,    atol=1e-6)
        assert torch.allclose(b_logits,  d_logits,  atol=1e-6)
        assert torch.allclose(b_probs,   d_probs,   atol=1e-6)
        assert torch.equal(b_idx, d_idx)

    def test_outputs_match_with_survival(self):
        brain, dsl = _build_pair()
        torch.manual_seed(2)
        x = torch.randn(8, 64)
        survival = torch.tensor([True, False, True, False, True, False, True, False])
        b_thought, b_bias, b_idx, b_logits, b_probs = brain(x, survival=survival)
        d_thought, d_bias, d_idx, d_logits, d_probs = dsl(x, survival=survival)
        assert torch.allclose(b_logits, d_logits, atol=1e-6)
        assert torch.allclose(b_probs,  d_probs,  atol=1e-6)
        assert torch.equal(b_idx, d_idx)
        # Survival sequences should pick FLEE (idx=4) since override is +5
        for i, surv in enumerate(survival):
            if surv:
                assert b_idx[i].item() == ACTION_INDEX["FLEE"]

    def test_action_bias_init_is_speak(self):
        """SPEAK should be the default action at init (bias=2.0)."""
        _, dsl = _build_pair()
        # With zero-input random weights, the SPEAK bias dominates
        x = torch.zeros(4, 64)
        _, _, idx, logits, probs = dsl(x)
        assert (idx == ACTION_INDEX["SPEAK"]).all()
        # Bias of 2 on SPEAK alone should dominate softmax
        assert (probs[:, ACTION_INDEX["SPEAK"]] > 0.5).all()


class TestGradientParity:
    def test_parameter_grads_match(self):
        """Backward pass must produce identical gradients on every learnable
        parameter — guards against the detach-vs-graph silent-divergence
        pattern documented in the memory ledger."""
        brain, dsl = _build_pair()
        x = torch.randn(8, 64)

        # Sum-of-squares loss across ALL trainable outputs (thought,
        # lang_bias, logits) so EVERY learnable param appears in the graph.
        # Using just `thought` excludes to_lang_bias + action_head — which
        # makes autograd.grad fail with "unused tensors".
        b_thought, b_bias, _, b_logits, _ = brain(x)
        d_thought, d_bias, _, d_logits, _ = dsl(x)
        b_loss = ((b_thought ** 2).mean() + (b_bias ** 2).mean()
                  + (b_logits ** 2).mean())
        d_loss = ((d_thought ** 2).mean() + (d_bias ** 2).mean()
                  + (d_logits ** 2).mean())

        b_grads = torch.autograd.grad(
            b_loss, [brain.proj[0].weight, brain.proj[0].bias,
                     brain.proj[2].weight, brain.proj[2].bias,
                     brain.to_lang_bias.weight, brain.to_lang_bias.bias,
                     brain.action_head.weight, brain.action_head.bias])
        d_grads = torch.autograd.grad(
            d_loss, [dsl.W1, dsl.b1, dsl.W2, dsl.b2,
                     dsl.Wlb, dsl.blb, dsl.Wah, dsl.bah])
        for (bg, dg, name) in zip(b_grads, d_grads,
                                  ["W1","b1","W2","b2","Wlb","blb","Wah","bah"]):
            assert torch.allclose(bg, dg, atol=1e-6), \
                f"grad for {name} diverged: max diff {(bg-dg).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
