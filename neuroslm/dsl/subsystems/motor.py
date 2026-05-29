# -*- coding: utf-8 -*-
"""DSL MotorCortex — bit-identical port of `neuroslm.modules.motor.MotorCortex`.

Brain's MotorCortex maps a chosen BG action embedding to:
  (a) `thought`   — d_sem conditioning vector for the language head
  (b) `lang_bias` — d_hidden additive bias on language cortex
  (c) `action_idx`/`action_logits`/`action_probs` — discrete action choice

All learnable parameters (proj 2-layer MLP, lang-bias linear, action head)
are expressed as DSL params with Brain's exact init (Xavier on Linear,
zero on `to_lang_bias`, zero bias + SPEAK=2.0 on `action_head`). The
forward composes only nn_ops atoms.

The survival-override branch (FLEE forced under threat) is gen-time
only and lives in Python — it never fires during training so it doesn't
need DSL coverage for trunk parity; we still implement it for inference.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.dsl import nn_ops
from neuroslm.dsl.nn_lang import _alloc

ACTION_NAMES = ("SPEAK", "REMAIN_SILENT", "RECALL", "PLAN", "FLEE")
N_ACTIONS = len(ACTION_NAMES)
ACTION_INDEX = {n: i for i, n in enumerate(ACTION_NAMES)}


# The 2-linear thought projection (Linear → GELU → Linear) expressed in DSL.
# Brain's `nn.Sequential(Linear(d_action,d_sem), GELU, Linear(d_sem,d_sem))`
# becomes a pure-DSL layer reading the trainable linears as params.
# Stored as a runtime-buildable string so the test can compile + sync.
_THOUGHT_PROJ_DSL = '''
layer ThoughtProjection(d_action, d_sem) {
    param W1: (d_sem, d_action) init=xavier
    param b1: (d_sem,)           init=zeros
    param W2: (d_sem, d_sem)     init=xavier
    param b2: (d_sem,)           init=zeros
    forward(action_emb) {
        h = gelu(linear(action_emb, W1, b1))
        return linear(h, W2, b2)
    }
}
'''


class DSLMotorCortex(nn.Module):
    """Pure-DSL MotorCortex bit-identical to `modules.motor.MotorCortex`.

    Parameter layout (copy_in/copy_out friendly for parity tests):
        W1, b1    — proj[0]  (Linear d_action → d_sem)
        W2, b2    — proj[2]  (Linear d_sem    → d_sem)
        Wlb, blb  — to_lang_bias (Linear d_action → d_hidden, zero-init)
        Wah, bah  — action_head  (Linear d_action → N_ACTIONS, SPEAK-biased)
    """

    def __init__(self, d_action: int, d_sem: int, d_hidden: Optional[int] = None):
        super().__init__()
        d_hidden = d_hidden or d_sem
        self.d_action = d_action
        self.d_sem = d_sem
        self.d_hidden = d_hidden

        # ── proj (2-layer MLP) — Xavier-init Linears ──
        self.W1 = nn.Parameter(_alloc("xavier", (d_sem, d_action)))
        self.b1 = nn.Parameter(_alloc("zeros",  (d_sem,)))
        self.W2 = nn.Parameter(_alloc("xavier", (d_sem, d_sem)))
        self.b2 = nn.Parameter(_alloc("zeros",  (d_sem,)))

        # ── to_lang_bias — Linear, zero-init weight + bias ──
        self.Wlb = nn.Parameter(_alloc("zeros", (d_hidden, d_action)))
        self.blb = nn.Parameter(_alloc("zeros", (d_hidden,)))

        # ── action_head — Linear, custom bias (SPEAK=2.0, rest 0) ──
        self.Wah = nn.Parameter(_alloc("xavier", (N_ACTIONS, d_action)))
        action_bias = torch.zeros(N_ACTIONS)
        action_bias[ACTION_INDEX["SPEAK"]] = 2.0
        self.bah = nn.Parameter(action_bias)

    def forward(self, action_emb: torch.Tensor,
                survival: Optional[torch.Tensor] = None):
        """Same signature/returns as `modules.motor.MotorCortex.forward`.

        Returns: (thought, lang_bias, action_idx, action_logits, action_probs)
        """
        # proj: Linear → GELU → Linear  (all DSL nn_ops)
        h = nn_ops.gelu(nn_ops.linear(action_emb, self.W1, self.b1))
        thought = nn_ops.linear(h, self.W2, self.b2)

        # to_lang_bias: single Linear
        lang_bias = nn_ops.linear(action_emb, self.Wlb, self.blb)

        # action_head + (optional) survival override
        logits = nn_ops.linear(action_emb, self.Wah, self.bah)
        if survival is not None:
            mask = survival.unsqueeze(-1).to(logits.dtype)
            override = torch.zeros_like(logits)
            override[:, ACTION_INDEX["FLEE"]] = 5.0
            override[:, ACTION_INDEX["REMAIN_SILENT"]] = 1.0
            override[:, ACTION_INDEX["SPEAK"]] = -5.0
            logits = logits * (1 - mask) + override * mask

        probs = nn_ops.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1)
        return thought, lang_bias, idx, logits, probs


# ── Weight-sync helper for parity tests ─────────────────────────────────

def sync_from_brain(dsl: DSLMotorCortex, brain_motor) -> None:
    """Copy parameters from a `modules.motor.MotorCortex` into a `DSLMotorCortex`.

    Used by the parity test to put both models in identical state before
    asserting forward + gradient equality.
    """
    with torch.no_grad():
        # proj is nn.Sequential(Linear, GELU, Linear) — indices 0 and 2
        dsl.W1.copy_(brain_motor.proj[0].weight)
        dsl.b1.copy_(brain_motor.proj[0].bias)
        dsl.W2.copy_(brain_motor.proj[2].weight)
        dsl.b2.copy_(brain_motor.proj[2].bias)
        dsl.Wlb.copy_(brain_motor.to_lang_bias.weight)
        dsl.blb.copy_(brain_motor.to_lang_bias.bias)
        dsl.Wah.copy_(brain_motor.action_head.weight)
        dsl.bah.copy_(brain_motor.action_head.bias)
