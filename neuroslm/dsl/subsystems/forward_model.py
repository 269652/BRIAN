# -*- coding: utf-8 -*-
"""DSL ForwardModel — bit-identical port of `neuroslm.modules.forward_model.ForwardModel`.

Cerebellum-like next-state predictor: takes `(z_world, z_self, action)`,
runs an n-layer MLP trunk (Linear→GELU repeated), then forks into two
parallel output heads (`world_head`, `self_head`). Pure-MLP; no state.

Parameter layout (sync-friendly):
    Wt[i], bt[i]   — trunk layer i (i in 0..n_layers-1)
    Ww, bw         — world_head
    Ws, bs         — self_head
"""
from __future__ import annotations

import torch
import torch.nn as nn

from neuroslm.dsl import nn_ops
from neuroslm.dsl.nn_lang import _alloc


class DSLForwardModel(nn.Module):
    def __init__(self, d_sem: int, d_action: int, n_layers: int):
        super().__init__()
        self.d_sem = d_sem
        self.d_action = d_action
        self.n_layers = n_layers
        in_dim = d_sem * 2 + d_action
        out_dim = d_sem * 2   # every trunk layer outputs 2*d_sem

        # Variable-depth trunk — Linear(cur,out) → GELU repeated. Layer 0
        # consumes in_dim; layers 1+ consume out_dim. Matches Brain's
        # `nn.Sequential` construction loop in `ForwardModel.__init__`.
        self.Wt = nn.ParameterList()
        self.bt = nn.ParameterList()
        cur = in_dim
        for _ in range(n_layers):
            self.Wt.append(nn.Parameter(_alloc("xavier", (out_dim, cur))))
            self.bt.append(nn.Parameter(_alloc("zeros",  (out_dim,))))
            cur = out_dim

        # Two parallel output heads — Linear, default Xavier init.
        self.Ww = nn.Parameter(_alloc("xavier", (d_sem, cur)))
        self.bw = nn.Parameter(_alloc("zeros",  (d_sem,)))
        self.Ws = nn.Parameter(_alloc("xavier", (d_sem, cur)))
        self.bs = nn.Parameter(_alloc("zeros",  (d_sem,)))

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor,
                action: torch.Tensor):
        # cat → trunk → split heads.  All ops are nn_ops atoms.
        x = torch.cat([z_world, z_self, action], dim=-1)
        h = x
        for W, b in zip(self.Wt, self.bt):
            h = nn_ops.gelu(nn_ops.linear(h, W, b))
        return (nn_ops.linear(h, self.Ww, self.bw),
                nn_ops.linear(h, self.Ws, self.bs))


def sync_from_brain(dsl: DSLForwardModel, brain_fm) -> None:
    """Copy params from `modules.forward_model.ForwardModel` into a DSL copy.

    Brain's trunk is `nn.Sequential(Linear, GELU, Linear, GELU, ...)` —
    Linears live at even indices (0, 2, 4, ...).
    """
    with torch.no_grad():
        for i in range(dsl.n_layers):
            lin = brain_fm.trunk[i * 2]
            dsl.Wt[i].copy_(lin.weight)
            dsl.bt[i].copy_(lin.bias)
        dsl.Ww.copy_(brain_fm.world_head.weight)
        dsl.bw.copy_(brain_fm.world_head.bias)
        dsl.Ws.copy_(brain_fm.self_head.weight)
        dsl.bs.copy_(brain_fm.self_head.bias)
