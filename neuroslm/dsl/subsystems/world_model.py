# -*- coding: utf-8 -*-
"""DSL RSSM — bit-identical port of `neuroslm.modules.world_model.RecurrentStateSpaceModel`.

Dreamer-V3 style world model with:
  • Deterministic path: GRU(h_{t-1}, proj([z_{t-1}; x_t]))
  • Stochastic prior:    z ~ Categorical(f_prior(h_t))
  • Stochastic posterior: z ~ Categorical(f_post(h_t, x_t))
  • Straight-through gradient through one-hot z
  • KL(posterior || prior) loss with free-nats clamp

`nn.GRU` is used as a primitive (the same way DSL uses `nn.Linear` —
it's a black-box atom; the DSL parity claim is "same params, same forward,
same gradient", which holds when both sides call the same GRU op on the
same input). Everything else is pure nn_ops composition.

The categorical sampler uses argmax + straight-through, so it's RNG-free
in training mode → no seed-sync needed for parity.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.dsl import nn_ops
from neuroslm.dsl.nn_lang import _alloc


class DSLCategoricalStraightThrough(nn.Module):
    """Bit-identical clone of `world_model.CategoricalStraightThrough`."""

    def __init__(self, n_cats: int, d_cat: int):
        super().__init__()
        self.n_cats = n_cats
        self.d_cat = d_cat

    def forward(self, logits: torch.Tensor, straight_through: bool = True):
        B = logits.shape[0]
        lg = logits.view(B, self.n_cats, self.d_cat)
        probs = nn_ops.softmax(lg, dim=-1)
        if self.training or straight_through:
            indices = probs.detach().argmax(-1)
            one_hot = F.one_hot(indices, self.d_cat).to(probs.dtype)
            sample = one_hot + probs - probs.detach()
        else:
            sample = probs
        sample = sample.view(B, self.n_cats * self.d_cat)
        return sample, probs


class DSLRecurrentStateSpaceModel(nn.Module):
    """DSL RSSM — same architecture, same params, bit-identical forward."""

    def __init__(self, d_sem: int, d_hidden: int, n_layers: int = 2,
                 n_cats: int = 8, d_cat: int = 16):
        super().__init__()
        self.d_sem = d_sem
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.n_cats = n_cats
        self.d_cat = d_cat
        self.d_z = n_cats * d_cat
        self.d_state = d_hidden + self.d_z

        # inp_proj: Linear(d_sem + d_z → d_hidden)
        self.Wip = nn.Parameter(_alloc("xavier", (d_hidden, d_sem + self.d_z)))
        self.bip = nn.Parameter(_alloc("zeros",  (d_hidden,)))

        # GRU — keep as primitive (same reason DSL uses nn.Linear as atom)
        self.rnn = nn.GRU(d_hidden, d_hidden, num_layers=n_layers,
                          batch_first=True)

        # prior_net: Linear → ELU → Linear  (d_hidden → d_hidden → d_z)
        self.Wp1 = nn.Parameter(_alloc("xavier", (d_hidden, d_hidden)))
        self.bp1 = nn.Parameter(_alloc("zeros",  (d_hidden,)))
        self.Wp2 = nn.Parameter(_alloc("xavier", (self.d_z, d_hidden)))
        self.bp2 = nn.Parameter(_alloc("zeros",  (self.d_z,)))

        # post_net: Linear → ELU → Linear  (d_hidden + d_sem → d_hidden → d_z)
        self.Wq1 = nn.Parameter(_alloc("xavier", (d_hidden, d_hidden + d_sem)))
        self.bq1 = nn.Parameter(_alloc("zeros",  (d_hidden,)))
        self.Wq2 = nn.Parameter(_alloc("xavier", (self.d_z, d_hidden)))
        self.bq2 = nn.Parameter(_alloc("zeros",  (self.d_z,)))

        # categorical sampler
        self.categorical = DSLCategoricalStraightThrough(n_cats, d_cat)

        # decoder: Linear → ELU → Linear  (d_state → d_hidden → d_sem)
        self.Wd1 = nn.Parameter(_alloc("xavier", (d_hidden, self.d_state)))
        self.bd1 = nn.Parameter(_alloc("zeros",  (d_hidden,)))
        self.Wd2 = nn.Parameter(_alloc("xavier", (d_sem, d_hidden)))
        self.bd2 = nn.Parameter(_alloc("zeros",  (d_sem,)))

        # world_proj: Linear(d_state → d_sem)
        self.Wwp = nn.Parameter(_alloc("xavier", (d_sem, self.d_state)))
        self.bwp = nn.Parameter(_alloc("zeros",  (d_sem,)))

        # predict_head: Linear(d_state → d_sem)
        self.Wph = nn.Parameter(_alloc("xavier", (d_sem, self.d_state)))
        self.bph = nn.Parameter(_alloc("zeros",  (d_sem,)))

    def init_state(self, batch_size: int, device, dtype=None) -> dict:
        if dtype is None:
            dtype = self.Wip.dtype
        return {
            "h": torch.zeros(self.n_layers, batch_size, self.d_hidden,
                             device=device, dtype=dtype),
            "z": torch.zeros(batch_size, self.d_z, device=device, dtype=dtype),
        }

    def forward(self, sensory: torch.Tensor,
                state: Optional[dict] = None
                ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        B = sensory.shape[0]
        if state is None:
            state = self.init_state(B, sensory.device)
        h_prev, z_prev = state["h"], state["z"]

        # Cast to weight dtype — match Brain's behavior
        w_dtype = self.Wip.dtype
        sensory = sensory.to(dtype=w_dtype)
        z_prev = z_prev.to(dtype=w_dtype)

        # 1. Deterministic GRU update
        inp = torch.cat([sensory, z_prev], dim=-1)
        inp = nn_ops.linear(inp, self.Wip, self.bip).unsqueeze(1)
        _, h_new = self.rnn(inp, h_prev)
        h_top = h_new[-1]

        # 2. Posterior z ~ q(z|h,x)
        post_inp = torch.cat([h_top, sensory], dim=-1)
        post_h = F.elu(nn_ops.linear(post_inp, self.Wq1, self.bq1))
        post_logits = nn_ops.linear(post_h, self.Wq2, self.bq2)
        z_post, post_probs = self.categorical(post_logits)

        # 3. Prior p(z|h) (for KL)
        prior_h = F.elu(nn_ops.linear(h_top, self.Wp1, self.bp1))
        prior_logits = nn_ops.linear(prior_h, self.Wp2, self.bp2)
        _, prior_probs = self.categorical(prior_logits, straight_through=False)

        # 4. World state and outputs
        world_state = torch.cat([h_top, z_post], dim=-1)
        z_world = nn_ops.linear(world_state, self.Wwp, self.bwp)
        pred_next = nn_ops.linear(world_state, self.Wph, self.bph)

        new_state = {"h": h_new, "z": z_post.detach(),
                     "_prior_probs": prior_probs, "_post_probs": post_probs,
                     "_world_state": world_state}
        return z_world, new_state, pred_next

    @staticmethod
    def kl_loss(state: dict, free_nats: float = 1.0) -> torch.Tensor:
        """Bit-identical to RecurrentStateSpaceModel.kl_loss."""
        prior_probs = state.get("_prior_probs")
        post_probs = state.get("_post_probs")
        if prior_probs is None or post_probs is None:
            return torch.tensor(0.0)
        eps = 1e-8
        kl = (post_probs * ((post_probs + eps).log()
                            - (prior_probs + eps).log())).sum(-1).sum(-1)
        return torch.mean(torch.clamp(kl, min=free_nats))


def sync_from_brain(dsl: DSLRecurrentStateSpaceModel, brain_rssm) -> None:
    """Copy every learnable param from a Brain RSSM into the DSL one.

    The GRU weights are shared by reference-copy since both sides use
    `nn.GRU`. Everything else copies tensor-by-tensor.
    """
    with torch.no_grad():
        # inp_proj
        dsl.Wip.copy_(brain_rssm.inp_proj.weight)
        dsl.bip.copy_(brain_rssm.inp_proj.bias)
        # GRU — copy all GRU param tensors (flat parameter layout)
        for (dst_p, src_p) in zip(dsl.rnn.parameters(), brain_rssm.rnn.parameters()):
            dst_p.copy_(src_p)
        # prior_net
        dsl.Wp1.copy_(brain_rssm.prior_net[0].weight)
        dsl.bp1.copy_(brain_rssm.prior_net[0].bias)
        dsl.Wp2.copy_(brain_rssm.prior_net[2].weight)
        dsl.bp2.copy_(brain_rssm.prior_net[2].bias)
        # post_net
        dsl.Wq1.copy_(brain_rssm.post_net[0].weight)
        dsl.bq1.copy_(brain_rssm.post_net[0].bias)
        dsl.Wq2.copy_(brain_rssm.post_net[2].weight)
        dsl.bq2.copy_(brain_rssm.post_net[2].bias)
        # decoder
        dsl.Wd1.copy_(brain_rssm.decoder[0].weight)
        dsl.bd1.copy_(brain_rssm.decoder[0].bias)
        dsl.Wd2.copy_(brain_rssm.decoder[2].weight)
        dsl.bd2.copy_(brain_rssm.decoder[2].bias)
        # world_proj
        dsl.Wwp.copy_(brain_rssm.world_proj.weight)
        dsl.bwp.copy_(brain_rssm.world_proj.bias)
        # predict_head
        dsl.Wph.copy_(brain_rssm.predict_head.weight)
        dsl.bph.copy_(brain_rssm.predict_head.bias)
