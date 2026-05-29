# -*- coding: utf-8 -*-
"""DSL RSSM ↔ Brain RSSM bit-identical parity."""
import pytest
import torch

from neuroslm.modules.world_model import RecurrentStateSpaceModel
from neuroslm.dsl.subsystems.world_model import (
    DSLRecurrentStateSpaceModel, sync_from_brain,
)


def _build_pair(d_sem=64, d_hidden=128, n_layers=2, n_cats=8, d_cat=16):
    torch.manual_seed(0)
    brain = RecurrentStateSpaceModel(d_sem, d_hidden, n_layers, n_cats, d_cat)
    dsl = DSLRecurrentStateSpaceModel(d_sem, d_hidden, n_layers, n_cats, d_cat)
    sync_from_brain(dsl, brain)
    brain.eval(); dsl.eval()
    return brain, dsl


class TestRSSMParity:
    def test_forward_matches_observation_mode(self):
        brain, dsl = _build_pair()
        torch.manual_seed(1)
        x = torch.randn(4, 64)
        b_zw, b_st, b_pn = brain(x)
        d_zw, d_st, d_pn = dsl(x)
        assert torch.allclose(b_zw, d_zw, atol=1e-5), \
            f"z_world max diff {(b_zw-d_zw).abs().max()}"
        assert torch.allclose(b_pn, d_pn, atol=1e-5), \
            f"pred_next max diff {(b_pn-d_pn).abs().max()}"
        assert torch.allclose(b_st["h"], d_st["h"], atol=1e-5), "h diverged"
        assert torch.allclose(b_st["z"], d_st["z"], atol=1e-5), "z diverged"
        assert torch.allclose(b_st["_prior_probs"], d_st["_prior_probs"], atol=1e-5)
        assert torch.allclose(b_st["_post_probs"],  d_st["_post_probs"],  atol=1e-5)

    def test_kl_loss_matches(self):
        brain, dsl = _build_pair()
        torch.manual_seed(2)
        x = torch.randn(4, 64)
        _, b_st, _ = brain(x)
        _, d_st, _ = dsl(x)
        b_kl = brain.kl_loss(b_st)
        d_kl = dsl.kl_loss(d_st)
        assert torch.allclose(b_kl, d_kl, atol=1e-5)

    def test_gradient_matches(self):
        brain, dsl = _build_pair()
        brain.train(); dsl.train()
        torch.manual_seed(3)
        x = torch.randn(4, 64)
        b_zw, b_st, b_pn = brain(x)
        d_zw, d_st, d_pn = dsl(x)
        # Use BOTH outputs to ensure every param participates
        b_loss = (b_zw ** 2).mean() + (b_pn ** 2).mean() + brain.kl_loss(b_st)
        d_loss = (d_zw ** 2).mean() + (d_pn ** 2).mean() + dsl.kl_loss(d_st)
        assert torch.allclose(b_loss, d_loss, atol=1e-5)

        # Compare a representative subset of params (full list is order-different)
        pairs = [
            (brain.inp_proj.weight,    dsl.Wip,  "Wip"),
            (brain.prior_net[0].weight, dsl.Wp1, "Wp1"),
            (brain.prior_net[2].weight, dsl.Wp2, "Wp2"),
            (brain.post_net[0].weight,  dsl.Wq1, "Wq1"),
            (brain.post_net[2].weight,  dsl.Wq2, "Wq2"),
            (brain.world_proj.weight,   dsl.Wwp, "Wwp"),
            (brain.predict_head.weight, dsl.Wph, "Wph"),
        ]
        b_grads = torch.autograd.grad(b_loss, [p[0] for p in pairs],
                                      retain_graph=True)
        d_grads = torch.autograd.grad(d_loss, [p[1] for p in pairs],
                                      retain_graph=True)
        for (bg, dg, (_, _, name)) in zip(b_grads, d_grads, pairs):
            assert torch.allclose(bg, dg, atol=1e-5), \
                f"{name}: grad max diff {(bg-dg).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
