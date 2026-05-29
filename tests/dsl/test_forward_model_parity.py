# -*- coding: utf-8 -*-
"""DSLForwardModel ↔ Brain ForwardModel parity tests."""
import pytest
import torch

from neuroslm.modules.forward_model import ForwardModel
from neuroslm.dsl.subsystems.forward_model import DSLForwardModel, sync_from_brain


def _build_pair(d_sem=64, d_action=32, n_layers=3):
    torch.manual_seed(0)
    brain = ForwardModel(d_sem, d_action, n_layers)
    dsl = DSLForwardModel(d_sem, d_action, n_layers)
    sync_from_brain(dsl, brain)
    return brain, dsl


class TestForwardModelParity:
    def test_forward_matches(self):
        brain, dsl = _build_pair()
        torch.manual_seed(1)
        z_world = torch.randn(8, 64)
        z_self  = torch.randn(8, 64)
        action  = torch.randn(8, 32)
        b_w, b_s = brain(z_world, z_self, action)
        d_w, d_s = dsl(z_world, z_self, action)
        assert torch.allclose(b_w, d_w, atol=1e-6), \
            f"world_head diverged: max {(b_w-d_w).abs().max()}"
        assert torch.allclose(b_s, d_s, atol=1e-6), \
            f"self_head diverged:  max {(b_s-d_s).abs().max()}"

    def test_gradient_matches(self):
        brain, dsl = _build_pair()
        torch.manual_seed(2)
        z_world = torch.randn(8, 64)
        z_self  = torch.randn(8, 64)
        action  = torch.randn(8, 32)
        b_w, b_s = brain(z_world, z_self, action)
        d_w, d_s = dsl(z_world, z_self, action)
        b_loss = (b_w ** 2).mean() + (b_s ** 2).mean()
        d_loss = (d_w ** 2).mean() + (d_s ** 2).mean()

        # Pair Brain params with their DSL counterparts by name (NOT by
        # .parameters() iteration order — Brain uses nn.Sequential which
        # interleaves W,b per layer, while DSL uses ParameterList which
        # groups Ws then bs).
        pairs = []
        for i in range(dsl.n_layers):
            pairs.append((brain.trunk[i * 2].weight, dsl.Wt[i], f"Wt[{i}]"))
            pairs.append((brain.trunk[i * 2].bias,   dsl.bt[i], f"bt[{i}]"))
        pairs += [
            (brain.world_head.weight, dsl.Ww, "Ww"),
            (brain.world_head.bias,   dsl.bw, "bw"),
            (brain.self_head.weight,  dsl.Ws, "Ws"),
            (brain.self_head.bias,    dsl.bs, "bs"),
        ]
        b_grads = torch.autograd.grad(b_loss, [p[0] for p in pairs])
        d_grads = torch.autograd.grad(d_loss, [p[1] for p in pairs])
        for (bg, dg, (_, _, name)) in zip(b_grads, d_grads, pairs):
            assert torch.allclose(bg, dg, atol=1e-6), \
                f"{name}: grad max diff {(bg-dg).abs().max()}"

    def test_variable_depth(self):
        """Verify n_layers ∈ {1, 3, 5} all match."""
        for n in (1, 3, 5):
            brain, dsl = _build_pair(n_layers=n)
            x_w = torch.randn(4, 64); x_s = torch.randn(4, 64); a = torch.randn(4, 32)
            b_w, b_s = brain(x_w, x_s, a)
            d_w, d_s = dsl(x_w, x_s, a)
            assert torch.allclose(b_w, d_w, atol=1e-6), f"n={n} world diverged"
            assert torch.allclose(b_s, d_s, atol=1e-6), f"n={n} self diverged"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
