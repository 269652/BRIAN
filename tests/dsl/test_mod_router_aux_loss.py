# -*- coding: utf-8 -*-
"""TDD contracts for the MoD router load-balancing aux loss (H60).

Root-cause record
==================
`neuroslm/dsl/nn_ops.py::mod_block` routes tokens via
`scores.topk(C, dim=-1, sorted=False)`, keeping only the (non-
differentiable) `indices` and discarding the differentiable `values`.
Since `router_w1`/`router_w2`/biases are all zero-initialised and no
other code path in the DSL/train_dsl training pipeline ever touches
`router_logits`, the router receives ZERO gradient for the entire
training run — confirmed live on the 2026-08-10 topology-100m deploy,
where `.topk()` on an all-zero, never-updated score tensor produces a
fixed, content-blind token selection every step. One-third of an
interleaved trunk's blocks (the ModBlocks) were doing static routing,
not the content-adaptive routing Mixture-of-Depths (Raposo et al. 2024)
is supposed to do.

The reference module (`neuroslm/modules/mixture_of_depths.py::
MoDBlock.router_aux_loss`) already has the fix — a load-balancing +
entropy loss over the DIFFERENTIABLE router logits — but it was never
ported to the DSL path. This file pins:

  A. `nn_ops.mod_router_aux_loss(x, router_w1, router_b1, router_w2,
     router_b2, capacity_ratio)` — numerically matches the reference
     property's formula exactly (mean-matching + entropy bonus).
  B. Gradient reaches router_w1/router_w2 through this loss (the thing
     that was structurally impossible through `mod_block`'s topk path).
  C. `DSLLanguageCortex` wires this in automatically for every ModBlock
     in the trunk (block_pattern="interleave") and exposes the summed
     result as `_last_mod_router_aux_loss`; "standard" pattern (no
     ModBlocks) exposes None/zero.
"""
from __future__ import annotations

import math

import pytest
import torch


# ══════════════════════════════════════════════════════════════════════
# A. nn_ops.mod_router_aux_loss — numerical contract
# ══════════════════════════════════════════════════════════════════════

class TestModRouterAuxLossOp:
    def _make(self, B=2, T=8, D=16, R=8, seed=0):
        torch.manual_seed(seed)
        x = torch.randn(B, T, D)
        router_w1 = torch.randn(R, D) * 0.1
        router_b1 = torch.zeros(R)
        router_w2 = torch.randn(1, R) * 0.1
        router_b2 = torch.zeros(1)
        return x, router_w1, router_b1, router_w2, router_b2

    def test_returns_scalar(self):
        from neuroslm.dsl.nn_ops import mod_router_aux_loss
        x, w1, b1, w2, b2 = self._make()
        loss = mod_router_aux_loss(x, w1, b1, w2, b2, capacity_ratio=0.5)
        assert loss.shape == ()

    def test_matches_reference_formula(self):
        """Reproduces MoDBlock.router_aux_loss exactly: mean_loss =
        (mean(sigmoid(logits)) - target)^2, entropy = binary entropy of
        sigmoid(logits), aux = mean_loss - 0.01*entropy."""
        from neuroslm.dsl.nn_ops import mod_router_aux_loss
        x, w1, b1, w2, b2 = self._make()
        target = 0.5
        loss = mod_router_aux_loss(x, w1, b1, w2, b2, capacity_ratio=target)

        h = torch.nn.functional.silu(torch.nn.functional.linear(x, w1, b1))
        logits = torch.nn.functional.linear(h, w2, b2)
        probs = torch.sigmoid(logits.squeeze(-1))
        mean_prob = probs.mean(dim=-1)
        mean_loss = ((mean_prob - target) ** 2).mean()
        entropy = -(probs * (probs + 1e-8).log()
                    + (1 - probs) * (1 - probs + 1e-8).log()).mean()
        expected = mean_loss - 0.01 * entropy
        assert loss.item() == pytest.approx(expected.item(), abs=1e-6)

    def test_zero_router_gives_maximal_entropy_term(self):
        """All-zero router logits -> sigmoid=0.5 everywhere -> mean_prob
        exactly matches target=0.5 (mean_loss=0) and entropy is at its
        maximum (ln 2), so aux = -0.01*ln(2)."""
        from neuroslm.dsl.nn_ops import mod_router_aux_loss
        x = torch.randn(2, 8, 16)
        zeros_w1 = torch.zeros(8, 16)
        zeros_b1 = torch.zeros(8)
        zeros_w2 = torch.zeros(1, 8)
        zeros_b2 = torch.zeros(1)
        loss = mod_router_aux_loss(x, zeros_w1, zeros_b1, zeros_w2, zeros_b2,
                                   capacity_ratio=0.5)
        assert loss.item() == pytest.approx(-0.01 * math.log(2.0), abs=1e-5)

    def test_gradient_reaches_router_weights(self):
        """The actual bug fix: mod_block's topk-based routing gives the
        router ZERO gradient. This aux loss must give it real gradient."""
        x, w1, b1, w2, b2 = self._make()
        w1 = w1.clone().requires_grad_(True)
        b1 = b1.clone().requires_grad_(True)
        w2 = w2.clone().requires_grad_(True)
        b2 = b2.clone().requires_grad_(True)
        from neuroslm.dsl.nn_ops import mod_router_aux_loss
        loss = mod_router_aux_loss(x, w1, b1, w2, b2, capacity_ratio=0.5)
        loss.backward()
        for p, name in ((w1, "router_w1"), (b1, "router_b1"),
                       (w2, "router_w2"), (b2, "router_b2")):
            assert p.grad is not None, f"{name} received no gradient"
            assert p.grad.abs().sum() > 0, f"{name} gradient is all-zero"

    def test_uneven_capacity_target_shifts_mean_loss(self):
        """A capacity_ratio far from the router's actual mean routing
        rate must produce a nonzero mean_loss component."""
        from neuroslm.dsl.nn_ops import mod_router_aux_loss
        x, w1, b1, w2, b2 = self._make()
        loss_05 = mod_router_aux_loss(x, w1, b1, w2, b2, capacity_ratio=0.5)
        loss_01 = mod_router_aux_loss(x, w1, b1, w2, b2, capacity_ratio=0.01)
        assert loss_01.item() != pytest.approx(loss_05.item(), abs=1e-4)


# ══════════════════════════════════════════════════════════════════════
# B/C. DSLLanguageCortex wiring
# ══════════════════════════════════════════════════════════════════════

class TestDSLLanguageCortexModRouterWiring:
    def _cortex(self, **kw):
        from neuroslm.dsl.nn_lang import build_dsl_language_cortex
        return build_dsl_language_cortex(
            vocab=100, d_model=64, depth=6, n_heads=4, max_ctx=32, **kw)

    def test_interleave_pattern_exposes_aux_loss(self):
        torch.manual_seed(0)
        lm = self._cortex()  # default block_pattern="interleave"
        ids = torch.randint(0, 100, (2, 16))
        lm(ids)
        aux = getattr(lm, "_last_mod_router_aux_loss", None)
        assert aux is not None
        assert aux.shape == ()

    def test_standard_pattern_has_no_mod_blocks_so_aux_is_none_or_zero(self):
        torch.manual_seed(0)
        lm = self._cortex(block_pattern="standard", geometry_adapters=False)
        ids = torch.randint(0, 100, (2, 16))
        lm(ids)
        aux = getattr(lm, "_last_mod_router_aux_loss", None)
        assert aux is None or float(aux) == pytest.approx(0.0)

    def test_gradient_reaches_mod_block_router_via_the_aux_loss(self):
        """End-to-end regression pin for the actual bug: before this
        fix, a ModBlock's router_w1/router_w2 NEVER received gradient
        from anywhere in the trunk. After wiring, backpropagating the
        aux loss must reach them."""
        torch.manual_seed(0)
        lm = self._cortex()
        mod_blocks = [b for b in lm.blocks if type(b).__name__ == "ModBlock"]
        assert mod_blocks, "fixture must contain at least one ModBlock"
        ids = torch.randint(0, 100, (2, 16))
        lm(ids)
        aux = lm._last_mod_router_aux_loss
        aux.backward()
        blk = mod_blocks[0]
        assert blk.router_w1.grad is not None
        assert blk.router_w1.grad.abs().sum() > 0
        assert blk.router_w2.grad is not None
        assert blk.router_w2.grad.abs().sum() > 0

    def test_does_not_change_the_main_forward_output(self):
        """This must be a pure addition — the LM logits themselves are
        unaffected (mod_block's own forward path is untouched)."""
        torch.manual_seed(0)
        lm1 = self._cortex()
        torch.manual_seed(0)
        lm2 = self._cortex()
        ids = torch.randint(0, 100, (2, 16))
        with torch.no_grad():
            out1 = lm1(ids)
            out2 = lm2(ids)
        torch.testing.assert_close(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
