# -*- coding: utf-8 -*-
"""H60: wiring the MoD router load-balancing aux loss into the harness.

Companion to tests/dsl/test_mod_router_aux_loss.py (which pins the
nn_ops op itself and DSLLanguageCortex's stashing of
``_last_mod_router_aux_loss``). This file pins the LAST leg: the
harness actually composing ``mod_router_aux_weight * aux_loss`` into
the total training loss — the piece a bugfix could ship without ever
touching the loss AdamW actually descends.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


def _make_harness(weight: float, d_sem: int = 8):
    from neuroslm.harness import BRIANHarness
    from neuroslm.dsl.training_config import TrainingConfig

    class _StubLM(nn.Module):
        """Exposes the stash attribute DSLLanguageCortex.forward() sets;
        avoids the full DSL compile pipeline (needs a tokenizer)."""

        def __init__(self, d):
            super().__init__()
            self.proj = nn.Linear(d, d)
            self._last_mod_router_aux_loss = None

    cfg = TrainingConfig()
    cfg.mod_router_aux_weight = weight
    lm = _StubLM(d_sem)
    h = BRIANHarness.from_language_model(
        language_model=lm, vocab_size=257, d_sem=d_sem, training_config=cfg,
    )
    return h


class TestTrainingConfigDefault:
    def test_defaults_to_small_positive(self):
        """Not a new optional feature — a bugfix restoring a mechanism
        (MoD routing) that's already on by default. Off (0.0) would
        silently leave the router dead again."""
        from neuroslm.dsl.training_config import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.mod_router_aux_weight == pytest.approx(0.01)

    def test_arch_neuro_parses_the_field(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config("mod_router_aux_weight: 0.05")
        assert cfg.mod_router_aux_weight == pytest.approx(0.05)


class TestModRouterAuxStep:
    def test_noop_when_weight_zero(self):
        h = _make_harness(weight=0.0)
        h.language_model._last_mod_router_aux_loss = torch.tensor(5.0)
        total = torch.tensor(1.0)
        out = h._mod_router_aux_step(total)
        assert out is total  # untouched, not just numerically equal

    def test_noop_when_stash_is_none(self):
        """block_pattern='standard' -> no ModBlocks -> stash is None."""
        h = _make_harness(weight=0.01)
        h.language_model._last_mod_router_aux_loss = None
        total = torch.tensor(1.0)
        out = h._mod_router_aux_step(total)
        assert out.item() == pytest.approx(1.0)

    def test_adds_weighted_aux_to_total(self):
        h = _make_harness(weight=0.5)
        aux = torch.tensor(2.0, requires_grad=True)
        h.language_model._last_mod_router_aux_loss = aux
        total = torch.tensor(1.0)
        out = h._mod_router_aux_step(total)
        assert out.item() == pytest.approx(1.0 + 0.5 * 2.0)

    def test_gradient_flows_from_total_to_router_params(self):
        """End-to-end: backprop through the composed total loss must
        reach the router weights — this is the actual fix, verified at
        the harness composition boundary, not just the isolated op."""
        from neuroslm.dsl.nn_lang import build_dsl_language_cortex
        from neuroslm.harness import BRIANHarness
        from neuroslm.dsl.training_config import TrainingConfig

        torch.manual_seed(0)
        lm = build_dsl_language_cortex(
            vocab=100, d_model=64, depth=6, n_heads=4, max_ctx=32)
        cfg = TrainingConfig()
        cfg.mod_router_aux_weight = 0.5
        h = BRIANHarness.from_language_model(
            language_model=lm, vocab_size=100, d_sem=64, training_config=cfg,
        )
        ids = torch.randint(0, 100, (2, 16))
        lm(ids)  # populates _last_mod_router_aux_loss
        total = torch.tensor(0.0)
        total = h._mod_router_aux_step(total)
        total.backward()
        mod_blocks = [b for b in lm.blocks if type(b).__name__ == "ModBlock"]
        assert mod_blocks
        assert mod_blocks[0].router_w2.grad is not None
        assert mod_blocks[0].router_w2.grad.abs().sum() > 0

    def test_publishes_metric_when_active(self):
        h = _make_harness(weight=0.5)
        h.language_model._last_mod_router_aux_loss = torch.tensor(3.0)
        h._mod_router_aux_step(torch.tensor(0.0))
        assert h._metrics.get("mod_router_aux_loss") == pytest.approx(3.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
