# -*- coding: utf-8 -*-
"""DSL Brain aggregator ↔ Brain formula bit-identical parity.

The aggregator IS Brain's brain.py:1802-1824 expression, refactored into
a reusable function. The reference function `brain_reference_total`
writes the formula out literally — they must produce torch.allclose
results for every (mat, weights, loss bundle) combination, AND the
gradient through any aux input must match between the two.
"""
import pytest
import torch

from neuroslm.dsl.brain_aggregator import (
    aggregate_total_loss, brain_reference_total, LossBundle,
)
from neuroslm.dsl.maturity import (
    TotalLossConfig, AuxWeights, MaturityTracker,
)
from neuroslm.dsl.subsystems.orchestrator_adapter import (
    MockOrchestrator, OrchestratorMetrics,
)


def _bundle(mat_seed=0):
    """A random LossBundle with every component populated."""
    g = torch.Generator().manual_seed(mat_seed)
    def s(): return torch.rand((), generator=g) * 3.0   # 0..3 like real losses
    return LossBundle(
        lm_loss=s(),
        pred_coding=s(),
        world=s(),
        forward=s(),
        motor=s(),
        kl_world=s(),
        novel=s(),
        cpc=s(),
        phi=s() - 1.0,   # phi can be slightly negative in practice
        orchestrator=OrchestratorMetrics(
            identity_drift=s() * 0.1,
            neural_calm=torch.tensor(0.8),
        ),
    )


class TestAggregatorFormulaParity:
    def test_full_bundle_at_low_maturity(self):
        bundle = _bundle(0)
        cfg = TotalLossConfig()
        for mat in (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0):
            got = aggregate_total_loss(bundle, mat, cfg)
            ref = brain_reference_total(bundle, mat, cfg)
            assert torch.allclose(got, ref, atol=1e-6), \
                f"mat={mat}: got={got.item()} ref={ref.item()}"

    def test_partial_bundle_only_lm(self):
        """No aux losses → total == w_lm * lm_loss exactly."""
        b = LossBundle(lm_loss=torch.tensor(3.5))
        for mat in (0.0, 0.5, 1.0):
            assert torch.allclose(aggregate_total_loss(b, mat),
                                  brain_reference_total(b, mat), atol=1e-9)
            # Also: with default w_lm=1.0, total must equal lm exactly
            assert torch.allclose(aggregate_total_loss(b, mat),
                                  torch.tensor(3.5), atol=1e-9)

    def test_lm_only_with_master_scale(self):
        """master_scale change must not affect the lm_loss term."""
        b = LossBundle(lm_loss=torch.tensor(2.0))
        cfg = TotalLossConfig(aux=AuxWeights(master_scale=0.5))
        for mat in (0.0, 0.5, 1.0):
            assert torch.allclose(aggregate_total_loss(b, mat, cfg),
                                  torch.tensor(2.0), atol=1e-9)

    def test_pred_coding_weighting_matches_formula(self):
        """Pure PCH contribution must equal aux_w * ph_pred * w_pred * pch."""
        from neuroslm.dsl.maturity import phase_gate
        b = LossBundle(lm_loss=torch.tensor(0.0),
                        pred_coding=torch.tensor(2.0))
        for mat in (0.0, 0.35, 0.5, 1.0):
            got = aggregate_total_loss(b, mat)
            # Formula: 1.0 (aux_w) * ph(mat, 0.35, 0.08) * 0.10 (w_pred) * 2.0
            expected = phase_gate(mat, 0.35, 0.08) * 0.10 * 2.0
            assert abs(got.item() - expected) < 1e-7, \
                f"mat={mat}: got={got.item()} expected={expected}"

    def test_orchestrator_contribution(self):
        """`0.01*id_drift + 0.01*(1-calm)` scaled by aux_w."""
        b = LossBundle(
            lm_loss=torch.tensor(0.0),
            orchestrator=OrchestratorMetrics(
                identity_drift=torch.tensor(0.5),
                neural_calm=torch.tensor(0.3)),
        )
        # 1.0 * (0.01*0.5 + 0.01*0.7) = 0.012
        got = aggregate_total_loss(b, mat=0.5)
        assert abs(got.item() - 0.012) < 1e-7

    def test_neutral_orchestrator_zero_contribution(self):
        """id_drift=0, calm=1 → orchestrator term is exactly 0."""
        b = LossBundle(
            lm_loss=torch.tensor(1.0),
            orchestrator=OrchestratorMetrics.neutral(),
        )
        assert torch.allclose(aggregate_total_loss(b, 0.5),
                              torch.tensor(1.0), atol=1e-9)


class TestAggregatorGradientParity:
    """Gradient through aux losses must match the reference formula."""

    def test_gradient_through_lm_loss(self):
        lm = torch.tensor(2.5, requires_grad=True)
        pch = torch.tensor(1.5, requires_grad=True)
        b = LossBundle(lm_loss=lm, pred_coding=pch)
        agg_total = aggregate_total_loss(b, mat=0.5)
        ref_total = brain_reference_total(b, mat=0.5)
        assert torch.allclose(agg_total, ref_total, atol=1e-7)
        g_agg_lm, g_agg_pch = torch.autograd.grad(agg_total, [lm, pch],
                                                  retain_graph=True)
        g_ref_lm, g_ref_pch = torch.autograd.grad(ref_total, [lm, pch])
        assert torch.allclose(g_agg_lm,  g_ref_lm,  atol=1e-7)
        assert torch.allclose(g_agg_pch, g_ref_pch, atol=1e-7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
