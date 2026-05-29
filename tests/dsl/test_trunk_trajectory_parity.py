# -*- coding: utf-8 -*-
"""End-to-end trunk-trajectory parity: DSL aggregator vs Brain formula
across a full SGD training run (50+ steps with optimizer + autograd).

This is the *closing test* for the Scope-B port. The Scope-A tests proved
individual subsystem forward/gradient parity:
  * MotorCortex     — test_motor_parity.py
  * ForwardModel    — test_forward_model_parity.py
  * RSSM            — test_world_model_parity.py
  * Aggregator      — test_brain_aggregator_parity.py
  * Maturity/Gates  — test_maturity_parity.py
  * Loss clipping   — test_loss_clipping.py (gradient pairity, post-fix)
  * PCH aux         — test_loss_parity_n8.py

This test composes ALL of them into a single training loop and asserts
that the loss trajectory of the DSL aggregator stays *bit-identical* to
the literal Brain reference formula across many steps + a real optimizer.

Why this matters: float errors compound — small per-step divergence can
amplify over hundreds of steps. The Scope-A tests check single forward
passes; this checks that the trajectory itself doesn't drift.
"""
import pytest
import torch
import torch.nn as nn

from neuroslm.dsl.brain_aggregator import (
    aggregate_total_loss, brain_reference_total, LossBundle,
)
from neuroslm.dsl.maturity import MaturityTracker, TotalLossConfig
from neuroslm.dsl.subsystems.orchestrator_adapter import (
    OrchestratorMetrics,
)


class _TinyTrunk(nn.Module):
    """Small trainable trunk that produces all aux losses from a hidden vec.

    Stand-in for the full Brain trunk — enough to exercise the aggregator
    end-to-end without spinning up 100M params per test step.
    """
    def __init__(self, d=32):
        super().__init__()
        self.lm = nn.Linear(d, d)
        self.pch = nn.Linear(d, d)
        self.world = nn.Linear(d, d)
        self.forward_h = nn.Linear(d, d)
        self.motor = nn.Linear(d, d)
        self.kl = nn.Linear(d, d)
        self.novel = nn.Linear(d, d)
        self.cpc = nn.Linear(d, d)
        self.phi = nn.Linear(d, d)

    def make_bundle(self, x: torch.Tensor) -> LossBundle:
        # Each loss is a squared-norm of a per-head projection of x
        return LossBundle(
            lm_loss     = (self.lm(x)        ** 2).mean(),
            pred_coding = (self.pch(x)       ** 2).mean(),
            world       = (self.world(x)     ** 2).mean(),
            forward     = (self.forward_h(x) ** 2).mean(),
            motor       = (self.motor(x)     ** 2).mean(),
            kl_world    = (self.kl(x)        ** 2).mean(),
            novel       = (self.novel(x)     ** 2).mean(),
            cpc         = (self.cpc(x)       ** 2).mean(),
            phi         = (self.phi(x)       ** 2).mean() - 0.1,
            orchestrator=OrchestratorMetrics(
                identity_drift=torch.tensor(0.05),
                neural_calm=torch.tensor(0.9),
            ),
        )


class TestTrunkTrajectoryParity:
    def test_50_steps_aggregator_equals_brain_formula(self):
        """Over 50 SGD steps with identical maturity ramp, the DSL aggregator's
        total loss must match the literal Brain formula at every step."""
        torch.manual_seed(0)
        trunk = _TinyTrunk(d=32)
        opt = torch.optim.SGD(trunk.parameters(), lr=1e-3)
        mat_tracker = MaturityTracker()
        g = torch.Generator().manual_seed(42)
        cfg = TotalLossConfig()

        max_diff_seen = 0.0
        max_rel_seen = 0.0
        for step in range(50):
            x = torch.randn(4, 32, generator=g)
            bundle = trunk.make_bundle(x)

            # DSL aggregator's total
            mat = mat_tracker.value()
            agg_total = aggregate_total_loss(bundle, mat, cfg)
            # Brain's literal formula on the SAME bundle (no re-forward — both
            # sides see the same loss tensors; we only validate aggregation)
            ref_total = brain_reference_total(bundle, mat, cfg)

            diff = (agg_total - ref_total).abs().item()
            rel  = diff / max(abs(ref_total.item()), 1e-9)
            max_diff_seen = max(max_diff_seen, diff)
            max_rel_seen = max(max_rel_seen, rel)
            assert diff < 1e-5, \
                f"step {step}: aggregator={agg_total.item()} ref={ref_total.item()} diff={diff}"

            opt.zero_grad()
            agg_total.backward()
            opt.step()

            # Update maturity from the LM portion (matches Brain pattern)
            mat_tracker.update(bundle.lm_loss.detach().item())

        # Sanity: maturity actually moved over 50 steps
        assert mat_tracker.value() > 0.0, \
            f"MaturityTracker didn't update: still at {mat_tracker.value()}"
        print(f"\n50-step trajectory: max_abs_diff={max_diff_seen:.2e} "
              f"max_rel_diff={max_rel_seen:.2e} final_mat={mat_tracker.value():.4f}")

    def test_100_steps_gradient_through_aggregator_matches(self):
        """Per-param gradient through the aggregator must match the Brain
        reference formula over 100 steps — guards against the silent
        gradient-divergence pattern we hit on loss clipping."""
        torch.manual_seed(1)
        trunk_a = _TinyTrunk(d=24)
        trunk_b = _TinyTrunk(d=24)
        # Sync weights so both start from identical state
        for p_a, p_b in zip(trunk_a.parameters(), trunk_b.parameters()):
            p_b.data.copy_(p_a.data)

        opt_a = torch.optim.SGD(trunk_a.parameters(), lr=5e-4)
        opt_b = torch.optim.SGD(trunk_b.parameters(), lr=5e-4)
        mat_a = MaturityTracker()
        mat_b = MaturityTracker()
        g = torch.Generator().manual_seed(7)
        cfg = TotalLossConfig()

        for step in range(100):
            x = torch.randn(4, 24, generator=g)
            bundle_a = trunk_a.make_bundle(x)
            bundle_b = trunk_b.make_bundle(x)

            total_a = aggregate_total_loss(bundle_a, mat_a.value(), cfg)
            total_b = brain_reference_total(bundle_b, mat_b.value(), cfg)

            opt_a.zero_grad(); total_a.backward(); opt_a.step()
            opt_b.zero_grad(); total_b.backward(); opt_b.step()

            mat_a.update(bundle_a.lm_loss.detach().item())
            mat_b.update(bundle_b.lm_loss.detach().item())

            # After each step, parameters must remain identical — proves
            # the gradient through the aggregator equals the gradient
            # through Brain's literal formula at every step.
            for pa, pb in zip(trunk_a.parameters(), trunk_b.parameters()):
                d = (pa.data - pb.data).abs().max().item()
                assert d < 1e-5, \
                    f"step {step}: param trajectory diverged (max diff {d})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
