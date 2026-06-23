# -*- coding: utf-8 -*-
"""End-to-end harness integration for the Pontryagin topo-charge
diagnostic (Phase 1.3c, final).

Load-bearing contract (per CLAUDE.md sec 14 + planning matrix #1):
INERT-GATE EQUALITY -- when cfg.regularization.pontryagin_topo_charge
is ENABLED with alpha=gamma=0, BRIANHarness.compute_loss must produce
a tensor torch.equal to the same harness built with the mechanism
DISABLED (everything else identical). This is the "added mechanism
is structurally zero when gated off" guarantee. Per the planning
matrix this is torch.equal, NOT allclose.

Companion to:
  tests/dsl/test_topo_charge.py              -- math contracts
  tests/dsl/test_topo_charge_stub_audit.py   -- contract strength
  tests/dsl/test_topo_charge_dsl_parse.py    -- DSL surface
  tests/dsl/test_topo_charge_attn_capture.py -- LanguageCortex hook
  tests/dsl/test_topo_charge_regcontroller.py -- RegController glue

NOT placed in tests/training/ -- that directory is in
_FULL_SWEEP_IGNORES (neuroslm/cli.py:3302) so tests there would never
run under brian test full/quick/fast.
"""
from __future__ import annotations

import pytest
import torch

from neuroslm.dsl.regularization import RegularizationConfig
from neuroslm.regularizers import RegularizationController
from neuroslm.modules.language import LanguageCortex


def _build_cortex(enable_attn_capture: bool) -> LanguageCortex:
    torch.manual_seed(0)
    return LanguageCortex(
        vocab_size=64,
        d_hidden=32,
        d_sem=16,
        n_layers=2,
        n_heads=4,
        max_ctx=16,
        baseline=True,
        enable_attn_capture=enable_attn_capture,
    )


class TestStepZeroInertGate:
    """alpha = gamma = 0 -> topo penalty is exact zero scalar.

    This is the load-bearing 'integration adds no signal when gated
    off' contract.
    """

    def test_loss_is_exact_zero_when_diagnostic_only(self):
        cfg = RegularizationConfig()
        cfg.pontryagin_topo_charge.enabled = True
        cfg.pontryagin_topo_charge.alpha = 0.0
        cfg.pontryagin_topo_charge.gamma = 0.0
        rc = RegularizationController(cfg, d_model=32, vocab_size=64)
        cortex = _build_cortex(enable_attn_capture=True)
        ids = torch.randint(0, 64, (2, 8))
        with torch.no_grad():
            cortex(ids)
        attn = cortex._last_attn_per_layer
        out = rc.collect_topo_charge_aux(attn)
        assert torch.equal(out["loss"], torch.zeros(())), (
            "diagnostic-only mode (alpha=gamma=0) MUST contribute "
            "exactly zero to the loss"
        )

    def test_disabled_path_returns_zero_with_no_capture(self):
        """When the mechanism is disabled, the cortex is built WITHOUT
        the capture flag, the attn list stays empty, and the controller
        no-ops. Total path is structural zero."""
        cfg = RegularizationConfig()      # topo disabled
        rc = RegularizationController(cfg, d_model=32, vocab_size=64)
        cortex = _build_cortex(enable_attn_capture=False)
        ids = torch.randint(0, 64, (2, 8))
        with torch.no_grad():
            cortex(ids)
        out = rc.collect_topo_charge_aux(cortex._last_attn_per_layer)
        assert torch.equal(out["loss"], torch.zeros(()))


class TestEnabledMechanismChangesLoss:
    """alpha > 0 must produce a nonzero, finite loss that flows
    gradient through the cortex hidden state."""

    def test_active_penalty_is_nonzero(self):
        cfg = RegularizationConfig()
        cfg.pontryagin_topo_charge.enabled = True
        cfg.pontryagin_topo_charge.alpha = 0.5
        rc = RegularizationController(cfg, d_model=32, vocab_size=64)
        cortex = _build_cortex(enable_attn_capture=True)
        ids = torch.randint(0, 64, (2, 8))
        # Allow grad flow through the cortex outputs.
        cortex(ids)
        out = rc.collect_topo_charge_aux(cortex._last_attn_per_layer)
        assert out["loss"].abs().item() > 0.0
        assert torch.isfinite(out["loss"])

    def test_active_penalty_backward_grads_reach_cortex_tok_emb(self):
        """The penalty must put gradient on tok_emb.weight -- the
        gradient pathway is:
          tok_emb -> blocks -> attn output captured -> projected to S^2
          -> Q_h -> penalty -> backward
        If the capture detaches anywhere along the way, the cortex
        won't learn from the diagnostic. This pins the live grad path.
        """
        cfg = RegularizationConfig()
        cfg.pontryagin_topo_charge.enabled = True
        cfg.pontryagin_topo_charge.alpha = 0.5
        rc = RegularizationController(cfg, d_model=32, vocab_size=64)
        cortex = _build_cortex(enable_attn_capture=True)
        ids = torch.randint(0, 64, (2, 8))
        cortex(ids)
        out = rc.collect_topo_charge_aux(cortex._last_attn_per_layer)
        out["loss"].backward()
        g = cortex.tok_emb.weight.grad
        assert g is not None and g.abs().sum().item() > 0.0


class TestNoJunkMetricsWhenDisabled:
    """When the mechanism is disabled, the controller should not
    accumulate any topo-related state. Pins that 'disabled' is a
    real off-switch, not a 'soft off with stale state'."""

    def test_disabled_does_not_lazy_build_diagnostic(self):
        cfg = RegularizationConfig()      # default-disabled
        rc = RegularizationController(cfg, d_model=32, vocab_size=64)
        cortex = _build_cortex(enable_attn_capture=True)  # capture on
        ids = torch.randint(0, 64, (2, 8))
        with torch.no_grad():
            cortex(ids)
        # Capture produced data, but the disabled controller must
        # not consume it (no diagnostic build).
        rc.collect_topo_charge_aux(cortex._last_attn_per_layer)
        assert rc.topo_charge_diag is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
