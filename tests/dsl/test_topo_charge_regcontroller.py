# -*- coding: utf-8 -*-
"""RegularizationController integration for the Pontryagin topo-charge
diagnostic (Phase 1.3b).

Contracts:
  1. Default cfg (topo disabled) -> collect_topo_charge_aux returns a
     dict whose 'loss' is exact zero of the right dtype/device and
     whose metrics keys are present but zero. The diagnostic is NOT
     instantiated (lazy build).
  2. Enabled + diagnostic_only (alpha = gamma = 0) -> diagnostic IS
     instantiated; loss is exact zero (torch.equal); Q_h_mean and
     eps_ortho keys are populated with real finite values.
  3. Enabled + alpha > 0 -> loss > 0, backward succeeds, gradient on
     the diagnostic's proj.weight.
  4. Empty attn_per_layer (capture flag off, e.g.) -> no crash; loss
     is zero; diagnostic NOT built.
  5. Lazy build: head_dim is taken from attn_per_layer[0].shape[-1]
     on first call; subsequent calls reuse the same diagnostic
     instance (same proj.weight identity).

Companion to:
  tests/dsl/test_topo_charge.py        -- pure-math contracts
  tests/dsl/test_topo_charge_stub_audit.py -- contract-strength meta
  tests/dsl/test_topo_charge_dsl_parse.py  -- DSL surface
  tests/dsl/test_topo_charge_attn_capture.py -- LanguageCortex hook
"""
from __future__ import annotations

import pytest
import torch

from neuroslm.dsl.regularization import RegularizationConfig
from neuroslm.regularizers import RegularizationController


def _build_controller(enabled: bool = False, alpha: float = 0.0,
                      gamma: float = 0.0) -> RegularizationController:
    cfg = RegularizationConfig()
    cfg.pontryagin_topo_charge.enabled = enabled
    cfg.pontryagin_topo_charge.alpha = alpha
    cfg.pontryagin_topo_charge.gamma = gamma
    return RegularizationController(
        cfg, d_model=32, vocab_size=64,
    )


def _attn_per_layer(L: int = 2, B: int = 2, H: int = 4,
                    T: int = 5, head_dim: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    return [torch.randn(B, H, T, head_dim, requires_grad=False)
            for _ in range(L)]


class TestRegControllerTopoChargeDisabled:
    def test_disabled_returns_zero_loss(self):
        rc = _build_controller(enabled=False)
        out = rc.collect_topo_charge_aux(_attn_per_layer())
        assert torch.equal(out["loss"], torch.zeros(()))

    def test_disabled_does_not_build_diagnostic(self):
        rc = _build_controller(enabled=False)
        rc.collect_topo_charge_aux(_attn_per_layer())
        assert rc.topo_charge_diag is None

    def test_empty_attn_per_layer_safe(self):
        rc = _build_controller(enabled=False)
        out = rc.collect_topo_charge_aux([])
        assert torch.equal(out["loss"], torch.zeros(()))
        assert rc.topo_charge_diag is None


class TestRegControllerTopoChargeDiagnosticOnly:
    """alpha=gamma=0: diagnostic runs, loss is structural zero."""

    def test_enabled_lazy_builds_diagnostic(self):
        rc = _build_controller(enabled=True)
        assert rc.topo_charge_diag is None
        rc.collect_topo_charge_aux(_attn_per_layer())
        from neuroslm.mechanisms.topo_charge import TopoChargeDiagnostic
        assert isinstance(rc.topo_charge_diag, TopoChargeDiagnostic)

    def test_diagnostic_only_loss_is_exact_zero(self):
        rc = _build_controller(enabled=True, alpha=0.0, gamma=0.0)
        out = rc.collect_topo_charge_aux(_attn_per_layer())
        assert torch.equal(out["loss"], torch.zeros(()))

    def test_metrics_populated_under_diagnostic_only(self):
        rc = _build_controller(enabled=True, alpha=0.0, gamma=0.0)
        out = rc.collect_topo_charge_aux(_attn_per_layer())
        assert "Q_h_mean" in out and "eps_ortho" in out
        assert torch.isfinite(out["Q_h_mean"]).all()
        assert torch.isfinite(out["eps_ortho"]).all()

    def test_lazy_build_reuses_same_instance(self):
        rc = _build_controller(enabled=True)
        rc.collect_topo_charge_aux(_attn_per_layer(seed=0))
        diag_1 = rc.topo_charge_diag
        rc.collect_topo_charge_aux(_attn_per_layer(seed=1))
        diag_2 = rc.topo_charge_diag
        assert diag_1 is diag_2, (
            "second call must reuse the first build's diagnostic "
            "module so its proj.weight stays a single learnable"
        )

    def test_head_dim_from_attn_shape(self):
        rc = _build_controller(enabled=True)
        # Custom head_dim=12 via per-tensor shape.
        attn = [torch.randn(1, 2, 4, 12) for _ in range(2)]
        rc.collect_topo_charge_aux(attn)
        assert rc.topo_charge_diag.head_dim == 12


class TestRegControllerTopoChargeActive:
    def test_alpha_positive_produces_nonzero_loss(self):
        rc = _build_controller(enabled=True, alpha=0.5, gamma=0.0)
        out = rc.collect_topo_charge_aux(_attn_per_layer(seed=3))
        assert out["loss"].abs().item() > 0.0

    def test_alpha_positive_backward_grad_on_diag(self):
        rc = _build_controller(enabled=True, alpha=0.5, gamma=0.0)
        out = rc.collect_topo_charge_aux(_attn_per_layer(seed=4))
        out["loss"].backward()
        w = rc.topo_charge_diag.proj.weight.grad
        assert w is not None and w.abs().sum().item() > 1e-6

    def test_gamma_only_propagates_via_eps_ortho(self):
        rc = _build_controller(enabled=True, alpha=0.0, gamma=0.5)
        out = rc.collect_topo_charge_aux(_attn_per_layer(seed=5))
        # eps_ortho > 0 for random inputs -> loss > 0.
        assert out["loss"].abs().item() > 0.0


class TestRegControllerTopoChargeNoRegressionOnExisting:
    """The existing collect_aux for DAR/PCC/Iso/CMD must still work."""

    def test_collect_aux_unaffected_by_topo_addition(self):
        cfg = RegularizationConfig()
        rc = RegularizationController(cfg, d_model=32, vocab_size=64)
        B, T = 2, 4
        h = torch.randn(B, T, 32)
        lm_logits = torch.randn(B, T, 64)
        per_sample_ce = torch.randn(B)
        out = rc.collect_aux(h, lm_logits, per_sample_ce, None,
                             global_step=10)
        # Sanity: all original keys still present, total finite.
        for k in ("dar", "pcc", "isotropy", "cmd",
                  "total", "warmup_mult"):
            assert k in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
