# -*- coding: utf-8 -*-
"""RegularizationController integration for the Liouville Symplectic
Residual diagnostic (Phase 2).

Contracts:
  1. Default cfg (symplectic disabled) -> collect_symplectic_aux returns
     exact-zero loss; no lazy block is built.
  2. Enabled + noether_strength=0 -> diagnostic-only: block IS built,
     loss is EXACT zero (torch.equal), H_diff is a finite scalar.
  3. Enabled + noether_strength>0 -> loss > 0, backward succeeds,
     gradient reaches the block's dtau Parameter.
  4. Odd d_model -> returns gracefully with zero loss (no crash).
  5. Lazy build: block is built on first call, reused on second.
"""
from __future__ import annotations

import pytest
import torch

from neuroslm.dsl.regularization import RegularizationConfig
from neuroslm.regularizers import RegularizationController


def _build_controller(
    enabled: bool = False,
    noether_strength: float = 0.0,
    dtau_init: float = 0.1,
    potential_kind: str = "quadratic",
) -> RegularizationController:
    cfg = RegularizationConfig()
    cfg.liouville_symplectic.enabled = enabled
    cfg.liouville_symplectic.noether_strength = noether_strength
    cfg.liouville_symplectic.dtau_init = dtau_init
    cfg.liouville_symplectic.potential_kind = potential_kind
    return RegularizationController(cfg, d_model=32, vocab_size=64)


def _h(d: int = 32, B: int = 2, T: int = 5, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, T, d, requires_grad=True)


class TestRegControllerSymplecticDisabled:
    def test_disabled_returns_zero_loss(self):
        rc = _build_controller(enabled=False)
        out = rc.collect_symplectic_aux(_h())
        assert torch.equal(out["loss"], torch.zeros(()))

    def test_disabled_does_not_build_block(self):
        rc = _build_controller(enabled=False)
        rc.collect_symplectic_aux(_h())
        assert rc._symplectic_block is None


class TestRegControllerSymplecticDiagnosticOnly:
    """noether_strength=0: block runs, loss is structural zero."""

    def test_enabled_lazy_builds_block(self):
        rc = _build_controller(enabled=True)
        assert rc._symplectic_block is None
        rc.collect_symplectic_aux(_h())
        from neuroslm.mechanisms.liouville_symplectic import LiouvilleSymplecticBlock
        assert isinstance(rc._symplectic_block, LiouvilleSymplecticBlock)

    def test_diagnostic_only_loss_is_exact_zero(self):
        rc = _build_controller(enabled=True, noether_strength=0.0)
        out = rc.collect_symplectic_aux(_h(seed=1))
        assert torch.equal(out["loss"], torch.zeros(()))

    def test_H_diff_is_finite(self):
        rc = _build_controller(enabled=True)
        out = rc.collect_symplectic_aux(_h(seed=2))
        assert torch.isfinite(out["H_diff"])

    def test_lazy_build_reuses_same_instance(self):
        rc = _build_controller(enabled=True)
        rc.collect_symplectic_aux(_h(seed=0))
        blk_1 = rc._symplectic_block
        rc.collect_symplectic_aux(_h(seed=1))
        blk_2 = rc._symplectic_block
        assert blk_1 is blk_2


class TestRegControllerSymplecticActive:
    def test_noether_strength_positive_gives_nonzero_loss(self):
        rc = _build_controller(enabled=True, noether_strength=1.0)
        out = rc.collect_symplectic_aux(_h(seed=3))
        assert out["loss"].abs().item() > 0.0

    def test_backward_grad_reaches_dtau(self):
        rc = _build_controller(enabled=True, noether_strength=1.0)
        h = _h(seed=4)
        out = rc.collect_symplectic_aux(h)
        out["loss"].backward()
        g = rc._symplectic_block.dtau.grad
        assert g is not None and g.abs().item() > 0.0


class TestRegControllerSymplecticOddDModel:
    def test_odd_d_model_returns_zero_gracefully(self):
        rc = _build_controller(enabled=True, noether_strength=1.0)
        h_odd = torch.randn(2, 5, 7, requires_grad=True)
        out = rc.collect_symplectic_aux(h_odd)
        assert torch.equal(out["loss"], torch.zeros(()))
        assert rc._symplectic_block is None


class TestRegControllerSymplecticNoRegressionOnExisting:
    def test_collect_aux_unaffected_by_symplectic_addition(self):
        cfg = RegularizationConfig()
        rc = RegularizationController(cfg, d_model=32, vocab_size=64)
        B, T = 2, 4
        h = torch.randn(B, T, 32)
        lm_logits = torch.randn(B, T, 64)
        per_sample_ce = torch.randn(B)
        out = rc.collect_aux(h, lm_logits, per_sample_ce, None,
                             global_step=10)
        for k in ("dar", "pcc", "isotropy", "cmd", "total", "warmup_mult"):
            assert k in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
