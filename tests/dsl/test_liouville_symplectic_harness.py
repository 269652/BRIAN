# -*- coding: utf-8 -*-
"""Harness integration tests for the Liouville Symplectic Residual
auxiliary step (Phase 2, _symplectic_aux_step).

Contracts:
  1. Disabled path: _symplectic_aux_step is a pure no-op (torch.equal).
  2. Enabled + noether_strength=0: total unchanged (torch.equal).
  3. Enabled + noether_strength>0: total strictly changes.
  4. No noether_loss / noether_H_diff keys leak into _metrics when
     symplectic is disabled.
  5. Keys ARE set when enabled.

Uses tiny stub objects to avoid constructing a full BRIANHarness.
Matches the pattern in test_topo_charge_harness_integration.py.
"""
from __future__ import annotations

import torch

from neuroslm.dsl.regularization import RegularizationConfig
from neuroslm.regularizers import RegularizationController


# ── Minimal stubs ─────────────────────────────────────────────────────

class _FakeLM:
    """Minimal language-model stub that exposes _last_hidden."""
    def __init__(self, d: int = 32, B: int = 2, T: int = 5, seed: int = 0):
        torch.manual_seed(seed)
        self._last_hidden = torch.randn(B, T, d, requires_grad=True)


class _FakeHarness:
    """Minimal harness stub that exposes reg_controller + language_model."""
    def __init__(self, rc: RegularizationController, lm: _FakeLM):
        self.reg_controller = rc
        self.language_model = lm
        self._metrics: dict = {}

    def _symplectic_aux_step(self, total: torch.Tensor) -> torch.Tensor:
        # Import the real method and bind it to self for test purposes.
        from neuroslm.harness import BRIANHarness
        return BRIANHarness._symplectic_aux_step(self, total)


def _build_harness(enabled: bool = False, noether_strength: float = 0.0,
                   d: int = 32, seed: int = 0) -> _FakeHarness:
    cfg = RegularizationConfig()
    cfg.liouville_symplectic.enabled = enabled
    cfg.liouville_symplectic.noether_strength = noether_strength
    rc = RegularizationController(cfg, d_model=d, vocab_size=64)
    lm = _FakeLM(d=d, seed=seed)
    return _FakeHarness(rc, lm)


class TestSymplecticAuxStepDisabled:
    def test_disabled_is_pure_noop(self):
        h = _build_harness(enabled=False)
        total = torch.tensor(3.14)
        out = h._symplectic_aux_step(total)
        assert torch.equal(out, total)

    def test_disabled_leaks_no_metrics_keys(self):
        h = _build_harness(enabled=False)
        total = torch.tensor(0.0)
        h._symplectic_aux_step(total)
        assert "noether_loss" not in h._metrics
        assert "noether_H_diff" not in h._metrics


class TestSymplecticAuxStepDiagnosticOnly:
    """noether_strength=0: step runs but contributes zero to total."""

    def test_diagnostic_only_total_unchanged(self):
        h = _build_harness(enabled=True, noether_strength=0.0)
        total = torch.tensor(2.71828)
        out = h._symplectic_aux_step(total)
        assert torch.equal(out, total)

    def test_diagnostic_only_metrics_keys_set(self):
        h = _build_harness(enabled=True, noether_strength=0.0)
        h._symplectic_aux_step(torch.tensor(0.0))
        assert "noether_loss" in h._metrics
        assert "noether_H_diff" in h._metrics


class TestSymplecticAuxStepActive:
    def test_positive_strength_changes_total(self):
        h = _build_harness(enabled=True, noether_strength=1.0, seed=7)
        total = torch.tensor(0.0)
        out = h._symplectic_aux_step(total)
        # total must change (Noether residual is nonzero for random input).
        assert not torch.equal(out, total)

    def test_backward_from_total_does_not_crash(self):
        h = _build_harness(enabled=True, noether_strength=1.0, seed=8)
        total = torch.tensor(0.0, requires_grad=True)
        out = h._symplectic_aux_step(total)
        out.backward()  # must not raise


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
