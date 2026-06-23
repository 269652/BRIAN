# -*- coding: utf-8 -*-
"""Harness integration tests for the KJPLA aux step (Phase 3).

Contracts:
  1. Disabled: _kjpla_aux_step is a pure no-op (torch.equal on total).
  2. Disabled: no kjpla_* / josephson_* keys leak into _metrics.
  3. Enabled, josephson_strength=0: total unchanged (torch.equal).
  4. Enabled, josephson_strength>0: total changes from non-trivial phases.
  5. Metrics keys ARE set when enabled.
  6. backward from total does not crash.

Uses minimal stubs, same pattern as test_liouville_symplectic_harness.py.
"""
from __future__ import annotations

import torch
from neuroslm.dsl.regularization import RegularizationConfig
from neuroslm.regularizers import RegularizationController
from neuroslm.mechanisms.kjpla import KJPLAttention
from neuroslm.harness import BRIANHarness

DIM = 32
N_HEADS = 4
N_LAYERS = 3
HEAD_DIM = DIM // N_HEADS


def _make_kjpla_layers(n: int, k_fill: float = 0.0) -> list:
    layers = []
    for _ in range(n):
        m = KJPLAttention(DIM, N_HEADS, max_ctx=16, n_layers=N_LAYERS)
        with torch.no_grad():
            m.K_h.fill_(k_fill)
            m.beta_h.fill_(0.1)
        layers.append(m)
    return layers


def _make_phi_list(n: int, B: int = 1, T: int = 8) -> list:
    return [torch.randn(B, N_HEADS, T).to(torch.bfloat16) for _ in range(n)]


class _FakeLM:
    """Exposes _last_kjpla_phi_list and _last_kjpla_layers for harness stub."""
    def __init__(self, n_layers: int = N_LAYERS, B: int = 1, T: int = 8,
                 k_fill: float = 0.0):
        self._last_kjpla_phi_list = _make_phi_list(n_layers, B, T)
        self._last_kjpla_layers = _make_kjpla_layers(n_layers, k_fill)


class _FakeHarness:
    def __init__(self, rc: RegularizationController, lm: _FakeLM):
        self.reg_controller = rc
        self.language_model = lm
        self._metrics: dict = {}

    def _kjpla_aux_step(self, total: torch.Tensor) -> torch.Tensor:
        return BRIANHarness._kjpla_aux_step(self, total)


def _build(enabled: bool = False, josephson_strength: float = 0.0,
           k_fill: float = 0.0) -> _FakeHarness:
    cfg = RegularizationConfig()
    cfg.kjpla_phase_lattice.enabled = enabled
    cfg.kjpla_phase_lattice.josephson_strength = josephson_strength
    rc = RegularizationController(cfg, d_model=DIM, vocab_size=64)
    lm = _FakeLM(k_fill=k_fill)
    return _FakeHarness(rc, lm)


class TestKJPLAAuxHarnessDisabled:
    def test_disabled_is_pure_noop(self):
        h = _build(enabled=False)
        total = torch.tensor(3.14)
        out = h._kjpla_aux_step(total)
        assert torch.equal(out, total)

    def test_disabled_leaks_no_keys(self):
        h = _build(enabled=False)
        h._kjpla_aux_step(torch.tensor(0.0))
        assert not any("kjpla" in k or "josephson" in k for k in h._metrics)


class TestKJPLAAuxHarnessDiagnostic:
    def test_diagnostic_total_unchanged(self):
        h = _build(enabled=True, josephson_strength=0.0, k_fill=1.0)
        total = torch.tensor(1.0)
        out = h._kjpla_aux_step(total)
        assert torch.equal(out, total)

    def test_diagnostic_metrics_set(self):
        h = _build(enabled=True, josephson_strength=0.0, k_fill=1.0)
        h._kjpla_aux_step(torch.tensor(0.0))
        assert any("josephson" in k or "kjpla" in k for k in h._metrics)


class TestKJPLAAuxHarnessActive:
    def test_positive_strength_changes_total(self):
        torch.manual_seed(9)
        h = _build(enabled=True, josephson_strength=1.0, k_fill=1.0)
        total = torch.tensor(0.0)
        out = h._kjpla_aux_step(total)
        assert not torch.equal(out, total)

    def test_backward_does_not_crash(self):
        torch.manual_seed(10)
        h = _build(enabled=True, josephson_strength=1.0, k_fill=0.5)
        total = torch.tensor(0.0, requires_grad=True)
        out = h._kjpla_aux_step(total)
        out.backward()
