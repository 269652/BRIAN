# -*- coding: utf-8 -*-
"""RED-first RegularizationController tests for KJPLA aux step (Phase 3).

Contracts:
  1. Disabled: collect_kjpla_aux is a pure no-op (torch.equal on total).
  2. Disabled: no josephson_* / kjpla_* keys leak into metrics.
  3. Enabled, josephson_strength=0: total unchanged (torch.equal).
  4. Enabled, josephson_strength>0: total changes.
  5. Metrics keys present when enabled.
  6. Backward from total.backward() does not crash when enabled.
"""
from __future__ import annotations

import torch
from neuroslm.dsl.regularization import RegularizationConfig, KJPLAPhaseLatticeConfig
from neuroslm.regularizers import RegularizationController
from neuroslm.mechanisms.kjpla import KJPLAttention


DIM = 32
N_HEADS = 4
N_LAYERS = 3
HEAD_DIM = DIM // N_HEADS


def _make_kjpla(k_fill: float = 0.0, beta_fill: float = 0.1) -> KJPLAttention:
    m = KJPLAttention(DIM, N_HEADS, max_ctx=16, n_layers=N_LAYERS)
    with torch.no_grad():
        m.K_h.fill_(k_fill)
        m.beta_h.fill_(beta_fill)
    return m


def _make_phi_list(n_layers: int, B: int = 1, T: int = 8) -> list:
    return [torch.randn(B, N_HEADS, T).to(torch.bfloat16) for _ in range(n_layers)]


def _build_rc(enabled: bool, josephson_strength: float = 0.0) -> RegularizationController:
    cfg = RegularizationConfig()
    cfg.kjpla_phase_lattice.enabled = enabled
    cfg.kjpla_phase_lattice.josephson_strength = josephson_strength
    return RegularizationController(cfg, d_model=DIM, vocab_size=64)


class TestKJPLAAuxDisabled:
    def test_disabled_is_pure_noop(self):
        rc = _build_rc(enabled=False)
        layers = [_make_kjpla() for _ in range(N_LAYERS)]
        phi_list = _make_phi_list(N_LAYERS)
        total = torch.tensor(3.14)
        out = rc.collect_kjpla_aux(total, phi_list, layers)
        assert torch.equal(out, total)

    def test_disabled_leaks_no_metrics(self):
        rc = _build_rc(enabled=False)
        layers = [_make_kjpla() for _ in range(N_LAYERS)]
        phi_list = _make_phi_list(N_LAYERS)
        metrics: dict = {}
        rc.collect_kjpla_aux(torch.tensor(0.0), phi_list, layers, metrics=metrics)
        assert not any("kjpla" in k or "josephson" in k for k in metrics)


class TestKJPLAAuxDiagnosticOnly:
    """josephson_strength=0: mechanism runs but contributes zero to total."""

    def test_diagnostic_only_total_unchanged(self):
        rc = _build_rc(enabled=True, josephson_strength=0.0)
        layers = [_make_kjpla(k_fill=1.0) for _ in range(N_LAYERS)]
        phi_list = _make_phi_list(N_LAYERS)
        total = torch.tensor(2.71)
        out = rc.collect_kjpla_aux(total, phi_list, layers)
        assert torch.equal(out, total)

    def test_diagnostic_only_metrics_present(self):
        rc = _build_rc(enabled=True, josephson_strength=0.0)
        layers = [_make_kjpla(k_fill=1.0) for _ in range(N_LAYERS)]
        phi_list = _make_phi_list(N_LAYERS)
        metrics: dict = {}
        rc.collect_kjpla_aux(torch.tensor(0.0), phi_list, layers, metrics=metrics)
        assert "josephson_loss" in metrics or "kjpla_josephson_loss" in metrics


class TestKJPLAAuxActive:
    def test_positive_strength_changes_total(self):
        torch.manual_seed(5)
        rc = _build_rc(enabled=True, josephson_strength=1.0)
        layers = [_make_kjpla(k_fill=1.0) for _ in range(N_LAYERS)]
        phi_list = _make_phi_list(N_LAYERS)
        total = torch.tensor(0.0)
        out = rc.collect_kjpla_aux(total, phi_list, layers)
        # Josephson loss should be non-zero for random phases with K=1.
        assert not torch.equal(out, total)

    def test_backward_from_total_does_not_crash(self):
        torch.manual_seed(6)
        rc = _build_rc(enabled=True, josephson_strength=1.0)
        layers = [_make_kjpla(k_fill=0.5) for _ in range(N_LAYERS)]
        phi_list = _make_phi_list(N_LAYERS)
        total = torch.tensor(0.0, requires_grad=True)
        out = rc.collect_kjpla_aux(total, phi_list, layers)
        out.backward()  # must not raise
