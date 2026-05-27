# -*- coding: utf-8 -*-
"""Phase N7/N8 — observability metrics computed from the DSL model.

Φ (integrated information), λ₁ (Fiedler / graph connectivity), GWS
ignition, oscillation bands, NT system, trophic state, meso learning
gain — each computed from the DSL model's own activations and
computation graph. This is the IIT/hypershape-aligned approach: the
metrics are genuine measures of *this* model, not copies of Brain's
module-specific values.

These populate the native log format so a DSL run reports the same
metric columns as the hand-written trainer.
"""
import math
import pytest
import torch

from neuroslm.dsl import metrics as M


def _fake_layer_acts(n_layers=4, B=2, T=16, D=32):
    return [torch.randn(B, T, D) for _ in range(n_layers)]


# ── Φ proxy ────────────────────────────────────────────────────────────

class TestPhiProxy:
    def test_nonnegative_scalar(self):
        phi = M.phi_proxy(_fake_layer_acts())
        assert isinstance(phi, float)
        assert phi >= 0.0

    def test_correlated_halves_higher_phi(self):
        # If the two halves of the representation are correlated, Φ
        # (integration between halves) should exceed the independent case.
        B, T, D = 4, 16, 32
        indep = [torch.randn(B, T, D)]
        base = torch.randn(B, T, D // 2)
        corr = torch.cat([base, base + 0.01 * torch.randn(B, T, D // 2)], dim=-1)
        phi_indep = M.phi_proxy(indep)
        phi_corr = M.phi_proxy([corr])
        assert phi_corr > phi_indep


# ── Fiedler λ₁ ─────────────────────────────────────────────────────────

class TestFiedler:
    def test_chain_graph_value(self):
        # A depth-L residual transformer is a chain; its normalized-Laplacian
        # Fiedler value is positive and decreases with depth.
        lam4 = M.fiedler_lambda(n_layers=4)
        lam8 = M.fiedler_lambda(n_layers=8)
        assert lam4 > 0
        assert lam8 > 0
        assert lam8 < lam4   # deeper chain → smaller algebraic connectivity

    def test_single_layer(self):
        # Degenerate but must not crash
        assert M.fiedler_lambda(n_layers=1) >= 0.0


# ── GWS ignition ───────────────────────────────────────────────────────

class TestIgnition:
    def test_in_unit_range(self):
        act = torch.randn(2, 16, 32)
        ign = M.gws_ignition(act)
        assert 0.0 <= ign <= 1.0

    def test_peaky_activation_higher(self):
        # A sharply-peaked activation should ignite more than uniform noise.
        peaky = torch.zeros(2, 16, 32)
        peaky[:, :, 0] = 10.0
        flat = torch.randn(2, 16, 32) * 0.01
        assert M.gws_ignition(peaky) >= M.gws_ignition(flat)


# ── Oscillation bands ──────────────────────────────────────────────────

class TestOscillations:
    def test_returns_three_bands(self):
        osc = M.OscillationTracker()
        for _ in range(64):
            osc.observe(torch.randn(2, 8, 16))
        bands = osc.bands()
        assert set(bands.keys()) == {"δ", "θ", "γ"}
        for v in bands.values():
            assert v >= 0.0

    def test_empty_history_safe(self):
        osc = M.OscillationTracker()
        bands = osc.bands()  # no observations yet
        assert set(bands.keys()) == {"δ", "θ", "γ"}


# ── NT system ──────────────────────────────────────────────────────────

class TestNTSystem:
    def test_seven_neurotransmitters(self):
        nt = M.NTSystem()
        levels = nt.levels()
        assert set(levels.keys()) == {"DA", "NE", "5HT", "ACh", "eCB", "Glu", "GABA"}
        for v in levels.values():
            assert 0.0 <= v <= 1.0

    def test_baselines_from_arch(self):
        # Initialised from arch.neuro base_concentrations (dopamine=0.10, ...)
        nt = M.NTSystem()
        assert abs(nt.levels()["DA"] - 0.10) < 0.05

    def test_step_keeps_bounded(self):
        nt = M.NTSystem()
        for _ in range(100):
            nt.step(activity=0.5)
        for v in nt.levels().values():
            assert 0.0 <= v <= 1.0


# ── Trophic system ─────────────────────────────────────────────────────

class TestTrophic:
    def test_tracks_layers(self):
        tr = M.TrophicSystem(n_projections=4)
        for _ in range(10):
            tr.step(_fake_layer_acts(n_layers=4))
        s = tr.stats()
        assert s["n_projections"] == 4
        assert 0 <= s["n_active"] <= 4
        assert s["trophic_mean"] >= 0.0


# ── MetricObserver (bundles everything) ────────────────────────────────

class TestMetricObserver:
    def test_produces_full_metric_dict(self):
        obs = M.MetricObserver(n_layers=4)
        acts = _fake_layer_acts(n_layers=4)
        m = obs.observe(layer_acts=acts, loss=5.0)
        # Every key the native log expects
        for key in ("phi", "fiedler", "ignition", "meso_lg",
                    "troph_active", "troph_total", "troph_mean", "nt", "osc"):
            assert key in m
        assert isinstance(m["nt"], dict) and len(m["nt"]) == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
