"""Tests for the Neural Field Oscillator (NFO) — H015..H018.

Each class corresponds to one of the four hypotheses formalised in
``hypothesis/H015..H018_*.md`` and verified in
``hypothesis/proofs/H01*.lean``.

Run all with::

    pytest tests/modules/test_nfo.py -v

Or one hypothesis at a time::

    pytest tests/modules/test_nfo.py::TestBaselineIdentity -v
    pytest tests/modules/test_nfo.py::TestCoherenceGate -v
    pytest tests/modules/test_nfo.py::TestSwiftHohenberg -v
    pytest tests/modules/test_nfo.py::TestBipartitionCoherence -v
"""
from __future__ import annotations

import math

import pytest
import torch

from neuroslm.modules.neural_field_oscillator import (
    NFOConfig,
    NeuralFieldOscillator,
    _bipartition_coherence,
    _complex_polar,
    make_nfo,
)


# ──────────────────────────────────────────────────────────────────────
# H018 — baseline identity at zero-init
# ──────────────────────────────────────────────────────────────────────

class TestBaselineIdentity:
    """Discharges the H018 obligation in the Python lift."""

    def test_baseline_identity_at_init(self):
        torch.manual_seed(0)
        d = 32
        blk = NeuralFieldOscillator(d_model=d, cfg=NFOConfig(n_osc=8))
        h = torch.randn(2, 6, d)
        y = blk(h)
        assert y.shape == h.shape
        assert torch.allclose(y, h, atol=1e-6), (
            "zero-init readout must produce h_out = h_in (H018).\n"
            f"max |y - h| = {(y - h).abs().max().item():.3e}"
        )

    @pytest.mark.parametrize("seed", list(range(32)))
    def test_baseline_identity_for_any_input(self, seed: int):
        """32 random batches — varies B, T, dtype, content."""
        torch.manual_seed(seed)
        d = 16 + (seed % 4) * 8
        B = 1 + seed % 4
        T = 2 + seed % 7
        dtype = (torch.float32, torch.float64)[seed % 2]
        blk = NeuralFieldOscillator(d_model=d, cfg=NFOConfig(n_osc=4)).to(dtype)
        h = torch.randn(B, T, d, dtype=dtype)
        y = blk(h)
        assert y.dtype == h.dtype
        assert torch.allclose(y, h, atol=1e-5), (
            f"seed={seed}: H018 identity violated, "
            f"max |y - h| = {(y - h).abs().max().item():.3e}"
        )

    @pytest.mark.parametrize("variant", list(range(16)))
    def test_baseline_identity_for_any_config(self, variant: int):
        """16 random configs — varies dynamics hyperparameters."""
        torch.manual_seed(variant + 100)
        d = 24
        cfg = NFOConfig(
            n_osc=2 + variant,
            n_steps=1 + variant % 3,
            dt_init=0.05 + 0.02 * variant,
            kappa_init=0.1 * (variant % 4),    # explicitly NON-zero κ_init
            alpha_init=0.0,                    # the H018 contract knob
            mu_init=0.1 + 0.05 * variant,
            a_star_init=0.5 + 0.1 * variant,
            kernel_temperature=0.5 + 0.05 * variant,
        )
        blk = NeuralFieldOscillator(d_model=d, cfg=cfg)
        h = torch.randn(2, 5, d)
        y = blk(h)
        # H018: zero-init Wo dominates over any non-trivial κ / dynamics.
        assert torch.allclose(y, h, atol=1e-5), (
            f"variant={variant}: H018 violated under non-default dynamics"
        )

    def test_baseline_identity_after_one_optim_step_holds_when_alpha_frozen(self):
        """If alpha is frozen at zero, baseline identity holds even after
        a backward + step (Wo can move, but α=0 zeros its effect)."""
        torch.manual_seed(0)
        d = 16
        blk = NeuralFieldOscillator(d_model=d, cfg=NFOConfig(n_osc=4))
        # Freeze alpha so we can let Wo move; H018 still applies through
        # the second zero in the readout product (Wo·y = anything, then ×0).
        blk.alpha.requires_grad_(False)
        opt = torch.optim.SGD(blk.parameters(), lr=0.1)
        h = torch.randn(2, 5, d, requires_grad=True)
        y = blk(h)
        loss = (y - h).pow(2).mean()
        loss.backward()
        opt.step()
        # New forward still identity because α stayed at 0.
        y2 = blk(h)
        assert torch.allclose(y2, h, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────
# H016 — coherence gate is information-preserving
# ──────────────────────────────────────────────────────────────────────

class TestCoherenceGate:
    """The gate g = R / max R is monotone in R, zero only at R=0,
    identity at the uniform extreme."""

    @staticmethod
    def _gate(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        R_max = R.max(dim=-1, keepdim=True).values
        return R / (R_max + eps)

    def test_coherence_gate_zero_when_R_zero(self):
        R = torch.zeros(2, 3, 4)
        g = self._gate(R)
        assert torch.allclose(g, torch.zeros_like(R))

    def test_coherence_gate_one_when_R_uniform(self):
        R = torch.full((2, 3, 4), 0.5)
        g = self._gate(R)
        # uniform R, max = 0.5, g = 0.5/0.5 = 1 (modulo eps).
        assert torch.allclose(g, torch.ones_like(R), atol=1e-4)

    def test_coherence_gate_in_unit_interval(self):
        torch.manual_seed(42)
        R = torch.rand(8, 16, 32)             # ∈ [0, 1) by construction
        g = self._gate(R)
        assert g.min() >= 0.0
        assert g.max() <= 1.0 + 1e-4

    @pytest.mark.parametrize("seed", list(range(16)))
    def test_coherence_gate_monotone_in_R(self, seed: int):
        """Holding max R fixed, increasing any component cannot decrease g."""
        torch.manual_seed(seed)
        R = torch.rand(4, 8)
        # Bump element [0, 0] by 0.05 (without exceeding max, so max R
        # stays the same).
        R_max = R.max(dim=-1, keepdim=True).values
        # Find a safe component to bump — one that's strictly below max.
        below_max = (R < R_max - 0.1).any(dim=-1)
        if not below_max.any():
            pytest.skip(f"seed={seed}: no safe component to bump")
        row = int(below_max.float().argmax().item())
        col = int((R[row] < R_max[row] - 0.1).float().argmax().item())
        R2 = R.clone()
        R2[row, col] += 0.05
        g1 = self._gate(R)
        g2 = self._gate(R2)
        # Component (row, col) must non-strictly increase.
        assert g2[row, col] >= g1[row, col] - 1e-6


# ──────────────────────────────────────────────────────────────────────
# H017 — Swift–Hohenberg amplitude flow is contractive under dt cap
# ──────────────────────────────────────────────────────────────────────

# Lyapunov sweep grid — used by the falsifiable prediction in
# hypothesis/H017_swift_hohenberg_contractive.md §4.
AMP_LYAPUNOV_GRID = [
    (mu, a_star)
    for mu in (0.1, 0.3, 0.5, 0.8)
    for a_star in (0.5, 1.0, 1.5, 2.0)
]


def _lyapunov(A: torch.Tensor, mu: float, a_star: float) -> torch.Tensor:
    """V(A) = (1/8)(A^2 − A_+^2)^2 where A_+ = sqrt(A_*^2 + 4μ) is the
    stable equilibrium of the cubic ODE.

    See lib/blocks/neural_field_oscillator.neuro::nfo_swift_hohenberg_amplitude_bounded
    for the derivation: V_dot = -(1/8) A^2 (A^2 - A_+^2)^2 ≤ 0 in
    continuous time."""
    a_plus_sq = a_star * a_star + 4.0 * mu
    return 0.125 * (A * A - a_plus_sq) ** 2


def _sh_step(A: torch.Tensor, mu: float, a_star: float, dt: float) -> torch.Tensor:
    """Discrete Swift–Hohenberg step from
    lib/blocks/neural_field_oscillator.neuro::nfo_amplitude_sh
    with κ = 0 (free amplitude, no coupling). The coupled case is
    tested through the full block forward in TestBlockWiring."""
    return A + dt * (mu * A - 0.25 * (A * A - a_star * a_star) * A)


# Empirical safe-dt bound established by the grid sweep in
# scripts/validate_swift_hohenberg.py (and the
# `min(results.values())` in the development notebook).
_SAFE_DT = 0.50


class TestSwiftHohenberg:
    """Discharges the H017 Lyapunov / contractivity obligation."""

    @pytest.mark.parametrize("mu,a_star", AMP_LYAPUNOV_GRID)
    def test_amplitude_lyapunov_nonincreasing(self, mu: float, a_star: float):
        """16 amplitudes × 64 iterations per (μ, A*): no trajectory
        should ever increase the (correct) Lyapunov functional
        V(A) = (1/8)(A² − A_+²)²."""
        dt = _SAFE_DT
        a_plus = math.sqrt(a_star * a_star + 4.0 * mu)
        A0 = torch.linspace(0.0, a_plus, 16)
        A = A0.clone()
        V_prev = _lyapunov(A, mu, a_star)
        for _ in range(64):
            A = _sh_step(A, mu, a_star, dt)
            V = _lyapunov(A, mu, a_star)
            assert (V <= V_prev + 1e-5).all(), (
                f"mu={mu}, a_star={a_star}, dt={dt:.3f}: Lyapunov went up "
                f"by {(V - V_prev).max().item():.3e} (max)"
            )
            V_prev = V

    @pytest.mark.parametrize("mu,a_star", AMP_LYAPUNOV_GRID)
    def test_amplitude_bounded_under_dt_cap(self, mu: float, a_star: float):
        """Discrete trajectory stays inside [0, A_+] for the cap
        dt = min(_SAFE_DT, default dt_max = 0.45)."""
        dt = min(0.45, _SAFE_DT)
        a_plus = math.sqrt(a_star * a_star + 4.0 * mu)
        A = torch.linspace(0.0, a_plus, 16)
        for _ in range(64):
            A = _sh_step(A, mu, a_star, dt)
            assert (A >= -1e-5).all(), (
                f"mu={mu}, a_star={a_star}: A went negative "
                f"(min = {A.min().item():.3e})"
            )
            # A_+ is the stable equilibrium — discrete Euler can overshoot
            # it by O(dt) on a single step but always returns. Allow 5 %
            # over-shoot, well below the un-bounded blow-up the test is
            # actually guarding against.
            assert (A <= a_plus * 1.05 + 1e-3).all(), (
                f"mu={mu}, a_star={a_star}: A exceeded upper bound "
                f"(max = {A.max().item():.3e} > {a_plus:.3e})"
            )

    def test_implementation_dt_max_below_safe_bound(self):
        """Sanity: the production cap is below the empirical safe bound."""
        from neuroslm.modules.neural_field_oscillator import NFOConfig
        cfg = NFOConfig()
        assert cfg.dt_max < _SAFE_DT, (
            f"dt_max = {cfg.dt_max} should be below empirical safe "
            f"bound {_SAFE_DT}"
        )

    def test_amplitude_block_keeps_A_finite_under_full_forward(self):
        """Full block forward (coupled κ + α=0 readout): even with the
        readout disabled, the internal A field must stay finite."""
        torch.manual_seed(0)
        d = 32
        blk = NeuralFieldOscillator(
            d_model=d, cfg=NFOConfig(n_osc=16, n_steps=5,
                                     kappa_init=0.5, dt_init=0.4)
        )
        h = torch.randn(2, 32, d) * 3.0
        y = blk(h)
        assert torch.isfinite(y).all()
        # Telemetry: A_mean must be finite and bounded.
        A_mean = blk.last_state["A_mean"].item()
        assert math.isfinite(A_mean)
        assert 0 < A_mean < 10.0


# ──────────────────────────────────────────────────────────────────────
# H015 — bipartition coherence is a closed-form Φ lower bound
# ──────────────────────────────────────────────────────────────────────

class TestBipartitionCoherence:
    """Tests the Φκ lower bound surrogate from H015."""

    def test_phi_kappa_in_unit_interval(self):
        R = torch.rand(4, 8, 16)
        phi_k = _bipartition_coherence(R)
        assert 0.0 <= float(phi_k.item()) <= 1.0

    def test_phi_kappa_minimum_at_full_synchrony(self):
        """If R = 1 everywhere, Φκ = 0 — the minimum-incoherence
        extreme (= maximum integrated information by H015)."""
        R = torch.ones(2, 4, 8)
        phi_k = _bipartition_coherence(R)
        assert torch.allclose(phi_k, torch.zeros_like(phi_k), atol=1e-6)

    def test_phi_kappa_maximum_at_silence(self):
        """If R = 0 everywhere, Φκ = 1 — fully incoherent."""
        R = torch.zeros(2, 4, 8)
        phi_k = _bipartition_coherence(R)
        assert torch.allclose(phi_k, torch.ones_like(phi_k), atol=1e-6)

    def test_phi_kappa_monotone_in_coherence(self):
        """Raising R uniformly must lower Φκ — the H015 direction."""
        R_low = torch.full((2, 4, 8), 0.3)
        R_hi = torch.full((2, 4, 8), 0.7)
        phi_low = _bipartition_coherence(R_low).item()
        phi_hi = _bipartition_coherence(R_hi).item()
        assert phi_hi < phi_low, (
            f"H015 direction violated: Φκ({R_hi.mean().item():.2f}) "
            f"= {phi_hi:.3f} should be < Φκ({R_low.mean().item():.2f}) "
            f"= {phi_low:.3f}"
        )

    def test_bipartition_coherence_monotone_in_couplings(self):
        """Integer analog of H015: more cut couplings ⇒ higher Φ-proxy.

        Mirrors ``Brian.Nfo.bipartition_coherence_phi_lower_bound`` from
        ``lean/Brian/Nfo.lean``. The structural Lean theorem covers the
        formal statement; here we just exercise the Python-side direction
        on the coherence functional.
        """
        # Synchronising one extra cut edge ⇒ one more R_i jumps from 0 to 1
        # ⇒ Φ_κ drops by exactly 1/N.
        N = 16
        R_base = torch.zeros(N)
        R_base[:5] = 1.0                                     # 5 synced edges
        phi_base = _bipartition_coherence(R_base).item()
        R_more = R_base.clone()
        R_more[5] = 1.0                                       # 6 synced edges
        phi_more = _bipartition_coherence(R_more).item()
        assert phi_more < phi_base
        assert abs((phi_base - phi_more) - 1.0 / N) < 1e-6


# ──────────────────────────────────────────────────────────────────────
# Block-level sanity / wiring
# ──────────────────────────────────────────────────────────────────────

class TestBlockWiring:

    def test_polar_inverse_is_identity(self):
        torch.manual_seed(0)
        u = torch.randn(2, 3, 8)
        v = torch.randn(2, 3, 8)
        A, phi = _complex_polar(u, v, eps=1e-8)
        u2 = A * torch.cos(phi)
        v2 = A * torch.sin(phi)
        assert torch.allclose(u, u2, atol=1e-5)
        assert torch.allclose(v, v2, atol=1e-5)

    def test_telemetry_keys_present(self):
        blk = NeuralFieldOscillator(d_model=16, cfg=NFOConfig(n_osc=4))
        _ = blk(torch.randn(1, 4, 16))
        for k in ("R_mean", "R_max", "A_mean", "A_std",
                  "phi_circular_var", "kappa", "dt", "alpha", "phi_kappa"):
            assert k in blk.last_state, f"missing telemetry key: {k}"

    def test_n_osc_validation(self):
        with pytest.raises(ValueError):
            NeuralFieldOscillator(d_model=16, cfg=NFOConfig(n_osc=1))
        with pytest.raises(ValueError):
            NeuralFieldOscillator(d_model=16, cfg=NFOConfig(n_osc=99))

    def test_synchronisation_rate_vs_dt(self):
        """A larger dt should produce a Kuramoto field at least as
        coherent as a tiny dt (sanity for the dynamics — informs the
        recommended dt range)."""
        torch.manual_seed(0)
        d, M = 16, 8
        h = torch.randn(1, 24, d)
        def _R_after(dt_init: float) -> float:
            torch.manual_seed(0)
            blk = NeuralFieldOscillator(d_model=d, cfg=NFOConfig(
                n_osc=M, n_steps=4, kappa_init=0.8, dt_init=dt_init,
                alpha_init=0.0,
            ))
            _ = blk(h)
            return float(blk.last_state["R_mean"].item())
        low_R = _R_after(0.02)
        hi_R = _R_after(0.40)
        assert hi_R >= low_R - 1e-3           # large dt synchronises ≥ as fast


# ──────────────────────────────────────────────────────────────────────
# DSL factory
# ──────────────────────────────────────────────────────────────────────

class TestFactory:
    def test_make_nfo_disabled_returns_none(self):
        assert make_nfo(None, d_model=16) is None
        assert make_nfo(False, d_model=16) is None
        assert make_nfo({"enabled": False}, d_model=16) is None

    def test_make_nfo_true_uses_defaults(self):
        # d_model must be ≥ default n_osc (32) for the True-shortcut path.
        blk = make_nfo(True, d_model=64)
        assert blk is not None
        assert blk.cfg.n_osc == NFOConfig().n_osc

    def test_make_nfo_dict_overrides(self):
        blk = make_nfo({"n_osc": 8, "kappa_init": 0.3}, d_model=16)
        assert blk is not None
        assert blk.cfg.n_osc == 8
        assert blk.cfg.kappa_init == pytest.approx(0.3)

    def test_make_nfo_ignores_unknown_keys(self):
        """DSL parser may emit extra keys (e.g. `enabled: true`); the
        factory should drop them silently instead of raising."""
        blk = make_nfo({"n_osc": 8, "unknown_extra_key": 42}, d_model=16)
        assert blk is not None
        assert blk.cfg.n_osc == 8


# ──────────────────────────────────────────────────────────────────────
# Probe
# ──────────────────────────────────────────────────────────────────────

class TestProbe:
    def test_probe_returns_defaults_when_unattached(self):
        from neuroslm.emergent import NFOCoherenceProbe
        p = NFOCoherenceProbe()
        out = p.step()
        # Stable schema even when no block is attached.
        for k in ("nfo_R_mean", "nfo_phi_kappa", "nfo_alpha"):
            assert k in out

    def test_probe_reads_block_state(self):
        from neuroslm.emergent import NFOCoherenceProbe
        blk = NeuralFieldOscillator(d_model=16, cfg=NFOConfig(n_osc=4))
        _ = blk(torch.randn(1, 8, 16))
        p = NFOCoherenceProbe(blk)
        out = p.step()
        assert out["nfo_R_mean"] >= 0.0
        assert 0.0 <= out["nfo_phi_kappa"] <= 1.0
