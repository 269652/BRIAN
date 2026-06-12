# -*- coding: utf-8 -*-
"""Algebraic contracts for ``TopologicalDifferentialWorkspace`` (TDW).

TDW is an **opt-in** drop-in replacement for ``GlobalWorkspace`` that
applies a differential gate at the slot level. The forward computes
two Hopfield retrievals from the same query set against the same
candidates but with two different inverse temperatures::

    Gamma_1 = Hopfield(slots, candidates, beta_1)   # "sharp" retrieval
    Gamma_2 = Hopfield(slots, candidates, beta_2)   # "blurry" retrieval
    Gamma_syn = Gamma_1 - lambda * Gamma_2          # differential (DIFF-style)

with optional:
  * ``synergy_gate``: scale each slot row by
    ``||Gamma_syn||^2 / (||Gamma_1||^2 + eps)`` clamped to ``[0, 1]``
    (a scalar proxy for "what fraction of the sharp retrieval survives
    cancellation by the smoothed retrieval");
  * ``tonnetz``: project the slots onto the column-space of a learnable
    QR-orthonormalised basis ``U`` so the slots inherit a positive
    spectral gap by construction (sigma_min of an orthonormal-column
    matrix is 1.0 → gap = 1.0 - 0.0 = 1.0).

Scope of this suite (per CLAUDE.md §13: mechanism only, no perf claims):

  * **Construction**: defaults, shape, dtype.
  * **Algebraic invariants**: common-mode cancellation, orthogonal
    differential preservation, synergy mask bounds.
  * **Spectral**: when tonnetz is on, the QR basis is column-orthonormal
    so sigma_min >= 1 - tol.
  * **Drop-in**: output shape + dtype match GlobalWorkspace so it can be
    swapped at the call-site without touching downstream code.
  * **Gradient flow**: lambda, both betas, and the basis receive
    non-zero gradients on a tiny LM-style loss.
  * **No NaNs/infs** on tiny / oddly-shaped inputs.

Any **system-level** claim ("improves OOD", "reduces hallucinations")
is out of scope here and lives only in ``docs/findings.md`` after a
recorded run produces an artefact.
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn


# ────────────────────────────────────────────────────────────────────────
# Construction & shape
# ────────────────────────────────────────────────────────────────────────
class TestConstruction:
    """The class exists, instantiates with safe defaults, and emits the
    same shape as ``GlobalWorkspace`` so it is drop-in compatible."""

    def test_module_importable(self):
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        assert TopologicalDifferentialWorkspace is not None

    def test_defaults_construct_without_args(self):
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        tdw = TopologicalDifferentialWorkspace(d_sem=32, n_slots=4)
        # Defaults must not raise; expose lambda / both beta logs so the
        # caller can probe the differential gate.
        assert hasattr(tdw, "log_lambda")
        assert hasattr(tdw, "log_beta_1")
        assert hasattr(tdw, "log_beta_2")

    def test_forward_shape_matches_global_workspace(self):
        """TDW(candidates) shape must equal GWS(candidates) so it is a
        true drop-in at the brain.py call-sites."""
        from neuroslm.modules.workspace import (
            GlobalWorkspace, TopologicalDifferentialWorkspace,
        )
        d_sem, n_slots, K, B = 32, 4, 6, 2
        gws = GlobalWorkspace(d_sem=d_sem, n_slots=n_slots)
        tdw = TopologicalDifferentialWorkspace(d_sem=d_sem, n_slots=n_slots)
        torch.manual_seed(0)
        x = torch.randn(B, K, d_sem)
        gws_out = gws(x)
        tdw_out = tdw(x)
        assert tdw_out.shape == gws_out.shape == (B, n_slots, d_sem)

    def test_forward_preserves_dtype(self):
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        tdw = TopologicalDifferentialWorkspace(d_sem=16, n_slots=4)
        # Don't test bf16 on CPU (matmul fallbacks differ); just confirm
        # float32 in -> float32 out (no silent .float() cast escapes).
        x = torch.randn(1, 4, 16, dtype=torch.float32)
        out = tdw(x)
        assert out.dtype == torch.float32, (
            f"TDW should preserve input dtype; got {out.dtype}"
        )

    def test_accepts_ne_temp_like_global_workspace(self):
        """Drop-in contract: ``ne_temp`` kwarg must be accepted (used by
        brain.py: ``self.gws(candidates, ne_temp=nt[:, NT_NAMES.index('NE')])``)."""
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        tdw = TopologicalDifferentialWorkspace(d_sem=16, n_slots=4)
        x = torch.randn(2, 4, 16)
        ne = torch.tensor([1.0, 1.5])
        out = tdw(x, ne_temp=ne)
        assert out.shape == (2, 4, 16)

    def test_forward_is_finite_on_tiny_input(self):
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        tdw = TopologicalDifferentialWorkspace(d_sem=8, n_slots=2)
        x = torch.randn(1, 2, 8) * 1e-3
        out = tdw(x)
        assert torch.isfinite(out).all(), "TDW output had non-finite values"


# ────────────────────────────────────────────────────────────────────────
# Algebraic invariants — the whole point of the differential gate.
# ────────────────────────────────────────────────────────────────────────
class TestAlgebraicInvariants:
    """The differential gate must cancel common-mode components and
    preserve differential components. These are math identities, not
    performance claims."""

    def test_common_mode_cancellation_when_betas_equal_and_lambda_one(self):
        """When both Hopfield pathways are identical (same beta, same
        candidates, same init slots) and lambda = 1, the differential
        ``Gamma_1 - lambda * Gamma_2`` must be exactly zero **before**
        the synergy mask and the output norm fire. The pre-norm tensor
        is exposed via ``_last_diff`` for this test."""
        import torch.nn.functional as F
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        # The two pathways use SLIGHTLY different beta formulas:
        #   beta_1 = softplus(log_beta_1) + 0.5    (sharp, has +0.5 floor)
        #   beta_2 = softplus(log_beta_2)          (blurry, no floor)
        # So to force them numerically equal we calibrate log_beta_2 to
        # match the +0.5 offset:
        #   softplus(log_beta_2) := softplus(log_beta_1) + 0.5
        #   log_beta_2 = ln(exp(target) - 1) where target = beta_1.
        tdw = TopologicalDifferentialWorkspace(
            d_sem=16, n_slots=4,
            synergy_gate=False, tonnetz=False,
        )
        with torch.no_grad():
            target = float(F.softplus(tdw.log_beta_1).item()) + 0.5
            inv_b2 = math.log(math.exp(target) - 1.0)
            tdw.log_beta_2.fill_(inv_b2)
            # Set lambda = 1 exactly: softplus(x) = 1 <=> x = ln(e - 1).
            tdw.log_lambda.fill_(math.log(math.e - 1.0))
        x = torch.randn(2, 6, 16)
        tdw(x)   # populates _last_diff
        assert tdw._last_diff is not None
        max_abs = float(tdw._last_diff.abs().max())
        assert max_abs < 1e-5, (
            f"Common-mode (identical pathways, lambda=1) must cancel to "
            f"~0; got max|diff| = {max_abs}"
        )

    def test_orthogonal_pathways_preserve_signal(self):
        """When the two pathways produce well-separated retrievals (one
        sharp, one blurry on the same candidates) the differential must
        retain a non-trivial signal — otherwise the gate would erase
        legitimate information, not just noise."""
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        # Wide beta spread: sharp beta_1 >> blurry beta_2 ⇒ Gamma_1 is
        # close to a single-candidate retrieval, Gamma_2 is close to the
        # candidate mean. Their difference is a non-trivial signal even
        # at lambda = 1.
        tdw = TopologicalDifferentialWorkspace(
            d_sem=16, n_slots=4,
            synergy_gate=False, tonnetz=False,
        )
        with torch.no_grad():
            # beta_1 = softplus(2.0) + 0.5 ~= 2.62 (sharp).
            # beta_2 = softplus(-3.0)      ~= 0.049 (very blurry).
            tdw.log_beta_1.fill_(2.0)
            tdw.log_beta_2.fill_(-3.0)
        torch.manual_seed(7)
        x = torch.randn(2, 6, 16)
        out = tdw(x)
        # Pre-norm tensor used for the assertion — the final layernorm
        # would mask differences in magnitude.
        assert tdw._last_diff is not None
        assert tdw._last_diff.abs().mean() > 1e-3, (
            "Sharp/blurry pathways should leave a non-trivial residual; "
            f"got mean|diff| = {tdw._last_diff.abs().mean():.2e}"
        )
        # And the post-norm output is still well-formed.
        assert torch.isfinite(out).all()

    def test_synergy_mask_in_unit_interval(self):
        """The synergy proxy is ``clamp(||diff||^2 / ||gamma_1||^2, 0, 1)``
        and must stay in [0, 1] for any input."""
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        tdw = TopologicalDifferentialWorkspace(
            d_sem=16, n_slots=4, synergy_gate=True, tonnetz=False,
        )
        torch.manual_seed(0)
        x = torch.randn(3, 8, 16)
        tdw(x)
        m = tdw._last_synergy_mask
        assert m is not None
        assert float(m.min()) >= 0.0
        assert float(m.max()) <= 1.0 + 1e-5

    def test_synergy_gate_off_equals_pure_differential_in_pre_norm(self):
        """``synergy_gate=False`` must short-circuit the mask multiply.
        Pre-norm diff equals the differential without any scaling."""
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        tdw_on  = TopologicalDifferentialWorkspace(
            d_sem=16, n_slots=4, synergy_gate=True,  tonnetz=False)
        tdw_off = TopologicalDifferentialWorkspace(
            d_sem=16, n_slots=4, synergy_gate=False, tonnetz=False)
        # Force identical params so the only difference is the gate.
        tdw_off.load_state_dict(tdw_on.state_dict())
        torch.manual_seed(0)
        x = torch.randn(2, 4, 16)
        tdw_off(x)
        tdw_on(x)
        # The pre-norm differential is captured BEFORE the synergy mask,
        # so it must agree exactly regardless of the gate.
        assert torch.allclose(tdw_off._last_diff, tdw_on._last_diff,
                              atol=1e-6)
        # And the synergy-mask buffer is None when the gate is off.
        assert tdw_off._last_synergy_mask is None
        assert tdw_on._last_synergy_mask  is not None


# ────────────────────────────────────────────────────────────────────────
# Tonnetz basis — spectral gap by construction
# ────────────────────────────────────────────────────────────────────────
class TestTonnetzBasis:
    """When ``tonnetz=True``, slots are projected onto the column-space
    of a QR-orthonormalised learnable basis ``U`` (d_sem, n_slots).
    Column-orthonormal => sigma_min(U) = 1 => spectral gap = 1 - 0 = 1
    by construction. The class exposes ``spectral_gap()`` for the
    verifier and a buffer ``_last_basis_smin`` for this test."""

    def test_tonnetz_basis_columns_are_orthonormal_after_forward(self):
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        tdw = TopologicalDifferentialWorkspace(
            d_sem=32, n_slots=8, tonnetz=True,
        )
        x = torch.randn(2, 5, 32)
        tdw(x)
        # U_orth has shape (d_sem, n_slots); U_orth^T @ U_orth must be
        # the n_slots-by-n_slots identity to within QR numerical noise.
        u = tdw._last_basis_orth
        assert u is not None and u.shape == (32, 8)
        gram = u.T @ u
        eye = torch.eye(8, dtype=u.dtype, device=u.device)
        max_err = float((gram - eye).abs().max())
        assert max_err < 1e-4, (
            f"Tonnetz basis columns must be orthonormal after QR; "
            f"||U^T U - I||_inf = {max_err}"
        )

    def test_spectral_gap_is_positive_when_tonnetz_enabled(self):
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        tdw = TopologicalDifferentialWorkspace(
            d_sem=32, n_slots=4, tonnetz=True,
        )
        x = torch.randn(1, 4, 32)
        tdw(x)
        gap = tdw.spectral_gap()
        # By construction (orthonormal columns), sigma_min == 1 so the
        # spectral gap to zero is 1.0. Verify it is at least 0.5 to leave
        # numerical slack.
        assert gap >= 0.5, (
            f"Tonnetz spectral gap should be ~1.0; got {gap:.3f}"
        )

    def test_tonnetz_disabled_reports_zero_gap(self):
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        tdw = TopologicalDifferentialWorkspace(
            d_sem=32, n_slots=4, tonnetz=False,
        )
        x = torch.randn(1, 4, 32)
        tdw(x)
        # No basis -> spectral_gap() returns 0.0 (telemetry says "off").
        assert tdw.spectral_gap() == 0.0


# ────────────────────────────────────────────────────────────────────────
# Gradient flow — the differential pieces are reachable from a loss.
# ────────────────────────────────────────────────────────────────────────
class TestGradientFlow:
    """Every learnable knob (lambda, both betas, the optional basis,
    the slot queries) must receive a non-zero gradient under a plain
    sum-loss. If any of them is unreachable, the corresponding
    sub-mechanism is dead weight in the architecture."""

    def test_lambda_receives_gradient(self):
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        tdw = TopologicalDifferentialWorkspace(
            d_sem=16, n_slots=4,
            synergy_gate=False, tonnetz=False,
        )
        x = torch.randn(2, 4, 16, requires_grad=False)
        out = tdw(x)
        out.sum().backward()
        assert tdw.log_lambda.grad is not None
        assert float(tdw.log_lambda.grad.abs().sum()) > 0.0

    def test_both_betas_receive_gradient(self):
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        tdw = TopologicalDifferentialWorkspace(
            d_sem=16, n_slots=4,
            synergy_gate=False, tonnetz=False,
        )
        x = torch.randn(2, 4, 16)
        tdw(x).sum().backward()
        for name, p in [("log_beta_1", tdw.log_beta_1),
                         ("log_beta_2", tdw.log_beta_2)]:
            assert p.grad is not None, f"{name} grad is None"
            assert float(p.grad.abs().sum()) > 0.0, (
                f"{name} grad is zero — pathway is unreachable from loss"
            )

    def test_tonnetz_basis_receives_gradient_when_enabled(self):
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        tdw = TopologicalDifferentialWorkspace(
            d_sem=16, n_slots=4,
            synergy_gate=False, tonnetz=True,
        )
        x = torch.randn(2, 4, 16)
        tdw(x).sum().backward()
        assert tdw.basis.grad is not None
        assert float(tdw.basis.grad.abs().sum()) > 0.0


# ────────────────────────────────────────────────────────────────────────
# Determinism + no-NaN safety net
# ────────────────────────────────────────────────────────────────────────
class TestSafety:

    def test_deterministic_under_seed(self):
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        torch.manual_seed(42)
        tdw1 = TopologicalDifferentialWorkspace(d_sem=16, n_slots=4)
        torch.manual_seed(42)
        tdw2 = TopologicalDifferentialWorkspace(d_sem=16, n_slots=4)
        x = torch.randn(2, 4, 16)
        # Same seed during construction + same input => same output.
        assert torch.allclose(tdw1(x), tdw2(x), atol=1e-6)

    def test_no_nans_under_zero_input(self):
        from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
        tdw = TopologicalDifferentialWorkspace(d_sem=16, n_slots=4)
        x = torch.zeros(2, 4, 16)
        out = tdw(x)
        assert torch.isfinite(out).all(), (
            "TDW emitted non-finite output on a zero-input — synergy "
            "ratio denominator likely missing eps guard."
        )
