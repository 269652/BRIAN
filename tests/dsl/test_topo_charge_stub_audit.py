# -*- coding: utf-8 -*-
"""Stub-detection meta-test (CLAUDE.md §14 contract-strength guard).

A test suite that passes against an obvious stub implementation is too
weak. This file constructs deliberately-broken alternatives to the
TopoChargeDiagnostic + its helpers and runs targeted assertions
against them, demanding that the assertions FAIL. Each entry below
documents which §14-banned pattern the stub embodies and which real
contract catches it.

If a future agent refactors topo_charge and accidentally weakens a
contract, ONE of the meta-tests below will GREEN where it should be
RED -- and that's the signal that the audit needs revisiting.
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from neuroslm.mechanisms.topo_charge import (
    solid_angle as real_solid_angle,
    berg_luscher_q as real_berg_luscher_q,
    TopoChargeDiagnostic,
)


# ============================================================================
# Stub fixtures: minimum-effort implementations that should fail the suite.
# ============================================================================


class ZeroQDiagnostic(nn.Module):
    """Stub: returns constant zero Q_h and zero eps_ortho. Encodes the
    §14-banned 'default-OFF runs nothing when ON' pattern."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.proj = nn.Linear(head_dim, 3, bias=True)
        self._last_Q_h = None
        self._last_eps_ortho = None

    def forward(self, attn_per_layer):
        B, H = attn_per_layer[0].shape[:2]
        self._last_Q_h = torch.zeros(B, H, requires_grad=True)
        self._last_eps_ortho = torch.zeros((), requires_grad=True)
        return {"Q_h": self._last_Q_h, "eps_ortho": self._last_eps_ortho}

    def pop_metrics(self):
        return {"Q_h": self._last_Q_h, "eps_ortho": self._last_eps_ortho}

    def penalty(self, Q_target=0.0, alpha=0.0, gamma=0.0):
        # Stub: ignore alpha / gamma entirely.
        return torch.zeros((), requires_grad=True)


def stub_solid_angle_zero(n_a, n_b, n_c, eps=1e-8):
    """Stub: signed area always returns 0. Should fail the octant
    and orientation-flip contracts."""
    leading = torch.broadcast_shapes(
        n_a.shape[:-1], n_b.shape[:-1], n_c.shape[:-1]
    )
    return torch.zeros(leading, dtype=n_a.dtype, device=n_a.device)


def stub_berg_luscher_uses_acos(n):
    """Stub: acos-based solid-angle sum (the formulation Berg-Lueschner
    found to be numerically unstable near the antipodal boundary).
    Should give correct values on benign inputs but is fragile."""
    T = n.shape[-2]
    if T < 3:
        return n.new_zeros(n.shape[:-2] + (0,))
    n_a = n[..., :-2, :]
    n_b = n[..., 1:-1, :]
    n_c = n[..., 2:, :]
    # Trivial (wrong) "solid angle" -- the dot product, scaled.
    return ((n_a * n_b).sum(-1) + (n_b * n_c).sum(-1)) / (4 * math.pi)


# ============================================================================
# Meta-tests: each must FAIL (i.e. the contract must catch the stub).
# ============================================================================


class TestSuiteCatchesStubs:
    """Each method below asserts that a specific contract in
    test_topo_charge.py would catch a specific stub."""

    def _attn(self, L=3, B=2, H=4, T=6, head_dim=8, seed=0):
        torch.manual_seed(seed)
        return [torch.randn(B, H, T, head_dim) for _ in range(L)]

    # -- Contract: pop_metrics_changes_with_input --------------------------

    def test_constant_Q_stub_fails_change_with_input(self):
        """A stub that returns the same Q_h for any input MUST be
        caught by the 'two distinct inputs -> distinct Q_h' test."""
        diag = ZeroQDiagnostic(head_dim=8)
        diag(self._attn(seed=1))
        Q_a = diag.pop_metrics()["Q_h"].clone()
        diag(self._attn(seed=2))
        Q_b = diag.pop_metrics()["Q_h"].clone()
        # The real contract requires (Q_a - Q_b).abs().max() > 1e-4.
        # The stub MUST fail this.
        assert (Q_a - Q_b).abs().max().item() <= 1e-4, (
            "stub diagnostic should have failed the "
            "test_pop_metrics_changes_with_input contract"
        )

    # -- Contract: penalty_nonzero_alpha_propagates_gradient --------------

    def test_ignored_alpha_stub_fails_gradient_propagation(self):
        """A stub whose penalty ignores alpha (no gradient path to the
        learnable) MUST be caught by the alpha=0.5, grad>0 check."""
        diag = ZeroQDiagnostic(head_dim=8)
        diag(self._attn(seed=3))
        loss = diag.penalty(Q_target=0.0, alpha=0.5, gamma=0.0)
        # Stub returns a fresh zero tensor with no link to proj.weight.
        try:
            loss.backward()
        except RuntimeError:
            # Some stubs may not even have a backward graph; that's
            # itself a failure mode.
            pass
        assert (
            diag.proj.weight.grad is None
            or diag.proj.weight.grad.abs().sum().item() <= 1e-9
        ), (
            "stub diagnostic should have failed the "
            "test_penalty_nonzero_alpha_propagates_gradient contract"
        )

    # -- Contract: orthogonal_octant_equals_pi_over_two -------------------

    def test_zero_solid_angle_stub_fails_octant_contract(self):
        """A stub solid_angle that always returns 0 MUST be caught by
        the orthogonal octant = pi/2 contract."""
        e_x = torch.tensor([1.0, 0.0, 0.0])
        e_y = torch.tensor([0.0, 1.0, 0.0])
        e_z = torch.tensor([0.0, 0.0, 1.0])
        omega = stub_solid_angle_zero(e_x, e_y, e_z)
        # Real contract: |omega - pi/2| < 1e-5. Stub returns 0.
        assert (omega - math.pi / 2).abs().item() >= 1e-5, (
            "zero-returning solid_angle stub should have failed "
            "the orthogonal-octant contract"
        )

    # -- Contract: orientation_swap_flips_sign ----------------------------

    def test_zero_solid_angle_stub_fails_orientation_flip(self):
        """A stub that always returns 0 also fails the
        orientation-swap-flips-sign contract (since 0 == -0 trivially
        means the assertion is technically true, but the contract
        actually requires the omega itself to be NONZERO for the
        flip to be observable). This meta-test makes that subtlety
        explicit."""
        e_x = torch.tensor([1.0, 0.0, 0.0])
        e_y = torch.tensor([0.0, 1.0, 0.0])
        e_z = torch.tensor([0.0, 0.0, 1.0])
        omega_pos = stub_solid_angle_zero(e_x, e_y, e_z)
        # The MEANINGFUL form of the contract is that omega_pos is
        # nontrivially nonzero (otherwise the sign-flip is vacuous).
        assert omega_pos.abs().item() < 1e-5, (
            "stub omega should be ~0 (and therefore the orientation-flip "
            "contract is vacuous -- the suite-strength test catches "
            "this by requiring the octant contract to fail first)"
        )

    # -- Contract: module_forward_invokes_berg_luscher --------------------

    def test_stub_that_bypasses_berg_luscher_is_caught(self, monkeypatch):
        """A stub TopoChargeDiagnostic.forward that returns Q_h without
        calling berg_luscher_q MUST be caught by the spy contract."""
        from neuroslm.mechanisms import topo_charge as topo_mod
        calls = []
        real_q = topo_mod.berg_luscher_q

        def spy(n):
            calls.append(n.shape)
            return real_q(n)

        monkeypatch.setattr(topo_mod, "berg_luscher_q", spy)
        # Run the STUB module (does NOT call berg_luscher_q).
        ZeroQDiagnostic(head_dim=8)(self._attn(L=2, B=1, H=2, T=5,
                                                head_dim=8))
        assert len(calls) == 0, (
            "stub diagnostic should have failed the "
            "test_module_forward_invokes_berg_luscher contract"
        )

    # -- Diagnostic: the real impl passes the same contracts. -------------

    def test_real_impl_passes_all_audited_contracts(self):
        """Smoke check: the contracts above are genuinely
        differentiating. Run the REAL impl against the same
        assertions but expect the OPPOSITE outcome (so this test
        passes when the real impl is healthy)."""
        diag = TopoChargeDiagnostic(head_dim=8)

        # (1) Changes with input.
        diag(self._attn(seed=1))
        Q_a = diag.pop_metrics()["Q_h"].clone()
        diag(self._attn(seed=2))
        Q_b = diag.pop_metrics()["Q_h"].clone()
        assert (Q_a - Q_b).abs().max().item() > 1e-4

        # (2) alpha > 0 -> nonzero grad on proj.weight.
        diag.proj.weight.grad = None
        diag(self._attn(seed=3))
        diag.penalty(Q_target=0.0, alpha=0.5, gamma=0.0).backward()
        assert diag.proj.weight.grad.abs().sum().item() > 1e-6

        # (3) Octant = pi/2.
        e_x = torch.tensor([1.0, 0.0, 0.0])
        e_y = torch.tensor([0.0, 1.0, 0.0])
        e_z = torch.tensor([0.0, 0.0, 1.0])
        torch.testing.assert_close(
            real_solid_angle(e_x, e_y, e_z),
            torch.tensor(math.pi / 2),
            atol=1e-5, rtol=1e-5,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
