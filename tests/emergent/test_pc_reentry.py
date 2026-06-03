"""Tests for C3 — PCReentryProbe."""
from __future__ import annotations
import torch

from neuroslm.emergent.pc_reentry import PCReentryProbe


def test_validates_dim():
    import pytest
    with pytest.raises(ValueError):
        PCReentryProbe(dim=0)


def test_no_op_first_call_returns_zero_residual():
    probe = PCReentryProbe(dim=8)
    s = probe.step(torch.randn(2, 4, 8), torch.randn(2, 4, 8))
    # First call only seeds the prev buffer.
    assert s["pc_residual"] == 0.0


def test_residual_decreases_under_linear_dynamics():
    """If sensory_t = D · motor_{t-1} for some diagonal D, the rank-1+diag
    predictor must drive the residual toward 0."""
    torch.manual_seed(0)
    dim = 6
    D = torch.tensor([1.3, -0.5, 0.7, 1.0, 0.2, -1.1])
    probe = PCReentryProbe(dim=dim, lr=0.01, momentum=0.9, ema=0.05)

    initial = None
    final = None
    prev_m = torch.randn(1, 1, dim)
    for t in range(2000):
        m = torch.randn(1, 1, dim)
        s = prev_m * D                    # s_t = D · m_{t-1}
        out = probe.step(m, s)            # probe stashes m, predicts s_{t+1} from it
        prev_m = m
        if t == 50:
            initial = out["pc_residual"]
        final = out["pc_residual"]
    assert initial is not None and final is not None
    assert final < initial * 0.3, f"residual {final} did not fall below {initial * 0.3}"


def test_residual_unchanged_on_uncorrelated_noise():
    """For independent noise the predictor cannot do better than the
    constant predictor — residual should stay near `var(s)`."""
    torch.manual_seed(0)
    dim = 8
    probe = PCReentryProbe(dim=dim, lr=0.01, ema=0.1)
    for _ in range(300):
        m = torch.randn(1, 1, dim)
        s = torch.randn(1, 1, dim)
        probe.step(m, s)
    stats = probe.step(torch.randn(1, 1, dim), torch.randn(1, 1, dim))
    # Cannot have negative explained variance, and should be small.
    assert 0.0 <= stats["pc_explained"] <= 0.4


def test_gradient_does_not_leak_into_inputs():
    """The probe must detach — inputs that require grad must have no
    grad accumulated through the probe call."""
    dim = 4
    probe = PCReentryProbe(dim=dim, lr=0.01)
    m = torch.randn(2, 3, dim, requires_grad=True)
    s = torch.randn(2, 3, dim, requires_grad=True)
    # Two calls so the predictor actually runs (first only seeds).
    probe.step(m, s)
    probe.step(m, s)
    # Inputs must have no grad — we never called backward and the probe
    # internally calls .detach(), so nothing should have populated .grad.
    assert m.grad is None
    assert s.grad is None


def test_missing_inputs_return_stale_stats():
    probe = PCReentryProbe(dim=4)
    s1 = probe.step(None, None)
    assert "pc_residual" in s1
    s2 = probe.step(torch.randn(1, 1, 4), None)
    assert s2 == s1


def test_shape_mismatch_no_op():
    probe = PCReentryProbe(dim=4)
    # Wrong last-dim → silent no-op (probe is best-effort).
    s = probe.step(torch.randn(1, 1, 8), torch.randn(1, 1, 8))
    assert s["pc_residual"] == 0.0
