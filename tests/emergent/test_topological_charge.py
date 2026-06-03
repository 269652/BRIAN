"""Tests for C4 — TopologicalChargeProbe."""
from __future__ import annotations
import math
import torch

from neuroslm.emergent.topological_charge import TopologicalChargeProbe


def test_init_validates_dim():
    import pytest
    with pytest.raises(ValueError):
        TopologicalChargeProbe(dim=1)


def test_short_sequence_returns_zero():
    probe = TopologicalChargeProbe(dim=8)
    h = torch.randn(2, 1, 8)
    s = probe.step(h)
    assert s["Q_walls"] == 0.0
    assert s["Q_total"] == 0.0


def test_none_input_returns_last():
    probe = TopologicalChargeProbe(dim=4)
    s1 = probe.step(None)
    assert "Q_total" in s1


def test_R_is_skew_symmetric():
    probe = TopologicalChargeProbe(dim=16, seed=1)
    R = probe._R
    asym = (R + R.T).abs().max().item()
    assert asym < 1e-5


def test_walls_counted_on_antipodal_flip():
    """Build a sequence where h_t and h_{t+1} are anti-parallel every
    other step → ⟨h_t, h_{t+1}⟩ < 0 → wall."""
    probe = TopologicalChargeProbe(dim=6, seed=0)
    v = torch.randn(6)
    v = v / v.norm()
    T = 10
    h = torch.stack([(-1.0) ** t * v for t in range(T)], dim=0)
    h = h.unsqueeze(0)             # (1, T, D)
    s = probe.step(h)
    # Every transition is a wall → T-1 walls per sequence.
    assert s["Q_walls"] == float(T - 1)


def test_no_walls_for_smooth_rotation():
    """A slowly rotating field in the (e0, e1) plane has all inner
    products ≥ cos(small angle) > 0 → no walls.

    Uses centred=False so the test signal (absolute orientation in the
    e0/e1 plane) isn't washed out by the per-sequence mean.
    """
    dim = 8
    probe = TopologicalChargeProbe(dim=dim, seed=0, centred=False)
    T = 64
    angles = torch.linspace(0.0, math.pi / 2, T)        # quarter turn
    h = torch.zeros(1, T, dim)
    h[0, :, 0] = torch.cos(angles)
    h[0, :, 1] = torch.sin(angles)
    s = probe.step(h)
    assert s["Q_walls"] == 0.0


def test_rotation_produces_nonzero_winding():
    """Same smooth rotation as above — winding number should be non-zero
    in magnitude (sign depends on R's chirality, which is seeded)."""
    dim = 4
    probe = TopologicalChargeProbe(dim=dim, seed=42, centred=False)
    T = 128
    angles = torch.linspace(0.0, 4.0 * math.pi, T)
    h = torch.zeros(1, T, dim)
    h[0, :, 0] = torch.cos(angles)
    h[0, :, 1] = torch.sin(angles)
    s = probe.step(h)
    # |Q| should be non-trivial (well above the integer-quantisation
    # noise floor for the chosen sequence).
    assert s["Q_abs"] > 0.1


def test_plateau_length_correct_with_known_walls():
    """Sequence length T-1 transitions, W walls → expected plateau
    length T-1 / (W + 1)."""
    probe = TopologicalChargeProbe(dim=4, seed=0, centred=False)
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    # 8 transitions: smooth, smooth, FLIP, smooth, smooth, smooth, FLIP, smooth
    h_seq = [v, v, v, -v, -v, -v, -v, v, v]
    h = torch.stack(h_seq).unsqueeze(0)             # (1, 9, 4)
    s = probe.step(h)
    # walls = 2 ⟹ plateau_len = 8 / 3 ≈ 2.667
    assert s["Q_walls"] == 2.0
    assert abs(s["Q_plateau_len"] - 8.0 / 3.0) < 0.01


def test_centred_mode_exposes_winding_in_high_dc_field():
    """A trunk-like signal: large constant DC + small structured noise.

    Without centring, the inner products of adjacent unit-vectors are
    pinned near +1 (the DC dominates), so |Q| collapses toward zero
    even though the underlying perturbation field has real structure.
    With centring (the default), the DC is removed and the structure
    becomes visible as a much larger winding magnitude.
    """
    dim = 32
    T = 256
    torch.manual_seed(0)
    # Strong DC + weak structured oscillation in a 2-plane.
    dc = torch.randn(dim) * 5.0
    angles = torch.linspace(0.0, 6.0 * math.pi, T)
    osc = torch.zeros(T, dim)
    osc[:, 0] = torch.cos(angles)
    osc[:, 1] = torch.sin(angles)
    h = (dc.unsqueeze(0) + osc).unsqueeze(0)         # (1, T, D)

    # Use the SAME R-seed for both so the comparison is fair.
    plain = TopologicalChargeProbe(dim=dim, seed=0, centred=False)
    s_plain = plain.step(h)
    centred = TopologicalChargeProbe(dim=dim, seed=0, centred=True)
    s_centred = centred.step(h)

    # The DC pins re ≈ +1 in the un-centred view → walls suppressed
    # AND |Q| compressed.  After centring both should grow.
    assert s_centred["Q_abs"] > 5.0 * max(s_plain["Q_abs"], 1e-6), (
        f"centring should amplify |Q| by ≥5×: plain={s_plain}, "
        f"centred={s_centred}")
    # Either the wall count rises, or |Q| itself crosses a sane floor.
    assert (s_centred["Q_walls"] > s_plain["Q_walls"]
            or s_centred["Q_abs"] > 0.2), (
        f"centred probe should reveal structure: {s_centred}")


def test_dim_mismatch_raises():
    probe = TopologicalChargeProbe(dim=4)
    import pytest
    with pytest.raises(ValueError):
        probe.step(torch.randn(1, 5, 8))


def test_raw_winding_shapes():
    probe = TopologicalChargeProbe(dim=4)
    h = torch.randn(2, 5, 4)
    w = probe.raw_winding(h)
    assert w["re"].shape == (2, 4)
    assert w["im"].shape == (2, 4)
    assert w["Q"].shape == (2,)
