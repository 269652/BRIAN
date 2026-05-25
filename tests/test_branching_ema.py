"""Tests for Surprise-Gated Branching EMA (neuroslm/intelligence/branching_ema.py).

Covers each mechanism individually + the full integration against the
synth-v1 divergence trace.

Run:
    .venv/Scripts/python.exe tests/test_branching_ema.py
"""
from __future__ import annotations
import sys, os, math
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroslm.intelligence.branching_ema import BranchingEMA


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

def _make_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))


def _params_dict(m: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def _mutate(m: nn.Module, scale: float = 0.01) -> None:
    """Simulate a training step: small random perturbation to all params."""
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * scale)


def _max_abs_diff(a: dict, b: dict) -> float:
    return max((a[k] - b[k]).abs().max().item() for k in a)


# ──────────────────────────────────────────────────────────────────────
# Unit tests — one mechanism at a time
# ──────────────────────────────────────────────────────────────────────

def test_velocity_direction():
    """PPL rising → positive velocity; PPL falling → negative velocity."""
    m = _make_model()
    bema = BranchingEMA(m, history_len=10)
    # Rising trajectory
    for step, ppl in enumerate([50, 80, 120, 200, 350, 500, 700]):
        bema.maybe_update(m, ppl=ppl, step=step)
    v_rising = bema._ppl_velocity()
    assert v_rising > 0.1, f"rising should produce positive velocity, got {v_rising}"

    # Reset for falling
    m2 = _make_model()
    bema2 = BranchingEMA(m2, history_len=10)
    for step, ppl in enumerate([5000, 3000, 1500, 800, 400, 200, 100]):
        bema2.maybe_update(m2, ppl=ppl, step=step)
    v_falling = bema2._ppl_velocity()
    assert v_falling < -0.1, f"falling should produce negative velocity, got {v_falling}"
    print(f"[1] velocity direction: rising={v_rising:+.3f}, falling={v_falling:+.3f}  PASS")


def test_alpha_gates_on_rising_not_falling():
    """alpha_eff during a rising-PPL window should be lower than during
    a flat-PPL window AT THE SAME PPL LEVEL.

    Isolates the gate term by holding the 1/avg_ppl base equal across
    the two scenarios. The bare-gate ratio must be sigmoid-dampened at
    least 3x by gamma=5 against a 1.6x-per-step rise (velocity =~ 0.5).
    """
    # Flat trace: ppl bounces around 500 (no trend → velocity =~ 0)
    flat = [500, 510, 490, 505, 495, 500, 510, 495, 500, 505]
    # Rising trace: ppl 1.6x per step, centered around 500 (so 1/avg_ppl
    # base is comparable). Starts at ~80, ends at ~3200, geometric mean ~500.
    rising = [80, 130, 200, 320, 510, 800, 1280, 2000, 3000, 4500]

    m = _make_model()
    bema_flat = BranchingEMA(m, history_len=10, gamma=5.0, base_alpha_cap=0.05)
    a_flat = []
    for step, ppl in enumerate(flat):
        a_flat.append(bema_flat.maybe_update(m, ppl=ppl, step=step)["alpha_eff"])

    m2 = _make_model()
    bema_rise = BranchingEMA(m2, history_len=10, gamma=5.0, base_alpha_cap=0.05)
    a_rise = []
    for step, ppl in enumerate(rising):
        a_rise.append(bema_rise.maybe_update(m2, ppl=ppl, step=step)["alpha_eff"])

    # Compare only the tail (history is full → velocity is stable)
    avg_flat = sum(a_flat[5:]) / 5
    avg_rise = sum(a_rise[5:]) / 5
    ratio = avg_flat / max(1e-12, avg_rise)
    assert ratio > 3.0, (
        f"flat-alpha should be >=3x rising-alpha at matched ppl regime; "
        f"flat={avg_flat:.6f}, rising={avg_rise:.6f}, ratio={ratio:.2f}")

    print(f"[2] alpha gating (matched-regime): "
          f"flat avg={avg_flat:.6f}, rising avg={avg_rise:.6f}  "
          f"(damping ratio={ratio:.1f}x)  PASS")


def test_best_tracker_captures_dip():
    """best_ema_ppl must track down through a descent to ~50,
    NOT lag at 800+ as in v1 of the implementation."""
    m = _make_model()
    bema = BranchingEMA(m, history_len=10, gamma=5.0, base_alpha_cap=0.05)
    descent = [5000, 1000, 500, 300, 200, 150, 100, 80, 65, 55, 50, 48, 52, 50, 49]
    for step, ppl in enumerate(descent):
        _mutate(m)
        bema.maybe_update(m, ppl=ppl, step=step)
    assert bema._best_ema_ppl < 100, (
        f"best_ema_ppl should reach < 100 after descent to 50, "
        f"got {bema._best_ema_ppl:.1f}")
    print(f"[3] best tracker: best_ema_ppl = {bema._best_ema_ppl:.1f} (< 100 required)  PASS")


def test_collapse_triggers_on_spike():
    """After descending to ~50, a spike to 500 should trigger
    collapse-to-best (current_ppl > 3 × best_ema_ppl)."""
    m = _make_model()
    bema = BranchingEMA(m, history_len=10, gamma=5.0, base_alpha_cap=0.05)
    # First descend to a low PPL so best snapshot is taken
    for step, ppl in enumerate([5000, 1000, 500, 200, 100, 60, 50, 48, 52, 50]):
        _mutate(m)
        bema.maybe_update(m, ppl=ppl, step=step)
    snapshot_before_spike = _params_dict(m)

    # Now simulate post-spike weights: mutate the LIVE model hard, then
    # check the collapse pulls it back to the snapshot taken at the low
    _mutate(m, scale=1.0)   # big perturbation = corrupted model state
    pre_collapse = _params_dict(m)

    triggered = bema.maybe_collapse_to_best(m, current_ppl=500.0, require_history=5)
    assert triggered, "collapse should trigger when current_ppl=500 > 3 * best_ppl"

    post_collapse = _params_dict(m)
    diff_to_snap = _max_abs_diff(post_collapse, snapshot_before_spike)
    diff_to_pre  = _max_abs_diff(post_collapse, pre_collapse)
    assert diff_to_snap < 1e-5, (
        f"post-collapse model should match best snapshot, max diff={diff_to_snap}")
    assert diff_to_pre > 0.1, (
        f"collapse should have undone the big perturbation, max diff={diff_to_pre}")
    print(f"[4] collapse: triggered={triggered}, "
          f"diff_to_snapshot={diff_to_snap:.2e}, "
          f"diff_to_pre_collapse={diff_to_pre:.3f}  PASS")


def test_collapse_does_not_trigger_during_descent():
    """While PPL is descending, no collapse should fire even though
    PPL is still much higher than the eventual minimum."""
    m = _make_model()
    bema = BranchingEMA(m, history_len=10, gamma=5.0, base_alpha_cap=0.05)
    descent = [5000, 1000, 500, 300, 200, 150, 100, 80]
    any_collapse = False
    for step, ppl in enumerate(descent):
        _mutate(m)
        bema.maybe_update(m, ppl=ppl, step=step)
        if bema.maybe_collapse_to_best(m, current_ppl=ppl, require_history=3):
            any_collapse = True
    assert not any_collapse, (
        "collapse must not fire during normal descent — it would prevent "
        "the model from ever finding lower-PPL basins")
    print("[5] no spurious collapse during descent  PASS")


def test_stable_ema_absorbs_during_descent():
    """params_stable should ACCUMULATE updates during descent."""
    m = _make_model()
    bema = BranchingEMA(m, history_len=10, gamma=5.0, base_alpha_cap=0.05)
    stable_initial = {k: v.clone() for k, v in bema._stable.items()}

    descent = [5000, 1000, 500, 200, 100, 60, 50, 50, 49, 50, 51, 49, 50, 50]
    for step, ppl in enumerate(descent):
        _mutate(m, scale=0.05)
        bema.maybe_update(m, ppl=ppl, step=step)

    # Stable shadow should have drifted from its initial snapshot
    drift = max(
        (bema._stable[k] - stable_initial[k]).abs().max().item()
        for k in stable_initial
    )
    assert drift > 1e-4, (
        f"params_stable should accumulate >1e-4 drift over a 14-step "
        f"descent, got {drift:.2e}")
    print(f"[6] stable shadow absorbs during descent (drift={drift:.4f})  PASS")


def test_synth_v1_real_trace():
    """End-to-end on the actual synth-v1 trajectory: descent to 59,
    spike to 426, never recovers. BEMA should snapshot at step 9, then
    collapse during the late spikes."""
    ppl_trace = [
        5000, 1000, 500, 300, 200, 150, 100, 80, 65, 59,   # descent to 59 (step 9)
        111, 90, 95, 257, 187, 247, 210, 426, 256, 228     # divergence
    ]
    m = _make_model()
    bema = BranchingEMA(m, history_len=10, gamma=5.0, base_alpha_cap=0.05)

    n_collapses = 0
    best_after_descent = None
    for step, ppl in enumerate(ppl_trace):
        _mutate(m, scale=0.02)
        bema.maybe_update(m, ppl=ppl, step=step)
        if step == 9:
            best_after_descent = bema._best_ema_ppl
        triggered = bema.maybe_collapse_to_best(m, current_ppl=ppl, require_history=5)
        if triggered:
            n_collapses += 1

    assert best_after_descent is not None
    assert best_after_descent < 200, (
        f"after descent to 59, best_ema_ppl should be < 200, "
        f"got {best_after_descent:.1f}")
    assert n_collapses >= 1, (
        f"the post-step-13 spikes (257, 426) should trigger >=1 collapse, "
        f"got {n_collapses}")
    print(f"[7] synth-v1 trace: best_after_descent={best_after_descent:.1f}, "
          f"n_collapses={n_collapses}, n_freezes={bema._n_freezes}  PASS")


def test_state_dict_roundtrip():
    """Saving and reloading should preserve the stable + best shadows."""
    m = _make_model()
    bema = BranchingEMA(m, history_len=10, gamma=5.0, base_alpha_cap=0.05)
    for step, ppl in enumerate([5000, 1000, 500, 100, 60, 50, 50, 50]):
        _mutate(m, scale=0.02)
        bema.maybe_update(m, ppl=ppl, step=step)
    sd = bema.state_dict()
    stable_orig = {k: v.clone() for k, v in bema._stable.items()}
    best_orig = {k: v.clone() for k, v in bema._best.items()}

    # Roundtrip
    m_new = _make_model()
    bema_new = BranchingEMA(m_new, history_len=10, gamma=5.0, base_alpha_cap=0.05)
    bema_new.load_state_dict(sd, m_new)
    for k in stable_orig:
        assert torch.allclose(bema_new._stable[k], stable_orig[k]), (
            f"stable shadow mismatch on {k}")
        assert torch.allclose(bema_new._best[k], best_orig[k]), (
            f"best shadow mismatch on {k}")
    assert bema_new._best_ema_ppl == bema._best_ema_ppl
    assert list(bema_new.history) == list(bema.history)
    print("[8] state_dict roundtrip preserves shadows  PASS")


def test_invariance_to_ppl_magnitude():
    """Gamma should NOT need re-tuning when PPL scale changes — using
    log-PPL velocity makes this scale-invariant."""
    # Same RELATIVE divergence at two scales
    trace_low = [50, 80, 130, 220, 360, 600]     # ~1.6x per step
    trace_high = [500, 800, 1300, 2200, 3600, 6000]   # also ~1.6x

    m_l = _make_model()
    bema_l = BranchingEMA(m_l, history_len=10, gamma=5.0, base_alpha_cap=0.05)
    for step, ppl in enumerate(trace_low):
        _mutate(m_l)
        bema_l.maybe_update(m_l, ppl=ppl, step=step)
    v_low = bema_l._ppl_velocity()

    m_h = _make_model()
    bema_h = BranchingEMA(m_h, history_len=10, gamma=5.0, base_alpha_cap=0.05)
    for step, ppl in enumerate(trace_high):
        _mutate(m_h)
        bema_h.maybe_update(m_h, ppl=ppl, step=step)
    v_high = bema_h._ppl_velocity()

    rel_diff = abs(v_low - v_high) / max(abs(v_low), abs(v_high), 1e-9)
    assert rel_diff < 0.05, (
        f"log-PPL velocity should be scale-invariant; got v_low={v_low:.3f}, "
        f"v_high={v_high:.3f}, rel_diff={rel_diff:.3f}")
    print(f"[9] scale-invariance: v_low={v_low:.3f} =~ v_high={v_high:.3f}  PASS")


def test_handles_param_shape_growth():
    """BDNF growth resizes parameters mid-training (kern_a/kern_b grow
    along the rank dim). BEMA must NOT crash on the shape mismatch and
    must keep training going."""
    # Model with a tensor we'll grow
    class GrowingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
            # Initial low-rank kernel: (8, 4) shaped like NeuralGeometryAdapter
            self.kern_a = nn.Parameter(torch.randn(8, 4) * 0.01)
        def forward(self, x):
            return self.fc(x)

    m = GrowingModel()
    bema = BranchingEMA(m, history_len=10, gamma=5.0, base_alpha_cap=0.05)

    # Normal pre-growth updates
    for step in range(5):
        _mutate(m, scale=0.01)
        bema.maybe_update(m, ppl=100.0, step=step)

    # BDNF growth: kern_a goes from (8, 4) -> (8, 8)
    with torch.no_grad():
        new_block = torch.zeros(8, 4)
        m.kern_a = nn.Parameter(torch.cat([m.kern_a.data, new_block], dim=1))

    # This MUST NOT crash now — BEMA should handle the shape mismatch
    diag = bema.maybe_update(m, ppl=95.0, step=5)
    assert "alpha_eff" in diag, f"maybe_update should still return diagnostics after growth: {diag}"

    # After update, stable shadow shape should match the new param
    assert bema._stable["kern_a"].shape == m.kern_a.shape, (
        f"stable shadow should be reshaped to {m.kern_a.shape}, "
        f"got {bema._stable['kern_a'].shape}")

    # Continue training, more updates should work normally
    for step in range(6, 12):
        _mutate(m, scale=0.01)
        d = bema.maybe_update(m, ppl=90.0 - step, step=step)
        assert "alpha_eff" in d

    # And collapse should not crash either (even if best snapshot was at old shape)
    triggered = bema.maybe_collapse_to_best(m, current_ppl=500.0, require_history=5)
    # Whether it triggers depends on best_ema_ppl; the point is NO crash
    print(f"[10] handles BDNF param shape growth (kern_a: (8,4) -> (8,8))  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Branching EMA tests")
    print("=" * 60)
    test_velocity_direction()
    test_alpha_gates_on_rising_not_falling()
    test_best_tracker_captures_dip()
    test_collapse_triggers_on_spike()
    test_collapse_does_not_trigger_during_descent()
    test_stable_ema_absorbs_during_descent()
    test_synth_v1_real_trace()
    test_state_dict_roundtrip()
    test_invariance_to_ppl_magnitude()
    test_handles_param_shape_growth()
    print("=" * 60)
    print("ALL TESTS PASSED")
