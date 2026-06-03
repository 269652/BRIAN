"""Tests for C1 — DrivenNTSystem (homeostatic neuromodulatory field).

The standardiser is an EMA of mean and variance, so by construction a
**constant** input produces z=0 (no signal to react to).  Real training
data is noisy — the tests below mimic that by adding stochastic
fluctuation to all warmup drivers, then measuring channel response to
a sustained-level shift on top of the noise.

Guarded properties:
  * channels stay in the OPEN unit interval under any input (no rails)
  * each channel responds to its intended driver in the expected
    direction after standardiser warmup
  * constant good loss does NOT cause monotonic 5HT drift (OU leak)
  * removing drivers causes exponential relaxation back to baseline
  * compat shim ``step(activity=...)`` matches the legacy interface
"""
from __future__ import annotations
import math
import random

from neuroslm.emergent.driven_nt import DrivenNTSystem


_EPS = 1e-6


def _all_in_open_unit(d):
    return all(_EPS < v < 1.0 - _EPS for v in d.values())


def _noisy_warmup(nt, n=80, seed=0, **base):
    """Run ``n`` step_full calls with small gaussian noise on each
    driver, so the standardiser builds a non-degenerate variance
    estimate before the test signal is applied."""
    rng = random.Random(seed)
    for _ in range(n):
        kw = {}
        for k, v in base.items():
            if v is None:
                continue
            kw[k] = float(v) + 0.2 * rng.gauss(0.0, 1.0) * max(0.1, abs(v))
        nt.step_full(**kw)


# ── basic schema / construction ───────────────────────────────────────

def test_initial_levels_match_baselines():
    nt = DrivenNTSystem()
    assert nt.levels() == nt.baselines
    assert _all_in_open_unit(nt.levels())


def test_keys_match_legacy_baselines():
    nt = DrivenNTSystem()
    assert set(nt.levels()) == {"DA", "NE", "5HT", "ACh", "eCB", "Glu", "GABA"}


def test_levels_returns_fresh_dict():
    nt = DrivenNTSystem()
    a = nt.levels()
    a["DA"] = 999.0
    assert nt.levels()["DA"] != 999.0


def test_custom_baselines_respected():
    nt = DrivenNTSystem(baselines={"DA": 0.5})
    assert math.isclose(nt.baselines["DA"], 0.5)
    assert math.isclose(nt.levels()["DA"], 0.5)


# ── boundedness (the "no rails" guarantee) ───────────────────────────

def test_channels_strictly_bounded_under_extreme_drivers():
    """Sigmoid readout + z-scored drivers ⇒ no channel can ever rail."""
    nt = DrivenNTSystem()
    rng = random.Random(0)
    for _ in range(400):
        nt.step_full(
            loss=1e6 + 1e3 * rng.gauss(0, 1),
            grad_norm=1e6 + 1e3 * rng.gauss(0, 1),
            activation=1e6 + 1e3 * rng.gauss(0, 1),
            ignition_rate=0.95 + 0.04 * rng.random(),
            attn_entropy_norm=0.05 * rng.random(),
        )
        assert _all_in_open_unit(nt.levels())


def test_no_channel_pinned_at_rails_under_realistic_training():
    """A realistic training trace where loss falls + gnorm spikes. The
    OLD per-channel squashes railed eCB/Glu at 1.0 and collapsed GABA
    to 0. Here we assert NO channel sits within 0.02 of either rail."""
    nt = DrivenNTSystem()
    rng = random.Random(0)
    for t in range(600):
        L = 10.0 - 9.0 * (1.0 - math.exp(-t / 80.0))
        nt.step_full(
            loss=L + 0.3 * rng.gauss(0, 1),
            grad_norm=abs(0.8 + 0.5 * rng.gauss(0, 1)),
            activation=0.2 + 0.05 * t / 600.0 + 0.1 * rng.gauss(0, 1),
            ignition_rate=0.6 + 0.3 * rng.random(),
            attn_entropy_norm=0.4 + 0.2 * rng.random(),
        )
    final = nt.levels()
    for k, v in final.items():
        assert 0.02 < v < 0.98, (
            f"channel {k} railed at {v:.4f} under realistic-training trace")


# ── per-channel directional response (post-warmup) ───────────────────

def test_ne_rises_on_grad_norm_spike():
    nt = DrivenNTSystem()
    _noisy_warmup(nt, n=80, grad_norm=0.5)
    base = nt.levels()["NE"]
    rng = random.Random(1)
    for _ in range(40):
        nt.step_full(grad_norm=5.0 + 0.5 * rng.gauss(0, 1))
    high = nt.levels()["NE"]
    assert high > base + 0.05, f"NE failed to rise: {base:.3f} → {high:.3f}"


def test_gaba_rises_with_workspace_ignition():
    """Bio-correct sign: high ignition recruits inhibition (homeostatic
    E/I), unlike the old `GABA = 1 − ignition` which took the brakes
    off when the workspace was on fire."""
    nt = DrivenNTSystem()
    _noisy_warmup(nt, n=80, ignition_rate=0.3)
    low_ign_gaba = nt.levels()["GABA"]
    rng = random.Random(2)
    for _ in range(40):
        nt.step_full(ignition_rate=0.9 + 0.05 * rng.random())
    high_ign_gaba = nt.levels()["GABA"]
    assert high_ign_gaba > low_ign_gaba + 0.03


def test_ach_rises_with_sharper_attention():
    nt = DrivenNTSystem()
    _noisy_warmup(nt, n=80, attn_entropy_norm=0.7)
    diffuse = nt.levels()["ACh"]
    rng = random.Random(3)
    for _ in range(40):
        nt.step_full(attn_entropy_norm=0.1 + 0.05 * rng.random())
    sharp = nt.levels()["ACh"]
    assert sharp > diffuse + 0.03


def test_da_spikes_on_negative_surprise():
    nt = DrivenNTSystem()
    _noisy_warmup(nt, n=80, loss=5.0)
    baseline = nt.levels()["DA"]
    # Sudden drop in loss = positive surprise → DA up.
    rng = random.Random(4)
    for _ in range(8):
        nt.step_full(loss=2.0 + 0.1 * rng.gauss(0, 1))
    da_good = nt.levels()["DA"]
    assert da_good > baseline + 0.03


def test_glu_and_ecb_respond_to_activation_spike():
    nt = DrivenNTSystem()
    _noisy_warmup(nt, n=80, activation=0.2)
    base_glu = nt.levels()["Glu"]
    base_ecb = nt.levels()["eCB"]
    rng = random.Random(5)
    for _ in range(40):
        nt.step_full(activation=2.0 + 0.2 * rng.gauss(0, 1))
    assert nt.levels()["Glu"] > base_glu + 0.05
    assert nt.levels()["eCB"] > base_ecb + 0.05


# ── the "no ratchet" guarantee (kills problem #2) ────────────────────

def test_5ht_does_not_monotonically_drift_under_constant_loss():
    """5HT must not behave as an unbounded integrator. Sustained
    improvement raises it transiently while the standardiser baseline
    catches up; over very long horizons it stabilises (or relaxes)
    rather than climbing toward 1.0."""
    nt = DrivenNTSystem()
    _noisy_warmup(nt, n=300, loss=10.0)
    # Sustained improvement
    rng = random.Random(6)
    for _ in range(500):
        nt.step_full(loss=2.0 + 0.2 * rng.gauss(0, 1))
    mid = nt.levels()["5HT"]
    # Many more steps at the same level → mean catches up, z relaxes
    for _ in range(3000):
        nt.step_full(loss=2.0 + 0.2 * rng.gauss(0, 1))
    late = nt.levels()["5HT"]
    assert late < 0.99, f"5HT railed at {late:.4f} — leak failed"
    # Sustained level should not produce ever-increasing 5HT.
    assert late <= mid + 0.05, (
        f"5HT still drifting upward: mid={mid:.3f} late={late:.3f}")


def test_5ht_responds_to_sustained_loss_improvement():
    """The homeostat reports regime *changes*: 5HT must rise during
    the transition from a high-loss to a low-loss regime. After the
    standardiser baseline catches up, 5HT relaxes — by design — so we
    check the PEAK within the transition window, not the endpoint."""
    nt = DrivenNTSystem()
    _noisy_warmup(nt, n=200, loss=10.0)
    early = nt.levels()["5HT"]
    rng = random.Random(7)
    peak = early
    for _ in range(400):
        nt.step_full(loss=2.0 + 0.2 * rng.gauss(0, 1))
        peak = max(peak, nt.levels()["5HT"])
    assert peak > early + 0.05, f"5HT did not respond: early={early:.3f} peak={peak:.3f}"


# ── the "guaranteed recovery" guarantee (kills problem #3) ────────────

def test_all_channels_recover_to_baseline_when_drivers_stop():
    """After any driving regime, removing all drivers exponentially
    relaxes every channel back to its baseline via the OU leak."""
    nt = DrivenNTSystem()
    rng = random.Random(8)
    # Push channels far from baseline with noisy extreme drivers.
    for _ in range(150):
        nt.step_full(
            loss=2.0 + 0.5 * rng.gauss(0, 1),     # +surprise → DA/5HT up
            grad_norm=5.0 + rng.gauss(0, 1),      # → NE up
            activation=2.0 + 0.3 * rng.gauss(0, 1),  # → Glu/eCB up
            ignition_rate=0.9 + 0.05 * rng.random(),  # → GABA up, ACh down
            attn_entropy_norm=0.1 + 0.05 * rng.random(),  # → ACh up
        )
    pushed = nt.levels()
    base = nt.baselines
    moved = max(abs(pushed[k] - base[k]) for k in base)
    assert moved > 0.05, f"setup failed — no channel moved (max Δ={moved:.4f})"
    # Now let it relax for a long time with NO drivers.
    for _ in range(5000):
        nt.step_full()                             # all None ⇒ pure leak
    final = nt.levels()
    for k in base:
        assert abs(final[k] - base[k]) < 0.05, (
            f"channel {k} failed to relax: {final[k]:.4f} vs base {base[k]:.4f}")


def test_gaba_bidirectional_excursion_and_recovery():
    """The old system had GABA stuck at exactly 0.0 forever. Here we
    drive it both up and down and assert it stays in the interior and
    its peak excursion is in the expected direction."""
    nt = DrivenNTSystem()
    rng = random.Random(9)
    # Push up: high ignition + high gnorm. Track peak — once driver
    # mean catches up, z relaxes and GABA returns toward baseline.
    peak_up = nt.levels()["GABA"]
    for _ in range(60):
        nt.step_full(
            grad_norm=5.0 + rng.gauss(0, 1),
            ignition_rate=0.9 + 0.05 * rng.random(),
        )
        peak_up = max(peak_up, nt.levels()["GABA"])
    assert peak_up > nt.baselines["GABA"] + 0.03, (
        f"GABA failed to rise during high-ignition window: peak={peak_up:.3f}")
    # Push down: opposite drivers.
    trough = nt.levels()["GABA"]
    for _ in range(150):
        nt.step_full(
            grad_norm=0.05 + 0.02 * rng.random(),
            ignition_rate=0.05 + 0.02 * rng.random(),
        )
        trough = min(trough, nt.levels()["GABA"])
    assert trough > 0.001, f"GABA collapsed to {trough} — sigmoid is degenerate"


# ── compat shim ───────────────────────────────────────────────────────

def test_compat_shim_step_only_drives_activation():
    """Legacy step(activity=...) only feeds the activation driver;
    other channels experience pure OU leak toward baseline."""
    nt = DrivenNTSystem()
    rng = random.Random(10)
    # Push NE far from baseline via step_full with noisy gnorm.
    _noisy_warmup(nt, n=60, grad_norm=0.5)
    for _ in range(40):
        nt.step_full(grad_norm=8.0 + rng.gauss(0, 1))
    pushed_ne = nt.levels()["NE"]
    base_ne = nt.baselines["NE"]
    assert pushed_ne > base_ne + 0.1, f"setup failed: NE={pushed_ne:.3f}"
    # Many shim-step calls without grad_norm should relax NE back via leak.
    for _ in range(400):
        nt.step(activity=0.1)
    final_ne = nt.levels()["NE"]
    assert abs(final_ne - base_ne) < abs(pushed_ne - base_ne) / 2
