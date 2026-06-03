"""Phase-7 integration tests: emergent layer wired into MetricObserver.

These assert the contract from `docs/EMERGENT_TOPOLOGY.md §4`:

  * default behaviour (enable_emergent=False) is byte-identical to legacy
  * enable_emergent=True adds the new keys while keeping all legacy keys
  * the new NT values actually MOVE under a synthetic training-like
    stream (the load-bearing fix that motivates this whole layer)
"""
from __future__ import annotations
import torch

from neuroslm.dsl.metrics import MetricObserver


def _fake_layer_acts(B=2, T=8, D=16, n_layers=4, scale=1.0):
    return [torch.randn(B, T, D) * scale for _ in range(n_layers)]


def test_default_observer_keys_unchanged():
    """The DEFAULT observer (enable_emergent=True after 2026-06-03) keeps
    all legacy keys AND emits the emergent ones — so the log shape grows
    but never shrinks."""
    obs = MetricObserver(n_layers=4)
    m = obs.observe(_fake_layer_acts(), loss=5.0)
    # Legacy keys still present.
    legacy = {"phi", "fiedler", "ignition", "meso_lg",
              "troph_active", "troph_total", "troph_mean", "nt", "osc"}
    assert legacy.issubset(set(m))
    # And at least one canonical emergent key is too.
    assert "nt_driven" in m
    assert "pac" in m


def test_legacy_only_observer_keys_exact():
    """If a caller explicitly asks for the legacy observer they get the
    pre-emergent dict shape."""
    obs = MetricObserver(n_layers=4, enable_emergent=False)
    m = obs.observe(_fake_layer_acts(), loss=5.0)
    expected = {"phi", "fiedler", "ignition", "meso_lg",
                "troph_active", "troph_total", "troph_mean",
                "nt", "osc"}
    assert set(m) == expected


def test_emergent_observer_adds_new_keys():
    obs = MetricObserver(n_layers=4, enable_emergent=True, emergent_dim=16)
    m = obs.observe(_fake_layer_acts(), loss=5.0, grad_norm=1.0,
                    attn_entropy_norm=0.4, class_label=0)
    # All legacy keys still present.
    for k in ("phi", "fiedler", "ignition", "meso_lg",
              "troph_active", "troph_total", "troph_mean", "nt", "osc"):
        assert k in m, f"missing legacy key {k}"
    # All emergent keys present.
    for k in ("nt_driven", "ign_rate", "ign_strength", "ign_threshold",
              "Q_total", "Q_walls", "Q_plateau_len",
              "pac", "pac_pref_phase"):
        assert k in m, f"missing emergent key {k}"


def test_emergent_nt_values_actually_move():
    """The whole point: with emergent ON, the NT column must respond to
    training-state inputs and not stay pinned at baseline."""
    torch.manual_seed(0)
    obs = MetricObserver(n_layers=4, enable_emergent=True, emergent_dim=16)
    nts = []
    for step in range(200):
        # Synthetic descending loss + variable grad norm.
        loss = 10.0 - 0.04 * step + 0.5 * float(torch.randn(()).item())
        gnorm = 1.0 + abs(float(torch.randn(()).item()))
        acts = _fake_layer_acts(scale=1.0 + 0.01 * step)
        m = obs.observe(acts, loss=loss, grad_norm=gnorm,
                        attn_entropy_norm=0.5)
        nts.append(m["nt"]["NE"])
    # Range across the run should be non-trivial.
    span = max(nts) - min(nts)
    assert span > 0.05, f"NE moved only {span:.4f} across 200 steps"


def test_emergent_off_keeps_log_byte_identical():
    """When the flag is off, every value matches the legacy path."""
    torch.manual_seed(7)
    obs_a = MetricObserver(n_layers=4, enable_emergent=False)
    obs_b = MetricObserver(n_layers=4, enable_emergent=False)
    acts = _fake_layer_acts()
    m_a = obs_a.observe(acts, loss=3.0)
    m_b = obs_b.observe(acts, loss=3.0)
    assert m_a == m_b


def test_pc_probe_constructed_from_motor_dim():
    obs = MetricObserver(n_layers=4, enable_emergent=True)
    motor = torch.randn(1, 4, 12)
    sensory = torch.randn(1, 4, 12)
    m1 = obs.observe(_fake_layer_acts(D=16), loss=3.0,
                     h_motor=motor, h_sensory=sensory)
    assert m1["pc_residual"] == 0.0          # first call seeds
    m2 = obs.observe(_fake_layer_acts(D=16), loss=3.0,
                     h_motor=motor * 0.9, h_sensory=sensory * 0.9)
    assert "pc_residual" in m2


def test_topo_charge_emitted_under_emergent():
    obs = MetricObserver(n_layers=4, enable_emergent=True)
    m = obs.observe(_fake_layer_acts(B=2, T=16, D=8), loss=3.0)
    # Sequence is long enough; charge should compute.
    assert "Q_walls" in m
    assert m["Q_walls"] >= 0.0


def test_lattice_keys_present_when_class_label_given():
    obs = MetricObserver(n_layers=4, enable_emergent=True)
    m = obs.observe(_fake_layer_acts(D=16), loss=3.0, class_label=1)
    assert "lattice_spec" in m
    assert m["lattice_spec"] >= 1.0


def test_lattice_keys_absent_without_class_label():
    obs = MetricObserver(n_layers=4, enable_emergent=True)
    m = obs.observe(_fake_layer_acts(D=16), loss=3.0)
    assert "lattice_spec" not in m
