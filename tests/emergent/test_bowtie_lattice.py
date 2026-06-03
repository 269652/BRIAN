"""Tests for C5 — BowtieLatticeProbe."""
from __future__ import annotations
import torch

from neuroslm.emergent.bowtie_lattice import BowtieLatticeProbe


def test_init_validates():
    import pytest
    with pytest.raises(ValueError):
        BowtieLatticeProbe(dim=10, K=3)        # 10 not divisible by 3
    with pytest.raises(ValueError):
        BowtieLatticeProbe(dim=0)


def test_initial_stats_safe():
    p = BowtieLatticeProbe(dim=16, K=4, n_classes=4)
    s = p.stats()
    assert s["lattice_spec"] == 1.0
    assert s["lattice_active_k"] == 0


def test_slice_winner_shapes():
    p = BowtieLatticeProbe(dim=12, K=3, n_classes=2)
    x = torch.randn(2, 5, 12)
    w = p.slice_winner(x)
    assert w.shape == (2, 5)
    assert int(w.min().item()) >= 0
    assert int(w.max().item()) <= 2


def test_perfect_specialisation_when_class_one_to_one():
    """If class c always activates slice c only, lift = K and spec = K."""
    K = 4
    dim = 16
    p = BowtieLatticeProbe(dim=dim, K=K, n_classes=K, history=512)
    # Build inputs where class c puts all mass in slice c.
    for c in range(K):
        x = torch.zeros(1, 4, dim)
        x[..., c * (dim // K):(c + 1) * (dim // K)] = 5.0
        for _ in range(20):
            p.step(x, class_label=c)
    s = p.stats()
    assert abs(s["lattice_spec"] - float(K)) < 0.1


def test_no_specialisation_for_random_inputs_across_classes():
    """Random inputs distributed across classes → spec close to 1."""
    torch.manual_seed(0)
    K = 4
    dim = 16
    p = BowtieLatticeProbe(dim=dim, K=K, n_classes=K, history=4096)
    for _ in range(2000):
        c = int(torch.randint(K, (1,)).item())
        x = torch.randn(1, 4, dim)
        p.step(x, class_label=c)
    s = p.stats()
    # Should not be far from 1 (and certainly well below K).
    assert s["lattice_spec"] < 1.5


def test_entropy_high_when_slices_equally_used():
    K = 4
    dim = 16
    p = BowtieLatticeProbe(dim=dim, K=K, n_classes=K)
    # Cycle through slices uniformly.
    for k in range(K):
        x = torch.zeros(1, 1, dim)
        x[..., k * (dim // K):(k + 1) * (dim // K)] = 5.0
        for _ in range(40):
            p.step(x, class_label=0)
    s = p.stats()
    # All four slices active, equal usage → normalised entropy ≈ 1.
    assert s["lattice_entropy"] > 0.9
    assert s["lattice_active_k"] == K


def test_missing_inputs_no_op():
    p = BowtieLatticeProbe(dim=8, K=2, n_classes=2)
    s1 = p.step(None, None)
    s2 = p.step(torch.randn(1, 1, 8), None)
    s3 = p.step(None, 0)
    assert s1 == s2 == s3


def test_out_of_range_class_label_ignored():
    p = BowtieLatticeProbe(dim=8, K=2, n_classes=2)
    p.step(torch.randn(1, 1, 8), class_label=99)
    # No counts recorded.
    assert p.stats()["lattice_active_k"] == 0
