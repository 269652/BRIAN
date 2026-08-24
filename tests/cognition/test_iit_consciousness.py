# -*- coding: utf-8 -*-
"""Tests for neuroslm.cognition.consciousness (architecture.md §14.11) —
a real IIT-flavored Φ and a GWT-flavored broadcast-strength proxy for
CognitiveRuntime's tick loop.

Mirrors tests/test_phi.py's comparative-not-absolute pattern: these
guard that compute_phi_iit reuses the SAME Gaussian-MI minimum-
information-partition estimator NeuralOrchestrator uses for Brain's own
Φ (via the public NeuralOrchestrator.gaussian_mi_mip_phi), not a new,
competing, ad-hoc heuristic.
"""
from __future__ import annotations

import pytest
import torch

from neuroslm.cognition.consciousness import (
    bucket_reduce,
    compute_broadcast_strength,
    compute_phi_iit,
)


class TestBucketReduce:
    def test_output_length_always_k(self):
        assert len(bucket_reduce([1, 2, 3], k=8)) == 8
        assert len(bucket_reduce(list(range(100)), k=8)) == 8
        assert len(bucket_reduce([1, 2, 3, 4, 5, 6, 7, 8], k=8)) == 8

    def test_uniform_segment_reproduces_bucket_mean(self):
        # 8 values into 4 buckets -> each bucket is the mean of its pair.
        vec = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
        out = bucket_reduce(vec, k=4)
        assert out == pytest.approx([2.0, 6.0, 10.0, 14.0])

    def test_fewer_elements_than_buckets_does_not_crash(self):
        out = bucket_reduce([1.0, 2.0], k=8)
        assert len(out) == 8

    def test_empty_vector_returns_k_zeros(self):
        assert bucket_reduce([], k=8) == [0.0] * 8


class TestComputePhiIit:
    def test_requires_at_least_two_modules(self):
        phi, n = compute_phi_iit({"a": [1.0] * 8}, k=8)
        assert phi == 0.0
        assert n == 1
        phi0, n0 = compute_phi_iit({}, k=8)
        assert phi0 == 0.0
        assert n0 == 0

    def test_higher_for_correlated_than_decorrelated_modules(self):
        """Coupled: every module is a scaled copy of the same base vector
        -> MI cannot be reduced by any bipartition. Independent: unrelated
        noise per module -> MI lower bound close to zero. Deterministic
        construction (fixed seeds), comparative assertion — avoids
        flakiness, matches tests/test_phi.py::test_phi_higher_for_coupled_outputs.
        """
        g = torch.Generator().manual_seed(0)
        base = torch.randn(32, generator=g).tolist()
        coupled = {f"m{i}": [x * (0.5 + 0.1 * i) for x in base]
                   for i in range(4)}
        gi = torch.Generator().manual_seed(1)
        independent = {f"m{i}": torch.randn(32, generator=gi).tolist()
                       for i in range(4)}
        phi_c, n_c = compute_phi_iit(coupled, k=8)
        phi_i, n_i = compute_phi_iit(independent, k=8)
        assert n_c == 4 and n_i == 4
        assert phi_c > phi_i, f"coupled {phi_c} should exceed independent {phi_i}"

    def test_calls_orchestrator_gaussian_mi_mip_phi(self, monkeypatch):
        """Structural reuse pin: compute_phi_iit must not reimplement the
        MI/MIP math — it delegates to NeuralOrchestrator's public
        estimator."""
        from neuroslm.intelligence.orchestrator import NeuralOrchestrator

        calls = []
        real = NeuralOrchestrator.gaussian_mi_mip_phi

        def spy(M):
            calls.append(M)
            return real(M)

        monkeypatch.setattr(NeuralOrchestrator, "gaussian_mi_mip_phi", spy)
        compute_phi_iit({"a": [1.0] * 8, "b": [2.0] * 8}, k=8)
        assert len(calls) == 1
        assert tuple(calls[0].shape) == (2, 8)

    def test_never_returns_nan_or_inf(self):
        # Degenerate input: every module identical -> singular covariance.
        modules = {"a": [1.0] * 8, "b": [1.0] * 8, "c": [1.0] * 8}
        phi, n = compute_phi_iit(modules, k=8)
        assert n == 3
        assert phi == phi  # not NaN
        assert phi not in (float("inf"), float("-inf"))


class TestComputeBroadcastStrength:
    def test_identical_winner_and_context_is_one(self):
        vec = [1.0, 2.0, 3.0, 4.0]
        modules = {"thought": vec, "nt": vec, "recall": vec}
        s = compute_broadcast_strength(modules, "thought", k=4)
        assert s == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_winner_is_near_zero(self):
        modules = {"thought": [1.0, 0.0, 0.0, 0.0],
                  "nt": [0.0, 1.0, 0.0, 0.0],
                  "recall": [0.0, 0.0, 1.0, 0.0]}
        s = compute_broadcast_strength(modules, "thought", k=4)
        assert abs(s) < 1e-6

    def test_missing_winner_key_returns_zero(self):
        assert compute_broadcast_strength({"nt": [1.0, 2.0]}, "thought") == 0.0

    def test_single_module_returns_zero(self):
        assert compute_broadcast_strength({"thought": [1.0, 2.0]}, "thought") == 0.0

    def test_empty_modules_returns_zero(self):
        assert compute_broadcast_strength({}, "thought") == 0.0
