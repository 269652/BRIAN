# -*- coding: utf-8 -*-
"""Contracts for LogitNorm calibration (H30 guardrail).

Why this exists
===============
Run 43133274 (logits_mixture, H28) overfit catastrophically: the standalone
trunk's held-out WikiText OOD ppl reached 175k — CE ≈ 12 nats, ABOVE the
uniform ceiling ln(50257) = 10.82. ppl > uniform means the model is
*confidently wrong* off-distribution: it puts large probability mass on the
wrong token. That can only happen if the logit MAGNITUDE is unbounded.

LogitNorm (Wei et al., ICML 2022) removes the magnitude degree of freedom:
training minimises CE on ``f / (τ·‖f‖)`` instead of ``f``. The network can no
longer lower the loss by inflating ‖f‖, so it stops manufacturing
overconfidence, and at inference the logits stay bounded → OOD CE is capped
near (not above) uniform.

The defining property is **scale-invariance**: ``logit_norm(c·f) ==
logit_norm(f)`` for any c>0. That is the whole mechanism — confidence is set
by the logit *direction*, never its *magnitude*. These tests pin that and
the surrounding math.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402


class TestLogitNormMath:
    def test_importable(self):
        from neuroslm.regularizers import logit_norm
        assert callable(logit_norm)

    def test_scale_invariance(self):
        """THE property: scaling the logits by any positive constant leaves
        the normalised logits unchanged — so confidence cannot be inflated by
        growing ‖f‖. This is what makes 'confidently wrong on OOD' impossible.
        """
        from neuroslm.regularizers import logit_norm
        torch.manual_seed(0)
        f = torch.randn(4, 7, 50)
        a = logit_norm(f, tau=0.05)
        b = logit_norm(7.3 * f, tau=0.05)
        c = logit_norm(1000.0 * f, tau=0.05)
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(a, c, rtol=1e-4, atol=1e-5)

    def test_output_norm_is_one_over_tau(self):
        """‖f/(τ‖f‖)‖ = 1/τ along the vocab axis (the norm is pinned)."""
        from neuroslm.regularizers import logit_norm
        torch.manual_seed(1)
        f = torch.randn(3, 5, 40) * 12.0
        out = logit_norm(f, tau=0.04)
        norms = out.norm(dim=-1)
        torch.testing.assert_close(
            norms, torch.full_like(norms, 1.0 / 0.04), rtol=1e-3, atol=1e-3)

    def test_argmax_preserved(self):
        """Normalisation is a positive scaling — it must not change the
        predicted token (ranking is invariant)."""
        from neuroslm.regularizers import logit_norm
        torch.manual_seed(2)
        f = torch.randn(6, 11, 60)
        assert torch.equal(
            f.argmax(-1), logit_norm(f, tau=0.05).argmax(-1))

    def test_confidence_depends_on_direction_not_magnitude(self):
        """Two logit vectors with the same direction but very different
        magnitude must yield the SAME softmax (so a model cannot become more
        confident merely by scaling up its logits)."""
        from neuroslm.regularizers import logit_norm
        torch.manual_seed(3)
        f = torch.randn(2, 3, 30)
        p_small = F.softmax(logit_norm(f, tau=0.05), dim=-1)
        p_huge = F.softmax(logit_norm(500.0 * f, tau=0.05), dim=-1)
        torch.testing.assert_close(p_small, p_huge, rtol=1e-4, atol=1e-5)

    def test_gradient_flows(self):
        from neuroslm.regularizers import logit_norm
        f = torch.randn(2, 4, 20, requires_grad=True)
        out = logit_norm(f, tau=0.05)
        out.sum().backward()
        assert f.grad is not None
        assert torch.isfinite(f.grad).all()

    def test_shape_and_dtype_preserved(self):
        from neuroslm.regularizers import logit_norm
        f = torch.randn(3, 5, 64, dtype=torch.float32)
        out = logit_norm(f, tau=0.05)
        assert out.shape == f.shape
        assert out.dtype == f.dtype

    def test_zero_logits_safe(self):
        """An all-zero logit row (‖f‖=0) must not NaN — eps guards it."""
        from neuroslm.regularizers import logit_norm
        f = torch.zeros(2, 3, 10)
        out = logit_norm(f, tau=0.05)
        assert torch.isfinite(out).all()


class TestLogitNormWiredIntoLoss:
    """The calibration invariance must reach the actual LM loss: when
    ``logit_norm_tau > 0`` the harness cross-entropy is scale-invariant in the
    logits (the model cannot lower its loss by becoming over-confident); when
    off, it is not. Exercises ``BRIANHarness._compute_loss_from_logits`` via a
    minimal stub (it only reads ``training_config`` / ``_gif`` / ``_metrics``).
    """

    def _loss(self, tau, scale, logits, targets):
        import types
        from neuroslm.harness import BRIANHarness
        from neuroslm.dsl.training_config import TrainingConfig
        cfg = TrainingConfig()
        cfg.logit_norm_tau = tau
        stub = types.SimpleNamespace(
            training_config=cfg, _gif=None, _metrics={})
        return BRIANHarness._compute_loss_from_logits(
            stub, scale * logits, targets).item()

    def test_off_loss_changes_with_scale(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 16, 50)
        targets = torch.randint(0, 50, (2, 16))
        a = self._loss(0.0, 1.0, logits, targets)
        b = self._loss(0.0, 12.0, logits, targets)
        assert abs(a - b) > 0.1, (
            "baseline: without logit_norm, scaling the logits must change CE")

    def test_on_loss_invariant_to_scale(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 16, 50)
        targets = torch.randint(0, 50, (2, 16))
        a = self._loss(0.05, 1.0, logits, targets)
        b = self._loss(0.05, 12.0, logits, targets)
        assert abs(a - b) < 1e-3, (
            "with logit_norm the LM loss must be scale-invariant — the "
            "calibration guarantee that caps OOD over-confidence")
