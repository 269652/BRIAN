# -*- coding: utf-8 -*-
"""Contracts for Jacobian-consistency distillation (H30 elegant core).

Why this exists
===============
Run 43133274 (H28) showed pointwise distillation transfers the teacher's
training-point VALUES (lm_ema→cx_ema) but not its generalising FUNCTION — the
trunk memorised and its OOD exploded. The fix (Srinivas & Fleuret, ICML'18):
matching the teacher under INPUT PERTURBATION is the first-order equivalent of
matching its input-Jacobian, which transfers the teacher's local function and
provably closes generalisation gaps. Concretely:

    L_consist = T² · KL( softmax(teacher(x)/T)  ‖  softmax(student(x+δ)/T) )

with δ Gaussian noise on the student's input embedding and the teacher
detached. It forces the student to match the teacher even from a perturbed
input → it cannot spike to confidently-wrong values nearby.

Two cores are pinned here:
  1. ``consistency_distill_loss`` — the temperature-scaled, teacher-detached KL.
  2. the trunk's ``embed_noise_std`` forward hook — perturbs the input
     embedding so the consistency pass measures the function, not the point.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402


class TestConsistencyKL:
    def test_importable(self):
        from neuroslm.regularizers import consistency_distill_loss
        assert callable(consistency_distill_loss)

    def test_zero_when_student_matches_teacher(self):
        from neuroslm.regularizers import consistency_distill_loss
        torch.manual_seed(0)
        f = torch.randn(2, 5, 30)
        loss = consistency_distill_loss(f.clone(), f.clone(), temperature=2.0)
        assert abs(loss.item()) < 1e-5, "KL must vanish when student == teacher"

    def test_positive_when_they_differ(self):
        from neuroslm.regularizers import consistency_distill_loss
        torch.manual_seed(1)
        t = torch.randn(2, 5, 30)
        s = torch.randn(2, 5, 30)
        assert consistency_distill_loss(t, s, temperature=2.0).item() > 0.0

    def test_gradient_flows_to_student_not_teacher(self):
        """The teacher is the (clean) target — it must be detached; gradient
        flows only into the perturbed student."""
        from neuroslm.regularizers import consistency_distill_loss
        torch.manual_seed(2)
        teacher = torch.randn(2, 4, 20, requires_grad=True)
        student = torch.randn(2, 4, 20, requires_grad=True)
        loss = consistency_distill_loss(teacher, student, temperature=2.0)
        loss.backward()
        assert student.grad is not None and torch.isfinite(student.grad).all()
        assert teacher.grad is None, "teacher must be detached (it's the target)"

    def test_temperature_squared_scaling(self):
        """Hinton T² scaling: the loss carries a (T²) factor so its gradient
        magnitude is temperature-independent."""
        from neuroslm.regularizers import consistency_distill_loss
        torch.manual_seed(3)
        t = torch.randn(2, 4, 16)
        s = torch.randn(2, 4, 16)
        l1 = consistency_distill_loss(t, s, temperature=1.0).item()
        # Build the same KL by hand at T=1 and compare (T=1 ⇒ factor 1).
        kl = F.kl_div(F.log_softmax(s, -1), F.softmax(t, -1),
                      reduction="batchmean").item()
        assert abs(l1 - kl) < 1e-4


class TestEmbedNoiseHook:
    """The trunk's forward must accept ``embed_noise_std`` and perturb the
    input embedding by that amount — the mechanism that turns one extra
    forward into a local function probe. σ=0 is an exact no-op."""

    def _tiny_trunk(self):
        from neuroslm.dsl.nn_lang import build_dsl_language_cortex
        torch.manual_seed(0)
        return build_dsl_language_cortex(
            vocab=32, d_model=16, depth=1, n_heads=2, max_ctx=8)

    def test_sigma_zero_is_identity(self):
        trunk = self._tiny_trunk()
        trunk.eval()
        ids = torch.randint(0, 32, (1, 6))
        with torch.no_grad():
            a = trunk(ids)
            b = trunk(ids, embed_noise_std=0.0)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    def test_sigma_positive_perturbs_output(self):
        trunk = self._tiny_trunk()
        trunk.eval()
        ids = torch.randint(0, 32, (1, 6))
        torch.manual_seed(7)
        with torch.no_grad():
            clean = trunk(ids)
            noised = trunk(ids, embed_noise_std=0.5)
        assert (clean - noised).abs().max().item() > 1e-3, (
            "embed_noise_std>0 must perturb the trunk output")


class TestStashPreservingForward:
    """The consistency pass re-runs the trunk on noised input. That second
    forward overwrites the trunk's ``_last*`` / ``last_*`` stashes — which
    LATER aux losses (topo/symplectic/kjpla) still consume this step. The
    helper must restore them to their clean values."""

    def test_stashes_restored_after_noised_forward(self):
        import torch.nn as nn
        from neuroslm.harness import BRIANHarness

        class _FakeTrunk(nn.Module):
            def __init__(self):
                super().__init__()
                self._last_hidden = torch.zeros(1)          # clean stash
                self.last_token_surprise = torch.ones(1)

            def forward(self, ids, embed_noise_std=0.0):
                # the real trunk mutates these on every forward:
                self._last_hidden = torch.full((1,), 99.0)
                self.last_token_surprise = torch.full((1,), 77.0)
                return torch.randn(1, ids.shape[1], 4) + embed_noise_std

        lm = _FakeTrunk()
        clean_hidden = lm._last_hidden.clone()
        clean_surprise = lm.last_token_surprise.clone()
        ids = torch.zeros(1, 5, dtype=torch.long)

        out = BRIANHarness._trunk_forward_preserving_stashes(lm, ids, 0.5)
        assert out.shape == (1, 5, 4), "must return the noised forward output"
        torch.testing.assert_close(lm._last_hidden, clean_hidden)
        torch.testing.assert_close(lm.last_token_surprise, clean_surprise)


class TestConsistencyWiredIntoAuxStep:
    """§14 contract: enabling ``consistency_weight`` must actually add a
    positive term to the loss and publish the ``consistency_kl`` metric —
    the toggle is not a no-op. Differential test (off vs on) on the REAL
    ``_cortex_fusion_aux_step`` glue with a REAL noise-aware trunk; the
    cortex stash is set directly so we don't have to run the cortex stack.
    """

    _VOCAB = 64
    _DSEM = 32

    def _harness(self):
        from neuroslm.harness import BRIANHarness
        from neuroslm.dsl.nn_lang import build_dsl_language_cortex
        from neuroslm.dsl.training_config import (
            MultiCortexConfig, TrainingConfig)
        torch.manual_seed(0)
        trunk = build_dsl_language_cortex(
            vocab=self._VOCAB, d_model=self._DSEM, depth=1,
            n_heads=2, max_ctx=16)
        cfg = TrainingConfig()
        mc = MultiCortexConfig()
        mc.enabled = True
        mc.n_cortices = 2
        mc.domains = ["general", "code"]
        mc.weights = "stub"
        mc.freeze_weights = False
        mc.lexical_bias_weight = 0.0
        mc.bema_tau = 0.5
        mc.router_d_model = self._DSEM
        mc.fusion_mode = "logits_mixture"
        mc.fusion_init = 0.3
        mc.distillation_temperature = 2.0
        cfg.multi_cortex = mc
        return BRIANHarness.from_language_model(
            language_model=trunk, vocab_size=self._VOCAB,
            d_sem=self._DSEM, training_config=cfg)

    def _set_stashes(self, h, ids):
        # The cortex teacher + trunk pre-fusion logits the aux step consumes.
        torch.manual_seed(1)
        B, T = ids.shape
        h._last_pre_fusion_lm_logits = torch.randn(B, T, self._VOCAB)
        h._last_pre_fusion_cortex_logits = torch.randn(B, T, self._VOCAB)

    def test_off_is_noop_on_is_positive(self):
        h = self._harness()
        ids = torch.randint(0, self._VOCAB, (2, 12))
        targets = torch.randint(0, self._VOCAB, (2, 12))
        base = torch.zeros((), dtype=torch.float32)

        # OFF: weight 0 → no consistency term, no metric.
        h.training_config.multi_cortex.consistency_weight = 0.0
        h._metrics.pop("consistency_kl", None)
        self._set_stashes(h, ids)
        total_off = h._cortex_fusion_aux_step(base.clone(), targets, ids=ids)
        assert "consistency_kl" not in h._metrics

        # ON: weight 2.0 → positive term added + metric published.
        h.training_config.multi_cortex.consistency_weight = 2.0
        self._set_stashes(h, ids)
        total_on = h._cortex_fusion_aux_step(base.clone(), targets, ids=ids)
        assert "consistency_kl" in h._metrics
        assert h._metrics["consistency_kl"] > 0.0
        assert total_on.item() > total_off.item() + 1e-6, (
            "enabling consistency_weight must increase the aux-step total")

    def test_requires_ids(self):
        """ids=None (no token ids available) must skip the term cleanly —
        the consistency pass needs the raw ids to re-embed with noise."""
        h = self._harness()
        ids = torch.randint(0, self._VOCAB, (2, 12))
        targets = torch.randint(0, self._VOCAB, (2, 12))
        h.training_config.multi_cortex.consistency_weight = 2.0
        h._metrics.pop("consistency_kl", None)
        self._set_stashes(h, ids)
        total = h._cortex_fusion_aux_step(
            torch.zeros(()), targets, ids=None)
        assert "consistency_kl" not in h._metrics
        assert torch.isfinite(total).all()
