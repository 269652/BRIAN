# -*- coding: utf-8 -*-
"""Two-layer doctrine: `trunk_pretrain` decouples LM training from cognition.

Design decision (2026-08-11, conversation w/ user): the LM trunk is the
language cortex — LM-dataset training teaches IT, with detached expert
KL-distillation as the only assist. The neuroanatomical layer (pred-coding
/world/motor stashes, PC-reentry, MSPCC cascade, STE criticality,
topo-charge/symplectic/KJPLA physics losses, genetic Φ) is runtime
cognition: it thinks in natural language ON a trained trunk and must not
leak gradient into LM pretraining. Inference/eval was already isolated
(`eval_surface: "trunk_only"`); this flag isolates TRAINING the same way.

Contracts:
  A. `trunk_pretrain` parses from the DSL `training{}` block, default False
     (all existing arches keep bit-identical behaviour).
  B. With the flag ON, compute_loss excludes the cognition-bucket aux
     losses (representative: the `_last_pred_coding_loss` stash) —
     total == w_lm · CE. With the flag OFF the same stash contributes.
  C. KL-distillation (the "distil assist") still fires with the flag ON —
     it is a trunk-training aid, not cognition.
  D. Forward fusion mixing becomes identity under the flag (CE gradient
     flows through the ISOLATED trunk logits, matching the trunk-only
     inference surface), while the pre-fusion stashes the KL needs are
     the caller's responsibility and unaffected.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from neuroslm.dsl.training_config import TrainingConfig, parse_training_config
from neuroslm.harness import BRIANHarness


V, D, B, T = 64, 16, 2, 6


class _TinyLM(nn.Module):
    """Deterministic ids→logits LM that stashes a cognition aux loss the
    way the DSL Brain aggregator does (`_last_pred_coding_loss`)."""

    def __init__(self, aux_value: float = 0.0):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(V, D)
        self.head = nn.Linear(D, V)
        self.aux_value = aux_value

    def forward(self, ids):
        h = self.embed(ids)
        self._last_hidden = h
        if self.aux_value > 0.0:
            # grad-carrying, like the real pred-coding stash
            self._last_pred_coding_loss = (
                self.head.weight.sum() * 0.0 + self.aux_value)
        return self.head(h)


def _make_harness(cfg: TrainingConfig, aux_value: float = 0.0) -> BRIANHarness:
    lm = _TinyLM(aux_value=aux_value)
    h = BRIANHarness.from_language_model(
        lm, vocab_size=V, d_sem=D, training_config=cfg)
    # Make the pred_coding phase gate wide open regardless of maturity so
    # the aux weight is unambiguously nonzero when the stash is present.
    h.total_loss_config.aux.pred_coding = (0.5, 0.0, 0.01)
    return h


def _ids():
    torch.manual_seed(1)
    ids = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))
    return ids, targets


# ── A. Config surface ─────────────────────────────────────────────────

class TestTrunkPretrainConfig:
    def test_default_is_false(self):
        assert TrainingConfig().trunk_pretrain is False

    def test_empty_block_default_is_false(self):
        assert parse_training_config("").trunk_pretrain is False

    def test_parses_true_from_dsl(self):
        cfg = parse_training_config("trunk_pretrain: true")
        assert cfg.trunk_pretrain is True


# ── B. Cognition-bucket aux gated out of compute_loss ─────────────────

class TestCognitionAuxGate:
    def test_stash_aux_contributes_when_flag_off(self):
        cfg = TrainingConfig()
        assert cfg.trunk_pretrain is False
        h = _make_harness(cfg, aux_value=7.0)
        ids, targets = _ids()
        total_with_aux = h.compute_loss(ids, targets)
        h_no_aux = _make_harness(TrainingConfig(), aux_value=0.0)
        baseline = h_no_aux.compute_loss(ids, targets)
        assert total_with_aux.item() > baseline.item() + 0.5, (
            "flag OFF must preserve existing behaviour: the pred_coding "
            "stash (7.0 nats at weight ≥0.25) must appear in total")

    def test_stash_aux_excluded_when_flag_on(self):
        cfg = TrainingConfig()
        cfg.trunk_pretrain = True
        h = _make_harness(cfg, aux_value=7.0)
        ids, targets = _ids()
        total = h.compute_loss(ids, targets)
        cfg2 = TrainingConfig()
        cfg2.trunk_pretrain = True
        h2 = _make_harness(cfg2, aux_value=0.0)
        baseline = h2.compute_loss(ids, targets)
        assert abs(total.item() - baseline.item()) < 1e-5, (
            "trunk_pretrain must exclude the cognition stash losses from "
            "the trunk's training loss entirely")

    def test_trunk_pretrain_total_is_pure_lm_ce(self):
        cfg = TrainingConfig()
        cfg.trunk_pretrain = True
        h = _make_harness(cfg, aux_value=7.0)
        ids, targets = _ids()
        total = h.compute_loss(ids, targets)
        with torch.no_grad():
            logits = h.language_model(ids)
            ce = F.cross_entropy(
                logits.reshape(-1, V).float(), targets.reshape(-1))
        assert abs(total.item() - ce.item()) < 1e-4, (
            "with every optional mechanism at defaults, trunk_pretrain "
            "total must equal w_lm·CE exactly")


# ── C. Distillation assist SURVIVES the gate ──────────────────────────

class TestDistillAssistSurvives:
    def _distill_harness(self, trunk_pretrain: bool) -> BRIANHarness:
        cfg = TrainingConfig()
        cfg.trunk_pretrain = trunk_pretrain
        h = _make_harness(cfg, aux_value=0.0)
        # Flip the fusion master switch POST-construction so no ensemble
        # is built — _cortex_fusion_aux_step only needs cfg.enabled plus
        # the pre-fusion stashes.
        h.training_config.multi_cortex.enabled = True
        h.training_config.multi_cortex.distillation_enabled = True
        return h

    def _stash_teacher_student(self, h):
        torch.manual_seed(2)
        ids = torch.randint(0, V, (B, T))
        targets = torch.randint(0, V, (B, T))
        # student: uniform (bad); teacher: peaked on the target (good) —
        # gap ≥ ceiling ⇒ λ = lambda_max ⇒ the KL term must be > 0.
        student = torch.zeros(B, T, V, requires_grad=True)
        teacher = torch.full((B, T, V), -10.0)
        teacher.scatter_(-1, targets.unsqueeze(-1), 10.0)
        h._last_pre_fusion_lm_logits = student
        h._last_pre_fusion_cortex_logits = teacher
        return ids, targets, student

    def test_distill_fires_with_flag_on(self):
        h = self._distill_harness(trunk_pretrain=True)
        ids, targets, student = self._stash_teacher_student(h)
        total0 = torch.tensor(0.0)
        total = h._cortex_fusion_aux_step(total0, targets, ids=ids)
        assert total.item() > total0.item() + 1e-4, (
            "KL distillation is the doctrine's trunk-training assist — "
            "trunk_pretrain must NOT gate it")
        total.backward()
        assert student.grad is not None and student.grad.abs().sum() > 0, (
            "distillation gradient must reach the trunk (student) logits")

    def test_distill_off_stays_off(self):
        cfg = TrainingConfig()
        cfg.trunk_pretrain = True
        assert cfg.multi_cortex.distillation_enabled is False
        h = _make_harness(cfg, aux_value=0.0)
        ids, targets, _ = self._stash_teacher_student(h)
        total0 = torch.tensor(0.0)
        total = h._cortex_fusion_aux_step(total0, targets, ids=ids)
        assert abs(total.item() - total0.item()) < 1e-6


# ── D. Fusion mixing becomes identity (isolated-trunk CE) ─────────────

class TestFusionIdentityUnderTrunkPretrain:
    def _logits(self):
        torch.manual_seed(3)
        lm = torch.randn(B, T, V)
        cortex = torch.randn(B, T, V)
        alpha = torch.tensor(0.3)
        return lm, cortex, alpha

    def test_maybe_fuse_is_identity_when_flag_on(self):
        cfg = TrainingConfig()
        cfg.trunk_pretrain = True
        h = _make_harness(cfg)
        lm, cortex, alpha = self._logits()
        out = h._maybe_fuse(lm, cortex, alpha, "logits_mixture")
        assert out is lm, (
            "trunk_pretrain: the training CE must see the ISOLATED trunk "
            "logits — mixing in cortex logits trains a surface that "
            "inference (trunk-only) never runs")

    def test_maybe_fuse_matches_fuse_logits_when_flag_off(self):
        cfg = TrainingConfig()
        h = _make_harness(cfg)
        lm, cortex, alpha = self._logits()
        out = h._maybe_fuse(lm, cortex, alpha, "logits_mixture")
        ref = BRIANHarness._fuse_logits(lm, cortex, alpha, "logits_mixture")
        assert torch.allclose(out, ref), (
            "flag OFF must preserve the existing fusion bit-for-bit")
