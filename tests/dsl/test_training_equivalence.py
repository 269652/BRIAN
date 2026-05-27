# -*- coding: utf-8 -*-
"""Training-equivalence validation: DSL run vs hand-written rcc_bowtie.

Three levels of "follows exactly as the hand-written implementation":

  1. LEARNING RATE — exact, validated in test_lr_parity.py (DSL schedule
     == train.cosine_lr bit-for-bit). Re-asserted here end-to-end.
  2. LOSS FUNCTION — the harness's CE + per-sample clipping matches a
     manual reference computation exactly.
  3. DETERMINISM — same seed → identical loss sequence (reproducible
     training, a prerequisite for any curve-matching).

LOSS-CURVE parity with Brain (identical loss *values* per step) requires
the full bit-identical port of every Brain subsystem with matched init
and data ordering — the N8 arc. That test is provided as a scaffold,
skipped with a clear gate, so it activates the moment N8 lands.
"""
import math
import pytest
import torch
import torch.nn.functional as F

from neuroslm.dsl.nn_lang import build_language_model
from neuroslm.dsl.training_config import TrainingConfig, LossClippingConfig
from neuroslm.harness import BRIANHarness, cosine_warmup_lr


def _make_harness(seed: int, clip: bool = False):
    torch.manual_seed(seed)
    lm = build_language_model(vocab=48, d_model=32, depth=2,
                              n_heads=4, max_ctx=32)
    cfg = TrainingConfig()
    cfg.learning_rate = 3e-4
    cfg.grad_accum = 1
    if clip:
        cfg.loss_clipping = LossClippingConfig(enabled=True, factor=3.0)
    return BRIANHarness.from_language_model(lm, vocab_size=48, d_sem=32,
                                            training_config=cfg)


# ── 1. Learning-rate parity, end-to-end through the optimizer ─────────

class TestLearningRateParity:
    def test_optimizer_lr_follows_schedule(self):
        h = _make_harness(seed=0)
        warmup, total, peak, min_ratio = 50, 500, 3e-4, 0.1
        h.set_schedule(warmup=warmup, total=total, min_lr_ratio=min_ratio)
        ids = torch.randint(0, 48, (4, 16))
        targets = torch.randint(0, 48, (4, 16))
        for step in range(1, 120):
            h.train_step(ids, targets)
            applied = h._optimizer.param_groups[0]["lr"]
            expected = cosine_warmup_lr(step, peak, warmup, total, min_ratio)
            assert abs(applied - expected) < 1e-12, \
                f"step {step}: applied {applied} != schedule {expected}"


# ── 2. Loss-function correctness ──────────────────────────────────────

class TestLossFunction:
    def test_unclipped_equals_cross_entropy(self):
        h = _make_harness(seed=1, clip=False)
        ids = torch.randint(0, 48, (4, 16))
        targets = torch.randint(0, 48, (4, 16))
        logits = h(ids)
        ref = F.cross_entropy(logits.reshape(-1, 48), targets.reshape(-1),
                              label_smoothing=h.training_config.label_smoothing)
        got = h._compute_loss_from_logits(logits, targets)
        assert torch.allclose(got, ref, atol=1e-6)

    def test_clipped_matches_manual_per_sample_clip(self):
        h = _make_harness(seed=2, clip=True)
        ids = torch.randint(0, 48, (4, 16))
        targets = torch.randint(0, 48, (4, 16))
        logits = h(ids)

        # Manual per-sample clip reference
        B, T, V = logits.shape
        per_tok = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                                  reduction="none").reshape(B, T)
        per_seq = per_tok.mean(dim=1)
        thresh = (per_seq.detach().median() * 3.0).clamp(min=1e-8)
        ref = torch.minimum(per_seq, thresh).mean()

        got = h._compute_loss_from_logits(logits, targets)
        assert torch.allclose(got, ref, atol=1e-6)


# ── 3. Determinism (reproducible training) ────────────────────────────

class TestDeterminism:
    def test_same_seed_same_loss_sequence(self):
        def run(seed):
            h = _make_harness(seed=seed)
            h.set_schedule(warmup=10, total=100, min_lr_ratio=0.1)
            g = torch.Generator().manual_seed(123)
            losses = []
            for _ in range(30):
                ids = torch.randint(0, 48, (4, 16), generator=g)
                targets = torch.randint(0, 48, (4, 16), generator=g)
                losses.append(h.train_step(ids, targets))
            return losses

        a = run(7)
        b = run(7)
        for i, (x, y) in enumerate(zip(a, b)):
            assert abs(x - y) < 1e-9, f"step {i}: {x} != {y} (non-deterministic)"


# ── Loss-curve parity with Brain — gated on N8 full port ──────────────

@pytest.mark.skip(reason="N8: requires bit-identical port of all Brain "
                         "subsystems + matched init/data; scaffold below "
                         "activates once N8 lands")
def test_loss_curve_parity_with_brain():
    """When N8 completes (full Brain expressed in DSL with matched init),
    a DSL run and a Brain run on the same seed + data must produce the
    same per-step loss. The comparison structure:

        brain = Brain(rcc_bowtie_30m_p4())
        dsl   = build_full_brain_dsl("rcc_bowtie_30m_p4")
        sync_init(dsl, brain)
        for batch in same_data_stream(seed):
            lb = brain.forward_lm(batch.ids, batch.targets)["loss"]
            ld = dsl_harness.compute_loss(batch.ids, batch.targets)
            assert torch.allclose(ld, lb, atol=1e-4)
    """
    pytest.fail("not reachable until N8")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
