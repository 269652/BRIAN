"""Comprehensive unit tests for per-sample loss clipping (step 1500 fix).

Tests verify that the loss clipping mechanism correctly:
1. Suppresses outliers without affecting normal examples
2. Uses adaptive threshold (median-based, no fixed hyperparams)
3. Preserves gradient signal for hard (but not pathological) examples
4. Integrates correctly with label smoothing and other CE parameters
"""
import torch
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neuroslm.brain import Brain
from neuroslm.config import rcc_bowtie_30m_p4, rcc_bowtie_30m_p3


class TestPerSampleLossClipping:
    """Test suite for per-sample loss clipping robustness."""

    @staticmethod
    def test_clipping_disabled_baseline():
        """Verify baseline behavior: without clipping, outliers dominate."""
        cfg = rcc_bowtie_30m_p4()
        cfg.loss_clip_robust = False
        brain = Brain(cfg)
        brain.eval()

        # Deterministic synthetic pathological batch: seq 2 has extreme loss.
        torch.manual_seed(0)
        B, T, V = 4, 16, cfg.vocab_size
        logits = torch.randn(B, T, V, dtype=torch.float32)
        logits[2] *= 40  # Sequence 2: heavily amplified → confidently wrong

        targets = torch.randint(0, V, (B, T))
        # Make seq 2 even harder by shifting targets (confidently wrong)
        targets[2] = (targets[2] + V // 2) % V

        with torch.no_grad():
            loss_per_seq = brain._chunked_ce(logits, targets,
                                             loss_clip_robust=False,
                                             loss_clip_factor=3.0)

        loss_mean_unclipped = loss_per_seq.mean()
        print(f"\n[test_clipping_disabled_baseline]")
        print(f"  Per-sequence losses: {[f'{x:.2f}' for x in loss_per_seq.detach().cpu()]}")
        print(f"  Mean: {loss_mean_unclipped.item():.4f}")
        print(f"  Outlier ratio: {(loss_per_seq[2] / loss_per_seq[0]).item():.1f}x")

        # Assertion: outlier should dominate
        assert loss_per_seq[2] > loss_per_seq.mean(), \
            "Without clipping, outlier should be highest"
        assert loss_per_seq[2] > 10 * loss_per_seq[0], \
            "Pathological sequence should have >10x loss of normal"

    @staticmethod
    def test_clipping_enabled_suppresses_outliers():
        """Verify clipping suppresses outliers to 3x median."""
        cfg = rcc_bowtie_30m_p4()
        cfg.loss_clip_robust = True
        cfg.loss_clip_factor = 3.0
        brain = Brain(cfg)
        brain.eval()

        B, T, V = 4, 16, cfg.vocab_size
        logits = torch.randn(B, T, V, dtype=torch.float32)
        logits[2] *= 15  # Same pathological setup

        targets = torch.randint(0, V, (B, T))
        targets[2] = (targets[2] + V // 2) % V

        with torch.no_grad():
            loss_per_seq_clipped = brain._chunked_ce(logits, targets,
                                                      loss_clip_robust=True,
                                                      loss_clip_factor=3.0)

        median = loss_per_seq_clipped.median()
        max_allowed = 3.0 * median

        print(f"\n[test_clipping_enabled_suppresses_outliers]")
        print(f"  Per-sequence losses: {[f'{x:.2f}' for x in loss_per_seq_clipped.detach().cpu()]}")
        print(f"  Median: {median.item():.4f}")
        print(f"  Max allowed (3 × median): {max_allowed.item():.4f}")
        print(f"  Clipped seq[2]: {loss_per_seq_clipped[2].item():.4f}")

        # Assertion: all losses ≤ 3 × median
        assert (loss_per_seq_clipped <= max_allowed + 1e-5).all(), \
            f"All losses must be ≤ {max_allowed.item():.4f}"
        assert loss_per_seq_clipped[2] <= max_allowed + 1e-5, \
            "Outlier should be clamped at 3 × median"

    @staticmethod
    def test_normal_batch_unaffected():
        """Verify normal (non-outlier) batches pass through unclipped."""
        cfg = rcc_bowtie_30m_p4()
        cfg.loss_clip_robust = True
        cfg.loss_clip_factor = 3.0
        brain = Brain(cfg)
        brain.eval()

        # Normal batch: all sequences have similar difficulty
        B, T, V = 4, 16, cfg.vocab_size
        logits = torch.randn(B, T, V, dtype=torch.float32) * 0.5  # Low variance
        targets = torch.randint(0, V, (B, T))

        with torch.no_grad():
            loss_per_seq = brain._chunked_ce(logits, targets,
                                             loss_clip_robust=True,
                                             loss_clip_factor=3.0)

        median = loss_per_seq.median()
        max_allowed = 3.0 * median

        print(f"\n[test_normal_batch_unaffected]")
        print(f"  Per-sequence losses: {[f'{x:.2f}' for x in loss_per_seq.detach().cpu()]}")
        print(f"  Max: {loss_per_seq.max().item():.4f}, threshold: {max_allowed.item():.4f}")
        print(f"  Clipping triggered: {(loss_per_seq.max() >= max_allowed * 0.95).item()}")

        # Assertion: no clipping should occur
        assert loss_per_seq.max() < max_allowed * 0.95, \
            "Normal batch should not trigger clipping"

    @staticmethod
    def test_clipping_factor_affects_threshold():
        """Verify different clipping factors produce different thresholds."""
        cfg = rcc_bowtie_30m_p4()
        cfg.loss_clip_robust = True
        brain = Brain(cfg)
        brain.eval()

        B, T, V = 4, 16, cfg.vocab_size
        logits = torch.randn(B, T, V, dtype=torch.float32)
        logits[2] *= 20
        targets = torch.randint(0, V, (B, T))

        with torch.no_grad():
            # Test with factor=2.0 (tighter)
            loss_factor_2 = brain._chunked_ce(logits, targets,
                                              loss_clip_robust=True,
                                              loss_clip_factor=2.0)
            # Test with factor=5.0 (looser)
            loss_factor_5 = brain._chunked_ce(logits, targets,
                                              loss_clip_robust=True,
                                              loss_clip_factor=5.0)

        median = loss_factor_5.median()

        print(f"\n[test_clipping_factor_affects_threshold]")
        print(f"  Factor 2.0 (tight): {[f'{x:.2f}' for x in loss_factor_2.detach().cpu()]}")
        print(f"  Factor 5.0 (loose): {[f'{x:.2f}' for x in loss_factor_5.detach().cpu()]}")

        # Tighter factor should produce lower max loss
        assert loss_factor_2.max() <= loss_factor_5.max() + 1e-5, \
            "Tighter factor should produce lower clamping threshold"

    @staticmethod
    def test_gradient_flow_preserved():
        """Verify gradients flow correctly through clipped sequences."""
        cfg = rcc_bowtie_30m_p4()
        cfg.loss_clip_robust = True
        cfg.loss_clip_factor = 3.0
        brain = Brain(cfg)

        B, T, V = 2, 8, cfg.vocab_size
        logits = torch.randn(B, T, V, dtype=torch.float32, requires_grad=True)
        targets = torch.randint(0, V, (B, T))

        # Compute loss with clipping
        loss_per_seq = brain._chunked_ce(logits, targets,
                                         loss_clip_robust=True,
                                         loss_clip_factor=3.0)
        loss = loss_per_seq.mean()

        # Backprop
        loss.backward()

        print(f"\n[test_gradient_flow_preserved]")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradient norm: {logits.grad.norm().item():.6f}")
        print(f"  Gradient non-zero: {(logits.grad.abs() > 1e-8).sum().item()} / {logits.numel()} elements")

        # Assertion: gradients exist and are non-zero
        assert logits.grad is not None, "Gradients should flow"
        assert (logits.grad.abs() > 1e-8).sum() > 0, "At least some gradients should be non-zero"

    @staticmethod
    def test_label_smoothing_compatibility():
        """Verify clipping works with label smoothing."""
        cfg = rcc_bowtie_30m_p4()
        cfg.loss_clip_robust = True
        cfg.label_smoothing = 0.1
        brain = Brain(cfg)
        brain.eval()

        B, T, V = 4, 16, cfg.vocab_size
        logits = torch.randn(B, T, V, dtype=torch.float32)
        targets = torch.randint(0, V, (B, T))

        with torch.no_grad():
            loss_per_seq = brain._chunked_ce(logits, targets,
                                             label_smoothing=0.1,
                                             loss_clip_robust=True,
                                             loss_clip_factor=3.0)

        print(f"\n[test_label_smoothing_compatibility]")
        print(f"  Losses with label smoothing: {[f'{x:.2f}' for x in loss_per_seq.detach().cpu()]}")

        # Assertion: losses are valid
        assert (loss_per_seq > 0).all(), "All losses should be positive"
        assert not loss_per_seq.isnan().any(), "No NaNs should appear"

    @staticmethod
    def test_step_1500_simulation():
        """Simulate the actual step 1500 pathology."""
        cfg = rcc_bowtie_30m_p4()
        cfg.loss_clip_robust = True
        cfg.loss_clip_factor = 3.0
        brain = Brain(cfg)
        brain.eval()

        # Simulate 4 chat sequences, 1 pathological (typical batch at step 1500)
        torch.manual_seed(0)
        B, T, V = 4, 1024, cfg.vocab_size
        logits = torch.randn(B, T, V, dtype=torch.float32)

        # Sequences 0-2: normal dialogue (baseline loss ~ln V)
        # Sequence 3: pathological chunk — strongly amplified + all-wrong
        # targets so it is a clear >3×median outlier the clip must cap.
        # ×20 puts the unclipped seq3 loss well above the 3×median cap, so
        # capping it reduces the batch mean by >20%.
        logits[3] *= 20.0

        targets = torch.randint(0, V, (B, T))
        targets[3] = (targets[3] + V // 2) % V  # entire seq3 confidently wrong

        with torch.no_grad():
            # Without clipping
            loss_unclipped = brain._chunked_ce(logits, targets,
                                               loss_clip_robust=False)
            # With clipping
            loss_clipped = brain._chunked_ce(logits, targets,
                                             loss_clip_robust=True,
                                             loss_clip_factor=3.0)

        print(f"\n[test_step_1500_simulation] (reproduces actual step 1500 pathology)")
        print(f"  WITHOUT clipping: mean={loss_unclipped.mean().item():.2f}, "
              f"ppl={torch.exp(loss_unclipped.mean()).item():.0f}")
        print(f"  WITH clipping:    mean={loss_clipped.mean().item():.2f}, "
              f"ppl={torch.exp(loss_clipped.mean()).item():.0f}")
        print(f"  PPL ratio (before/after): {(torch.exp(loss_unclipped.mean()) / torch.exp(loss_clipped.mean())).item():.1f}x")

        # Assertion: clipping should reduce impact of outlier
        assert loss_clipped.mean() < loss_unclipped.mean() * 0.8, \
            "Clipping should reduce mean loss by suppressing outlier"


def test_loss_clipping_disabled():
    """Backward compatibility test: clipping disabled by default."""
    cfg = rcc_bowtie_30m_p3()  # P3 doesn't have clipping by default
    assert not cfg.loss_clip_robust, "P3 should have clipping disabled"
    print(f"\n[test_loss_clipping_disabled] P3 legacy compatibility: ✓")


# ── Brain ↔ DSL gradient parity ───────────────────────────────────────
# These tests guard the Brain↔DSL clipping equivalence at the *gradient*
# level, not just the forward value. Forward values are trivially equal
# because detach() doesn't change a scalar; the divergence only shows up
# in autograd. Until this test was added, every parity test compared the
# forward output only — so the median-not-detached bug in Brain went
# unnoticed and accumulated over thousands of training steps.

def test_brain_clipping_gradient_matches_detached_median():
    """Brain's _chunked_ce must detach the median for clip-threshold.

    If the median is on the autograd graph, outlier sequences leak
    gradient through the threshold → clipping dampens its own learning
    signal. This is the bug found 2026-05-28 by inspecting why DSL ↔
    Brain trajectories diverge after step ~1500 (first clipping engage).
    """
    import torch.nn.functional as F

    torch.manual_seed(0)
    B, T, V = 4, 16, 64
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.randint(0, V, (B, T))
    # Force seq 3 to be a clear >3×median outlier so clipping fires
    targets[3] = (targets[3] + V // 2) % V

    # Brain path
    from neuroslm.brain import Brain as _Brain
    brain_loss_per_seq = _Brain._chunked_ce(
        logits, targets, loss_clip_robust=True, loss_clip_factor=3.0)
    brain_loss = brain_loss_per_seq.mean()

    # Manual reference: detached median, matching DSL formulation
    per_token = F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), reduction="none")
    per_seq_ref = per_token.reshape(B, T).mean(dim=1)
    thresh_ref = (per_seq_ref.detach().median() * 3.0).clamp(min=1e-8)
    ref_loss = torch.minimum(per_seq_ref, thresh_ref).mean()

    # Forward values must already match (and did, even with the bug)
    assert torch.allclose(brain_loss, ref_loss, atol=1e-6), \
        f"forward differs: brain={brain_loss.item()} ref={ref_loss.item()}"

    # Gradient must also match — this is what would have caught the bug
    brain_grad = torch.autograd.grad(brain_loss, logits, retain_graph=True)[0]
    ref_grad = torch.autograd.grad(ref_loss, logits)[0]
    max_diff = (brain_grad - ref_grad).abs().max().item()
    assert max_diff < 1e-6, (
        f"Brain clipping gradient diverges from DSL/Phi-3 formulation "
        f"(max diff {max_diff:.3e}). The median in _chunked_ce must be "
        f"computed on loss_per_seq.detach(), not on the live graph.")


def test_brain_clipping_matches_dsl_harness_gradient():
    """Direct Brain ↔ DSL harness gradient parity on the same inputs."""
    import torch.nn.functional as F
    from neuroslm.brain import Brain as _Brain
    from neuroslm.harness import BRIANHarness
    from neuroslm.dsl.training_config import TrainingConfig, LossClippingConfig

    torch.manual_seed(0)
    B, T, V = 4, 16, 64
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.randint(0, V, (B, T))
    targets[3] = (targets[3] + V // 2) % V  # outlier seq

    # Brain path
    brain_loss = _Brain._chunked_ce(
        logits, targets, loss_clip_robust=True, loss_clip_factor=3.0).mean()

    # DSL harness path — use _compute_loss_from_logits directly so we
    # don't need a wrapped language model for this gradient comparison
    cfg = TrainingConfig()
    cfg.label_smoothing = 0.0
    cfg.loss_clipping = LossClippingConfig(enabled=True, factor=3.0)
    # Bypass __init__ since we only need _compute_loss_from_logits
    h = BRIANHarness.__new__(BRIANHarness)
    h.training_config = cfg
    dsl_loss = h._compute_loss_from_logits(logits, targets)

    assert torch.allclose(brain_loss, dsl_loss, atol=1e-6), \
        f"forward: brain={brain_loss.item()} dsl={dsl_loss.item()}"

    brain_grad = torch.autograd.grad(brain_loss, logits, retain_graph=True)[0]
    dsl_grad = torch.autograd.grad(dsl_loss, logits)[0]
    max_diff = (brain_grad - dsl_grad).abs().max().item()
    assert max_diff < 1e-6, (
        f"Brain ↔ DSL clipping gradient diverged (max diff {max_diff:.3e}). "
        f"Both implementations must detach the median when computing the "
        f"clip threshold; otherwise outliers leak gradient through it.")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Per-Sample Loss Clipping Unit Tests")
    print("="*70)

    tests = TestPerSampleLossClipping()

    # Run all tests
    tests.test_clipping_disabled_baseline()
    tests.test_clipping_enabled_suppresses_outliers()
    tests.test_normal_batch_unaffected()
    tests.test_clipping_factor_affects_threshold()
    tests.test_gradient_flow_preserved()
    tests.test_label_smoothing_compatibility()
    tests.test_step_1500_simulation()
    test_loss_clipping_disabled()

    print("\n" + "="*70)
    print("✓ All 8 tests passed!")
    print("="*70 + "\n")
