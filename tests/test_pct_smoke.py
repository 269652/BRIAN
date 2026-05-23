"""Smoke test for arch/predictive-coding-trunk.

Verifies:
  1. Baseline (PCT off) is unchanged — forward shape + loss path intact.
  2. PCT loss_only mode: forward runs, free-energy is non-zero, predictors
     receive gradients on backward, baseline params still get gradients.
  3. PCT feedback mode: same as 2, plus the error_proj receives gradient
     (because it sits on the forward path).
  4. PCT with use_calm=False (inference path) computes FE without crash.

CPU-only. Should finish in <30s.
"""
from __future__ import annotations
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroslm.config import BrainConfig
from neuroslm.modules.language import LanguageCortex


def _make_cortex(use_pct: bool, mode: str = "loss_only",
                 baseline: bool = False) -> LanguageCortex:
    """Build a small LanguageCortex for smoke testing."""
    torch.manual_seed(0)
    return LanguageCortex(
        vocab_size=64,
        d_hidden=32,
        d_sem=16,
        n_layers=3,
        n_heads=4,
        max_ctx=16,
        n_kv_heads=None,
        n_nt=0,
        hebbian_rank=0,
        geometry_expansion=1.0,
        gradient_checkpointing=False,
        mod_capacity=1.0,
        baseline=baseline,
        enable_memory_xattn=False,
        use_attention_pool=False,
        dropout=0.0,
        use_predictive_coding_trunk=use_pct,
        pct_mode=mode,
        pct_feedback_alpha=0.1,
        pct_hidden_mult=0.5,
        pct_lambda_fe=0.5,
        pct_include_embedding_predictor=True,
    )


def _grads_present(module: torch.nn.Module) -> bool:
    """Return True if ANY param of the module has a non-zero gradient."""
    for p in module.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0.0:
            return True
    return False


def _all_grads_present(module: torch.nn.Module) -> tuple[bool, list[str]]:
    """Return (all_present, list_of_missing_param_names)."""
    missing = []
    for name, p in module.named_parameters():
        if p.requires_grad and (p.grad is None):
            missing.append(name)
    return len(missing) == 0, missing


def test_1_baseline_unchanged():
    """PCT off → no PCT params, FE loss path is None, output shape unchanged."""
    cx = _make_cortex(use_pct=False, baseline=True)
    assert cx.pct is None, "baseline cortex should have no PCT"
    ids = torch.randint(0, 64, (2, 8))
    out = cx(ids)
    assert len(out) == 4, f"baseline forward returns 4-tuple, got {len(out)}"
    logits, sem, h, pc_loss = out
    assert logits.shape == (2, 8, 64)
    assert sem.shape == (2, 16)
    assert h.shape == (2, 8, 32)
    assert torch.isfinite(pc_loss).item() and pc_loss.item() >= 0.0
    assert cx._last_fe_loss is None, "no FE breakdown when PCT off"
    print("[1] non-PCT path unchanged: PASS")


def test_2_pct_loss_only():
    """PCT loss_only: forward runs, FE loss non-zero, predictors get grads."""
    cx = _make_cortex(use_pct=True, mode="loss_only", baseline=False)
    assert cx.pct is not None, "PCT cortex should have a PCT trunk"
    # n_layers=3 + include_embedding_predictor → 3 predictors
    assert len(cx.pct.predictors) == 3, (
        f"expected 3 predictors, got {len(cx.pct.predictors)}")

    cx.train()
    ids = torch.randint(0, 64, (2, 8))
    logits, sem, h, pc_loss = cx(ids)
    assert logits.shape == (2, 8, 64)
    assert pc_loss.requires_grad, "pc_loss should be a graph-attached tensor"
    assert cx._last_fe_loss is not None, "FE breakdown should be cached"
    assert cx._last_fe_loss.item() > 0.0, (
        "FE loss should be > 0 with random predictors")
    # Predictors are init_identity=True (fc2 zero) → initial FE comes ONLY
    # from the genuine inter-layer difference (h_{n+1} != h_n in general).
    # Confirm:
    fe_per_layer = cx._last_fe_breakdown["fe_per_layer"]
    assert len(fe_per_layer) == 3
    assert all(v > 0 for v in fe_per_layer), f"all FE values > 0: {fe_per_layer}"

    # Backward: predictors should accumulate grads, blocks should too.
    target = torch.randint(0, 64, (2, 8))
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 64), target.reshape(-1))
    total_loss = ce + pc_loss
    total_loss.backward()

    assert _grads_present(cx.pct), "PCT predictors should receive gradient"
    assert _grads_present(cx.blocks), "Blocks should still receive gradient"
    assert _grads_present(cx.tok_emb), "Token embedding should receive gradient"

    # Specifically check predictor[0].fc2 (zero-init): MUST get a gradient,
    # otherwise the FE loss isn't propagating through the predictor MLP.
    p0_fc2 = cx.pct.predictors[0].fc2.weight
    assert p0_fc2.grad is not None and p0_fc2.grad.abs().sum().item() > 0, (
        "predictor[0].fc2 must receive gradient from FE loss")

    # log_precision MUST also get a gradient (FE = precision * err^2)
    lp0 = cx.pct.predictors[0].log_precision
    assert lp0.grad is not None and lp0.grad.abs().sum().item() > 0, (
        "log_precision should receive gradient from FE loss")

    print("[2] PCT loss_only: PASS  "
          f"(fe_mean={cx._last_fe_breakdown['fe_mean']:.4f}, "
          f"per_layer={['%.4f' % v for v in fe_per_layer]})")


def test_3_pct_feedback():
    """PCT feedback: error_proj on forward path receives gradient through LM loss."""
    cx = _make_cortex(use_pct=True, mode="feedback", baseline=False)
    assert cx.pct.mode == "feedback"
    assert cx.pct.error_proj is not None
    # error_proj is zero-init: at step 0 forward is bit-identical to standard.
    assert cx.pct.error_proj.weight.abs().sum().item() == 0.0

    cx.train()
    ids = torch.randint(0, 64, (2, 8))
    logits, sem, h, pc_loss = cx(ids)
    target = torch.randint(0, 64, (2, 8))
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 64), target.reshape(-1))
    (ce + pc_loss).backward()

    # error_proj is on the forward path through the LM loss (not just the FE).
    # At zero-init it contributes zero, but its GRADIENT should be non-zero
    # because dL/dW = ∂L/∂(out) ⊗ (input error) — both factors are non-zero
    # for the LM path even when W=0.
    ep = cx.pct.error_proj.weight
    assert ep.grad is not None, "error_proj must receive gradient"
    assert ep.grad.abs().sum().item() > 0.0, (
        "error_proj gradient must be non-zero "
        "(LM loss flows through the projection even though W=0)")

    print("[3] PCT feedback: PASS  "
          f"(error_proj.grad.norm={ep.grad.norm().item():.4f})")


def test_4_pct_eval_mode():
    """PCT in eval mode: no CALM crash, FE loss still computed."""
    cx = _make_cortex(use_pct=True, mode="loss_only", baseline=False)
    cx.eval()
    ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        logits, sem, h, pc_loss = cx(ids)
    assert logits.shape == (2, 8, 64)
    # In eval mode pc_loss is computed but doesn't require grad
    assert pc_loss.item() >= 0.0
    print("[4] PCT eval mode: PASS")


def test_5_legacy_default_fallback():
    """A BrainConfig without PCT flags (loaded from old ckpt) defaults to False."""
    cfg = BrainConfig()
    assert cfg.use_predictive_coding_trunk is False
    assert cfg.pct_mode == "loss_only"
    assert cfg.pct_lambda_fe == 0.1
    print("[5] BrainConfig defaults: PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("PCT smoke tests")
    print("=" * 60)
    test_1_baseline_unchanged()
    test_2_pct_loss_only()
    test_3_pct_feedback()
    test_4_pct_eval_mode()
    test_5_legacy_default_fallback()
    print("=" * 60)
    print("ALL TESTS PASSED")
