# -*- coding: utf-8 -*-
"""Phase A.2 — LR schedule (warmup + cosine) and mixed precision.

`Brain`'s training uses linear warmup followed by cosine decay over the
full step budget. The harness needs the same so DSL-driven runs have
comparable convergence.

Mixed precision: autocast over the harness forward + loss compute, with
bf16 on A100 (no GradScaler needed) and fp16 fallback on consumer GPUs.
"""
import math
import pytest
import torch

from pathlib import Path
from neuroslm.dsl.codegen import CodeGenerator
from neuroslm.dsl.multifile import compile_folder
from neuroslm.dsl.training_config import TrainingConfig
from neuroslm.harness import BRIANHarness, cosine_warmup_lr


ARCH_ROOT = Path(__file__).resolve().parent.parent / "architectures" / "master"


# ── Schedule math ──────────────────────────────────────────────────

class TestCosineWarmupLR:
    def test_warmup_starts_near_zero(self):
        # At step 0 we're at the start of warmup
        lr = cosine_warmup_lr(step=0, base_lr=1e-3, warmup=100, total=1000)
        assert lr <= 1e-3 * 0.02   # very small at step 0

    def test_warmup_peak_at_end_of_warmup(self):
        # At step==warmup, lr should be at its peak (== base_lr)
        lr = cosine_warmup_lr(step=100, base_lr=1e-3, warmup=100, total=1000)
        assert math.isclose(lr, 1e-3, rel_tol=1e-6)

    def test_cosine_decay_after_warmup(self):
        # Mid-decay: cosine should be at half between base and min
        lr_mid = cosine_warmup_lr(step=550, base_lr=1e-3, warmup=100, total=1000)
        # min_lr default = 0.1 * base. At mid-decay cosine = 0.5,
        # so lr ≈ 0.1 + 0.5 * 0.9 = 0.55 * base
        assert 0.4e-3 < lr_mid < 0.7e-3

    def test_end_at_min_lr(self):
        # At step==total, lr should be at min_lr
        lr = cosine_warmup_lr(step=1000, base_lr=1e-3, warmup=100, total=1000,
                              min_lr_ratio=0.1)
        assert math.isclose(lr, 1e-4, rel_tol=1e-4)

    def test_step_past_total_clamps(self):
        # Steps past total should clamp to min_lr (not negative, not oscillating)
        lr = cosine_warmup_lr(step=2000, base_lr=1e-3, warmup=100, total=1000,
                              min_lr_ratio=0.1)
        assert math.isclose(lr, 1e-4, rel_tol=1e-4)


# ── Schedule integrated into harness ───────────────────────────────

class TestHarnessAppliesSchedule:
    def _small_harness(self):
        ir = compile_folder(ARCH_ROOT)
        Cls = CodeGenerator(ir, module_name="SchedTestCircuit").compile_to_module()
        cfg = TrainingConfig()
        cfg.learning_rate = 1e-3
        cfg.grad_accum = 1
        return BRIANHarness(circuit=Cls(d_sem=64), vocab_size=256,
                           d_sem=64, training_config=cfg)

    def test_schedule_applied_when_total_steps_provided(self):
        h = self._small_harness()
        h.set_schedule(warmup=10, total=100, min_lr_ratio=0.1)

        ids = torch.randint(0, 256, (2, 8))
        targets = torch.randint(0, 256, (2, 8))

        # Step 1: LR should be at warmup-start (very small)
        h.train_step(ids, targets)
        lr_after_step1 = h._optimizer.param_groups[0]["lr"]
        assert lr_after_step1 < 1e-3 * 0.2

        # Run to step 10 (end of warmup); LR should be ~base
        for _ in range(9):
            h.train_step(ids, targets)
        lr_at_warmup = h._optimizer.param_groups[0]["lr"]
        assert math.isclose(lr_at_warmup, 1e-3, rel_tol=0.1)


# ── Mixed precision ────────────────────────────────────────────────

class TestMixedPrecision:
    def _small_harness(self, dtype: str):
        ir = compile_folder(ARCH_ROOT)
        Cls = CodeGenerator(ir, module_name=f"MPCircuit_{dtype}").compile_to_module()
        cfg = TrainingConfig()
        h = BRIANHarness(circuit=Cls(d_sem=64), vocab_size=256, d_sem=64,
                        training_config=cfg)
        h.enable_mixed_precision(dtype=dtype)
        return h

    def test_bf16_forward_runs(self):
        if not hasattr(torch, "bfloat16"):
            pytest.skip("bf16 unavailable")
        h = self._small_harness("bf16")
        ids = torch.randint(0, 256, (2, 8))
        # Forward should still produce float32 logits (autocast unwraps at return)
        logits = h(ids)
        assert logits.shape == (2, 8, 256)
        assert not torch.isnan(logits).any()

    def test_bf16_train_step_doesnt_crash(self):
        if not hasattr(torch, "bfloat16"):
            pytest.skip("bf16 unavailable")
        h = self._small_harness("bf16")
        ids = torch.randint(0, 256, (2, 8))
        targets = torch.randint(0, 256, (2, 8))
        loss = h.train_step(ids, targets)
        assert loss > 0
        assert not math.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
