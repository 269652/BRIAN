# -*- coding: utf-8 -*-
"""Exact learning-rate parity: DSL schedule == hand-written Brain schedule.

The DSL harness must drive the optimizer LR identically to the native
trainer so a DSL run's learning-rate column matches Brain's at every
step. This compares the harness's `cosine_warmup_lr` against the real
`neuroslm.train.cosine_lr` over a wide grid of (step, warmup, total,
peak, min_ratio) — asserting bit-equality (atol 1e-12).
"""
import ast
import math
from pathlib import Path

import pytest

from neuroslm.harness import cosine_warmup_lr


def _extract_brain_cosine_lr():
    """Pull the real `cosine_lr` out of neuroslm/train.py by source, without
    importing the module (train.py imports tiktoken/torch-heavy deps not
    present in a bare test env). This still validates against the *actual*
    hand-written function definition, byte-for-byte from source."""
    train_py = Path(__file__).resolve().parent.parent.parent / "neuroslm" / "train.py"
    tree = ast.parse(train_py.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "cosine_lr":
            mod = ast.Module(body=[node], type_ignores=[])
            ns = {"math": math}
            exec(compile(mod, "<train.cosine_lr>", "exec"), ns)
            return ns["cosine_lr"]
    raise RuntimeError("cosine_lr not found in train.py")


brain_cosine_lr = _extract_brain_cosine_lr()


class TestLRScheduleParity:
    @pytest.mark.parametrize("warmup,total,peak,min_ratio", [
        (1000, 10000, 3e-4, 0.1),
        (500, 5000, 6e-4, 0.0),
        (0, 1000, 1e-3, 0.1),
        (2000, 100000, 2.5e-4, 0.05),
        (100, 300, 1e-3, 0.5),
    ])
    def test_matches_brain_at_every_step(self, warmup, total, peak, min_ratio):
        # Sample across warmup, mid-decay, end, and past-total.
        steps = ([0, 1, warmup - 1 if warmup > 1 else 0, warmup, warmup + 1]
                 + list(range(0, total + 1, max(1, total // 20)))
                 + [total, total + 500])
        for s in steps:
            if s < 0:
                continue
            dsl = cosine_warmup_lr(step=s, base_lr=peak, warmup=warmup,
                                   total=total, min_lr_ratio=min_ratio)
            ref = brain_cosine_lr(step=s, warmup=warmup, total=total,
                                  peak=peak, min_ratio=min_ratio)
            assert abs(dsl - ref) < 1e-12, (
                f"step {s}: dsl={dsl!r} brain={ref!r} "
                f"(warmup={warmup} total={total} peak={peak} min={min_ratio})")


def test_full_10k_schedule_identical():
    """The exact schedule a 10k DSL run would use must match Brain's."""
    warmup, total, peak, min_ratio = 1000, 10000, 3e-4, 0.1
    for s in range(0, total + 1):
        dsl = cosine_warmup_lr(s, peak, warmup, total, min_ratio)
        ref = brain_cosine_lr(s, warmup, total, peak, min_ratio)
        assert abs(dsl - ref) < 1e-12, f"divergence at step {s}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
