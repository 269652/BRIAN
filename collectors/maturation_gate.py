"""Collect maturation-gate thresholds from the training loop.

Two values live in ``neuroslm/train.py`` and gate when the auxiliary
losses are allowed to ramp from the infancy weight (0.001) up to
target (1.0):

* ``MATURATION_STEP_THRESHOLD`` — historical step gate (``step < N``)
  documented in the maturation comment block. The continuous MAT signal
  replaced this, but the number is still referenced in the README as
  the rough "infancy window" length.
* ``MATURATION_LM_LOSS_THRESHOLD`` — value of ``_maturation_lm_threshold``
  in ``train.py``; the awakening transition fires when
  ``lm_loss < threshold`` and ``MAT > 0.3``.

Both are grepped from the source file so updating the training script
auto-updates the README.
"""
from __future__ import annotations

import re
from pathlib import Path

METRICS = [
    "MATURATION_STEP_THRESHOLD",
    "MATURATION_LM_LOSS_THRESHOLD",
]


def collect(root: Path) -> dict[str, str]:
    """Grep `_maturation_lm_threshold` + historical step gate."""
    metrics: dict[str, str] = {}

    train_py = root / "neuroslm" / "train.py"
    if not train_py.exists():
        return metrics
    text = train_py.read_text(encoding="utf-8", errors="replace")

    # 1) lm-loss threshold: `_maturation_lm_threshold = <float>`
    m = re.search(
        r"^\s*_maturation_lm_threshold\s*=\s*([\d.]+)",
        text,
        re.MULTILINE,
    )
    if m:
        val = m.group(1)
        # Strip trailing zeros for readability (7.5 not 7.500).
        try:
            f = float(val)
            metrics["MATURATION_LM_LOSS_THRESHOLD"] = (
                f"{f:g}" if f != int(f) else str(int(f))
            )
        except ValueError:
            metrics["MATURATION_LM_LOSS_THRESHOLD"] = val

    # 2) Historical step gate: documented as "step < N" in the
    #    Maturity-Driven Topological Maturation comment block.
    step = re.search(r'"step\s*<\s*(\d+)"', text)
    if step:
        metrics["MATURATION_STEP_THRESHOLD"] = step.group(1)

    return metrics
