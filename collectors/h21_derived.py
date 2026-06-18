"""Derived H21 metrics — aliases & deltas computed from existing values.

The README template references several H21 metrics under slightly
different names than the canonical values stored in either:

* ``docs/readme_metrics.toml`` ([h21] block), or
* the live ``layer_b_results`` collector ([b4] block).

Rather than duplicate the source numbers, this collector loads them
both and emits the README-template aliases plus computed percentage
deltas so the prose stays in sync with the underlying measurements.

Aliases:
    H21_CE_BEFORE        ← H21_ABSTAIN_CE_BEFORE
    H21_CE_AFTER         ← H21_ABSTAIN_CE_AFTER
    H21_TRAIN_PPL_BROKEN ← H21_BROKEN_TRAIN_PPL
    H21_OOD_PPL_BROKEN   ← H21_BROKEN_OOD_PPL

Deltas (computed from broken → B4):
    H21_TRAIN_PPL_DELTA  = -((1 - B4_TRAIN_PPL / H21_BROKEN_TRAIN_PPL) * 100)%
    H21_OOD_PPL_DELTA    = -((1 - B4_OOD_PPL   / H21_BROKEN_OOD_PPL  ) * 100)%

The cortex-EMA "delta" is a qualitative label (the broken value was
~0.001 and the fixed value is ~0.5, so we just report "fusion alive").
"""
from __future__ import annotations

from pathlib import Path

METRICS = [
    "H21_CE_BEFORE",
    "H21_CE_AFTER",
    "H21_TRAIN_PPL_BROKEN",
    "H21_OOD_PPL_BROKEN",
    "H21_TRAIN_PPL_DELTA",
    "H21_OOD_PPL_DELTA",
    "H21_CX_EMA_DELTA",
]


def _load_toml(root: Path) -> dict[str, object]:
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
    except Exception:
        return {}
    toml_path = root / "docs" / "readme_metrics.toml"
    if not toml_path.exists():
        return {}
    try:
        with open(toml_path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _get(cfg: dict, *keys: str) -> str | None:
    """Walk nested dict to a leaf string; return None if absent."""
    node: object = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return str(node) if node is not None else None


def _pct_drop(before: str, after: str) -> str | None:
    """Return "-NN%" given two positive PPL strings (after < before)."""
    try:
        b = float(str(before).replace(",", ""))
        a = float(str(after).replace(",", ""))
    except (TypeError, ValueError):
        return None
    if b <= 0:
        return None
    drop = (1.0 - a / b) * 100.0
    return f"-{drop:.0f}%"


def collect(root: Path) -> dict[str, str]:
    """Compute aliases + percentage deltas from existing TOML values."""
    metrics: dict[str, str] = {}
    cfg = _load_toml(root)

    h21 = cfg.get("h21") if isinstance(cfg.get("h21"), dict) else {}
    b4 = cfg.get("b4") if isinstance(cfg.get("b4"), dict) else {}

    # --- Aliases (rename so README template placeholders resolve) ----
    ce_before = h21.get("H21_ABSTAIN_CE_BEFORE") if isinstance(h21, dict) else None
    ce_after = h21.get("H21_ABSTAIN_CE_AFTER") if isinstance(h21, dict) else None
    train_broken = h21.get("H21_BROKEN_TRAIN_PPL") if isinstance(h21, dict) else None
    ood_broken = h21.get("H21_BROKEN_OOD_PPL") if isinstance(h21, dict) else None

    if ce_before is not None:
        metrics["H21_CE_BEFORE"] = str(ce_before)
    if ce_after is not None:
        metrics["H21_CE_AFTER"] = str(ce_after)
    if train_broken is not None:
        metrics["H21_TRAIN_PPL_BROKEN"] = str(train_broken)
    if ood_broken is not None:
        metrics["H21_OOD_PPL_BROKEN"] = str(ood_broken)

    # --- Computed percent deltas (broken → B4 fixed) -----------------
    b4_train = b4.get("B4_TRAIN_PPL") if isinstance(b4, dict) else None
    b4_ood = b4.get("B4_OOD_PPL") if isinstance(b4, dict) else None

    if train_broken is not None and b4_train is not None:
        delta = _pct_drop(str(train_broken), str(b4_train))
        if delta is not None:
            metrics["H21_TRAIN_PPL_DELTA"] = delta
    if ood_broken is not None and b4_ood is not None:
        delta = _pct_drop(str(ood_broken), str(b4_ood))
        if delta is not None:
            metrics["H21_OOD_PPL_DELTA"] = delta

    # --- Qualitative EMA delta (0.001 → 0.5 ≈ "fusion alive") --------
    if (
        isinstance(h21, dict)
        and "H21_CX_EMA_BROKEN" in h21
        and "H21_CX_EMA_FIXED" in h21
    ):
        metrics["H21_CX_EMA_DELTA"] = "fusion alive"

    return metrics
