"""Collect Layer B OOD results from results/*.json files."""
import json
from pathlib import Path

METRICS = [
    "B0_TRAINABLE", "B0_STEPS", "B0_TRAIN_PPL", "B0_OOD_PPL", "B0_GAP_RATIO", "B0_ARTIFACT",
    "B1_TRAINABLE", "B1_STEPS", "B1_TRAIN_PPL", "B1_OOD_PPL", "B1_GAP_RATIO", "B1_ARTIFACT",
    "B2FIX_TRAINABLE", "B2FIX_STEPS", "B2FIX_TRAIN_PPL", "B2FIX_OOD_PPL", "B2FIX_GAP_RATIO", "B2FIX_ARTIFACT",
    "B3_TRAINABLE", "B3_STEPS", "B3_TRAIN_PPL", "B3_OOD_PPL", "B3_GAP_RATIO", "B3_ARTIFACT",
    "LAYER_B_BASELINE_GAP_RATIO",
]

# Map result JSON filenames → variant prefix
_VARIANT_MAP = {
    "ood_baseline-80k_107M_step80000.json": "B0",
    "ood_recursive_108M_step5000.json": "B1",
    "ood_rezero-fixed_107M_step7000.json": "B2FIX",
    "ood_pct-30m_68M_step4000.json": "B3",
}


def collect(root: Path) -> dict[str, str]:
    """Read OOD result JSONs and emit per-variant metrics."""
    metrics: dict[str, str] = {}
    results_dir = root / "results"

    if not results_dir.exists():
        return metrics

    for filename, prefix in _VARIANT_MAP.items():
        fpath = results_dir / filename
        if not fpath.exists():
            continue

        with open(fpath) as f:
            data = json.load(f)

        n_params = data.get("n_params", 0)
        metrics[f"{prefix}_TRAINABLE"] = f"{n_params / 1e6:.1f}M" if n_params else ""
        metrics[f"{prefix}_STEPS"] = f"{data.get('step', 0):,}"
        metrics[f"{prefix}_TRAIN_PPL"] = f"{data.get('train_ppl', 0):.1f}"
        metrics[f"{prefix}_OOD_PPL"] = f"{data.get('ood_ppl', 0):.1f}"
        metrics[f"{prefix}_GAP_RATIO"] = f"{data.get('gap_ratio', 0):.2f}"
        metrics[f"{prefix}_ARTIFACT"] = f"results/{filename}"

        if prefix == "B0":
            metrics["LAYER_B_BASELINE_GAP_RATIO"] = f"{data.get('gap_ratio', 0):.2f}"

    return metrics
