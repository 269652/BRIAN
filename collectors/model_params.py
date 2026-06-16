"""Collect trainable + frozen param counts from the compiled PyTorch model."""
import sys
from pathlib import Path

METRICS = [
    "TRUNK_TRAINABLE_PARAMS",
    "TOTAL_FROZEN_PARAMS",
]


def _format_params(n: int) -> str:
    """Format param count as human-readable."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return str(n)


def collect(root: Path) -> dict[str, str]:
    """Instantiate the model at 'large' preset and count parameters."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    metrics: dict[str, str] = {}

    try:
        from neuroslm.config import PRESETS
        from neuroslm.brain import NeuralOrchestrator

        # Use 'large' preset (the standard training config)
        cfg = PRESETS["large"]()
        model = NeuralOrchestrator(cfg)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        metrics["TRUNK_TRAINABLE_PARAMS"] = _format_params(trainable)
        if frozen > 0:
            metrics["TOTAL_FROZEN_PARAMS"] = _format_params(frozen)

    except Exception:
        # Model instantiation may fail without GPU / dependencies
        # Fall through to TOML override
        pass

    return metrics
