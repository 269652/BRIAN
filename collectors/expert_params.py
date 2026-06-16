"""Collect expert parameter counts by inspecting HuggingFace model configs."""
import json
import sys
from pathlib import Path

METRICS = [
    "EXPERT_GENERAL_PARAMS",
    "EXPERT_CODE_PARAMS",
    "EXPERT_REASONING_PARAMS",
]

# HF cache directories where model configs live after first download
_HF_CACHE_DIRS = [
    Path.home() / ".cache" / "huggingface" / "hub",
]


def _count_params_from_config(config: dict) -> int | None:
    """Estimate param count from a transformers config dict."""
    # Common transformer param estimation:
    # params ≈ vocab_size * hidden_size + n_layers * (12 * hidden_size^2)
    vocab = config.get("vocab_size", 0)
    hidden = config.get("hidden_size") or config.get("n_embd", 0)
    n_layers = config.get("num_hidden_layers") or config.get("n_layer", 0)
    intermediate = config.get("intermediate_size") or (4 * hidden)

    if not all([vocab, hidden, n_layers]):
        return None

    # Embedding
    embed_params = vocab * hidden
    # Each transformer layer: attn (4 * h^2) + MLP (h*inter + inter*h) + norms
    layer_params = 4 * hidden * hidden + 2 * hidden * intermediate + 4 * hidden
    total = embed_params + n_layers * layer_params

    return total


def _format_params(n: int) -> str:
    """Format param count as human-readable string."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.0f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return str(n)


def collect(root: Path) -> dict[str, str]:
    """Try to read expert param counts from cached HF model configs."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    metrics: dict[str, str] = {}

    # First get the expert roster to know which models to look up
    try:
        from collectors.expert_roster import collect as get_roster
        roster = get_roster(root)
    except Exception:
        return metrics

    role_to_model = {}
    for role in ["GENERAL", "CODE", "REASONING"]:
        model_id = roster.get(f"EXPERT_{role}_MODEL")
        if model_id:
            role_to_model[role] = model_id

    # Try transformers library (most reliable)
    try:
        from transformers import AutoConfig

        for role, model_id in role_to_model.items():
            try:
                config = AutoConfig.from_pretrained(model_id)
                # Many configs expose num_parameters or we compute from architecture
                config_dict = config.to_dict()
                n_params = _count_params_from_config(config_dict)
                if n_params:
                    metrics[f"EXPERT_{role}_PARAMS"] = _format_params(n_params)
            except Exception:
                pass

        return metrics
    except ImportError:
        pass

    # Fallback: try to find config.json in HF cache
    for cache_dir in _HF_CACHE_DIRS:
        if not cache_dir.exists():
            continue

        for role, model_id in role_to_model.items():
            if f"EXPERT_{role}_PARAMS" in metrics:
                continue

            # HF cache stores models as models--owner--repo
            cache_name = "models--" + model_id.replace("/", "--")
            model_cache = cache_dir / cache_name

            if not model_cache.exists():
                continue

            # Find config.json in snapshots
            for config_file in model_cache.rglob("config.json"):
                try:
                    with open(config_file) as f:
                        config_dict = json.load(f)
                    n_params = _count_params_from_config(config_dict)
                    if n_params:
                        metrics[f"EXPERT_{role}_PARAMS"] = _format_params(n_params)
                        break
                except Exception:
                    pass

    return metrics
