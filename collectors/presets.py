"""Collect preset parameter counts from neuroslm/config.py."""
import re
from pathlib import Path

METRICS = [
    "PRESET_TINY_PARAMS",
    "PRESET_SMALL_PARAMS",
    "PRESET_MEDIUM_PARAMS",
    "PRESET_LARGE_PARAMS",
    "PRESET_XL_PARAMS",
    "PRESET_XXL_PARAMS",
]


def collect(root: Path) -> dict[str, str]:
    """Extract preset param counts from config.py docstrings."""
    metrics: dict[str, str] = {}

    config_path = root / "neuroslm" / "config.py"
    if not config_path.exists():
        return metrics

    text = config_path.read_text(encoding="utf-8", errors="replace")

    # Match: def preset_name() -> BrainConfig:\n    """~NNM params ..."""
    pattern = r'def\s+(\w+)\(\)\s*->.*?"""(.*?)"""'
    for match in re.finditer(pattern, text, re.DOTALL):
        name = match.group(1)
        docstring = match.group(2)
        param_match = re.search(r'~?(\d+(?:\.\d+)?[MBK])\s*param', docstring)
        if param_match:
            key = f"PRESET_{name.upper()}_PARAMS"
            metrics[key] = f"~{param_match.group(1)}"

    return metrics
