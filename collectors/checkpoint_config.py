"""Collect checkpoint storage config from brian.toml."""
from pathlib import Path

METRICS = [
    "CHECKPOINT_PUSH_BACKEND",
    "HF_REPO_ID",
]


def collect(root: Path) -> dict[str, str]:
    """Read push backend + repo from brian.toml."""
    metrics: dict[str, str] = {}

    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        toml_path = root / "brian.toml"
        if not toml_path.exists():
            return metrics

        with open(toml_path, "rb") as f:
            cfg = tomllib.load(f)

        defaults = cfg.get("defaults", {})
        if "push_backend" in defaults:
            metrics["CHECKPOINT_PUSH_BACKEND"] = str(defaults["push_backend"])
        if "hf_repo_id" in defaults:
            metrics["HF_REPO_ID"] = str(defaults["hf_repo_id"])

    except Exception:
        pass

    return metrics
