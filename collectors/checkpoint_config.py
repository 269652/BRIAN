"""Collect checkpoint storage config from ``brian.toml``.

Five values flow into the README template's "Checkpoints" section:

* ``CHECKPOINT_PUSH_BACKEND`` — ``defaults.push_backend`` (``hf``, etc).
* ``HF_REPO_ID``              — ``defaults.hf_repo_id`` (owner/repo).
* ``HF_SAVE_EVERY``           — ``defaults.save_every`` (local-disk cadence).
* ``HF_PUSH_EVERY``           — ``defaults.push_every`` (Hub upload cadence).
* ``CHECKPOINT_LOG_EVERY``    — ``defaults.log_every`` (stdout cadence).

The TOML is the single source of truth for these cadences so the
README never drifts from what training actually does.
"""
from pathlib import Path

METRICS = [
    "CHECKPOINT_PUSH_BACKEND",
    "HF_REPO_ID",
    "HF_SAVE_EVERY",
    "HF_PUSH_EVERY",
    "CHECKPOINT_LOG_EVERY",
]


def collect(root: Path) -> dict[str, str]:
    """Read push backend, repo, and cadence values from brian.toml."""
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
        if "save_every" in defaults:
            metrics["HF_SAVE_EVERY"] = str(defaults["save_every"])
        if "push_every" in defaults:
            metrics["HF_PUSH_EVERY"] = str(defaults["push_every"])
        if "log_every" in defaults:
            metrics["CHECKPOINT_LOG_EVERY"] = str(defaults["log_every"])

    except Exception:
        pass

    return metrics
