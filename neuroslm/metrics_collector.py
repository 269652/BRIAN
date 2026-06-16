"""
Metrics collector — discovers and runs all scripts in collectors/.

Each script in collectors/ must define:
  METRICS: list[str]          — metric names it provides
  collect(root: Path) -> dict[str, str]  — returns those metrics

This module:
  1. Discovers all .py files in collectors/ (excluding __init__.py)
  2. Imports each and calls collect(root)
  3. Merges results (later scripts override earlier on conflict)
  4. Returns the combined dict

Usage:
  from neuroslm.metrics_collector import collect_all
  metrics = collect_all()                   # all collectors
  metrics = collect_all(skip=["test_counts"])  # skip slow ones
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    """Walk up to find repo root (contains brian.toml)."""
    p = Path(__file__).resolve().parent.parent
    while p != p.parent:
        if (p / "brian.toml").exists():
            return p
        p = p.parent
    raise FileNotFoundError("Cannot find repo root (no brian.toml)")


def _discover_collectors(root: Path) -> list[Path]:
    """Find all .py collector scripts."""
    collectors_dir = root / "collectors"
    if not collectors_dir.is_dir():
        return []
    return sorted(
        p for p in collectors_dir.glob("*.py")
        if p.name != "__init__.py" and not p.name.startswith("_")
    )


def _load_and_run(script_path: Path, root: Path) -> dict[str, str]:
    """Import a collector script and call its collect() function."""
    module_name = f"collectors.{script_path.stem}"

    # Ensure collectors/ is importable
    collectors_parent = str(script_path.parent.parent)
    if collectors_parent not in sys.path:
        sys.path.insert(0, collectors_parent)

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        return {}

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod

    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  [collectors] WARN: failed to load {script_path.name}: {e}")
        return {}

    collect_fn = getattr(mod, "collect", None)
    if collect_fn is None:
        print(f"  [collectors] WARN: {script_path.name} has no collect() function")
        return {}

    try:
        result = collect_fn(root)
        if not isinstance(result, dict):
            return {}
        return result
    except Exception as e:
        print(f"  [collectors] WARN: {script_path.name}.collect() failed: {e}")
        return {}


def collect_all(
    root: Optional[Path] = None,
    *,
    skip: Optional[list[str]] = None,
) -> dict[str, str]:
    """Run all collector scripts and return merged metrics.

    Parameters
    ----------
    root : Path, optional
        Repo root. Auto-detected if not given.
    skip : list[str], optional
        Script names (without .py) to skip. E.g. ["test_counts"] to
        avoid the slow pytest collection pass.

    Returns
    -------
    dict[str, str]
        All live-collected metrics keyed by UPPER_SNAKE name.
    """
    root = root or _repo_root()
    skip_set = set(skip or [])
    merged: dict[str, str] = {}

    for script in _discover_collectors(root):
        if script.stem in skip_set:
            continue
        result = _load_and_run(script, root)
        merged.update(result)

    return merged


if __name__ == "__main__":
    """CLI: dump all collected metrics for debugging."""
    import argparse
    parser = argparse.ArgumentParser(description="Collect README metrics")
    parser.add_argument("--skip", nargs="*", default=[], help="Collectors to skip")
    args = parser.parse_args()

    metrics = collect_all(skip=args.skip)
    print(f"# Collected {len(metrics)} metrics from collectors/\n")
    for k in sorted(metrics.keys()):
        print(f'{k} = "{metrics[k]}"')
