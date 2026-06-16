"""
collectors/ — live metric collection scripts for README rendering.

Each .py file in this folder:
  1. Defines METRICS: list[str] — the metric names it provides
  2. Defines collect(root: Path) -> dict[str, str] — returns those metrics

The runner (neuroslm/metrics_collector.py) discovers all scripts, calls
collect() on each, and merges results. The TOML file (docs/readme_metrics.toml)
provides ONLY values that cannot be auto-collected (vast IDs, GPU costs, etc.)
and acts as a final override layer.

To add a new metric:
  1. Create a new .py file in this folder
  2. Define METRICS = ["MY_METRIC_NAME"]
  3. Define collect(root: Path) -> dict[str, str]
  4. Use ${MY_METRIC_NAME} in README.template.md
  5. Done — next `brian update-readme` picks it up automatically

Run all collectors standalone:
  py -3 -m neuroslm.metrics_collector
"""
