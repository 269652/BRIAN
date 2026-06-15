"""Metric collection for README.template rendering.

Public API:
  parse_layer_b_table(text)              → dict with baseline + best rows
  count_tests_from_durations_cache(path) → int
  build_metrics(repo_root)               → dict[str, str] of all placeholders
  render_template(template, metrics)     → str with ${PLACEHOLDER}s filled
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from string import Template
from typing import Any


# ── Layer B table parser ──────────────────────────────────────────────────────

_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_BACKTICK_RE = re.compile(r"`([^`]*)`")


def _strip_md(s: str) -> str:
    s = _BOLD_RE.sub(r"\1", s)
    s = _BACKTICK_RE.sub(r"\1", s)
    return s.strip()


def _try_float(s: str) -> float | None:
    cleaned = _strip_md(s).split()[0].rstrip("%")
    try:
        return float(cleaned)
    except (ValueError, IndexError):
        return None


def _row_id(raw: str) -> str | None:
    m = re.search(r"\bB(\d+(?:\.\w+)?)\b", raw)
    return f"B{m.group(1)}" if m else None


def parse_layer_b_table(text: str) -> dict[str, Any]:
    """Parse the Layer B reference table from findings.md text.

    Returns:
        {
            "baseline": {"row": "B0", "gap_ratio": 6.12, ...},
            "best":     {"row": "B4", "gap_ratio": 2.87, "improvement_pct": 53.1, ...},
            "all_rows": [...],
        }

    Raises ValueError if no baseline (B0) is found.
    """
    rows: list[dict] = []
    in_table = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if in_table:
                break
            continue

        cells = [c.strip() for c in stripped.split("|")[1:-1]]

        if "gap_ratio" in " ".join(cells) and "Row" in cells[0]:
            in_table = True
            continue

        if not in_table:
            continue

        # Separator rows: only dashes and spaces
        if all(re.fullmatch(r"[-: ]+", c) for c in cells):
            continue

        if len(cells) < 7:
            continue

        row_raw = cells[0]
        row_id = _row_id(row_raw)
        if row_id is None:
            continue

        train_ppl = _try_float(cells[4])
        ood_ppl = _try_float(cells[5])
        gap_ratio = _try_float(cells[6])
        verdict_raw = cells[7] if len(cells) > 7 else ""

        if gap_ratio is None:
            continue

        is_artifact = (
            "ARTIFACT" in verdict_raw.upper()
            or "load-bug" in row_raw.lower()
            or "buggy" in row_raw.lower()
        )
        is_baseline = row_id == "B0"

        rows.append(
            {
                "row": row_id,
                "train_ppl": train_ppl,
                "ood_ppl": ood_ppl,
                "gap_ratio": gap_ratio,
                "is_artifact": is_artifact,
                "is_baseline": is_baseline,
            }
        )

    baselines = [r for r in rows if r["is_baseline"]]
    if not baselines:
        raise ValueError(
            "No baseline row (B0) found in Layer B table. "
            "Make sure the text contains the Layer B reference table."
        )
    baseline = baselines[0]

    candidates = [r for r in rows if not r["is_artifact"] and not r["is_baseline"]]
    if not candidates:
        raise ValueError("No non-artifact, non-baseline rows found in Layer B table.")

    best = dict(min(candidates, key=lambda r: r["gap_ratio"]))
    best["improvement_pct"] = (
        (baseline["gap_ratio"] - best["gap_ratio"]) / baseline["gap_ratio"] * 100
    )

    return {"baseline": baseline, "best": best, "all_rows": rows}


# ── Test count ────────────────────────────────────────────────────────────────

def count_tests_from_durations_cache(cache_path: Path) -> int:
    """Count test items from .neuro/test_durations.json. Returns 0 on missing/corrupt."""
    try:
        data = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        return len(data) if isinstance(data, dict) else 0
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return 0


# ── Test counter (scan fallback) ──────────────────────────────────────────────

_SLOW_TEST_DIRS = {"training", "archive"}


def count_tests_by_scanning(tests_dir: Path) -> int:
    """Count test functions by scanning tests/**/*.py; skip known-slow dirs."""
    tests_dir = Path(tests_dir)
    if not tests_dir.is_dir():
        return 0
    count = 0
    for path in tests_dir.rglob("*.py"):
        if any(part in _SLOW_TEST_DIRS for part in path.parts):
            continue
        try:
            count += len(re.findall(r"(?:^|\n)\s*def test_", path.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            pass
    return count


# ── Metrics builder ───────────────────────────────────────────────────────────

_FINDINGS_PATH = Path("docs/FINDINGS.md")
_DURATIONS_PATH = Path(".neuro/test_durations.json")


def build_metrics(repo_root: Path | str) -> dict[str, str]:
    """Collect all README placeholder values from repo artifacts.

    Reads docs/FINDINGS.md for Layer B metrics, .neuro/test_durations.json
    for test count. Returns a dict of str→str suitable for render_template().
    """
    root = Path(repo_root)

    # Layer B metrics
    findings_path = root / _FINDINGS_PATH
    try:
        findings_text = findings_path.read_text(encoding="utf-8")
        table_data = parse_layer_b_table(findings_text)
        best = table_data["best"]
        baseline = table_data["baseline"]
    except (FileNotFoundError, ValueError):
        best = {"row": "?", "gap_ratio": 0.0, "train_ppl": 0.0, "ood_ppl": 0.0, "improvement_pct": 0.0}
        baseline = {"gap_ratio": 0.0}

    # Test count — durations cache (populated by `brian test full`) preferred;
    # fall back to scanning tests/ so the badge is never "0 passing".
    cache_path = root / _DURATIONS_PATH
    test_count = count_tests_from_durations_cache(cache_path)
    if test_count == 0:
        test_count = count_tests_by_scanning(root / "tests")

    def _fmt(v: float | None) -> str:
        if v is None:
            return "?"
        return str(int(round(v))) if v == int(v) else f"{v:.2f}".rstrip("0").rstrip(".")

    return {
        "LAYER_A_TEST_COUNT": str(test_count),
        "LAYER_B_BEST_ROW": best["row"],
        "LAYER_B_BEST_GAP_RATIO": _fmt(best.get("gap_ratio")),
        "LAYER_B_BEST_TRAIN_PPL": _fmt(best.get("train_ppl")),
        "LAYER_B_BEST_OOD_PPL": _fmt(best.get("ood_ppl")),
        "LAYER_B_BASELINE_GAP_RATIO": _fmt(baseline.get("gap_ratio")),
        "LAYER_B_IMPROVEMENT_PCT": str(int(round(best.get("improvement_pct", 0)))),
    }


# ── Template renderer ─────────────────────────────────────────────────────────

def render_template(template: str, metrics: dict[str, str]) -> str:
    """Substitute ${PLACEHOLDER} markers using safe_substitute (unknowns left intact)."""
    return Template(template).safe_substitute(metrics)
