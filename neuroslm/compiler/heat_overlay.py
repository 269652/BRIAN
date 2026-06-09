# -*- coding: utf-8 -*-
"""L6 — NFG heat overlay.

Reads a TrainingHeatmap-shaped artifact (dict, ``TrainingHeatmap``
instance, or on-disk JSON path) and converts it to per-element fill
colors that the NFG renderer can drop into its DOT output.

Color ramp (in HSV, mapped to hex):
  - heat=0.0 -> a near-white off-white (preserves the diagram structure)
  - heat=1.0 -> a saturated brick-red (#cc2222 family)
  - linear interpolation between the two endpoints

Public surface:
  - :func:`normalize_heat`     — dict[str, float] -> dict normalised to [0,1]
  - :func:`heat_to_fillcolor`  — float in [0,1] -> hex "#rrggbb"
  - :func:`load_heat_source`   — accepts dict | TrainingHeatmap | path,
                                 returns a normalized dict[str, float]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union


__all__ = [
    "normalize_heat",
    "heat_to_fillcolor",
    "load_heat_source",
]


# ── normalisation ─────────────────────────────────────────────────


def normalize_heat(raw: Dict[str, float]) -> Dict[str, float]:
    """Scale a raw heat dict to [0, 1] by dividing by ``max(values)``.

    If the dict is empty, returns an empty dict. If the maximum value is
    already ≤ 1.0 the input is assumed to be pre-normalised and returned
    unchanged. If the maximum is 0 (all-cold), returns an all-zero dict.
    """
    if not raw:
        return {}
    vmax = max(raw.values())
    if vmax <= 0.0:
        return {k: 0.0 for k in raw}
    if vmax <= 1.0:
        return dict(raw)
    return {k: v / vmax for k, v in raw.items()}


# ── colormap ──────────────────────────────────────────────────────


def heat_to_fillcolor(h: float) -> str:
    """Map a heat value in ``[0, 1]`` to an ``#rrggbb`` fill color.

    The ramp deliberately stays in the warm half of the spectrum so
    the overlay reads as "thermal" rather than as a generic palette:
      heat=0   -> ``#f7f7f7`` (near white)
      heat=0.5 -> ``#f0a060`` (orange)
      heat=1.0 -> ``#cc2222`` (saturated brick red)
    """
    h = max(0.0, min(1.0, float(h)))
    # Two-stop linear gradient (white -> red).
    # Cold endpoint:  (247, 247, 247)
    # Hot  endpoint:  (204,  34,  34)
    r0, g0, b0 = 247, 247, 247
    r1, g1, b1 = 204,  34,  34
    r = int(round(r0 + (r1 - r0) * h))
    g = int(round(g0 + (g1 - g0) * h))
    b = int(round(b0 + (b1 - b0) * h))
    return f"#{r:02x}{g:02x}{b:02x}"


# ── source-loading dispatch ───────────────────────────────────────


def load_heat_source(source: Any) -> Dict[str, float]:
    """Normalise any of the accepted heat-source types to a flat dict.

    Accepts:
      - ``None`` -> ``{}``
      - ``dict[str, float]`` -> normalised via :func:`normalize_heat`
      - ``TrainingHeatmap`` -> ``.normalized()``
      - ``str`` | ``Path`` -> read as JSON; expects either a top-level
        ``{"entries": {id: {"heat": v, ...}, ...}}`` (the
        :class:`TrainingHeatmap` on-disk format) or a flat
        ``{id: v}`` dict.
    """
    if source is None:
        return {}

    # dict-of-floats shortcut
    if isinstance(source, dict):
        return normalize_heat(source)

    # File path
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "entries" in data:
            flat = {
                k: float(v.get("heat", 0.0)) if isinstance(v, dict) else float(v)
                for k, v in data["entries"].items()
            }
            return normalize_heat(flat)
        # Bare {id: value}
        return normalize_heat({k: float(v) for k, v in data.items()})

    # TrainingHeatmap (duck-typed: anything with `.normalized()`)
    if hasattr(source, "normalized"):
        return source.normalized()

    raise TypeError(
        f"unsupported heat source type: {type(source).__name__}; "
        f"pass a dict, TrainingHeatmap, or a JSON file path"
    )
