# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["mechanics"])

_REPO = Path(__file__).parent.parent.parent.parent


def _load_mechanic_specs() -> list[dict]:
    """Parse all .neuro files in mechanics/, structures/, dynamics/ directories."""
    sys.path.insert(0, str(_REPO))
    try:
        from neuroslm.dsl.mechanic_parser import parse_mechanic_file, MechanicSpec
    except ImportError:
        return []

    dirs = [
        (_REPO / "mechanics", "mechanic"),
        (_REPO / "structures", "structure"),
        (_REPO / "dynamics", "dynamic"),
    ]
    specs: list[dict] = []
    for dirpath, default_type in dirs:
        if not dirpath.exists():
            continue
        for neuro_file in sorted(dirpath.glob("*.neuro")):
            try:
                src = neuro_file.read_text(encoding="utf-8")
                parsed = parse_mechanic_file(src)
                for spec in parsed:
                    specs.append({
                        "name": spec.name,
                        "category": spec.category or default_type,
                        "node_type": default_type,
                        "summary": spec.summary,
                        "equation": spec.equation,
                        "impl": spec.impl,
                        "zero_init": spec.zero_init,
                        "params": {
                            k: {
                                "default": v.default,
                                "type": v.type_hint,
                                "min": v.min_val,
                                "max": v.max_val,
                                "doc": v.doc,
                            }
                            for k, v in spec.params.items()
                        },
                        "when_to_use": spec.when_to_use,
                        "not_for": spec.not_for,
                        "properties": spec.properties,
                        "exported": spec.exported,
                        "file": neuro_file.name,
                        "dir": dirpath.name,
                    })
            except Exception:
                # Skip unparseable files
                pass
    return specs


_CACHE: list[dict] | None = None


def _get_specs() -> list[dict]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_mechanic_specs()
    return _CACHE


@router.get("", response_model=list[dict])
def list_mechanics() -> list[dict]:
    return _get_specs()


@router.get("/{name}")
def get_mechanic(name: str) -> dict:
    specs = _get_specs()
    match = next((s for s in specs if s["name"] == name), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Mechanic '{name}' not found")
    return match
