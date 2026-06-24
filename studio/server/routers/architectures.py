# -*- coding: utf-8 -*-
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from studio.server.neuro_parser import parse_arch

router = APIRouter(tags=["architectures"])

_REPO = Path(__file__).parent.parent.parent.parent
_ARCH_ROOT = _REPO / "architectures"


def _list_arch_dirs() -> list[Path]:
    if not _ARCH_ROOT.exists():
        return []
    return sorted(
        p for p in _ARCH_ROOT.iterdir()
        if p.is_dir() and (p / "arch.neuro").exists()
    )


class ArchSummary(BaseModel):
    name: str
    has_config: bool
    has_fitness: bool
    kind: str


class ArchDetail(BaseModel):
    name: str
    source: str
    nodes: list[dict]
    edges: list[dict]


class SaveRequest(BaseModel):
    source: str


@router.get("", response_model=list[ArchSummary])
def list_architectures() -> list[ArchSummary]:
    out = []
    for p in _list_arch_dirs():
        src = (p / "arch.neuro").read_text(encoding="utf-8")
        # Quick kind extraction
        import re
        m = re.search(r"kind\s*:\s*(\w+)", src)
        kind = m.group(1) if m else "custom"
        out.append(ArchSummary(
            name=p.name,
            has_config=(p / "config.neuro").exists(),
            has_fitness=(p / "fitness.neuro").exists(),
            kind=kind,
        ))
    return out


@router.get("/{name}", response_model=ArchDetail)
def get_architecture(name: str) -> ArchDetail:
    arch_dir = _ARCH_ROOT / name
    neuro_file = arch_dir / "arch.neuro"
    if not neuro_file.exists():
        raise HTTPException(status_code=404, detail=f"Architecture '{name}' not found")
    source = neuro_file.read_text(encoding="utf-8")
    parsed = parse_arch(source, name)
    return ArchDetail(**parsed)


@router.put("/{name}")
def save_architecture(name: str, body: SaveRequest) -> dict:
    arch_dir = _ARCH_ROOT / name
    arch_dir.mkdir(parents=True, exist_ok=True)
    (arch_dir / "arch.neuro").write_text(body.source, encoding="utf-8")
    return {"saved": True, "name": name}


@router.post("/{name}/compile")
def compile_architecture(name: str) -> dict:
    """Compile arch.neuro → DNA via brian dna compile."""
    result = subprocess.run(
        [sys.executable, "-m", "neuroslm.cli", "dna", "compile", f"architectures/{name}"],
        capture_output=True, text=True, cwd=_REPO
    )
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
