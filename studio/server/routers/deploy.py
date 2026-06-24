# -*- coding: utf-8 -*-
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["deploy"])

_REPO = Path(__file__).parent.parent.parent.parent


class DeployRequest(BaseModel):
    arch: str = ""
    steps: int = 10000
    label: str = ""
    platform: str = "vast"


class StatusResponse(BaseModel):
    instances: list[dict]


@router.post("")
def deploy(body: DeployRequest) -> dict:
    """Launch a training run via brian deploy. Requires explicit user intent."""
    cmd = [sys.executable, "-m", "neuroslm.cli", "deploy"]
    if body.arch:
        cmd.append(body.arch)
    cmd += ["--steps", str(body.steps)]
    if body.label:
        cmd += ["--label", body.label]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO, timeout=120)
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-2000:],
    }


@router.get("/status")
def get_status() -> dict:
    """List active vast.ai instances (read-only, no spend)."""
    result = subprocess.run(
        [sys.executable, "-m", "neuroslm.cli", "status"],
        capture_output=True, text=True, cwd=_REPO, timeout=30
    )
    return {
        "success": result.returncode == 0,
        "output": result.stdout,
        "stderr": result.stderr,
    }


@router.delete("/{instance_id}")
def destroy_instance(instance_id: str) -> dict:
    """Destroy a vast.ai instance by id."""
    result = subprocess.run(
        [sys.executable, "-m", "neuroslm.cli", "destroy", instance_id],
        capture_output=True, text=True, cwd=_REPO, timeout=60
    )
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
