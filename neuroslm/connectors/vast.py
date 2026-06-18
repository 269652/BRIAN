# -*- coding: utf-8 -*-
"""Vast.ai training connector.

Translates a :class:`~neuroslm.connectors.base.DeployConfig` into a
``bash scripts/vast_train.sh`` call with the appropriate environment
variables.  Uses the same git-bash path resolution as ``cmd_deploy_brain``
in ``cli.py`` so the script runs correctly on both Windows and POSIX.

Environment variables forwarded to ``vast_train.sh``
────────────────────────────────────────────────────
  USE_DSL               always "1" — DSL/DNA training path
  STEPS                 config.steps
  BRANCH                config.branch          (when set)
  ARCH                  config.arch            (when set)
  SCALE                 config.scale           (when set)
  LABEL_SUFFIX          config.label           (when set)
  RESUME_FROM           config.resume_from     (when set)
  BRIAN_SOURCE_DNA      config.source_dna      (when set; telemetry only)
  OOD_EVERY             config.ood_every       (when > 0)
  LOG_EVERY             config.log_every       (when > 0)
  SAVE_EVERY            config.save_every      (when > 0)
  PUSH_EVERY            config.push_every      (when > 0)
  CHECKPOINT_PUSH_BACKEND config.push_backend  (when set)
  HF_REPO_ID            config.hf_repo_id      (when set)
  + any key/value pairs in config.extra_env
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from neuroslm.connectors.base import BaseConnector, DeployConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


class VastConnector(BaseConnector):
    """Launch training on vast.ai via ``scripts/vast_train.sh``."""

    @classmethod
    def platform_name(cls) -> str:
        return "vast"

    def launch(self, config: DeployConfig) -> int:
        env = self._build_env(config)
        bash = self._find_bash()
        script = str(REPO_ROOT / "scripts" / "vast_train.sh")
        print(f"$ {bash} {script}")
        return subprocess.call([bash, script], cwd=str(REPO_ROOT), env=env)

    # ── internal helpers ────────────────────────────────────────────

    def _build_env(self, config: DeployConfig) -> dict:
        env = os.environ.copy()
        env["USE_DSL"] = "1"
        env["STEPS"] = str(config.steps)
        env["PYTHONIOENCODING"] = "utf-8"

        if config.branch:
            env["BRANCH"] = config.branch
        if config.arch:
            env["ARCH"] = config.arch
        if config.scale:
            env["SCALE"] = config.scale
        if config.label:
            env["LABEL_SUFFIX"] = config.label
        if config.resume_from:
            env["RESUME_FROM"] = config.resume_from
        if config.source_dna:
            env["BRIAN_SOURCE_DNA"] = config.source_dna

        if config.ood_every > 0:
            env["OOD_EVERY"] = str(config.ood_every)
        if config.log_every > 0:
            env["LOG_EVERY"] = str(config.log_every)
        if config.save_every > 0:
            env["SAVE_EVERY"] = str(config.save_every)
        if config.push_every > 0:
            env["PUSH_EVERY"] = str(config.push_every)

        if config.push_backend:
            env["CHECKPOINT_PUSH_BACKEND"] = config.push_backend
        if config.hf_repo_id:
            env["HF_REPO_ID"] = config.hf_repo_id

        env.update(config.extra_env)
        return env

    @staticmethod
    def _find_bash() -> str:
        """Git-bash on Windows, /bin/bash elsewhere — mirrors cli._bash()."""
        if sys.platform == "win32":
            candidates = [
                r"C:\Program Files\Git\bin\bash.exe",
                r"C:\Program Files (x86)\Git\bin\bash.exe",
            ]
            for c in candidates:
                if os.path.isfile(c):
                    return c
        return "bash"
