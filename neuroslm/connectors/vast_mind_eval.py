# -*- coding: utf-8 -*-
"""Vast.ai connector for the consciousness-indicator battery
(``brian deploy-mind-eval`` → ``brian eval mind`` on the box, §14.11).

A batch job, not an always-on mind: the box clones, bootstraps, runs
the battery on CUDA, pushes the result JSON + log back to git, and
self-destroys (unless ``--no-auto-destroy``). Offer search and
instance creation reuse ``scripts/vast_discover.sh`` verbatim, and the
self-destroy/leave-running blocks are the ones
:mod:`vast_discover` already ships.

The human-confirmation gate lives in the CLI layer
(``cmd_deploy_mind_eval``); this module never launches on its own.
Per CLAUDE.md §1, deploy scripts are verified by deploying; the parse
surface and onstart content are pinned in
tests/cognition/test_indicator_eval.py.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from neuroslm.connectors.vast_discover import (
    _LEAVE_RUNNING_BLOCK, _SELF_DESTROY_BLOCK, _current_branch)
from neuroslm.connectors.vast_mind import _DEFAULT_GPU_QUERY

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class MindEvalDeployConfig:
    expert: str = "smollm2_360m_instruct"
    eval_args: List[str] = field(default_factory=list)
    branch: Optional[str] = None
    label: str = "neuroslm-mind-eval"
    gpu_query: str = _DEFAULT_GPU_QUERY
    auto_destroy: bool = True
    push_interval: int = 120


_ONSTART_TEMPLATE = """\
set -e
export DEBIAN_FRONTEND=noninteractive
date -u +"vast_mind_eval boot @ %Y-%m-%dT%H:%M:%SZ"

(command -v git >/dev/null 2>&1 && command -v git-lfs >/dev/null 2>&1) \\
    || (apt-get update -y && apt-get install -y git git-lfs)
git lfs install --skip-smudge

export GH_TOKEN='__GH_TOKEN__' HF_TOKEN='__HF_TOKEN__'
mkdir -p /workspace && cd /workspace

echo "── cloning __BRANCH__ ──"
GIT_LFS_SKIP_SMUDGE=1 git clone --branch '__BRANCH__' --single-branch \\
    "https://x-access-token:${GH_TOKEN}@github.com/__REPO_SLUG__.git" brian
cd brian

echo "── bootstrap (pip deps only — no checkpoint pull needed) ──"
SKIP_LFS_RESUME=1 bash scripts/vast_bootstrap.sh

export BOOT_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p results/mind_eval logs/mind_eval
LOGFILE="logs/mind_eval/${BOOT_TIMESTAMP}.log"
touch "$LOGFILE"

echo "── starting log pusher (background, every __PUSH_INTERVAL__s) ──"
# A box that dies mid-run must not take the only copy of the log
# (mind-wandering traces stream into it tick by tick).
(
  while true; do
    sleep '__PUSH_INTERVAL__'
    TS="$(date -u +%H:%M:%SZ)" python3 -c "
import os
from neuroslm.genetic.modulation_pusher import push_artifacts
r = push_artifacts('.', ['results/mind_eval', '$LOGFILE'],
                   message='mind-eval: auto-push @ ' + os.environ.get('TS', ''))
print('[pusher]', r)
"
  done
) > /workspace/mind_eval_pusher.log 2>&1 &
PUSHER_PID=$!

echo "── running the indicator battery ──"
set +e
python3 -m neuroslm.cli eval mind --expert '__EXPERT__' --device cuda \\
    __EVAL_ARGS__ --out-dir results/mind_eval 2>&1 | tee "$LOGFILE"
EVAL_RC=${PIPESTATUS[0]}
set -e
echo "── eval exited rc=$EVAL_RC ──"
kill $PUSHER_PID 2>/dev/null || true
sleep 2

TS="$(date -u +%H:%M:%SZ)" python3 -c "
import os
from neuroslm.genetic.modulation_pusher import push_artifacts
r = push_artifacts('.', ['results/mind_eval', '$LOGFILE'],
                   message='mind-eval(__EXPERT__): results @ ' + os.environ.get('TS', ''))
print('[pusher] final:', r)
"

__SELF_DESTROY_BLOCK__
"""


def expert_slug(expert: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", expert).strip("-") or "expert"


def build_mind_eval_onstart(env: dict) -> str:
    """Container-side onstart script — same locally-expanded
    ``__PLACEHOLDER__`` pattern as the sibling connectors."""
    repo_url = env.get("REPO_URL") or "https://github.com/269652/BRIAN.git"
    repo_slug = (repo_url.removeprefix("https://github.com/")
                 .removesuffix(".git"))
    auto_destroy = str(env.get("AUTO_DESTROY", "1")) != "0"
    expert = str(env.get("EXPERT", "smollm2_360m"))
    result = _ONSTART_TEMPLATE.replace(
        "__SELF_DESTROY_BLOCK__",
        _SELF_DESTROY_BLOCK if auto_destroy else _LEAVE_RUNNING_BLOCK)
    for key, val in {
        "__GH_TOKEN__": str(env.get("GH_TOKEN", "")),
        "__HF_TOKEN__": str(env.get("HF_TOKEN", "")),
        "__BRANCH__": str(env.get("BRANCH", "master")),
        "__REPO_SLUG__": repo_slug,
        "__EXPERT__": expert,
        "__EVAL_ARGS__": str(env.get("EVAL_ARGS", "")),
        "__LABEL__": str(env.get("LABEL", "neuroslm-mind-eval")),
        "__PUSH_INTERVAL__": str(env.get("PUSH_INTERVAL", 120)),
    }.items():
        result = result.replace(key, val)
    return result


class VastMindEvalConnector:
    """Launch the indicator battery via ``scripts/vast_discover.sh``."""

    @staticmethod
    def _find_bash() -> str:
        from neuroslm.connectors.vast import VastConnector
        return VastConnector._find_bash()

    def launch(self, config: MindEvalDeployConfig) -> int:
        try:
            from neuroslm.utils.secrets import bootstrap_secrets
            bootstrap_secrets(
                ["GH_TOKEN", "HF_TOKEN", "VAST_API_KEY"],
                aliases={"GH_TOKEN": ("GITHUB_TOKEN", "GITHUB_PAT")},
                verbose=False)
        except Exception as exc:
            print(f"[deploy-mind-eval] (note) secrets bootstrap skipped: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
        if not os.environ.get("GH_TOKEN"):
            print("[deploy-mind-eval] ⚠ GH_TOKEN missing (checked process "
                  "env and .env) — the box can neither clone nor push "
                  "results.", file=sys.stderr)

        onstart = build_mind_eval_onstart({
            "GH_TOKEN": os.environ.get("GH_TOKEN", ""),
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "BRANCH": config.branch or _current_branch(),
            "REPO_URL": os.environ.get("REPO_URL", ""),
            "EXPERT": config.expert,
            "EVAL_ARGS": " ".join(shlex.quote(a) for a in config.eval_args),
            "LABEL": config.label,
            "AUTO_DESTROY": "1" if config.auto_destroy else "0",
            "PUSH_INTERVAL": config.push_interval,
        })
        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False, encoding="utf-8",
            newline="\n")
        try:
            tf.write(onstart)
            tf.flush()
            tf.close()
            env = os.environ.copy()
            env["ONSTART_FILE"] = tf.name
            env["VAST_LABEL"] = config.label
            env["GPU_QUERY"] = config.gpu_query
            bash = self._find_bash()
            script = str(REPO_ROOT / "scripts" / "vast_discover.sh")
            print(f"$ {bash} {script}")
            return subprocess.call([bash, script], cwd=str(REPO_ROOT),
                                   env=env, stdin=subprocess.DEVNULL)
        finally:
            try:
                os.unlink(tf.name)
            except OSError:
                pass
