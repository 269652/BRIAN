# -*- coding: utf-8 -*-
"""Vast.ai connector for ALWAYS-ON MIND deploys (``brian deploy-mind``).

Sibling of :mod:`vast_discover` for §14.5 mind boxes instead of
discovery jobs. Differences that justify a separate connector:

* **No self-destroy, ever.** A mind box is always-on by design — it
  bills until you run ``brian destroy <id>``. The onstart instead wraps
  the server in a crash-restart loop so a transient OOM/exception
  never silently kills the mind.
* **No artifact pusher.** The mind's episodic memory is in-process
  state (persistence is a future Layer-A rung); there is nothing to
  sync to git on a cadence.
* **Cheap-GPU default.** Expert inference (≤0.5B frozen models) needs
  ~4 GiB VRAM, not an A100 — default offer filter targets
  RTX 3060-class cards (~$0.05-0.10/hr).

Reuses ``scripts/vast_discover.sh`` verbatim for offer search +
instance creation — that script is fully env-driven
(``ONSTART_FILE`` / ``GPU_QUERY`` / ``VAST_LABEL``).

The server binds 127.0.0.1 on the box; reach it from a laptop with
``brian chat connect --tunnel <INSTANCE_ID>`` (SSH local forward — no
public port is ever opened).

Per CLAUDE.md §1, deploy scripts are verified by deploying, not
unit-tested; the CLI parse surface is pinned in
tests/cognition/test_mind_server.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# 2026-08-11: switched from the cheap-card default (gpu_ram>=8, <$0.15/hr)
# to A100 per explicit user preference — inference deploys should match
# the same card class as training, not the "expert inference is cheap"
# assumption the connector shipped with. Still overridable via
# --gpu-query / MindDeployConfig.gpu_query for anyone who wants the
# original cheap-card behaviour back.
_DEFAULT_GPU_QUERY = ("gpu_name=A100_SXM4 num_gpus=1 rentable=true "
                      "verified=true reliability>0.95")


@dataclass
class MindDeployConfig:
    expert: str = "smollm2_360m"
    mind: bool = True
    port: int = 7861
    branch: Optional[str] = None
    label: str = "neuroslm-mind"
    gpu_query: str = _DEFAULT_GPU_QUERY


_ONSTART_TEMPLATE = """\
set -e
export DEBIAN_FRONTEND=noninteractive
date -u +"vast_mind boot @ %Y-%m-%dT%H:%M:%SZ"

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

mkdir -p logs/mind
echo "── starting the always-on mind (crash-restart loop) ──"
# NO self-destroy: a mind box is always-on by design and bills until
# `brian destroy <id>`. The loop survives transient crashes; 10s
# backoff so a hard boot failure can't hot-spin the billing meter
# against a broken install.
while true; do
    date -u +"[mind-loop] (re)start @ %Y-%m-%dT%H:%M:%SZ"
    python3 -m neuroslm.cli chat --expert '__EXPERT__' __MIND_FLAG__ \\
        --serve --port __PORT__ --no-color \\
        2>&1 | tee -a "logs/mind/$(date -u +%Y%m%d)_mind.log"
    echo "[mind-loop] server exited rc=$? — restarting in 10s"
    sleep 10
done
"""


def build_mind_onstart(env: dict) -> str:
    """Container-side onstart script. Same locally-expanded-placeholder
    pattern as :func:`vast_discover.build_discover_onstart`."""
    result = _ONSTART_TEMPLATE
    repo_slug = env.get("REPO_URL") or "269652/BRIAN"
    repo_slug = (repo_slug.replace("https://github.com/", "")
                 .replace(".git", ""))
    replacements = {
        "__GH_TOKEN__": str(env.get("GH_TOKEN", "")),
        "__HF_TOKEN__": str(env.get("HF_TOKEN", "")),
        "__BRANCH__": str(env.get("BRANCH", "master")),
        "__REPO_SLUG__": repo_slug,
        "__EXPERT__": str(env.get("EXPERT", "smollm2_360m")),
        "__MIND_FLAG__": "--mind" if env.get("MIND") else "",
        "__PORT__": str(env.get("PORT", 7861)),
    }
    for key, val in replacements.items():
        result = result.replace(key, val)
    return result


class VastMindConnector:
    """Launch an always-on mind box via ``scripts/vast_discover.sh``."""

    @staticmethod
    def _find_bash() -> str:
        from neuroslm.connectors.vast import VastConnector
        return VastConnector._find_bash()

    def launch(self, config: MindDeployConfig) -> int:
        from neuroslm.connectors.vast_discover import _current_branch

        # ── §8.1: walk .env into os.environ BEFORE reading tokens ──
        # 2026-08-11 incident: `brian deploy-mind` run from a shell with
        # no GH_TOKEN exported produced a box whose onstart printed
        # "GH_TOKEN token not set" and never started the server — the
        # token was sitting in .env the whole time. Same walker
        # lightning.py already uses for this exact reason (bootstrap
        # secrets are process-env-only for the SDK; a .env-only value
        # is otherwise invisible). Never crashes the deploy chain — a
        # genuinely-missing token still reaches the box (fails there,
        # loudly) after we've warned locally, below.
        try:
            from neuroslm.utils.secrets import bootstrap_secrets
            bootstrap_secrets(
                ["GH_TOKEN", "HF_TOKEN", "VAST_API_KEY"],
                aliases={"GH_TOKEN": ("GITHUB_TOKEN", "GITHUB_PAT")},
                verbose=False)
        except Exception as exc:
            print(f"[deploy-mind] (note) secrets bootstrap skipped: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)

        if not os.environ.get("GH_TOKEN"):
            print("[deploy-mind] ⚠ GH_TOKEN missing (checked process env "
                  "and .env) — the box will fail to clone the repo. "
                  "Set it before deploying: export GH_TOKEN=ghp_... or "
                  "add it to .env.", file=sys.stderr)

        branch = config.branch or _current_branch()
        onstart_content = build_mind_onstart({
            "GH_TOKEN": os.environ.get("GH_TOKEN", ""),
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "BRANCH": branch,
            "REPO_URL": os.environ.get("REPO_URL", ""),
            "EXPERT": config.expert,
            "MIND": config.mind,
            "PORT": config.port,
        })

        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False, encoding="utf-8",
            newline="\n")
        try:
            tf.write(onstart_content)
            tf.flush()
            tf.close()

            env = os.environ.copy()
            env["ONSTART_FILE"] = tf.name
            env["VAST_LABEL"] = config.label
            env["GPU_QUERY"] = config.gpu_query

            bash = self._find_bash()
            script = str(REPO_ROOT / "scripts" / "vast_discover.sh")
            print(f"$ {bash} {script}")
            return subprocess.call(
                [bash, script],
                cwd=str(REPO_ROOT),
                env=env,
                stdin=subprocess.DEVNULL,
            )
        finally:
            try:
                os.unlink(tf.name)
            except OSError:
                pass
