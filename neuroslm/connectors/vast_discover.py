# -*- coding: utf-8 -*-
"""Vast.ai connector for DISCOVERY runs (``brian discover <mode> ...``).

Distinct from :mod:`neuroslm.connectors.vast` (full training): discover jobs
are shorter, mode-driven searches (update rules, trunk modulations, expert-
cortex modulations) rather than an arch/scale/steps training loop. No
checkpoint/OOD/HF-push machinery is needed; instead a MODE-AGNOSTIC
background timer pushes every artifact a discover run can produce (its log,
``modulations/``, the search ledger, ``heatmaps/``) on a short interval WHILE
the run is in progress — a discover mode's own internal push points (e.g.
``experts`` pushes every round) are not enough alone, since an interrupted or
destroyed instance would otherwise lose everything since the last internal
push.

Only ``experts``/``trunk``/``explore`` are deployable here — the other
discover modes (``optimizer``/``flow``/``qd``/``simplify``) are cheap,
fast, synthetic-benchmark searches that already finish on the free
interactive Colab GPU in seconds-to-minutes; renting a paid instance for
them would spend money for no benefit.

Modeled directly on :class:`neuroslm.connectors.vast.VastConnector`: Python
builds the onstart script content and writes it to a temp file (avoids the
bash heredoc pipe-buffer deadlock on Windows Git Bash — see that module's
docstring), ``scripts/vast_discover.sh`` reads it via ``ONSTART_FILE`` and
does the vast.ai offer search + instance create. Kept as a SEPARATE
script/module rather than extending ``vast_train.sh``/``VastConnector`` —
the same reasoning ``vast_train.sh`` itself documents for not modifying
``vast_deploy.sh``: a discover job has a genuinely different shape (mode +
search params, not arch/scale/steps), and a shared, delicate, tested file is
not the place to bolt on a second job type.

The human-confirmation gate (``neuroslm.cli._require_human_confirmation``)
lives in the CLI layer (``cmd_deploy_discover``), same as ``brian deploy`` —
this module never launches anything on its own.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

DEPLOYABLE_MODES = ("experts", "trunk", "explore")


@dataclass
class DiscoverDeployConfig:
    mode: str
    discover_args: List[str] = field(default_factory=list)
    branch: Optional[str] = None
    label: str = "neuroslm-discover"
    push_interval: int = 90
    gpu_query: str = ""

    def __post_init__(self) -> None:
        if self.mode not in DEPLOYABLE_MODES:
            raise ValueError(
                f"mode {self.mode!r} not deployable to vast.ai — choose one "
                f"of {DEPLOYABLE_MODES} (the other discover modes finish in "
                f"seconds/minutes on the free local Colab GPU; no rental "
                f"needed)")


_ONSTART_TEMPLATE = """\
set -e
export DEBIAN_FRONTEND=noninteractive
date -u +"vast_discover.sh boot @ %Y-%m-%dT%H:%M:%SZ"

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

echo "── starting artifact pusher (background, every __PUSH_INTERVAL__s) ──"
# Mode-agnostic safety net: independent of whatever a discover mode pushes
# internally (e.g. `experts` pushes every round already), so an interrupted
# instance never loses more than __PUSH_INTERVAL__ seconds of progress.
# push_artifacts() is best-effort and never raises.
export BOOT_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p logs/discover
LOGFILE="logs/discover/${BOOT_TIMESTAMP}___MODE__.log"
(
  while true; do
    sleep '__PUSH_INTERVAL__'
    TS="$(date -u +%H:%M:%SZ)" python3 -c "
import os
from neuroslm.genetic.modulation_pusher import push_artifacts
ts = os.environ.get('TS', '')
r = push_artifacts('.', ['$LOGFILE', 'modulations', '.neuro/search_ledger.json', 'heatmaps'],
                   message=f'discover(__MODE__): auto-push @ {ts}')
print('[pusher]', r)
"
  done
) > /workspace/discover_pusher.log 2>&1 &
PUSHER_PID=$!
echo "    pusher pid=$PUSHER_PID  logfile=$LOGFILE"

echo "── running: brian discover __MODE__ __DISCOVER_ARGS__ --push ──"
set +e
python3 -m neuroslm.cli discover __MODE__ __DISCOVER_ARGS__ --push 2>&1 | tee "$LOGFILE"
DISCOVER_RC=${PIPESTATUS[0]}
set -e
echo "── discover exited rc=$DISCOVER_RC ──"

echo "── stopping pusher, final push ──"
kill $PUSHER_PID 2>/dev/null || true
sleep 2
TS="$(date -u +%H:%M:%SZ)" python3 -c "
import os
from neuroslm.genetic.modulation_pusher import push_artifacts
ts = os.environ.get('TS', '')
r = push_artifacts('.', ['$LOGFILE', 'modulations', '.neuro/search_ledger.json', 'heatmaps'],
                   message=f'discover(__MODE__): final push @ {ts}')
print('[pusher] final:', r)
"

# ── Self-destroy the vast instance ────────────────────────────────
# Without this the container stays "running" after the discover run
# completes and bills you indefinitely (see neuroslm/connectors/vast.py's
# training onstart for the incident this pattern was written to prevent).
echo "── self-destroying instance ──"
if ! command -v vastai >/dev/null 2>&1; then
    pip install -q vastai 2>&1 | tail -3 || true
fi
if [ -n "${VAST_API_KEY:-}" ] && command -v vastai >/dev/null 2>&1; then
    vastai set api-key "$VAST_API_KEY" >/dev/null 2>&1 || true
    SELF_ID="${INSTANCE_ID:-${VAST_CONTAINER_ID:-}}"
    if [ -z "$SELF_ID" ]; then
        SELF_ID="$(vastai show instances --raw 2>/dev/null \\
            | python3 -c "import sys, json
data = json.load(sys.stdin)
for i in (data or []):
    if i.get('label') == '__LABEL__':
        print(i.get('id', '')); break" 2>/dev/null)"
    fi
    if [ -n "$SELF_ID" ]; then
        echo "[onstart] vastai destroy instance $SELF_ID"
        yes y | vastai destroy instance "$SELF_ID" 2>&1 || echo "[onstart] destroy failed"
        sleep 30
    else
        echo "[onstart] could not determine vast instance id; not destroying"
    fi
else
    echo "[onstart] VAST_API_KEY not in env or vastai CLI missing -- cannot self-destroy"
fi

echo "── discover run exited; FAILED to self-destroy. Run: vastai destroy instance <contract_id> ──"
"""


def build_discover_onstart(env: dict) -> str:
    """Return the container-side onstart bash script as a string.

    Mirrors ``VastConnector._build_onstart`` — locally-expanded Python
    values are substituted via ``__PLACEHOLDER__`` markers; bash variables
    that must appear literally in the container script (``$GH_TOKEN``,
    ``$LOGFILE``, ``$(date …)``, etc.) are left as plain characters.
    """
    repo_url = env.get("REPO_URL") or "https://github.com/269652/BRIAN.git"
    repo_slug = repo_url.removeprefix("https://github.com/").removesuffix(".git")
    result = _ONSTART_TEMPLATE
    result = result.replace("__GH_TOKEN__", env.get("GH_TOKEN", ""))
    result = result.replace("__HF_TOKEN__", env.get("HF_TOKEN", ""))
    result = result.replace("__BRANCH__", env.get("BRANCH", "master"))
    result = result.replace("__REPO_SLUG__", repo_slug)
    result = result.replace("__MODE__", env.get("MODE", ""))
    result = result.replace("__DISCOVER_ARGS__", env.get("DISCOVER_ARGS", ""))
    result = result.replace("__PUSH_INTERVAL__", str(env.get("PUSH_INTERVAL", 90)))
    result = result.replace("__LABEL__", env.get("LABEL", "neuroslm-discover"))
    return result


class VastDiscoverConnector:
    """Launch a discovery run on vast.ai via ``scripts/vast_discover.sh``."""

    @staticmethod
    def _find_bash() -> str:
        from neuroslm.connectors.vast import VastConnector
        return VastConnector._find_bash()

    def launch(self, config: DiscoverDeployConfig) -> int:
        branch = config.branch or _current_branch()
        discover_args_str = " ".join(shlex.quote(a) for a in config.discover_args)
        onstart_env = {
            "GH_TOKEN": os.environ.get("GH_TOKEN", ""),
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "BRANCH": branch,
            "REPO_URL": os.environ.get("REPO_URL", ""),
            "MODE": config.mode,
            "DISCOVER_ARGS": discover_args_str,
            "PUSH_INTERVAL": config.push_interval,
            "LABEL": config.label,
        }
        onstart_content = build_discover_onstart(onstart_env)

        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False, encoding="utf-8", newline="\n")
        try:
            tf.write(onstart_content)
            tf.flush()
            tf.close()

            env = os.environ.copy()
            env["ONSTART_FILE"] = tf.name
            env["VAST_LABEL"] = config.label
            if config.gpu_query:
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


def _current_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
        b = out.stdout.strip()
        return b if b and b != "HEAD" else "master"
    except Exception:
        return "master"
