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
  ONSTART_FILE          path to the temp file containing the onstart script
  + any key/value pairs in config.extra_env
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from neuroslm.connectors.base import (
    BaseConnector,
    DeployConfig,
    JobInfo,
    load_jobs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Container-side onstart script template.
#
# Variables locally expanded by Python before launch use __PLACEHOLDER__
# markers.  All bash variables that must appear literally in the container
# script ($BOOT_TIMESTAMP, ${GH_TOKEN}, $(date …), etc.) are written as plain
# characters — Python's str.replace() does not interpret $ or ${ }.
#
# Line-continuation backslashes for the container script are \\  (Python
# `\\` → one `\` in the string value → written as `\` to the file).
#
# This replaces the bash <<ONSTART heredoc that used to live in
# scripts/vast_train.sh.  On Windows Git Bash, bash's internal heredoc
# pipe writer fills the write-end synchronously before starting the reader
# command; the ~6 KB onstart script exceeds the pipe buffer (~4 KB), so
# the writer blocks and the whole process deadlocks.  Python's file I/O
# has no such limitation.
# ---------------------------------------------------------------------------
_ONSTART_TEMPLATE = """\
set -e
export DEBIAN_FRONTEND=noninteractive
date -u +"vast_train.sh boot @ %Y-%m-%dT%H:%M:%SZ"

# Make sure git + git-lfs are installed (image may not include them).
(command -v git >/dev/null 2>&1 && command -v git-lfs >/dev/null 2>&1) \\
    || (apt-get update -y && apt-get install -y git git-lfs)
git lfs install --skip-smudge

export GH_TOKEN='__GH_TOKEN__' HF_TOKEN='__HF_TOKEN__'
mkdir -p /workspace && cd /workspace

# Clone with LFS smudge skipped -- full LFS pull would fetch every old
# checkpoint over the network. We only need the resume target (if any),
# which the trainer fetches on its own when --resume is requested.
echo "── cloning __BRANCH__ ──"
GIT_LFS_SKIP_SMUDGE=1 git clone --branch '__BRANCH__' --single-branch \\
    "https://x-access-token:${GH_TOKEN}@github.com/__REPO_SLUG__.git" brian
cd brian

echo "── bootstrap (pip deps + targeted LFS pull) ──"
# When FRESH=1 we are not resuming, so skip the wholesale adamw-ckpt LFS
# pull in bootstrap step 6 (saves 5-10 min and 3-5 GB transfer).
if [ "__FRESH__" = "1" ]; then
  SKIP_LFS_RESUME=1 bash scripts/vast_bootstrap.sh
else
  bash scripts/vast_bootstrap.sh
fi

echo "── starting log-pusher (background) ──"
# Push the current training log to git every PUSH_INTERVAL seconds so
# progress is visible from any clone without SSH-ing into the instance.
# Default 300s ~ every ~200 train steps at typical ~1.5s/step.
#
# BOOT_TIMESTAMP is computed ONCE here so the filename prefix and the
# train_dsl boot-stamp line in the log share the same UTC moment. Both
# the background pusher (below) and the final one-shot pusher (after
# training) must see the same value, otherwise the final commit would
# write to a different filename than the snapshots.
export BOOT_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
echo "    BOOT_TIMESTAMP=$BOOT_TIMESTAMP (used in log filename prefix)"
ARCH='__ARCH__' INSTANCE_ID="$(hostname)" PUSH_INTERVAL='__LOG_PUSH__' \\
    BRANCH='__BRANCH__' REPO_SLUG='__REPO_SLUG__' \\
    BOOT_TIMESTAMP="$BOOT_TIMESTAMP" \\
    OOD_EVERY='__OOD_EVERY__' \\
    nohup bash scripts/log_pusher.sh > /workspace/log_pusher.log 2>&1 &
LOG_PUSHER_PID=$!
echo "    log_pusher pid=$LOG_PUSHER_PID"

echo "── starting training ──"
if [ "__USE_DSL__" = "1" ]; then
    echo "    DSL mode: arch=__ARCH__ scale=__SCALE__ steps=__STEPS__ batch=__BATCH__ seq_len=__SEQ_LEN__ d_sem=__D_SEM__ ood_every=__OOD_EVERY__ explore_every=__EXPLORE_EVERY__"
    ARCH='__ARCH__' SCALE='__SCALE__' STEPS='__STEPS__' BATCH='__BATCH__' \\
        SEQ_LEN='__SEQ_LEN__' D_SEM='__D_SEM__' \\
        SAVE_EVERY='__SAVE_EVERY__' LOG_EVERY='__LOG_EVERY__' \\
        OOD_EVERY='__OOD_EVERY__' \\
        EXPLORE_EVERY='__EXPLORE_EVERY__' EXPLORE_POP='__EXPLORE_POP__' \\
        EXPLORE_GENS='__EXPLORE_GENS__' EXPLORE_LEN='__EXPLORE_LEN__' \\
        EXPLORE_SITES='__EXPLORE_SITES__' USE_MODULATIONS='__USE_MODULATIONS__' \\
        bash scripts/vast_train_dsl_loop.sh 2>&1 | tee /workspace/train.log
else
    echo "    Brain mode: preset=__PRESET__ steps=__STEPS__ batch=__BATCH__ grad_accum=__GRAD_ACCUM__"
    PRESET='__PRESET__' STEPS='__STEPS__' BATCH='__BATCH__' GRAD_ACCUM='__GRAD_ACCUM__' \\
        OPT=adamw SAVE_EVERY='__SAVE_EVERY__' LOG_EVERY='__LOG_EVERY__' \\
        FRESH='__FRESH__' \\
        bash scripts/vast_train_loop.sh 2>&1 | tee /workspace/train.log
fi

echo "── stopping log-pusher ──"
kill $LOG_PUSHER_PID 2>/dev/null || true
sleep 2   # give it a moment to exit cleanly

# ── Final log push (one-shot) ──────────────────────────────────────
# Run a single iteration of the pusher loop with PUSH_INTERVAL=1 so it
# attempts exactly one commit+push of the complete train.log. The loop
# is unbounded so we wrap with timeout -- but we want it to exit ASAP,
# not poll for 60s. Solution: send SIGTERM after the first cycle (~5s
# at worst on a successful push).
#
# Reuse the same BOOT_TIMESTAMP as the background pusher so the final
# commit writes to the same filename, not a fresh one.
SOURCE_LOG=/workspace/train.log INSTANCE_ID="$(hostname)" \\
    PUSH_INTERVAL=1 BRANCH='__BRANCH__' REPO_SLUG='__REPO_SLUG__' \\
    BOOT_TIMESTAMP="$BOOT_TIMESTAMP" \\
    timeout 30 bash scripts/log_pusher.sh 2>&1 | head -10 \\
    || echo "[onstart] final log push: timeout/exit (best-effort)"

# ── Final checkpoint push (DSL trainer doesn't push on save) ──────
# vast_train_dsl_loop's train_dsl.py saves checkpoints to lfs_checkpoints/
# locally but doesn't push them -- unlike Brain's train.py. Push every
# dsl_arch_*.pt that landed during this run so the artefacts survive
# the instance destroy.
echo "── pushing final checkpoints ──"
cd /workspace/brian
ls -la lfs_checkpoints/dsl_arch_*.pt 2>/dev/null || echo "[onstart] no DSL checkpoints to push"
git config user.email "vast-train@brian.local" || true
git config user.name  "vast-train"             || true
for ckpt in lfs_checkpoints/dsl_arch_*.pt; do
    [ -e "$ckpt" ] || continue
    git add "$ckpt" 2>/dev/null || true
done
if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "chkpt(dsl): final push @ $(date -u +%Y-%m-%dT%H:%M:%SZ)" \\
        >/dev/null 2>&1 || echo "[onstart] commit failed"
    PUSH_URL="https://x-access-token:${GH_TOKEN}@github.com/__REPO_SLUG__.git"
    timeout 600 git push "$PUSH_URL" "HEAD:__BRANCH__" 2>&1 \\
        | sed "s#${GH_TOKEN}#***#g" \\
        || echo "[onstart] checkpoint push failed (will not block destroy)"
else
    echo "[onstart] no new checkpoints to commit"
fi

# ── Self-destroy the vast instance ────────────────────────────────
# Without this the container stays "running" after onstart exits and
# bills you indefinitely. Verified 2026-05-30 on 38469631 (8 hours
# idle after training completed, ~$10 wasted). Uses the INSTANCE_ID
# env var that vast.ai injects into every container, fallback to
# looking ourselves up by container label.
echo "── self-destroying instance ──"
if ! command -v vastai >/dev/null 2>&1; then
    pip install -q vastai 2>&1 | tail -3 || true
fi
if [ -n "${VAST_API_KEY:-}" ] && command -v vastai >/dev/null 2>&1; then
    vastai set api-key "$VAST_API_KEY" >/dev/null 2>&1 || true
    # Vast injects $CONTAINER_ID and $VAST_CONTAINERLABEL; the contract
    # id is usually exposed as $INSTANCE_ID. Try them in order.
    SELF_ID="${INSTANCE_ID:-${VAST_CONTAINER_ID:-}}"
    if [ -z "$SELF_ID" ]; then
        # Fall back: find our instance by label match.
        SELF_ID="$(vastai show instances --raw 2>/dev/null \\
            | python3 -c "import sys, json
data = json.load(sys.stdin)
for i in (data or []):
    if i.get('label') == 'neuroslm-full':
        print(i.get('id', '')); break" 2>/dev/null)"
    fi
    if [ -n "$SELF_ID" ]; then
        echo "[onstart] vastai destroy instance $SELF_ID"
        # `yes y` answers the interactive confirmation prompt (no -y
        # available in this vastai version). Without this the command
        # hangs waiting for stdin and the instance never destroys.
        yes y | vastai destroy instance "$SELF_ID" 2>&1 || echo "[onstart] destroy failed"
        # The destroy command kills our container -- anything past this
        # never executes. The echo below is reached only if destroy failed.
        sleep 30
    else
        echo "[onstart] could not determine vast instance id; not destroying"
    fi
else
    echo "[onstart] VAST_API_KEY not in env or vastai CLI missing -- cannot self-destroy"
fi

echo "── training exited; FAILED to self-destroy. Run: vastai destroy instance <contract_id> ──"
"""


class VastConnector(BaseConnector):
    """Launch training on vast.ai via ``scripts/vast_train.sh``."""

    @classmethod
    def platform_name(cls) -> str:
        return "vast"

    def launch(self, config: DeployConfig) -> int:
        env = self._build_env(config)

        # Build the container-side onstart script in Python (avoids bash
        # heredoc pipe-buffer deadlock on Windows Git Bash) and write it
        # to a temp file.  Pass the path via ONSTART_FILE so vast_train.sh
        # can read it with a simple `while read` loop (no pipe, no heredoc).
        onstart_content = self._build_onstart(env)
        tf = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".sh",
            delete=False,
            encoding="utf-8",
            newline="\n",   # LF-only; container is Linux
        )
        try:
            tf.write(onstart_content)
            tf.flush()
            tf.close()
            env["ONSTART_FILE"] = tf.name

            bash = self._find_bash()
            script = str(REPO_ROOT / "scripts" / "vast_train.sh")
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

    # ── ps integration (unified job registry) ───────────────────────
    #
    # Vast.ai already has rich live polling via ``vastai show instances``
    # in ``cmd_ps``; the unified registry surface is wired here for
    # completeness so ``brian ps --platform vast`` works symmetrically
    # with Lightning. We currently just expose any jobs persisted to
    # ``.brian/jobs/`` — the deploy path doesn't write them yet for
    # vast (the existing ``vast_train.sh`` flow is self-contained),
    # but future SSH-style refactors of vast will populate them.

    def list_jobs(self) -> List[JobInfo]:
        return load_jobs(platform=self.platform_name())

    # ── internal helpers ────────────────────────────────────────────

    @staticmethod
    def _build_onstart(env: dict) -> str:
        """Return the container-side onstart bash script as a string.

        Substitutes locally-expanded variables from *env* into the
        ``_ONSTART_TEMPLATE``.  Container-side variables ($BOOT_TIMESTAMP,
        ${GH_TOKEN}, $(date …), …) appear literally — Python str.replace()
        does not interpret $ characters.
        """
        repo_url = env.get("REPO_URL") or "https://github.com/269652/BRIAN.git"
        repo_slug = repo_url.removeprefix("https://github.com/").removesuffix(".git")

        subs = {
            "__GH_TOKEN__":     env.get("GH_TOKEN", ""),
            "__HF_TOKEN__":     env.get("HF_TOKEN", ""),
            "__BRANCH__":       env.get("BRANCH", "master"),
            "__REPO_SLUG__":    repo_slug,
            "__FRESH__":        env.get("FRESH", "1"),
            "__ARCH__":         env.get("ARCH", "current"),
            "__SCALE__":        env.get("SCALE", ""),
            "__STEPS__":        env.get("STEPS", ""),
            "__BATCH__":        env.get("BATCH", ""),
            "__GRAD_ACCUM__":   env.get("GRAD_ACCUM", ""),
            "__SEQ_LEN__":      env.get("SEQ_LEN", ""),
            "__D_SEM__":        env.get("D_SEM", ""),
            "__OOD_EVERY__":    env.get("OOD_EVERY", "0"),
            "__SAVE_EVERY__":   env.get("SAVE_EVERY", "1000"),
            "__LOG_EVERY__":    env.get("LOG_EVERY", "20"),
            "__PRESET__":       env.get("PRESET", ""),
            "__USE_DSL__":      env.get("USE_DSL", "1"),
            "__LOG_PUSH__":     env.get("LOG_PUSH_INTERVAL", "300"),
            "__EXPLORE_EVERY__":     env.get("EXPLORE_EVERY", "0"),
            "__EXPLORE_POP__":       env.get("EXPLORE_POP", "24"),
            "__EXPLORE_GENS__":      env.get("EXPLORE_GENS", "10"),
            "__EXPLORE_LEN__":       env.get("EXPLORE_LEN", "8"),
            "__EXPLORE_SITES__":     env.get("EXPLORE_SITES", "2"),
            "__USE_MODULATIONS__":   env.get("USE_MODULATIONS", "0"),
        }
        result = _ONSTART_TEMPLATE
        for placeholder, value in subs.items():
            result = result.replace(placeholder, value)
        return result

    def _build_env(self, config: DeployConfig) -> dict:
        env = os.environ.copy()
        env["USE_DSL"] = "1"
        env["STEPS"] = str(config.steps)
        env["PYTHONIOENCODING"] = "utf-8"

        if config.branch:
            env["BRANCH"] = config.branch
        if config.arch:
            # vast_train_dsl_loop.sh prepends "architectures/" to ARCH itself,
            # so strip the prefix here to avoid "architectures/architectures/…".
            arch_name = config.arch
            if arch_name.startswith("architectures/"):
                arch_name = arch_name[len("architectures/"):]
            env["ARCH"] = arch_name
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

        if config.explore_every > 0:
            env["EXPLORE_EVERY"] = str(config.explore_every)
            env["EXPLORE_POP"] = str(config.explore_pop)
            env["EXPLORE_GENS"] = str(config.explore_gens)
            env["EXPLORE_LEN"] = str(config.explore_len)
            env["EXPLORE_SITES"] = str(config.explore_sites)
            if config.use_modulations:
                env["USE_MODULATIONS"] = "1"

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
