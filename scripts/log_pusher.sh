#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# log_pusher.sh — periodically commit and push the training log to git.
#
# Runs in the background alongside `vast_train_loop.sh` on the vast instance.
# Every PUSH_INTERVAL seconds it copies the current train.log into
# logs/vast/<INSTANCE_ID>__neuroslm-full.log, commits, and pushes to the
# branch the runner cloned. This makes training progress visible from
# local clones without SSH'ing into the instance.
#
# Env vars (with defaults):
#   PUSH_INTERVAL=300     seconds between attempts (≈ every 200 train steps
#                          at typical ~1.5s/step)
#   SOURCE_LOG=/workspace/train.log
#   REPO_DIR=/workspace/brian
#   INSTANCE_ID=$(hostname)
#   BRANCH=<current git branch>
#   GITHUB=<PAT>          required for the push
#   REPO_SLUG=269652/BRIAN
#
# Failure modes:
#   * push rejected (someone else pushed) → log, retry next iteration
#   * no train.log yet → skip silently and retry
#   * log unchanged since last push → skip the commit (no empty commits)
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail

PUSH_INTERVAL="${PUSH_INTERVAL:-300}"
SOURCE_LOG="${SOURCE_LOG:-/workspace/train.log}"
REPO_DIR="${REPO_DIR:-/workspace/brian}"
INSTANCE_ID="${INSTANCE_ID:-$(hostname)}"
REPO_SLUG="${REPO_SLUG:-269652/BRIAN}"

: "${GITHUB:?log_pusher: GITHUB PAT must be exported}"

cd "$REPO_DIR" || { echo "[log_pusher] cannot cd to $REPO_DIR" >&2; exit 1; }

BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null)}"
[ -z "$BRANCH" ] && { echo "[log_pusher] no git branch detected" >&2; exit 1; }

LOG_REL="logs/vast/${INSTANCE_ID}__neuroslm-full.log"
mkdir -p "$(dirname "$LOG_REL")"

# In-container git identity for the commits.
git config user.email "vast-train@brian.local" >/dev/null 2>&1 || true
git config user.name  "vast-train"             >/dev/null 2>&1 || true

PUSH_URL="https://x-access-token:${GITHUB}@github.com/${REPO_SLUG}.git"

echo "[log_pusher] watching $SOURCE_LOG → $LOG_REL (every ${PUSH_INTERVAL}s)"

while true; do
    sleep "$PUSH_INTERVAL"

    if [ ! -f "$SOURCE_LOG" ]; then
        echo "[log_pusher] $(date -u +%H:%M:%SZ) source log not present yet"
        continue
    fi

    cp -f "$SOURCE_LOG" "$LOG_REL"

    # Stage first, then check for actual change vs the index/HEAD.
    # The previous check `git diff --quiet -- "$LOG_REL"` returned 0 for
    # *untracked* files (git diff only inspects tracked paths), so the very
    # first cp of a new log file looked like "no change" and was never
    # committed — zero logs ever made it to origin. Verified 2026-05-30
    # on instance 38469631 (172 KB local, 0 commits on origin).
    git add "$LOG_REL"
    if git diff --cached --quiet -- "$LOG_REL" 2>/dev/null; then
        echo "[log_pusher] $(date -u +%H:%M:%SZ) log unchanged, skipping"
        continue
    fi

    SIZE="$(wc -c <"$LOG_REL")"
    git commit -m "logs(${INSTANCE_ID}): tail sync @ $(date -u +%H:%M:%SZ) (${SIZE} B)" \
        >/dev/null 2>&1 || {
            echo "[log_pusher] $(date -u +%H:%M:%SZ) commit failed (probably nothing to commit)"
            continue
        }

    # Push, mask the PAT in any output. We attempt a fast push first; if
    # it's rejected (remote moved), pull --rebase the log path only and
    # retry. We don't touch any other files.
    if git push "$PUSH_URL" "${BRANCH}:${BRANCH}" 2>&1 | sed -E "s#${GITHUB}#***#g"; then
        echo "[log_pusher] $(date -u +%H:%M:%SZ) pushed (${SIZE} B)"
    else
        echo "[log_pusher] $(date -u +%H:%M:%SZ) push failed; pulling+retrying"
        if git pull --rebase "$PUSH_URL" "$BRANCH" 2>&1 | sed -E "s#${GITHUB}#***#g"; then
            git push "$PUSH_URL" "${BRANCH}:${BRANCH}" 2>&1 | sed -E "s#${GITHUB}#***#g" \
                || echo "[log_pusher] retry push also failed, will try again next cycle"
        else
            echo "[log_pusher] rebase failed; will try next cycle"
            # Reset the log commit so we don't accumulate broken state
            git reset --hard HEAD~1 >/dev/null 2>&1 || true
        fi
    fi
done
