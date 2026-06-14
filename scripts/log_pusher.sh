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
# Run identity, used to build a speaking filename like
# `<UTC_stamp>_<instance>_<arch>_<params>_<label>_step<cur>of<target>.log`
# Each env var is optional — missing ones get sensible defaults.
# 2026-06-14: bowtie arch folder renamed rcc_bowtie → master (canonical)
# with architectures/current as the live working-copy. Default to
# "current" so log filenames match `brian train` without --arch.
ARCH_NAME="${ARCH_NAME:-${ARCH:-current}}"
LABEL="${LABEL:-neuroslm-full}"
TOTAL_STEPS="${TOTAL_STEPS:-${STEPS:-?}}"
# UTC boot timestamp prefix. The deploy script (vast_train.sh) exports
# this before launching us so the timestamp matches the train_dsl boot
# stamp printed in the log itself. If unset (e.g. running standalone),
# we fall back to "now" — still better than no prefix, which lets two
# runs on the same vast.ai instance id silently overwrite each other
# (regression: deploy 40923107 clobbered 40921910 on instance 38569395).
BOOT_TIMESTAMP="${BOOT_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

: "${GITHUB:?log_pusher: GITHUB PAT must be exported}"

cd "$REPO_DIR" || { echo "[log_pusher] cannot cd to $REPO_DIR" >&2; exit 1; }

BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null)}"
[ -z "$BRANCH" ] && { echo "[log_pusher] no git branch detected" >&2; exit 1; }

mkdir -p "logs/vast"

# ── Filename builder ───────────────────────────────────────────────────
# Returns a "speaking" path that includes:
#   <instance>_<arch>_<params>_<label>_step<cur>of<target>.log
# Where params + cur step are parsed live from $SOURCE_LOG (defaults
# kick in when the log hasn't reported them yet).
_format_steps() {
    local n="$1"
    # Compact "10000" → "10k", "40000" → "40k", "?" stays "?".
    case "$n" in
        ?|"") echo "$n"; return ;;
    esac
    if [ "$n" -ge 1000 ] 2>/dev/null && [ $((n % 1000)) -eq 0 ]; then
        echo "$((n / 1000))k"
    else
        echo "$n"
    fi
}

_compose_logfile() {
    local cur_step="?" params="?"
    if [ -f "$SOURCE_LOG" ]; then
        # Latest "step    N" line (number-only field; we take the last hit)
        cur_step="$(grep -oE "^step[[:space:]]+[0-9]+" "$SOURCE_LOG" \
                        | tail -1 | awk '{print $2}')"
        cur_step="${cur_step:-0}"
        # "DSL-LM parameters: 134.8M" → "134M"
        params="$(grep -oE "DSL-LM parameters:[[:space:]]+[0-9.]+[MBG]" \
                        "$SOURCE_LOG" | tail -1 \
                        | awk '{print $3}' | sed -E 's/\.[0-9]+//')"
        params="${params:-?}"
    fi
    local cur="$(_format_steps "$cur_step")"
    local tgt="$(_format_steps "$TOTAL_STEPS")"
    # Strip the leading "neuroslm-full-" the deploy script prepends — it's
    # redundant with the path prefix logs/vast/. Keep only the suffix.
    local short_label="${LABEL#neuroslm-full-}"
    short_label="${short_label#neuroslm-full}"
    [ -z "$short_label" ] && short_label="run"
    # Timestamp prefix MUST come first so `ls logs/vast/` sorts
    # chronologically and reused vast instance ids never alias.
    echo "logs/vast/${BOOT_TIMESTAMP}_${INSTANCE_ID}_${ARCH_NAME}_${params}_${short_label}_step${cur}of${tgt}.log"
}

# Previous-file tracker so we can remove an old name when the step bumps.
_PREV_LOG=""

# In-container git identity for the commits.
git config user.email "vast-train@brian.local" >/dev/null 2>&1 || true
git config user.name  "vast-train"             >/dev/null 2>&1 || true

PUSH_URL="https://x-access-token:${GITHUB}@github.com/${REPO_SLUG}.git"

echo "[log_pusher] watching $SOURCE_LOG → logs/vast/${BOOT_TIMESTAMP}_${INSTANCE_ID}_..._stepNofN.log (every ${PUSH_INTERVAL}s)"

while true; do
    sleep "$PUSH_INTERVAL"

    if [ ! -f "$SOURCE_LOG" ]; then
        echo "[log_pusher] $(date -u +%H:%M:%SZ) source log not present yet"
        continue
    fi

    # Rebuild the filename every iteration — step + params info evolves.
    LOG_REL="$(_compose_logfile)"

    # If the filename changed since last push (step climbed, params learned),
    # remove the previous file so the directory doesn't accumulate stubs.
    if [ -n "$_PREV_LOG" ] && [ "$_PREV_LOG" != "$LOG_REL" ] && [ -f "$_PREV_LOG" ]; then
        git rm -f "$_PREV_LOG" >/dev/null 2>&1 || rm -f "$_PREV_LOG"
    fi
    _PREV_LOG="$LOG_REL"

    cp -f "$SOURCE_LOG" "$LOG_REL"

    git add "$LOG_REL"
    if git diff --cached --quiet 2>/dev/null; then
        echo "[log_pusher] $(date -u +%H:%M:%SZ) log unchanged, skipping"
        continue
    fi

    SIZE="$(wc -c <"$LOG_REL")"
    git commit -m "logs($(basename "$LOG_REL" .log)): tail sync @ $(date -u +%H:%M:%SZ) (${SIZE} B)" \
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
