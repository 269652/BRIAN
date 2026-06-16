#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# log_pusher.sh — periodically commit and push the training log to git.
#
# Runs in the background alongside `vast_train_loop.sh` on the vast instance.
# Copies the current train.log into the canonical per-run folder layout
#   logs/<YYYYMMDD>/<arch>/<HHMMSS>_<short-sha>/train.log
# commits, and pushes to the branch the runner cloned. This makes training
# progress visible from local clones without SSH'ing into the instance.
#
# Path layout (2026-06-15-eve): 3-level hierarchy — day → arch → per-run
# leaf. We write DIRECTLY into the folder used by ``brian migrate
# 0001_logs_to_run_folders``. Stable filename per run (always
# ``train.log``) means we just overwrite the file in place — no need
# to track ``_PREV_LOG`` and ``git rm`` the previous step's filename.
#
# Layout history:
#   pre-2026-06-15  flat:      logs/vast/<stamp>_<id>_..._stepNofN.log
#   2026-06-15-am   nested:    logs/vast/<YYYYMMDD>/<ARCH>/<stamp>_<id>_..._stepNofN.log
#   2026-06-15-pm   per-run:   logs/<YYYYMMDD>-<HHMMSS>_<arch>_<sha>/train.log (flat)
#   2026-06-15-eve hierarchy:  logs/<YYYYMMDD>/<arch>/<HHMMSS>_<sha>/train.log (3-level)
#
# The hierarchy puts day first because that's how humans search
# ("which day was that crash?"), arch second because it's the next-
# most-common search axis, and ``<time>_<sha>`` last as a per-run leaf.
#
# Cadence (2026-06-15): the pusher is now STEP-DRIVEN when ``LOG_EVERY``
# is exported. The poll is tight (POLL_INTERVAL=30s default) and the
# push only fires when the trainer's "step N" marker in the log crosses
# a LOG_EVERY boundary (so log push and the trainer's per-row print
# stay perfectly in sync — both at step 500, 1000, 1500, …). This
# decouples log pushes from checkpoint pushes, which are driven by
# PUSH_EVERY inside train_dsl.py (only push checkpoints, not logs).
#
# When LOG_EVERY is unset / 0, we fall back to the legacy time-based
# wall-clock loop using PUSH_INTERVAL — kept for back-compat with any
# standalone invocation that doesn't know the step cadence.
#
# Env vars (with defaults):
#   LOG_EVERY=0           step cadence (0 = legacy time-based mode)
#   POLL_INTERVAL=30      seconds between log inspections (step-driven mode)
#   PUSH_INTERVAL=300     seconds between attempts (legacy time-based mode,
#                          used only when LOG_EVERY=0)
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

LOG_EVERY="${LOG_EVERY:-0}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
PUSH_INTERVAL="${PUSH_INTERVAL:-300}"
SOURCE_LOG="${SOURCE_LOG:-/workspace/train.log}"
REPO_DIR="${REPO_DIR:-/workspace/brian}"
INSTANCE_ID="${INSTANCE_ID:-$(hostname)}"
REPO_SLUG="${REPO_SLUG:-269652/BRIAN}"
# Run identity, used to build the per-run folder name. Each env var
# is optional — missing ones get sensible defaults.
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

# Logs root; per-run folder created on demand inside _compose_logfile.
mkdir -p "logs"

# ── Filename builder ───────────────────────────────────────────────────
# Returns the canonical per-run folder path used across the project:
#
#   logs/<YYYYMMDD>/<arch>/<HHMMSS>_<short-sha>/train.log
#
# This is the SAME layout that the ``0001_logs_to_run_folders``
# migration produces when normalising old flat ``logs/vast/*.log``
# files. Writing here directly means:
#   * one stable filename per run (no _PREV_LOG cleanup needed)
#   * the migration becomes a no-op going forward
#   * ``ls logs/`` groups by day, ``ls logs/<day>/`` groups by arch —
#     matches how a human searches the directory tree
#
# History: pre-2026-06-15 we wrote to the nested layout
# ``logs/vast/<YYYYMMDD>/<ARCH_NAME>/<stamp>_<id>_..._stepNofN.log``.
# Two operational problems with that:
#   1. The filename changed every push (step count in the suffix), so
#      we had to ``git rm`` the previous path each iteration — a tiny
#      race that occasionally produced empty commits or stranded
#      stub files in the index.
#   2. The deeper nesting didn't survive the ``.gitignore`` whitelist
#      ``!logs/vast/*.log`` (regression 29522f9 had to add
#      ``!logs/vast/**/*.log``). The new layout ``logs/<folder>/train.log``
#      is whitelisted by a clean ``!logs/**/train.log`` pattern.
#
# The 2026-06-15-eve refactor split the per-run folder name from
# ``<date>-<time>_<arch>_<sha>`` (flat, all-in-one) into a 3-level
# hierarchy ``<date>/<arch>/<time>_<sha>`` (day → arch → leaf) so
# ``ls logs/`` is a short list of days instead of a long undifferentiated
# dump of run folders.
#
# BOOT_TIMESTAMP slicing: BOOT_TIMESTAMP=YYYYMMDDTHHMMSSZ →
#   day            = ${BOOT_TIMESTAMP:0:8}   # YYYYMMDD
#   time_of_day    = ${BOOT_TIMESTAMP:9:6}   # HHMMSS  (skips the 'T')
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

_start_step() {
    # Extract the FIRST "step N" from the log (for resumed runs this will
    # be > 0). Returns 0 if log is empty or unparseable.
    local n
    n="$(grep -oE "^step[[:space:]]+[0-9]+" "$SOURCE_LOG" 2>/dev/null \
                | head -1 | awk '{print $2}')"
    echo "${n:-0}"
}

_compose_logfile() {
    # New layout (2026-06-16): step range in filename, not generic train.log
    #
    # Layout (2-level hierarchy: day → arch):
    #   logs/<YYYYMMDD>/<arch>/<HHMMSS>_<start>_<end>.log
    #
    # Where:
    #   HHMMSS = boot time (UTC)
    #   start  = first step in log (0 for fresh runs, >0 for resumed)
    #   end    = current/final step
    #
    # The filename updates with each push as the step count advances.
    # Previous filenames are removed (git rm) to avoid accumulating
    # <time>_0_1000.log, <time>_0_2000.log, etc.
    local day time_of_day start_step cur_step
    # 20260615T185105Z → day=20260615  time_of_day=185105
    day="${BOOT_TIMESTAMP:0:8}"
    time_of_day="${BOOT_TIMESTAMP:9:6}"
    
    # Determine step range from log content
    start_step="$(_start_step)"
    cur_step="$(_current_step)"
    
    local folder="logs/${day}/${ARCH_NAME}"
    mkdir -p "$folder"
    echo "${folder}/${time_of_day}_${start_step}_${cur_step}.log"
}

# In-container git identity for the commits.
git config user.email "vast-train@brian.local" >/dev/null 2>&1 || true
git config user.name  "vast-train"             >/dev/null 2>&1 || true

PUSH_URL="https://x-access-token:${GITHUB}@github.com/${REPO_SLUG}.git"

# ── Observability markers (2026-06-15) ─────────────────────────────────
# /workspace/log_pusher.log is invisible to ``brian logs`` (it goes to a
# separate file, not the trainer's stdout) AND `vastai execute` blocks
# `cat`/`tail` on running instances. So we drop empty marker files at
# well-known paths that `vastai execute id 'ls /workspace/log_pusher_*'`
# CAN list. The mtime tells you when each phase last happened.
#
# Marker contract:
#   /workspace/log_pusher_alive          touched every poll iteration
#   /workspace/log_pusher_last_push_OK   touched on every successful push
#                                          (rename of _last_push_FAIL if any)
#   /workspace/log_pusher_last_push_FAIL touched on every push failure;
#                                          contains the git error message
#                                          (last 4KB) — readable via `du`
# When the user sees `du -h /workspace/log_pusher_last_push_FAIL` come
# back > 0 bytes, they know push is failing and can extract more info
# by stopping the instance + `vastai execute` (cat is allowed on stopped).
_MARK_ALIVE=/workspace/log_pusher_alive
_MARK_OK=/workspace/log_pusher_last_push_OK
_MARK_FAIL=/workspace/log_pusher_last_push_FAIL
# Persist the last 4KB of any push error so the user can post-mortem it
# after stopping the instance — vastai execute can't cat live but ls/du
# CAN reveal the file size as a "did push fail" signal.
_mark_fail() {
    local msg="$1"
    printf '%s\n' "$msg" | tail -c 4096 > "$_MARK_FAIL" 2>/dev/null || true
    rm -f "$_MARK_OK" 2>/dev/null || true
}
_mark_ok() {
    touch "$_MARK_OK" 2>/dev/null || true
    rm -f "$_MARK_FAIL" 2>/dev/null || true
}

# ── Defensive push helper (2026-06-15) ────────────────────────────────
# The previous "push-optimistically-rebase-on-fail" flow was the actual
# bug behind instance 41084160's silent push failure: the on-box repo
# was 3 commits behind origin (deploy commit + later NFG iterations),
# every push got rejected non-fast-forward, and the rebase retry had
# subtle interactions with untracked lfs_checkpoints/<RUN>/step*.pt
# files (matched by .gitattributes `*.pt filter=lfs`) that made the
# rebase silently fail too.
#
# The fix is to ALWAYS rebase --autostash BEFORE the commit, so the
# log file is committed on top of a known-fresh origin tip. Push is
# then guaranteed fast-forward.
#
# Returns 0 on push success, 1 on push failure (with $_MARK_FAIL
# populated). Caller is responsible for staging the file FIRST.
_safe_push() {
    local commit_msg="$1"
    # Step A: rebase any local commits onto origin/$BRANCH, stashing
    # working-tree changes (untracked lfs_checkpoints, modified
    # pointer files, etc.) so the rebase is clean. --autostash
    # re-applies the stash after a successful rebase.
    local pull_out
    if ! pull_out="$(git pull --rebase --autostash \
                        "$PUSH_URL" "$BRANCH" 2>&1 \
                        | sed -E "s#${GITHUB}#***#g")"; then
        echo "[log_pusher] git pull --rebase --autostash FAILED:"
        echo "$pull_out"
        _mark_fail "pull-rebase failed:\n$pull_out"
        return 1
    fi
    # Step B: commit the staged log file (caller already did `git add`).
    if git diff --cached --quiet 2>/dev/null; then
        echo "[log_pusher] nothing staged after rebase, skipping push"
        return 0
    fi
    if ! git commit -m "$commit_msg" >/dev/null 2>&1; then
        echo "[log_pusher] commit failed"
        _mark_fail "commit failed: $commit_msg"
        return 1
    fi
    # Step C: push — now guaranteed fast-forward.
    local push_out
    if push_out="$(git push "$PUSH_URL" "${BRANCH}:${BRANCH}" 2>&1 \
                       | sed -E "s#${GITHUB}#***#g")"; then
        _mark_ok
        return 0
    fi
    echo "[log_pusher] git push FAILED:"
    echo "$push_out"
    _mark_fail "push failed:\n$push_out"
    return 1
}

# ── ONESHOT mode (deterministic exit code for the deploy gate) ─────
# 2026-06-15: instance 41048619 trace showed `timeout 120 log_pusher
# | head -30` always returned 141 (SIGPIPE) — turning a successful
# final-log push into a false-positive that tripped the gated-
# self-destroy contract. ONESHOT=1 runs ONE push iteration and exits
# with the real status (0 = pushed or nothing-to-push, 1 = push
# actually failed). Locked by tests/test_deploy_failure_safety.py
# ::TestLogPusherOneshotMode + ::TestDeployUsesOneshotForFinalLog.
if [ "${ONESHOT:-0}" = "1" ]; then
    if [ ! -f "$SOURCE_LOG" ]; then
        echo "[log_pusher] ONESHOT: source log not present at $SOURCE_LOG"
        exit 0   # nothing to push isn't a failure
    fi
    LOG_REL="$(_compose_logfile)"
    cp -f "$SOURCE_LOG" "$LOG_REL"
    git add "$LOG_REL"
    if git diff --cached --quiet 2>/dev/null; then
        echo "[log_pusher] ONESHOT: log unchanged, nothing to push"
        exit 0
    fi
    SIZE="$(wc -c <"$LOG_REL")"
    # Re-stage AFTER the rebase (rebase may have changed HEAD and
    # unstaged us). The _safe_push helper does the commit itself.
    if _safe_push "logs($(basename "$LOG_REL" .log)): final sync @ $(date -u +%H:%M:%SZ) (${SIZE} B)"; then
        echo "[log_pusher] ONESHOT: pushed (${SIZE} B)"
        exit 0
    fi
    echo "[log_pusher] ONESHOT: push failed (see $_MARK_FAIL on box)"
    exit 1
fi

# Compute the per-run destination once for the startup banner — the
# banner is informational only, the actual writes go through
# _compose_logfile every iteration.
_BANNER_DEST="$(_compose_logfile)"

if [ "$LOG_EVERY" -gt 0 ] 2>/dev/null; then
    echo "[log_pusher] STEP-DRIVEN: push every ${LOG_EVERY} train steps (poll every ${POLL_INTERVAL}s)"
    echo "[log_pusher] watching $SOURCE_LOG → $_BANNER_DEST"
else
    echo "[log_pusher] TIME-DRIVEN (legacy): push every ${PUSH_INTERVAL}s"
    echo "[log_pusher] watching $SOURCE_LOG → $_BANNER_DEST"
fi

# ── Step parser ────────────────────────────────────────────────────────
# Returns the most recent "step N" reported by the trainer in the live
# log (matching ``_format_metrics_line`` in train_dsl.py — format
# ``step <N> | loss …``). Returns 0 when the log is empty / no row yet.
_current_step() {
    local n
    n="$(grep -oE "^step[[:space:]]+[0-9]+" "$SOURCE_LOG" 2>/dev/null \
                | tail -1 | awk '{print $2}')"
    echo "${n:-0}"
}

# Tracks the highest LOG_EVERY-bucket we've already pushed (integer
# division). Initialised to -1 so step 0 is always considered a fresh
# bucket. Only consulted in step-driven mode.
LAST_PUSHED_BUCKET=-1

while true; do
    # Poll cadence:
    #   * step-driven: tight POLL_INTERVAL (default 30s), gate decides
    #   * legacy time-driven: slow PUSH_INTERVAL (default 300s), always push
    if [ "$LOG_EVERY" -gt 0 ] 2>/dev/null; then
        sleep "$POLL_INTERVAL"
    else
        sleep "$PUSH_INTERVAL"
    fi

    # Liveness marker — touched every iteration, so the user can
    # `vastai execute id 'ls -la /workspace/log_pusher_alive'` and see
    # whether the bg process is still alive (mtime ≈ now).
    touch "$_MARK_ALIVE" 2>/dev/null || true

    if [ ! -f "$SOURCE_LOG" ]; then
        echo "[log_pusher] $(date -u +%H:%M:%SZ) source log not present yet"
        continue
    fi

    # ── Step-bucket gate (only active when LOG_EVERY > 0) ─────────────
    # Push iff the trainer has crossed into a new LOG_EVERY-sized bucket
    # since our last push. e.g. LOG_EVERY=500 → push exactly at the
    # step-500, step-1000, step-1500, … log rows. This keeps the on-
    # disk log in lock-step with what the user sees in ``brian logs``.
    if [ "$LOG_EVERY" -gt 0 ] 2>/dev/null; then
        CUR_STEP="$(_current_step)"
        CUR_BUCKET=$((CUR_STEP / LOG_EVERY))
        if [ "$CUR_BUCKET" -le "$LAST_PUSHED_BUCKET" ]; then
            # Still in the same bucket — no new log_every boundary crossed.
            continue
        fi
        # Cross detected. Fall through to commit+push; update tracker
        # only AFTER a successful push so a transient git failure
        # retries on the next poll.
        NEW_BUCKET="$CUR_BUCKET"
    fi

    # Compute new log path with current step range
    LOG_REL="$(_compose_logfile)"
    
    # Remove any previous log files for this run (same boot time, different step range)
    # Pattern: logs/<day>/<arch>/<boot_time>_*_*.log
    day="${BOOT_TIMESTAMP:0:8}"
    time_of_day="${BOOT_TIMESTAMP:9:6}"
    OLD_PATTERN="logs/${day}/${ARCH_NAME}/${time_of_day}_*_*.log"
    for old_log in $OLD_PATTERN; do
        if [ -f "$old_log" ] && [ "$old_log" != "$LOG_REL" ]; then
            git rm -f "$old_log" 2>/dev/null || rm -f "$old_log"
        fi
    done
    
    # Copy current log to new path
    cp -f "$SOURCE_LOG" "$LOG_REL"

    git add "$LOG_REL"
    if git diff --cached --quiet 2>/dev/null; then
        echo "[log_pusher] $(date -u +%H:%M:%SZ) log unchanged, skipping"
        continue
    fi

    SIZE="$(wc -c <"$LOG_REL")"

    # Push via the defensive rebase-first helper. It internally:
    #   1. git pull --rebase --autostash (always succeeds since the
    #      trainer doesn't modify tracked files)
    #   2. commits the staged log file
    #   3. git push (guaranteed fast-forward after step 1)
    # On failure, it populates /workspace/log_pusher_last_push_FAIL
    # with the git error and clears /workspace/log_pusher_last_push_OK
    # so the user can detect the regression via `vastai execute … 'ls'`.
    PUSH_OK=0
    if _safe_push "logs($(basename "$LOG_REL" .log)): tail sync @ $(date -u +%H:%M:%SZ) (${SIZE} B)"; then
        echo "[log_pusher] $(date -u +%H:%M:%SZ) pushed (${SIZE} B)"
        PUSH_OK=1
    else
        echo "[log_pusher] $(date -u +%H:%M:%SZ) push failed; will retry next cycle"
    fi

    # Only advance the bucket tracker on a SUCCESSFUL push. A transient
    # network blip stays in the current bucket so the next poll retries.
    if [ "$LOG_EVERY" -gt 0 ] 2>/dev/null && [ "$PUSH_OK" -eq 1 ]; then
        LAST_PUSHED_BUCKET="$NEW_BUCKET"
    fi
done

