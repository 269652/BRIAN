#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# vast_deploy.sh — spin up TWO A100 instances on Vast.ai and auto-train both
# the FULL bio model and the param-matched BASELINE for STEPS steps.
#
# Run this from your LOCAL machine, a Colab cell, or any Jupyter terminal —
# NOT on the GPU box. It uses the `vastai` CLI to create the instances; each
# instance then bootstraps + trains headless via its onstart command, and
# auto-pushes checkpoints to Git LFS (concurrent-push safe — see train.py).
#
# Secrets + config come from a local `.env` (copy from .env.example):
#   VAST_API_KEY, GITHUB, [HF_TOKEN], [PRESET STEPS BATCH GRAD_ACCUM ...]
#
# Usage:
#   cp .env.example .env && edit .env       # add VAST_API_KEY + GITHUB
#   bash scripts/vast_deploy.sh
#
# Re-run:    bash scripts/vast_deploy.sh             # keep healthy, replace stuck
#            bash scripts/vast_deploy.sh --recreate  # destroy ALL + redeploy
#                                                      (picks up a new git push)
#            bash scripts/vast_deploy.sh --fresh     # wipe ALL checkpoints +
#                                                      redeploy from step 0
#                                                      (use after a config change)
# To stop:   bash scripts/vast_deploy.sh --destroy   # tear everything down
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$HERE/.env}"

# ── Load .env ─────────────────────────────────────────────────────────────
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
else
  echo "✗ $ENV_FILE not found. cp .env.example .env and fill it in." >&2
  exit 1
fi

# Accept common alias names so either convention in .env works:
#   VAST_API_KEY | VAST_AI        GITHUB | GITHUB_PAT | GH_TOKEN
VAST_API_KEY="${VAST_API_KEY:-${VAST_AI:-}}"
GITHUB="${GITHUB:-${GITHUB_PAT:-${GH_TOKEN:-}}}"

REPO_URL="${REPO_URL:-https://github.com/269652/BRIAN.git}"
REPO_SLUG="${REPO_URL#https://github.com/}"; REPO_SLUG="${REPO_SLUG%.git}"
PRESET="${PRESET:-large}"
STEPS="${STEPS:-100000}"
# Optional git branch for the instance to train (default: repo default branch).
# Set BRANCH=<name> to deploy an experiment branch instead of master.
BRANCH="${BRANCH:-}"
BATCH="${BATCH:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-20}"
VAST_IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
VAST_DISK="${VAST_DISK:-60}"
# Vast query fragment selecting the GPU. Use the `in [...]` operator (the
# `~` "contains" operator is NOT supported) with underscore aliases — vast
# matches them against the space-named GPUs (A100 SXM4 / A100 PCIE).
VAST_GPU_QUERY="${VAST_GPU_QUERY:-gpu_name in [A100_SXM4,A100_PCIE,A100_SXM,A100X]}"

# ── Resolve python + vastai CLI (cross-platform: Linux, Colab, Win git-bash)
# Pitfalls handled:
#   • git-bash ships /usr/bin/python3 WITHOUT pip — must skip it.
#   • Windows venv exposes `python` (not `python3`) with pip.
#   • `python -m vastai` doesn't work (package has no __main__) — so we
#     locate the `vastai` console-script next to the chosen python binary.
# Strategy: pick the FIRST interpreter that actually has pip, preferring the
# active venv ($VIRTUAL_ENV), then `python`, then `python3`, then `py`.
# Normalise a possibly-Windows path (C:\...) to a git-bash path (/c/...).
_norm_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -u "$1" 2>/dev/null || printf '%s' "${1//\\//}"
  else
    printf '%s' "${1//\\//}"
  fi
}
_pick_python() {
  local c ve
  # 1. Active venv ($VIRTUAL_ENV may be a backslash Windows path → cygpath).
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    ve="$(_norm_path "$VIRTUAL_ENV")"
    for c in "$ve/Scripts/python.exe" "$ve/Scripts/python" "$ve/bin/python"; do
      [ -x "$c" ] && "$c" -m pip --version >/dev/null 2>&1 && { printf '%s' "$c"; return; }
    done
  fi
  # 2. Any repo-local venv (.venv, .venv-1, .venv-2, …) — robust even when
  #    VIRTUAL_ENV isn't exported into bash and PATH python lacks pip.
  for ve in "$HERE"/.venv*; do
    [ -d "$ve" ] || continue
    for c in "$ve/Scripts/python.exe" "$ve/bin/python"; do
      [ -x "$c" ] && "$c" -m pip --version >/dev/null 2>&1 && { printf '%s' "$c"; return; }
    done
  done
  # 3. Windows per-user installs (typical when invoked from PowerShell — `py`
  #    may default to a 2.7 install and `python` in PATH may be the Store stub
  #    or Python 2.7). Probe these BEFORE the PATH fallback so we always grab
  #    a working 3.x with pip.
  local _home_u
  _home_u="$(_norm_path "${USERPROFILE:-$HOME}")"
  for c in "$_home_u"/AppData/Local/Programs/Python/Python3*/python.exe \
           "/c/Program Files/Python3"*/python.exe \
           "/c/Python3"*/python.exe; do
    [ -x "$c" ] || continue
    "$c" -m pip --version >/dev/null 2>&1 && { printf '%s' "$c"; return; }
  done
  # 4. PATH pythons that have pip.
  for c in python python.exe python3 py; do
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" -m pip --version >/dev/null 2>&1 && { printf '%s' "$c"; return; }
  done
  # 5. Last resort: ask the `py` launcher for a Python 3 by ABSOLUTE PATH
  #    (some PowerShell setups have `py` defaulting to 2.7, so a bare `py`
  #    test above returns 2.7-without-pip; but `py -3 -c "..."` still hits a
  #    3.x install and gives us its real python.exe path).
  if command -v py >/dev/null 2>&1; then
    local _py3
    _py3="$(py -3 -c 'import sys,os; print(os.path.normpath(sys.executable))' 2>/dev/null)"
    if [ -n "$_py3" ]; then
      _py3="$(_norm_path "$_py3")"
      [ -x "$_py3" ] && "$_py3" -m pip --version >/dev/null 2>&1 && { printf '%s' "$_py3"; return; }
    fi
  fi
  printf ''
}
PYTHON="$(_pick_python)"
if [ -z "$PYTHON" ]; then
  echo "✗ no python with pip found." >&2
  echo "  VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>}" >&2
  echo "  tried (in order):" >&2
  echo "    1. \$VIRTUAL_ENV/Scripts/python.exe (and bin/python)" >&2
  echo "    2. $HERE/.venv*/Scripts/python.exe" >&2
  echo "    3. ~/AppData/Local/Programs/Python/Python3*/python.exe" >&2
  echo "       /c/Program Files/Python3*/python.exe, /c/Python3*/python.exe" >&2
  echo "    4. python, python.exe, python3, py in PATH" >&2
  echo "    5. py -3 → resolved-absolute python.exe" >&2
  echo "  Fix: activate a venv that has pip, or install Python 3 from python.org," >&2
  echo "       or run from Colab/Linux." >&2
  echo "  (git-bash's /usr/bin/python3 has no pip and is intentionally skipped.)" >&2
  exit 1
fi
echo "  using python: $PYTHON"

# Invoke vastai by importing its entry point and calling main() — exactly
# what the console script does. This is path-independent (no need to find a
# vastai.exe, which pip may install to a user-scripts dir that sysconfig
# doesn't report) AND keeps subcommand registration working (unlike
# `python -m vastai.cli.main`, where __name__=='__main__' breaks it).
vastai() {
  "$PYTHON" -c 'import sys; from vastai.cli.main import main; sys.exit(main())' "$@"
}

# Run a vastai command and emit ONLY its JSON payload. Recent vastai CLI
# versions print a "DEPRECATED: ..." banner line BEFORE the JSON for
# `show instances --raw`, which makes a naive json.load() fail (it then
# silently treats the result as zero instances → the reconcile deploys
# duplicates and never destroys the old ones). Strip anything before the
# first JSON bracket so parsing is robust to such banners.
_vast_json() {
  vastai "$@" 2>/dev/null | "$PYTHON" -c '
import sys
buf = sys.stdin.read()
starts = [i for i in (buf.find("["), buf.find("{")) if i != -1]
sys.stdout.write(buf[min(starts):] if starts else "")
'
}

if ! "$PYTHON" -c 'import vastai.cli.main' >/dev/null 2>&1; then
  echo "── installing vastai CLI (via $PYTHON -m pip) ──"
  "$PYTHON" -m pip install -q --upgrade vastai
fi
if ! "$PYTHON" -c 'import vastai.cli.main' >/dev/null 2>&1; then
  echo "✗ vastai not importable after install. Try: $PYTHON -m pip install vastai" >&2
  exit 1
fi
echo "  using vastai: $PYTHON -c '…vastai.cli.main:main' ($(vastai --version 2>/dev/null))"

: "${VAST_API_KEY:?set VAST_API_KEY (or VAST_AI) in .env}"
: "${GITHUB:?set GITHUB (or GITHUB_PAT) in .env}"
vastai set api-key "$VAST_API_KEY" >/dev/null

# ── Modes (flag-style, order-independent) ───────────────────────────────
#   --destroy            tear everything down and exit
#   --recreate           destroy + redeploy BOTH roles (even if healthy)
#   --recreate-full      destroy + redeploy ONLY the full role
#   --recreate-baseline  destroy + redeploy ONLY the baseline role
#   --best-step N        BEFORE recreating, delete checkpoints with step > N
#                        from the role(s) being recreated (uses git rm +
#                        commit + push so the new instance pulls the
#                        cleaned LFS state). Optional.
RECREATE_ROLES=""
BEST_STEP=""
FRESH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --destroy)
      echo "── destroying instances labelled neuroslm-* ──"
      _vast_json show instances --raw \
        | "$PYTHON" -c "import sys,json;[print(i['id']) for i in json.load(sys.stdin) if 'neuroslm' in (i.get('label') or '')]" \
        | while read -r id; do echo "destroy $id"; vastai destroy instance "$id" -y; done
      exit 0
      ;;
    --recreate)          RECREATE_ROLES="full baseline"; shift ;;
    --recreate-full)     RECREATE_ROLES="${RECREATE_ROLES} full";     shift ;;
    --recreate-baseline) RECREATE_ROLES="${RECREATE_ROLES} baseline"; shift ;;
    --best-step)         BEST_STEP="${2:?--best-step requires a number}"; shift 2 ;;
    --best-step=*)       BEST_STEP="${1#--best-step=}"; shift ;;
    # --fresh: wipe checkpoints + redeploy from step 0 (train from scratch
    # with a new config instead of inheriting old weights + optimizer state).
    # Targets whichever roles are selected by --recreate* flags; if none are
    # given it defaults to BOTH roles (see the post-parse block below). So
    # `--fresh --recreate-full` does a fresh start of ONLY the full role.
    --fresh)             FRESH=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
# Dedupe RECREATE_ROLES tokens (allow `--recreate-full --recreate-baseline`)
RECREATE_ROLES="$(printf '%s\n' $RECREATE_ROLES | awk '!seen[$0]++' | tr '\n' ' ')"
RECREATE_ROLES="${RECREATE_ROLES% }"
# --fresh with no explicit role flag → both roles (back-compat default).
if [ "$FRESH" = "1" ] && [ -z "$RECREATE_ROLES" ]; then
  RECREATE_ROLES="full baseline"
fi
if [ -n "$RECREATE_ROLES" ]; then
  echo "── recreate roles: ${RECREATE_ROLES} ──"
fi

# --fresh wipes EVERY step-numbered checkpoint for the recreated roles so a
# crash-restart's `--resume latest` can't pick up a stale high-step file.
# Implemented by reusing the prune+push path with max-step -1 (deletes all
# step > -1). The instances then bootstrap with FRESH=1 → train from step 0.
if [ "$FRESH" = "1" ] && [ -z "$BEST_STEP" ]; then
  echo "── --fresh: wiping ALL checkpoints for roles: ${RECREATE_ROLES} ──"
  BEST_STEP=-1
fi

# ── Optional: prune > best-step from the role(s) being recreated ────────
# Does git rm + commit + push so the freshly-created Vast instance, when
# it bootstraps and does `git lfs pull`, only sees checkpoints ≤ BEST_STEP
# and `--resume latest` picks the right one.
if [ -n "$BEST_STEP" ] && [ -n "$RECREATE_ROLES" ]; then
  for role in $RECREATE_ROLES; do
    echo "── prune ${role} checkpoints > step ${BEST_STEP} ──"
    "$PYTHON" -m neuroslm.tools.prune_ckpts \
      --dirs "$HERE/lfs_checkpoints" \
      --max-step "$BEST_STEP" --only "$role" --git || true
  done
  # Push the deletion commits so the new instance sees the cleaned state.
  # Use the PAT from .env so this never prompts on Windows credential manager.
  # Skip if prune deleted nothing (no local commits ahead of origin) — avoids
  # firing the git-lfs pre-push hook on a no-op push.
  _branch="$(cd "$HERE" && git rev-parse --abbrev-ref HEAD)"
  _ahead="$(cd "$HERE" && git rev-list "origin/${_branch}..HEAD" --count 2>/dev/null || echo 0)"
  if [ "${_ahead:-0}" -eq 0 ]; then
    echo "── nothing to push (no commits ahead of origin/${_branch}) ──"
  else
    echo "── pushing ${_ahead} prune commit(s) to origin/${_branch} ──"
    PUSH_URL="https://x-access-token:${GITHUB}@github.com/${REPO_SLUG}.git"
    ( cd "$HERE" && git -c credential.helper= push "$PUSH_URL" HEAD 2>&1 \
        | sed "s#${GITHUB}#***#g" | tail -3 ) || true
  fi
elif [ -n "$BEST_STEP" ] && [ -z "$RECREATE_ROLES" ]; then
  echo "✗ --best-step given but no --recreate*; nothing to do. Pass " >&2
  echo "  --recreate / --recreate-full / --recreate-baseline too." >&2
  exit 2
fi

# ── Reconcile existing instances (idempotent re-runs) ─────────────────────
# Normal run: keep healthy ('running'/'loading'), destroy STUCK ones
# (created/stopped/exited/…), (re)deploy only what's needed.
# --recreate: destroy ALL neuroslm-<role> instances and redeploy fresh
# (so a new git push is picked up).
ROLES="full baseline"

_instances_json="$(_vast_json show instances --raw)"
declare -A NEEDS   # role -> 1 if we must (re)deploy it

for role in $ROLES; do
  # Emit "id<TAB>status" for this role's instance; status is trimmed +
  # lowercased in python so no stray CR/whitespace can corrupt the match.
  _info="$(printf '%s' "$_instances_json" | ROLE="$role" "$PYTHON" -c "
import sys, json, os
role = os.environ['ROLE']
try: data = json.load(sys.stdin)
except Exception: data = []
for i in (data or []):
    if (i.get('label') or '') == 'neuroslm-'+role:
        st = i.get('actual_status') or i.get('cur_state') or i.get('intended_status') or 'unknown'
        sys.stdout.write('%s\t%s' % (i.get('id',''), str(st).strip().lower()))
        break
" 2>/dev/null)"
  iid=""; istatus=""
  IFS=$'\t' read -r iid istatus <<< "$_info"

  if [ -z "${iid:-}" ]; then
    echo "  neuroslm-$role: not present — deploying"
    NEEDS[$role]=1
    continue
  fi

  # --recreate-* flags: if THIS role is in the recreate set, destroy
  # whatever exists (healthy or not) and redeploy. Roles NOT in the set
  # fall through to the normal health-based reconcile below.
  if printf ' %s ' $RECREATE_ROLES | grep -qw "$role"; then
    echo "  neuroslm-$role: recreate → destroying (id $iid, status=$istatus) + redeploying"
    vastai destroy instance "$iid" -y >/dev/null 2>&1 || true
    NEEDS[$role]=1
    continue
  fi

  case "$istatus" in
    running|loading|*running*)
      # Hard guard: a running/loading instance is NEVER auto-destroyed.
      echo "  neuroslm-$role: healthy (id $iid, status=$istatus) — keeping"
      NEEDS[$role]=0
      ;;
    created|stopped|exited|offline|error|unknown|"")
      echo "  neuroslm-$role: STUCK (id $iid, status='${istatus:-?}') — destroying + redeploying"
      vastai destroy instance "$iid" -y >/dev/null 2>&1 || true
      NEEDS[$role]=1
      ;;
    *)
      # Unrecognised status — be SAFE: keep it, don't destroy. Report so we
      # can extend the lists if a new healthy state shows up.
      echo "  neuroslm-$role: status='$istatus' not recognised — keeping (safe default)"
      NEEDS[$role]=0
      ;;
  esac
done

NEED_COUNT=0
for role in $ROLES; do [ "${NEEDS[$role]:-0}" = "1" ] && NEED_COUNT=$((NEED_COUNT+1)); done
if [ "$NEED_COUNT" -eq 0 ]; then
  echo "✓ all roles already healthy — nothing to deploy."
  echo "  (use --destroy to tear everything down)"
  exit 0
fi

# ── Pick $NEED_COUNT cheapest available A100 offers (distinct machines) ────
echo "── searching A100 offers (need $NEED_COUNT) ──"
OFFER_QUERY="${VAST_GPU_QUERY} num_gpus=1 rentable=true disk_space>=${VAST_DISK} reliability>0.95"
echo "  query: $OFFER_QUERY"
OFFERS="$(vastai search offers "$OFFER_QUERY" -o 'dph+' --raw 2>&1)"
case "$OFFERS" in
  \[*|\{*) : ;;
  *)
    echo "✗ vastai search did not return JSON. Output was:" >&2
    printf '%s\n' "$OFFERS" | head -5 >&2
    exit 1
    ;;
esac

PICKED="$(printf '%s' "$OFFERS" | "$PYTHON" -c "
import sys, json
need = int('$NEED_COUNT')
offers = json.load(sys.stdin)
seen, picked = set(), []
for o in offers:
    if 'A100' not in (o.get('gpu_name') or ''):
        continue
    mid = o.get('machine_id')
    if mid in seen:
        continue
    seen.add(mid); picked.append(str(o['id']))
    if len(picked) == need:
        break
print(' '.join(picked))
")"
# shellcheck disable=SC2206
OFFER_LIST=($PICKED)
if [ "${#OFFER_LIST[@]}" -lt "$NEED_COUNT" ]; then
  echo "✗ found ${#OFFER_LIST[@]} A100 offer(s), need $NEED_COUNT." >&2
  echo "  Try again later, lower VAST_DISK, or set VAST_GPU_QUERY." >&2
  exit 1
fi
echo "  picked offers: ${OFFER_LIST[*]}"

# ── Build the per-role onstart command ────────────────────────────────────
# Inline-clone (the scripts live in the repo), then bootstrap + train loop.
# Tokens arrive via --env so bootstrap/push can read them.
make_onstart() {
  local role="$1" extra="$2" ckpt_dir="$3"
  cat <<ONSTART
set -e
export DEBIAN_FRONTEND=noninteractive
# The pytorch/pytorch image ships without an ssh client/server. Vast's
# launcher (/.launch) and 'vastai execute' both want ssh; install it early
# (best-effort) so debugging connections work. The real fix for the
# /.launch 'ssh: command not found' loop is dropping --direct below.
(command -v ssh >/dev/null 2>&1 || (apt-get update -y && apt-get install -y openssh-client openssh-server)) || true
export GITHUB='${GITHUB}' HF_TOKEN='${HF_TOKEN:-}'
cd /workspace
git clone https://x-access-token:\${GITHUB}@github.com/${REPO_SLUG}.git brian || true
cd brian
if [ -n "${BRANCH}" ]; then
  echo "── checking out branch ${BRANCH} ──"
  git fetch origin "${BRANCH}" && git checkout "${BRANCH}" || echo "⚠ branch ${BRANCH} checkout failed; staying on default"
fi
bash scripts/vast_bootstrap.sh
echo "── launching ${role} training (live below; also in /workspace/train_${role}.log) ──"
# Foreground + tee so 'vastai logs ID' streams the actual training output
# (step/loss lines), not just the bootstrap. The instance persists while
# this runs; when it finishes (STEPS reached) the box idles until destroyed.
PRESET='${PRESET}' STEPS='${STEPS}' BATCH='${BATCH}' GRAD_ACCUM='${GRAD_ACCUM}' \
  OPT=adamw SAVE_EVERY='${SAVE_EVERY}' LOG_EVERY='${LOG_EVERY}' \
  CKPT_DIR='${ckpt_dir}' EXTRA_ARGS='${extra}' FRESH='${FRESH}' \
  bash scripts/vast_train_loop.sh 2>&1 | tee /workspace/train_${role}.log
ONSTART
}

ENV_ARG="-e GITHUB=${GITHUB} -e HF_TOKEN=${HF_TOKEN:-}"

create_instance() {
  local offer="$1" role="$2" onstart="$3"
  echo "── creating ${role} instance on offer ${offer} ──"
  # NOTE: --ssh WITHOUT --direct. Direct mode makes Vast's /.launch invoke an
  # in-container `ssh` client in a keepalive loop; the pytorch/pytorch image
  # has no ssh, so it spins on "ssh: command not found" and the onstart-cmd
  # never runs (instance idles, no training). Proxy ssh (plain --ssh) routes
  # through Vast and doesn't need the in-container client, so onstart runs.
  vastai create instance "$offer" \
    --image "$VAST_IMAGE" \
    --disk "$VAST_DISK" \
    --label "neuroslm-${role}" \
    --env "$ENV_ARG" \
    --onstart-cmd "$onstart" \
    --ssh
}

# Assign the freshly-picked offers to the roles that need (re)deploying.
oi=0
for role in $ROLES; do
  [ "${NEEDS[$role]:-0}" = "1" ] || continue
  offer="${OFFER_LIST[$oi]}"; oi=$((oi+1))
  if [ "$role" = "baseline" ]; then
    onstart="$(make_onstart baseline '--baseline' /workspace/brian/checkpoints_baseline)"
  else
    onstart="$(make_onstart full '' /workspace/brian/lfs_checkpoints)"
  fi
  create_instance "$offer" "$role" "$onstart"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Deploy reconciled. Roles healthy/deployed:"
echo "    • neuroslm-full     → full bio model ($PRESET, $STEPS steps)"
echo "    • neuroslm-baseline → param-matched vanilla baseline"
echo "  Checkpoints push to Git LFS (separate streams, concurrent-safe)."
echo ""
echo "  Watch:    bash scripts/vast.sh show instances"
echo "            bash scripts/vast.sh logs <id>"
echo "  Re-run:   bash scripts/vast_deploy.sh   # kills stuck, redeploys missing"
echo "  Compare:  python -m neuroslm.tools.compare_ckpts \\"
echo "              --full_dir lfs_checkpoints --baseline_dir lfs_checkpoints \\"
echo "              --preset $PRESET --device cpu   (after git lfs pull)"
echo "  STOP (avoid charges):  bash scripts/vast_deploy.sh --destroy"
echo "════════════════════════════════════════════════════════════════"
