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
# To watch:  vastai logs <instance_id>      (or open the instance's Jupyter)
# To stop:   vastai destroy instance <id>   (or `bash scripts/vast_deploy.sh --destroy`)
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
  # 3. PATH pythons that have pip.
  for c in python python.exe python3 py; do
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" -m pip --version >/dev/null 2>&1 && { printf '%s' "$c"; return; }
  done
  printf ''
}
PYTHON="$(_pick_python)"
if [ -z "$PYTHON" ]; then
  echo "✗ no python with pip found." >&2
  echo "  VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>}" >&2
  echo "  tried: \$VIRTUAL_ENV/Scripts/python.exe, python, python.exe, python3, py" >&2
  echo "  Fix: activate a venv that has pip, or run from Colab/Linux." >&2
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

# ── Optional: destroy mode ────────────────────────────────────────────────
if [ "${1:-}" = "--destroy" ]; then
  echo "── destroying instances labelled neuroslm-* ──"
  vastai show instances --raw \
    | "$PYTHON" -c "import sys,json;[print(i['id']) for i in json.load(sys.stdin) if 'neuroslm' in (i.get('label') or '')]" \
    | while read -r id; do echo "destroy $id"; vastai destroy instance "$id" -y; done
  exit 0
fi

# ── Reconcile existing instances (idempotent re-runs) ─────────────────────
# On re-run: keep a role's instance if it's healthy ('running'/'loading'),
# destroy it if it's STUCK (status not in the healthy set — e.g. stuck in
# 'created', 'offline', 'exited'), and (re)deploy only the roles that need
# it. So `bash scripts/vast_deploy.sh` can be run repeatedly to converge to
# exactly one healthy instance per role.
ROLES="full baseline"

_instances_json="$(vastai show instances --raw 2>/dev/null)"
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
export GITHUB='${GITHUB}' HF_TOKEN='${HF_TOKEN:-}'
cd /workspace
git clone https://x-access-token:\${GITHUB}@github.com/${REPO_SLUG}.git brian || true
cd brian
bash scripts/vast_bootstrap.sh
echo "── launching ${role} training (live below; also in /workspace/train_${role}.log) ──"
# Foreground + tee so `vastai logs <id>` streams the actual training output
# (step/loss lines), not just the bootstrap. The instance persists while
# this runs; when it finishes (STEPS reached) the box idles until destroyed.
PRESET='${PRESET}' STEPS='${STEPS}' BATCH='${BATCH}' GRAD_ACCUM='${GRAD_ACCUM}' \
  OPT=adamw SAVE_EVERY='${SAVE_EVERY}' LOG_EVERY='${LOG_EVERY}' \
  CKPT_DIR='${ckpt_dir}' EXTRA_ARGS='${extra}' \
  bash scripts/vast_train_loop.sh 2>&1 | tee /workspace/train_${role}.log
ONSTART
}

ENV_ARG="-e GITHUB=${GITHUB} -e HF_TOKEN=${HF_TOKEN:-}"

create_instance() {
  local offer="$1" role="$2" onstart="$3"
  echo "── creating ${role} instance on offer ${offer} ──"
  vastai create instance "$offer" \
    --image "$VAST_IMAGE" \
    --disk "$VAST_DISK" \
    --label "neuroslm-${role}" \
    --env "$ENV_ARG" \
    --onstart-cmd "$onstart" \
    --ssh --direct
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
