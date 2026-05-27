#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# vast_train.sh — DEPLOY ONE TRAINING INSTANCE TO VAST.AI.
#
# Replaces the multi-role / recreate / reconcile / fresh-flag machinery of
# `vast_deploy.sh` with a single straightforward path: pick a healthy host,
# create one instance labeled neuroslm-full, run training, push checkpoints.
#
# WHO THIS IS FOR
#   Any LLM agent or human who wants to launch a 10k training run without
#   reading 500 lines of bash and figuring out which env vars matter.
#
# USAGE (the only forms supported)
#   bash scripts/vast_train.sh                      # uses current branch
#   bash scripts/vast_train.sh <preset>             # +preset override
#   bash scripts/vast_train.sh <preset> <steps>     # +steps override
#
# Required in .env (or environment):
#   GITHUB              - GitHub PAT with repo+lfs write access
#   VAST_API_KEY        - vast.ai API key
#   HF_TOKEN            - optional, for HuggingFace dataset downloads
#
# Optional env overrides (sensible defaults):
#   PRESET=rcc_bowtie_30m_p1     - any preset from neuroslm/config.py:PRESETS
#   STEPS=10000                  - target step count
#   BRANCH=<current git branch>  - which branch to train
#   BATCH=4
#   GRAD_ACCUM=4
#   FRESH=1                      - start from step 0 (no --resume latest)
#   GPU_QUERY="…"                - vast offer filter; default A100/4090 verified>0.99
#
# OUTPUT
#   Prints instance ID + the three follow-up commands you'll need:
#     - bash scripts/vast.sh logs <id>
#     - yes | bash scripts/vast.sh destroy instance <id>
#     - git fetch origin <branch>
#
# WHY THIS IS A NEW SCRIPT (not a refactor of vast_deploy.sh)
#   vast_deploy.sh is 500+ lines covering multi-role training, recreate
#   flags, instance reconciliation, fresh-wipe flows. It's mature for the
#   workflows it knows, but is too complex to safely modify for a single
#   training launch. This script does ONLY launch.
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$HERE/.env"
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }

# ─── Required secrets ────────────────────────────────────────────────────
VAST_API_KEY="${VAST_API_KEY:-${VAST_AI:-}}"
GITHUB="${GITHUB:-${GITHUB_PAT:-${GH_TOKEN:-}}}"
: "${VAST_API_KEY:?✗ set VAST_API_KEY in .env}"
: "${GITHUB:?✗ set GITHUB (PAT with repo write) in .env}"

# ─── Configuration (positional args > env > default) ─────────────────────
PRESET="${1:-${PRESET:-rcc_bowtie_30m_p1}}"
STEPS="${2:-${STEPS:-10000}}"
BRANCH="${BRANCH:-$(git -C "$HERE" rev-parse --abbrev-ref HEAD 2>/dev/null || echo master)}"
BATCH="${BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-20}"
FRESH="${FRESH:-1}"
# Push the running training log to git every N seconds. Defaults to 300s
# which works out to roughly every ~200 train steps at typical pace.
LOG_PUSH_INTERVAL="${LOG_PUSH_INTERVAL:-300}"

REPO_URL="${REPO_URL:-https://github.com/269652/BRIAN.git}"
REPO_SLUG="${REPO_URL#https://github.com/}"; REPO_SLUG="${REPO_SLUG%.git}"
VAST_IMAGE="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
VAST_DISK=60
# Default GPU filter: any A100 variant or 4090, verified host, reliability >0.99.
# Some bad hosts have stuck container loops; this filter cuts most of them.
# Override with GPU_QUERY env var if you need different hardware.
GPU_QUERY="${GPU_QUERY:-gpu_name in [A100_SXM4,A100_PCIE,A100_SXM,A100X,RTX_4090] num_gpus=1 rentable=true verified=true reliability>0.99}"

# ─── Resolve a python that has vastai installed ──────────────────────────
_pick_python() {
  local c ve
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    for c in "$VIRTUAL_ENV/Scripts/python.exe" "$VIRTUAL_ENV/bin/python"; do
      [ -x "$c" ] && "$c" -c 'import vastai.cli.main' >/dev/null 2>&1 && { echo "$c"; return; }
    done
  fi
  for ve in "$HERE"/.venv*; do [ -d "$ve" ] || continue
    for c in "$ve/Scripts/python.exe" "$ve/bin/python"; do
      [ -x "$c" ] && "$c" -c 'import vastai.cli.main' >/dev/null 2>&1 && { echo "$c"; return; }
    done
  done
  for c in python python.exe python3 py; do
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" -c 'import vastai.cli.main' >/dev/null 2>&1 && { echo "$c"; return; }
  done
  echo ""
}
PYTHON="$(_pick_python)"
[ -z "$PYTHON" ] && { echo "✗ no python with vastai found. Install: pip install vastai" >&2; exit 1; }
vastai() { "$PYTHON" -c 'import sys; from vastai.cli.main import main; sys.exit(main())' "$@"; }
vastai set api-key "$VAST_API_KEY" >/dev/null 2>&1 || true

# ─── Header ──────────────────────────────────────────────────────────────
cat <<HDR

════════════════════════════════════════════════════════════════
  vast_train.sh
    branch  = $BRANCH
    preset  = $PRESET
    steps   = $STEPS
    fresh   = $FRESH
    batch   = $BATCH × grad_accum $GRAD_ACCUM
    gpu     = $GPU_QUERY
════════════════════════════════════════════════════════════════
HDR

# ─── Find cheapest healthy offer ─────────────────────────────────────────
echo "── searching offers ──"
OFFERS_JSON="$(vastai search offers "$GPU_QUERY disk_space>=$VAST_DISK" -o 'dph+' --raw 2>&1)"
case "$OFFERS_JSON" in \[*|\{*) : ;; *)
  echo "✗ vast offer search failed:" >&2
  printf '%s\n' "$OFFERS_JSON" | head -3 >&2
  exit 1 ;;
esac

OFFER_INFO="$(printf '%s' "$OFFERS_JSON" | "$PYTHON" -c "
import sys, json
offers = json.load(sys.stdin)
if not offers:
    sys.exit('no offers matched query')
o = offers[0]
print('%s\t%s\t%s\t%.4f\t%.3f' % (
    o.get('id',''),
    o.get('machine_id',''),
    o.get('gpu_name','?'),
    o.get('dph_total', 0),
    o.get('reliability2', 0),
))")"
[ -z "$OFFER_INFO" ] && { echo "✗ no offer info from search" >&2; exit 1; }

OFFER_ID="$(echo "$OFFER_INFO" | cut -f1)"
MACHINE_ID="$(echo "$OFFER_INFO" | cut -f2)"
GPU_NAME="$(echo "$OFFER_INFO" | cut -f3)"
DPH="$(echo "$OFFER_INFO" | cut -f4)"
RELI="$(echo "$OFFER_INFO" | cut -f5)"
echo "  offer  $OFFER_ID  ($GPU_NAME, machine $MACHINE_ID)"
echo "  cost   \$$DPH/hr  reliability $RELI"

# ─── Build the onstart script (runs INSIDE the vast container) ───────────
# Heredoc with normal expansion: $BRANCH etc. expand HERE (locally) so the
# values land in the onstart script as literals. Escape any in-container
# variables (\$1 etc.) if you add them later.
ONSTART="$(cat <<ONSTART
set -e
export DEBIAN_FRONTEND=noninteractive
date -u +"vast_train.sh boot @ %Y-%m-%dT%H:%M:%SZ"

# Make sure git + git-lfs are installed (image may not include them).
(command -v git >/dev/null 2>&1 && command -v git-lfs >/dev/null 2>&1) \\
    || (apt-get update -y && apt-get install -y git git-lfs)
git lfs install --skip-smudge

export GITHUB='${GITHUB}' HF_TOKEN='${HF_TOKEN:-}'
mkdir -p /workspace && cd /workspace

# Clone with LFS smudge skipped — full LFS pull would fetch every old
# checkpoint over the network. We only need the resume target (if any),
# which the trainer fetches on its own when --resume is requested.
echo "── cloning ${BRANCH} ──"
GIT_LFS_SKIP_SMUDGE=1 git clone --branch '${BRANCH}' --single-branch \\
    "https://x-access-token:\${GITHUB}@github.com/${REPO_SLUG}.git" brian
cd brian

echo "── bootstrap (pip deps + targeted LFS pull) ──"
# When FRESH=1 we are not resuming, so skip the wholesale adamw-ckpt LFS
# pull in bootstrap step 6 (saves 5-10 min and 3-5 GB transfer).
if [ "${FRESH}" = "1" ]; then
  SKIP_LFS_RESUME=1 bash scripts/vast_bootstrap.sh
else
  bash scripts/vast_bootstrap.sh
fi

echo "── starting log-pusher (background) ──"
# Push the current training log to git every PUSH_INTERVAL seconds so
# progress is visible from any clone without SSH-ing into the instance.
# Default 300s ≈ every ~200 train steps at typical ~1.5s/step.
INSTANCE_ID="\$(hostname)" PUSH_INTERVAL='${LOG_PUSH_INTERVAL}' \\
    BRANCH='${BRANCH}' REPO_SLUG='${REPO_SLUG}' \\
    nohup bash scripts/log_pusher.sh > /workspace/log_pusher.log 2>&1 &
LOG_PUSHER_PID=\$!
echo "    log_pusher pid=\$LOG_PUSHER_PID"

echo "── starting training ──"
echo "    preset=${PRESET} steps=${STEPS} batch=${BATCH} grad_accum=${GRAD_ACCUM}"
PRESET='${PRESET}' STEPS='${STEPS}' BATCH='${BATCH}' GRAD_ACCUM='${GRAD_ACCUM}' \\
    OPT=adamw SAVE_EVERY='${SAVE_EVERY}' LOG_EVERY='${LOG_EVERY}' \\
    FRESH='${FRESH}' \\
    bash scripts/vast_train_loop.sh 2>&1 | tee /workspace/train.log

echo "── stopping log-pusher ──"
kill \$LOG_PUSHER_PID 2>/dev/null || true
# Final push of the complete log
SOURCE_LOG=/workspace/train.log INSTANCE_ID="\$(hostname)" \\
    PUSH_INTERVAL=1 BRANCH='${BRANCH}' REPO_SLUG='${REPO_SLUG}' \\
    timeout 60 bash scripts/log_pusher.sh || echo "final log push timed out"

echo "── training exited; instance idle. Destroy with: vastai destroy instance \$(hostname) ──"
ONSTART
)"

# ─── Create the instance ─────────────────────────────────────────────────
echo "── creating instance ──"
CREATE_OUT="$(vastai create instance "$OFFER_ID" \
    --image "$VAST_IMAGE" \
    --disk "$VAST_DISK" \
    --label "neuroslm-full" \
    --env "-e GITHUB=$GITHUB -e HF_TOKEN=${HF_TOKEN:-}" \
    --onstart-cmd "$ONSTART" \
    --ssh 2>&1)"

# Parse JSON result. vastai prints a Python dict-literal, not strict JSON,
# so we use eval-via-python rather than json.loads.
INST_ID="$(printf '%s' "$CREATE_OUT" | "$PYTHON" -c "
import sys, re, ast
buf = sys.stdin.read()
m = re.search(r'\{[^}]*new_contract[^}]*\}', buf)
if m:
    try:
        d = ast.literal_eval(m.group(0))
        if d.get('success'):
            print(d.get('new_contract',''))
    except Exception:
        pass
")"

if [ -z "$INST_ID" ]; then
  echo "✗ instance create FAILED (no contract returned):" >&2
  printf '%s\n' "$CREATE_OUT" | sed -E "s#${GITHUB}#***#g" | head -10 >&2
  exit 1
fi

# Mask the PAT in case the dict was echoed earlier
printf '%s\n' "$CREATE_OUT" | sed -E "s#${GITHUB}#***#g" | grep -E "success|new_contract" | head -2

# ─── Done — print follow-up commands ─────────────────────────────────────
cat <<DONE

✓ launched instance $INST_ID (machine $MACHINE_ID, $GPU_NAME, \$$DPH/hr)

Follow-up commands:

  Watch boot + training logs:
    bash scripts/vast.sh logs $INST_ID

  Check status + cost:
    bash scripts/vast.sh show instance $INST_ID

  Destroy when done:
    yes | bash scripts/vast.sh destroy instance $INST_ID

  Pull a checkpoint locally (after first save_every=$SAVE_EVERY steps):
    git fetch origin $BRANCH
    git checkout origin/$BRANCH -- lfs_checkpoints/neuroslm_${PRESET}*.pt
    git lfs pull --include='lfs_checkpoints/neuroslm_${PRESET}*.pt'

Instance is now booting (~3-10 min for image pull + pip install). The
training loop starts once bootstrap completes.

DONE
