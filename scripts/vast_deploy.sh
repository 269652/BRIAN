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
VAST_GPU_NAME="${VAST_GPU_NAME:-A100}"

# ── vastai CLI ──────────────────────────────────────────────────────────
if ! command -v vastai >/dev/null 2>&1; then
  echo "── installing vastai CLI ──"
  pip install -q --upgrade vastai
fi
: "${VAST_API_KEY:?set VAST_API_KEY in .env}"
: "${GITHUB:?set GITHUB (PAT) in .env}"
vastai set api-key "$VAST_API_KEY" >/dev/null

# ── Optional: destroy mode ────────────────────────────────────────────────
if [ "${1:-}" = "--destroy" ]; then
  echo "── destroying instances labelled neuroslm-* ──"
  vastai show instances --raw \
    | python3 -c "import sys,json;[print(i['id']) for i in json.load(sys.stdin) if 'neuroslm' in (i.get('label') or '')]" \
    | while read -r id; do echo "destroy $id"; vastai destroy instance "$id"; done
  exit 0
fi

# ── Pick the 2 cheapest available on-demand A100 offers ───────────────────
echo "── searching A100 offers ──"
OFFERS=$(vastai search offers \
  "gpu_name~${VAST_GPU_NAME} num_gpus=1 rentable=true disk_space>=${VAST_DISK} reliability>0.95" \
  -o 'dph+' --raw)

read -r OFFER1 OFFER2 < <(printf '%s' "$OFFERS" | python3 -c "
import sys, json
offers = json.load(sys.stdin)
# already sorted by dollars-per-hour ascending; take 2 distinct machines
seen, picked = set(), []
for o in offers:
    mid = o.get('machine_id')
    if mid in seen:
        continue
    seen.add(mid); picked.append(str(o['id']))
    if len(picked) == 2:
        break
print(' '.join(picked))
")

if [ -z "${OFFER1:-}" ] || [ -z "${OFFER2:-}" ]; then
  echo "✗ could not find 2 available A100 offers. Try a different VAST_GPU_NAME" >&2
  echo "  or check 'vastai search offers \"gpu_name~A100 num_gpus=1 rentable=true\"'." >&2
  exit 1
fi
echo "  picked offers: FULL=$OFFER1  BASELINE=$OFFER2"

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
PRESET='${PRESET}' STEPS='${STEPS}' BATCH='${BATCH}' GRAD_ACCUM='${GRAD_ACCUM}' \
  OPT=adamw SAVE_EVERY='${SAVE_EVERY}' LOG_EVERY='${LOG_EVERY}' \
  CKPT_DIR='${ckpt_dir}' EXTRA_ARGS='${extra}' \
  nohup bash scripts/vast_train_loop.sh > /workspace/train_${role}.log 2>&1 &
echo "launched ${role} training; tail -f /workspace/train_${role}.log"
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

create_instance "$OFFER1" "full" \
  "$(make_onstart full '' /workspace/brian/lfs_checkpoints)"
create_instance "$OFFER2" "baseline" \
  "$(make_onstart baseline '--baseline' /workspace/brian/checkpoints_baseline)"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Two A100 instances launched:"
echo "    • neuroslm-full     → trains the full bio model ($PRESET, $STEPS steps)"
echo "    • neuroslm-baseline → trains the param-matched vanilla baseline"
echo "  Both push checkpoints to Git LFS (separate streams, concurrent-safe)."
echo ""
echo "  Watch:    vastai show instances        # get ids"
echo "            vastai logs <id>             # stream onstart + training log"
echo "  Compare:  python -m neuroslm.tools.compare_ckpts \\"
echo "              --full_dir lfs_checkpoints --baseline_dir checkpoints_baseline \\"
echo "              --preset $PRESET --device cuda   (after pulling the pushed ckpts)"
echo "  STOP (avoid charges):  bash scripts/vast_deploy.sh --destroy"
echo "════════════════════════════════════════════════════════════════"
