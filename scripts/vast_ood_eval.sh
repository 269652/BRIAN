#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# vast_ood_eval.sh — deploy a single throwaway vast.ai instance to run
# brian_ood_test.py on the best checkpoint of a given branch, commit the
# resulting JSON back to the branch, and idle.
#
# Use after a training run finishes (or any time) to get a clean,
# isolated OOD eval that doesn't disturb the training instance.
#
# Usage:
#   BRANCH=stabilize/trunk-grad-isolation \
#   CKPT=lfs_checkpoints/neuroslm_large_107M_adamw_mix_best.pt \
#   ROLE_TAG=rezero \
#     bash scripts/vast_ood_eval.sh
#
# Output JSON is pushed to the branch as `ood_results_<ROLE_TAG>.json` and
# also tee'd to `<json>.log` on the instance. The runner does NOT auto-
# destroy itself; tear down with `vastai destroy instance <id>` when done.
#
# Env vars (with defaults):
#   BRANCH              required — git branch with the trained checkpoint
#   CKPT                lfs_checkpoints/neuroslm_large_107M_adamw_mix_best.pt
#   ROLE_TAG            eval — used in label `neuroslm-ood-<tag>` and JSON name
#   MAX_OOD_WINDOWS     200
#   BATCH_SIZE          4
#   VAST_GPU_QUERY      (any A100/A40/A10/4090/3090 with reliability>0.95)
#   VAST_DISK           60
#   VAST_IMAGE          pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$HERE/.env}"
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }

VAST_API_KEY="${VAST_API_KEY:-${VAST_AI:-}}"
GITHUB="${GITHUB:-${GITHUB_PAT:-${GH_TOKEN:-}}}"
: "${VAST_API_KEY:?set VAST_API_KEY/VAST_AI in .env}"
: "${GITHUB:?set GITHUB/GITHUB_PAT in .env}"

: "${BRANCH:?BRANCH env required (the git branch with the trained checkpoint)}"
CKPT="${CKPT:-lfs_checkpoints/neuroslm_large_107M_adamw_mix_best.pt}"
ROLE_TAG="${ROLE_TAG:-eval}"
MAX_OOD_WINDOWS="${MAX_OOD_WINDOWS:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"

REPO_URL="${REPO_URL:-https://github.com/269652/BRIAN.git}"
REPO_SLUG="${REPO_URL#https://github.com/}"; REPO_SLUG="${REPO_SLUG%.git}"
VAST_IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
VAST_DISK="${VAST_DISK:-60}"
# Wider GPU pool than training (eval is light: 107M model + 200 windows).
VAST_GPU_QUERY="${VAST_GPU_QUERY:-gpu_name in [A100_SXM4,A100_PCIE,A100_SXM,A100X,A40,A10,RTX_4090,RTX_3090] num_gpus=1 rentable=true reliability>0.95}"
OUTPUT_FILE="ood_results_${ROLE_TAG}.json"

# ── Resolve python with vastai (lifted from vast_deploy.sh) ──────────────
_norm_path() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -u "$1" 2>/dev/null || printf '%s' "${1//\\//}"
  else printf '%s' "${1//\\//}"; fi
}
_pick_python() {
  local c ve
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    ve="$(_norm_path "$VIRTUAL_ENV")"
    for c in "$ve/Scripts/python.exe" "$ve/Scripts/python" "$ve/bin/python"; do
      [ -x "$c" ] && "$c" -m pip --version >/dev/null 2>&1 && { printf '%s' "$c"; return; }
    done
  fi
  for ve in "$HERE"/.venv*; do [ -d "$ve" ] || continue
    for c in "$ve/Scripts/python.exe" "$ve/bin/python"; do
      [ -x "$c" ] && "$c" -m pip --version >/dev/null 2>&1 && { printf '%s' "$c"; return; }
    done
  done
  for c in python python.exe python3 py; do
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" -m pip --version >/dev/null 2>&1 && { printf '%s' "$c"; return; }
  done
  printf ''
}
PYTHON="$(_pick_python)"
[ -n "$PYTHON" ] || { echo "✗ no python with pip found" >&2; exit 1; }

vastai() { "$PYTHON" -c 'import sys; from vastai.cli.main import main; sys.exit(main())' "$@"; }
_vast_json() {
  vastai "$@" 2>/dev/null | "$PYTHON" -c '
import sys
buf = sys.stdin.read()
starts = [i for i in (buf.find("["), buf.find("{")) if i != -1]
sys.stdout.write(buf[min(starts):] if starts else "")
'
}

if ! "$PYTHON" -c 'import vastai.cli.main' >/dev/null 2>&1; then
  "$PYTHON" -m pip install -q --upgrade vastai
fi
vastai set api-key "$VAST_API_KEY" >/dev/null

# ── Onstart: clone, checkout, lfs pull, install, run eval, push result ───
# Note: ${BRANCH}/${CKPT}/etc. expand in THIS script's scope; backslashed
# vars (\${GITHUB}, \$?) defer to instance-side expansion.
ONSTART=$(cat <<ONSTART
set -e
export DEBIAN_FRONTEND=noninteractive
(command -v git >/dev/null 2>&1 || apt-get update -y && apt-get install -y git git-lfs) || true
git lfs install || true
export GITHUB='${GITHUB}' HF_TOKEN='${HF_TOKEN:-}'
cd /workspace
git clone https://x-access-token:\${GITHUB}@github.com/${REPO_SLUG}.git brian
cd brian
git checkout ${BRANCH}
echo "── pulling LFS object: ${CKPT} ──"
git lfs pull --include="${CKPT}"
echo "── bootstrap (pip deps) ──"
bash scripts/vast_bootstrap.sh
echo "── running OOD eval (max_windows=${MAX_OOD_WINDOWS}, batch=${BATCH_SIZE}) ──"
python -u brian_ood_test.py \\
  --checkpoint "${CKPT}" \\
  --max_ood_windows ${MAX_OOD_WINDOWS} \\
  --batch_size ${BATCH_SIZE} \\
  --output "${OUTPUT_FILE}" 2>&1 | tee /workspace/brian/${OUTPUT_FILE}.log
echo "── committing + pushing result ──"
git config user.email "ood-eval@vast.local"
git config user.name "ood-eval-bot"
git add "${OUTPUT_FILE}" || true
git commit -m "ood eval (${ROLE_TAG}) on ${BRANCH}" || echo "nothing to commit"
PUSH_URL="https://x-access-token:\${GITHUB}@github.com/${REPO_SLUG}.git"
# Push retries against the race with concurrent training-instance LFS pushes.
for i in 1 2 3 4 5; do
  if git -c credential.helper= push "\${PUSH_URL}" ${BRANCH} 2>&1 | tee /tmp/push.log | grep -q "${BRANCH} -> ${BRANCH}"; then
    echo "✓ pushed"; break
  fi
  echo "push attempt \$i failed; fetching + retrying"
  git -c credential.helper= fetch "\${PUSH_URL}" ${BRANCH}
  git rebase FETCH_HEAD || true
  sleep 5
done
echo "── OOD eval done ──"
ONSTART
)

# ── Pick cheapest available offer matching the query ─────────────────────
OFFER_QUERY="${VAST_GPU_QUERY} disk_space>=${VAST_DISK}"
echo "── searching offers: $OFFER_QUERY ──"
OFFERS="$(vastai search offers "$OFFER_QUERY" -o 'dph+' --raw 2>&1)"
case "$OFFERS" in \[*|\{*) : ;; *)
  echo "✗ vastai search failed:" >&2; printf '%s\n' "$OFFERS" | head -5 >&2; exit 1 ;;
esac
OFFER_ID="$(printf '%s' "$OFFERS" | "$PYTHON" -c "
import sys, json
offers = json.load(sys.stdin)
for o in offers:
    print(o.get('id', '')); break
" )"
[ -n "$OFFER_ID" ] || { echo "✗ no matching offer" >&2; exit 1; }
echo "── picked offer: $OFFER_ID ──"

# ── Create the instance ──────────────────────────────────────────────────
ENV_ARG="-e GITHUB=${GITHUB} -e HF_TOKEN=${HF_TOKEN:-}"
echo "── creating ood-eval instance (label neuroslm-ood-${ROLE_TAG}) ──"
vastai create instance "$OFFER_ID" \
  --image "$VAST_IMAGE" \
  --disk "$VAST_DISK" \
  --label "neuroslm-ood-${ROLE_TAG}" \
  --env "$ENV_ARG" \
  --onstart-cmd "$ONSTART" \
  --ssh
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  OOD eval instance launched for branch=${BRANCH}"
echo "  checkpoint=${CKPT}"
echo "  output=${OUTPUT_FILE} (pushed to branch when done)"
echo ""
echo "  Watch:    bash scripts/vast.sh logs <id>"
echo "  When done, pull locally:"
echo "    git fetch origin ${BRANCH}"
echo "    git show origin/${BRANCH}:${OUTPUT_FILE}"
echo "  Then destroy: bash scripts/vast.sh destroy instance <id>"
echo "════════════════════════════════════════════════════════════════"
