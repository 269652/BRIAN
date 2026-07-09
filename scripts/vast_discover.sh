#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# vast_discover.sh — DEPLOY ONE `brian discover <mode>` RUN TO VAST.AI.
#
# Sibling of vast_train.sh, for discovery jobs instead of training. Not a
# modification of vast_train.sh — same reasoning that script's own docstring
# gives for not modifying vast_deploy.sh: a discover job has a genuinely
# different shape (mode + search params, not arch/scale/steps), and the
# well-tested training launcher is not the place to bolt on a second job type.
#
# NOT meant to be invoked directly — call `brian deploy-discover <mode>`,
# which builds the onstart script in Python (neuroslm/connectors/
# vast_discover.py) and sets ONSTART_FILE before running this.
#
# Required in .env (or environment):
#   GH_TOKEN            - GitHub PAT with repo write access
#   VAST_API_KEY         - vast.ai API key
#   HF_TOKEN            - optional, for HuggingFace downloads
#
# Optional env overrides:
#   GPU_QUERY   - vast offer filter; default is a single A100 (see below)
#   VAST_LABEL  - instance label (default neuroslm-discover)
#   VAST_DISK   - disk size GB (default 30 — no checkpoint corpus needed)
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$HERE/.env"
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }

VAST_API_KEY="${VAST_API_KEY:-${VAST_AI:-}}"
GH_TOKEN="${GH_TOKEN:-${GITHUB:-${GITHUB_PAT:-}}}"
: "${VAST_API_KEY:?✗ set VAST_API_KEY in .env}"
: "${GH_TOKEN:?✗ set GH_TOKEN in .env}"

VAST_LABEL="${VAST_LABEL:-neuroslm-discover}"
VAST_IMAGE="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
VAST_DISK="${VAST_DISK:-30}"
# A100 by default — a lower tier (e.g. RTX_3090) can land on a slow/congested
# host and spend most of the rental on image pull + bootstrap rather than the
# actual discover run. Override with GPU_QUERY for a cheaper card if desired.
GPU_QUERY="${GPU_QUERY:-gpu_name=A100_SXM4 num_gpus=1 rentable=true verified=true reliability>0.98}"

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

cat <<HDR

════════════════════════════════════════════════════════════════
  vast_discover.sh
    label   = $VAST_LABEL
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

trace() { printf '[stage] %s\n' "$*" >&2; }
trace "offer selected — loading onstart script"

# ─── Load the onstart script (runs INSIDE the vast container) ────────────
# VastDiscoverConnector.launch() wrote the fully-expanded onstart script to
# a temp file and passed its path via ONSTART_FILE (same pipe-buffer-deadlock
# avoidance as vast_train.sh — see neuroslm/connectors/vast.py's docstring).
: "${ONSTART_FILE:?ONSTART_FILE must be set — use brian deploy-discover, not this script directly}"
ONSTART=""
while IFS= read -r _onstart_line || [ -n "$_onstart_line" ]; do
    ONSTART="${ONSTART}${_onstart_line}"$'\n'
done < "$ONSTART_FILE"
unset _onstart_line
[ -n "$ONSTART" ] || { printf '✗ ONSTART empty — ONSTART_FILE=%s\n' "$ONSTART_FILE" >&2; exit 1; }
trace "onstart script loaded (${#ONSTART} chars)"

# ─── Create the instance ─────────────────────────────────────────────────
trace "calling: vastai create instance $OFFER_ID --image $VAST_IMAGE --disk $VAST_DISK"
echo "── creating instance ──"
_CREATE_TMP="$(mktemp -t vast_create.XXXXXX)"
trace "create call starting (live output below) ──"
PYTHONUNBUFFERED=1 timeout 120 "$PYTHON" -u -c \
    'import sys; from vastai.cli.main import main; sys.exit(main())' \
    create instance "$OFFER_ID" \
    --image "$VAST_IMAGE" \
    --disk "$VAST_DISK" \
    --label "$VAST_LABEL" \
    --env "-e GH_TOKEN=$GH_TOKEN -e HF_TOKEN=${HF_TOKEN:-} -e VAST_API_KEY=$VAST_API_KEY" \
    --onstart-cmd "$ONSTART" 2>&1 \
    | sed -E "s#${GH_TOKEN}#***#g" \
    | tee "$_CREATE_TMP"
_CREATE_RC=${PIPESTATUS[0]}
CREATE_OUT="$(cat "$_CREATE_TMP")"
rm -f "$_CREATE_TMP"
trace "create call exited rc=$_CREATE_RC"
if [ "$_CREATE_RC" = "124" ]; then
    echo "✗ vastai create instance TIMED OUT after 120s" >&2
    exit 1
fi
# Note: no --ssh — see vast_train.sh's comment on the same line; the
# pytorch/pytorch image has no openssh-client and --ssh spins forever.

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
  printf '%s\n' "$CREATE_OUT" | sed -E "s#${GH_TOKEN}#***#g" | head -10 >&2
  exit 1
fi

printf '%s\n' "$CREATE_OUT" | sed -E "s#${GH_TOKEN}#***#g" | grep -E "success|new_contract" | head -2

cat <<DONE

✓ launched discover instance $INST_ID (machine $MACHINE_ID, $GPU_NAME, \$$DPH/hr)

Follow-up commands:

  Watch boot + discover logs:
    bash scripts/vast.sh logs $INST_ID

  Check status + cost:
    bash scripts/vast.sh show instance $INST_ID

  Destroy early if needed (the run self-destroys on completion):
    yes | bash scripts/vast.sh destroy instance $INST_ID

  Pull discovered artifacts as they land:
    git fetch origin && git pull

Instance is now booting (~2-5 min for image pull + pip install). The
discover run starts once bootstrap completes; it pushes its log,
modulations/, and the search ledger every push interval WHILE running,
then self-destroys when the run finishes.

DONE
