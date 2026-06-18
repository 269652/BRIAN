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
BRANCH="${BRANCH:-$(git -C "$HERE" rev-parse --abbrev-ref HEAD 2>/dev/null || echo master)}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-20}"
FRESH="${FRESH:-1}"
# Push the running training log to git every N seconds. Defaults to 300s
# which works out to roughly every ~200 train steps at typical pace.
LOG_PUSH_INTERVAL="${LOG_PUSH_INTERVAL:-300}"
# USE_DSL=1 → train the architecture compiled from architectures/<ARCH>/
# via train_dsl.py + BRIANHarness instead of the hand-written Brain. Loss
# clipping etc. is configured in arch.neuro's `training { ... }` block.
USE_DSL="${USE_DSL:-0}"
# 2026-06-14: bowtie arch folder renamed rcc_bowtie → master (canonical),
# with architectures/current as the live working-copy `brian train`
# consumes by default. Match that here so vast.ai deploys land on the
# same arch.
ARCH="${ARCH:-current}"
# Training hyperparameters (STEPS, BATCH, GRAD_ACCUM, SEQ_LEN, D_SEM):
# the canonical defaults live in architectures/<ARCH>/arch.neuro's
# `training { ... }` block. Leave these unset here so vast_train_dsl_loop
# (USE_DSL=1) or vast_train_loop (Brain) falls back to the arch config.
# Pass an env override to force a specific value, e.g.
#   STEPS=5000 BATCH=16 bash scripts/vast_train.sh
STEPS="${2:-${STEPS:-}}"
BATCH="${BATCH:-}"
GRAD_ACCUM="${GRAD_ACCUM:-}"
SEQ_LEN="${SEQ_LEN:-}"
D_SEM="${D_SEM:-}"
# OOD_EVERY: mid-training WikiText-103 ppl snapshots every N steps.
# Toggle via `--ood [N]` flag (parsed below) or env override. 0 = off.
OOD_EVERY="${OOD_EVERY:-0}"

# ─── --ood [N] flag: turn on mid-training OOD eval every N steps ─────
# Pass `--ood` for the default 3000-step cadence, or `--ood 1000` etc.
_args_left=("$@")
_skip_next=0
shift_pos=2   # PRESET + STEPS were positional
for ((_i=shift_pos; _i<${#_args_left[@]}; _i++)); do
    if [ "$_skip_next" = "1" ]; then _skip_next=0; continue; fi
    case "${_args_left[$_i]}" in
        --ood)
            # Optional integer arg follows; default 3000.
            _next="${_args_left[$((_i+1))]:-}"
            if [[ "$_next" =~ ^[0-9]+$ ]]; then
                OOD_EVERY="$_next"
                _skip_next=1
            else
                OOD_EVERY=3000
            fi
            ;;
    esac
done

REPO_URL="${REPO_URL:-https://github.com/269652/BRIAN.git}"
REPO_SLUG="${REPO_URL#https://github.com/}"; REPO_SLUG="${REPO_SLUG%.git}"
VAST_IMAGE="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
VAST_DISK=60
# GPU filter: read from arch.neuro's `hardware {}` block. Falls back to a
# broad A100 filter if the arch doesn't declare hardware constraints.
# Override with GPU_QUERY env var if you need to force specific hardware.
if [ -z "${GPU_QUERY:-}" ]; then
  GPU_QUERY="$(python3 - "$ARCH" "${SCALE:-}" <<'PY' 2>/dev/null || echo ""
import sys
arch, scale = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""
try:
    from neuroslm.dsl.training_config import load_training_config_from_arch
    cfg = load_training_config_from_arch(f"architectures/{arch}")
    # If a scale variant has its own hardware block, use that.
    hw = cfg.hardware
    if scale and scale in (cfg.scales.variants or {}):
        v = cfg.scales.variants[scale]
        if hasattr(v, "hardware") and v.hardware and v.hardware.gpu_name:
            hw = v.hardware
    gpu = hw.gpu_name or "A100_SXM4"
    n = hw.num_gpus or 1
    rel = hw.min_reliability or 0.99
    mem = getattr(hw, "min_gpu_mem_gib", 0) or 0
    q = f"gpu_name={gpu} num_gpus={n} rentable=true verified=true reliability>{rel}"
    if mem > 0:
        q += f" gpu_ram>={mem}"
    print(q)
except Exception:
    print("")
PY
)"
  # Fallback if the python lookup failed or returned empty
  [ -z "$GPU_QUERY" ] && GPU_QUERY="gpu_name=A100_SXM4 num_gpus=1 rentable=true verified=true reliability>0.995 gpu_ram>=40"
fi

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
    scale   = ${SCALE:-<default from arch.neuro>}
    steps   = ${STEPS:-<from arch.neuro>}
    fresh   = $FRESH
    batch   = ${BATCH:-<from arch.neuro>} × grad_accum ${GRAD_ACCUM:-<from arch.neuro>}
    seq_len = ${SEQ_LEN:-<from arch.neuro>}
    use_dsl = $USE_DSL
    arch    = $ARCH
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
#
# BOOT_TIMESTAMP is computed ONCE here so the filename prefix and the
# train_dsl boot-stamp line in the log share the same UTC moment. Both
# the background pusher (below) and the final one-shot pusher (after
# training) must see the same value, otherwise the final commit would
# write to a different filename than the snapshots.
export BOOT_TIMESTAMP="\$(date -u +%Y%m%dT%H%M%SZ)"
echo "    BOOT_TIMESTAMP=\$BOOT_TIMESTAMP (used in log filename prefix)"
INSTANCE_ID="\$(hostname)" PUSH_INTERVAL='${LOG_PUSH_INTERVAL}' \\
    BRANCH='${BRANCH}' REPO_SLUG='${REPO_SLUG}' \\
    BOOT_TIMESTAMP="\$BOOT_TIMESTAMP" \\
    OOD_EVERY='${OOD_EVERY}' \\
    nohup bash scripts/log_pusher.sh > /workspace/log_pusher.log 2>&1 &
LOG_PUSHER_PID=\$!
echo "    log_pusher pid=\$LOG_PUSHER_PID"

echo "── starting training ──"
if [ "${USE_DSL}" = "1" ]; then
    echo "    DSL mode: arch=${ARCH} scale=${SCALE:-default} steps=${STEPS} batch=${BATCH} seq_len=${SEQ_LEN} d_sem=${D_SEM} ood_every=${OOD_EVERY}"
    ARCH='${ARCH}' SCALE='${SCALE:-}' STEPS='${STEPS}' BATCH='${BATCH}' \\
        SEQ_LEN='${SEQ_LEN}' D_SEM='${D_SEM}' \\
        SAVE_EVERY='${SAVE_EVERY}' LOG_EVERY='${LOG_EVERY}' \\
        OOD_EVERY='${OOD_EVERY}' \\
        bash scripts/vast_train_dsl_loop.sh 2>&1 | tee /workspace/train.log
else
    echo "    Brain mode: preset=${PRESET} steps=${STEPS} batch=${BATCH} grad_accum=${GRAD_ACCUM}"
    PRESET='${PRESET}' STEPS='${STEPS}' BATCH='${BATCH}' GRAD_ACCUM='${GRAD_ACCUM}' \\
        OPT=adamw SAVE_EVERY='${SAVE_EVERY}' LOG_EVERY='${LOG_EVERY}' \\
        FRESH='${FRESH}' \\
        bash scripts/vast_train_loop.sh 2>&1 | tee /workspace/train.log
fi

echo "── stopping log-pusher ──"
kill \$LOG_PUSHER_PID 2>/dev/null || true
sleep 2   # give it a moment to exit cleanly

# ── Final log push (one-shot) ──────────────────────────────────────
# Run a single iteration of the pusher loop with PUSH_INTERVAL=1 so it
# attempts exactly one commit+push of the complete train.log. The loop
# is unbounded so we wrap with timeout — but we want it to exit ASAP,
# not poll for 60s. Solution: send SIGTERM after the first cycle (~5s
# at worst on a successful push).
#
# Reuse the same BOOT_TIMESTAMP as the background pusher so the final
# commit writes to the same filename, not a fresh one.
SOURCE_LOG=/workspace/train.log INSTANCE_ID="\$(hostname)" \\
    PUSH_INTERVAL=1 BRANCH='${BRANCH}' REPO_SLUG='${REPO_SLUG}' \\
    BOOT_TIMESTAMP="\$BOOT_TIMESTAMP" \\
    timeout 30 bash scripts/log_pusher.sh 2>&1 | head -10 \\
    || echo "[onstart] final log push: timeout/exit (best-effort)"

# ── Final checkpoint push (DSL trainer doesn't push on save) ──────
# vast_train_dsl_loop's train_dsl.py saves checkpoints to lfs_checkpoints/
# locally but doesn't push them — unlike Brain's train.py. Push every
# dsl_arch_*.pt that landed during this run so the artefacts survive
# the instance destroy.
echo "── pushing final checkpoints ──"
cd /workspace/brian
ls -la lfs_checkpoints/dsl_arch_*.pt 2>/dev/null || echo "[onstart] no DSL checkpoints to push"
git config user.email "vast-train@brian.local" || true
git config user.name  "vast-train"             || true
for ckpt in lfs_checkpoints/dsl_arch_*.pt; do
    [ -e "\$ckpt" ] || continue
    git add "\$ckpt" 2>/dev/null || true
done
if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "chkpt(dsl): final push @ \$(date -u +%Y-%m-%dT%H:%M:%SZ)" \\
        >/dev/null 2>&1 || echo "[onstart] commit failed"
    PUSH_URL="https://x-access-token:\${GITHUB}@github.com/${REPO_SLUG}.git"
    timeout 600 git push "\$PUSH_URL" "HEAD:${BRANCH}" 2>&1 \\
        | sed "s#\${GITHUB}#***#g" \\
        || echo "[onstart] checkpoint push failed (will not block destroy)"
else
    echo "[onstart] no new checkpoints to commit"
fi

# ── Self-destroy the vast instance ────────────────────────────────
# Without this the container stays "running" after onstart exits and
# bills you indefinitely. Verified 2026-05-30 on 38469631 (8 hours
# idle after training completed, ~\$10 wasted). Uses the INSTANCE_ID
# env var that vast.ai injects into every container, fallback to
# looking ourselves up by container label.
echo "── self-destroying instance ──"
if ! command -v vastai >/dev/null 2>&1; then
    pip install -q vastai 2>&1 | tail -3 || true
fi
if [ -n "\${VAST_API_KEY:-}" ] && command -v vastai >/dev/null 2>&1; then
    vastai set api-key "\$VAST_API_KEY" >/dev/null 2>&1 || true
    # Vast injects \$CONTAINER_ID and \$VAST_CONTAINERLABEL; the contract
    # id is usually exposed as \$INSTANCE_ID. Try them in order.
    SELF_ID="\${INSTANCE_ID:-\${VAST_CONTAINER_ID:-}}"
    if [ -z "\$SELF_ID" ]; then
        # Fall back: find our instance by label match.
        SELF_ID="\$(vastai show instances --raw 2>/dev/null \\
            | python3 -c "import sys, json
data = json.load(sys.stdin)
for i in (data or []):
    if i.get('label') == 'neuroslm-full':
        print(i.get('id', '')); break" 2>/dev/null)"
    fi
    if [ -n "\$SELF_ID" ]; then
        echo "[onstart] vastai destroy instance \$SELF_ID"
        # `yes y` answers the interactive confirmation prompt (no -y
        # available in this vastai version). Without this the command
        # hangs waiting for stdin and the instance never destroys.
        yes y | vastai destroy instance "\$SELF_ID" 2>&1 || echo "[onstart] destroy failed"
        # The destroy command kills our container — anything past this
        # never executes. The echo below is reached only if destroy failed.
        sleep 30
    else
        echo "[onstart] could not determine vast instance id; not destroying"
    fi
else
    echo "[onstart] VAST_API_KEY not in env or vastai CLI missing — cannot self-destroy"
fi

echo "── training exited; FAILED to self-destroy. Run: vastai destroy instance <contract_id> ──"
ONSTART
)"

# ─── Create the instance ─────────────────────────────────────────────────
echo "── creating instance ──"
echo "  (this is a network call to vast.ai API; typically 5-30s, max 120s)"
# Stream output live to terminal AND capture for parsing. Without `tee` the
# raw `CREATE_OUT="$(...)"` capture buffers everything until the subshell
# exits, so a slow/hung create call shows zero feedback. Wrap with a 120s
# timeout so a true hang exits visibly instead of waiting forever.
_CREATE_TMP="$(mktemp -t vast_create.XXXXXX)"
timeout 120 vastai create instance "$OFFER_ID" \
    --image "$VAST_IMAGE" \
    --disk "$VAST_DISK" \
    --label "neuroslm-full" \
    --env "-e GITHUB=$GITHUB -e HF_TOKEN=${HF_TOKEN:-} -e VAST_API_KEY=$VAST_API_KEY" \
    --onstart-cmd "$ONSTART" 2>&1 \
    | sed -E "s#${GITHUB}#***#g" \
    | tee "$_CREATE_TMP"
_CREATE_RC=${PIPESTATUS[0]}
CREATE_OUT="$(cat "$_CREATE_TMP")"
rm -f "$_CREATE_TMP"
if [ "$_CREATE_RC" = "124" ]; then
    echo "✗ vastai create instance TIMED OUT after 120s" >&2
    exit 1
fi
    # Note: no --ssh. vast.ai /.launch spawns an ssh keepalive whenever
    # --ssh is set; the pytorch/pytorch image has no openssh-client so
    # /.launch spins on "ssh: command not found" forever and onstart-cmd
    # never runs (idle, billed). We don't need ssh — logs stream via
    # vastai logs and checkpoints push over HTTPS. Same fix as vast_deploy.sh.

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
