#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# vast_train_dsl_loop.sh — crash-resilient training loop for DSL-driven runs.
#
# Mirrors vast_train_loop.sh but invokes `neuroslm.train_dsl` instead of
# `neuroslm.train`. The model is built from `architectures/<ARCH>/` via
# the DSL compiler + BRIAN harness, with loss clipping and other
# pipeline behavior read from arch.neuro's `training { ... }` block.
#
# Source of truth for ARCH (highest priority first):
#   1. ARCH env var (legacy / CI overrides)
#   2. brian.toml [current].arch  (workspace config)
#   3. hardcoded fallback: "current"  (2026-06-14: was "rcc_bowtie";
#      the bowtie folder was renamed to master/, with current/ as the
#      live working-copy `brian train` consumes by default)
#
# Tunable env vars (with defaults):
#   ARCH=<name>                      architecture folder under architectures/
#                                     (auto-resolved from brian.toml if unset)
#   STEPS=10000                      target step count
#   BATCH=4 SEQ_LEN=256 D_SEM=256
#   LOG_EVERY=20 SAVE_EVERY=1000
#   CKPT_DIR=$REPO_DIR/lfs_checkpoints
#   MAX_RESTARTS=1000
#
# Resume semantics: the harness's checkpoint format is independent of
# Brain's. On restart we load the most recent dsl_arch_step*.pt from
# CKPT_DIR if present.
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
cd "$REPO_DIR"

# Resolve ARCH from brian.toml if not set in the environment.
if [ -z "${ARCH:-}" ]; then
    ARCH="$(python3 - <<'PY' 2>/dev/null || echo current
from neuroslm.project_config import load_project_config
cfg = load_project_config()
# arch field is "architectures/<name>" — extract the leaf
print(cfg.arch.split("/")[-1] if "/" in cfg.arch else cfg.arch)
PY
)"
fi
ARCH="${ARCH:-current}"
PRESET="${PRESET:-rcc_bowtie_30m_p4}"   # sizes the DSL LM trunk to match P4

# Read runtime hyperparameters from the architecture's `training { ... }`
# block in arch.neuro. The arch declares its own training conditions so
# the DSL trunk runs under the same setup Brain uses for the same arch —
# no need to remember/sync defaults across scripts.
_arch_root="architectures/${ARCH}"
_arch_default() {
    # Usage: _arch_default <attr> <fallback>
    python3 - "$_arch_root" "$1" "$2" <<'PY' 2>/dev/null || echo "$2"
import sys
arch_root, attr, fallback = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    from neuroslm.dsl.training_config import load_training_config_from_arch
    cfg = load_training_config_from_arch(arch_root)
    print(getattr(cfg, attr, fallback))
except Exception:
    print(fallback)
PY
}
STEPS="${STEPS:-$(_arch_default steps 10000)}"
BATCH="${BATCH:-$(_arch_default batch_size 4)}"
SEQ_LEN="${SEQ_LEN:-$(_arch_default seq_len 1024)}"
D_SEM="${D_SEM:-384}"                   # P4 d_hidden (overridden by PRESET)
DATA="${DATA:-real}"
MODE="${MODE:-mix}"
CHAT_RATIO="${CHAT_RATIO:-0.6}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
# OOD_EVERY > 0 → run a quick WikiText-103 ppl snapshot every N steps
# during training. Each snapshot writes a JSON to logs/vast/benchmarks/ood/
# that brian analyze-log picks up. Defaults to 0 (off).
OOD_EVERY="${OOD_EVERY:-0}"
CKPT_DIR="${CKPT_DIR:-$REPO_DIR/lfs_checkpoints}"
MAX_RESTARTS="${MAX_RESTARTS:-1000}"

mkdir -p "$CKPT_DIR"

ARCH_PATH="architectures/$ARCH"
if [ ! -f "$ARCH_PATH/arch.neuro" ]; then
    echo "✗ missing $ARCH_PATH/arch.neuro — is the architecture folder there?" >&2
    exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo "  DSL-driven training loop"
echo "  arch=$ARCH (folder: $ARCH_PATH)"
echo "  steps=$STEPS batch=$BATCH seq_len=$SEQ_LEN d_sem=$D_SEM"
echo "  ckpt_dir=$CKPT_DIR"
echo "════════════════════════════════════════════════════════════════"

restart=0
while [ "$restart" -lt "$MAX_RESTARTS" ]; do
    echo ""
    echo "▶ launch attempt $((restart+1)) @ $(date -u +%H:%M:%SZ)"

    python -u -m neuroslm.train_dsl \
        --arch "$ARCH_PATH" \
        --model dsl_lm \
        --preset "$PRESET" \
        --data "$DATA" \
        --mode "$MODE" \
        --chat_ratio "$CHAT_RATIO" \
        --steps "$STEPS" \
        --batch "$BATCH" \
        --seq_len "$SEQ_LEN" \
        --d_sem "$D_SEM" \
        --device cuda \
        --log_every "$LOG_EVERY" \
        --save_every "$SAVE_EVERY" \
        --ood_every "$OOD_EVERY" \
        --ckpt_dir "$CKPT_DIR" \
        --resume
    rc=$?

    if [ "$rc" -eq 0 ]; then
        echo "✓ training reached target ($STEPS steps). Done."
        break
    fi

    echo "⚠ training exited with code $rc — restarting in 15s..."
    restart=$((restart+1))
    sleep 15
done

if [ "$restart" -ge "$MAX_RESTARTS" ]; then
    echo "✗ hit MAX_RESTARTS=$MAX_RESTARTS; giving up." >&2
    exit 1
fi
