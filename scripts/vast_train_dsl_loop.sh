#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# vast_train_dsl_loop.sh — crash-resilient training loop for DSL-driven runs.
#
# Mirrors vast_train_loop.sh but invokes `neuroslm.train_dsl` instead of
# `neuroslm.train`. The model is built from `architectures/<ARCH>/` via
# the DSL compiler + BRIAN harness, with loss clipping and other
# pipeline behavior read from arch.neuro's `training { ... }` block.
#
# Tunable env vars (with defaults):
#   ARCH=rcc_bowtie                  architecture folder name under architectures/
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

ARCH="${ARCH:-rcc_bowtie}"
STEPS="${STEPS:-10000}"
BATCH="${BATCH:-4}"
SEQ_LEN="${SEQ_LEN:-256}"
D_SEM="${D_SEM:-256}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
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
        --steps "$STEPS" \
        --batch "$BATCH" \
        --seq_len "$SEQ_LEN" \
        --d_sem "$D_SEM" \
        --device cuda \
        --log_every "$LOG_EVERY" \
        --save_every "$SAVE_EVERY" \
        --ckpt_dir "$CKPT_DIR"
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
