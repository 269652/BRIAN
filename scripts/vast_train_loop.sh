#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# vast_train_loop.sh — crash-resilient long training run on Vast.ai
#
# Runs neuroslm.train in a loop: if the process dies (OOM, transient CUDA
# error, preempted spot instance) it auto-restarts and `--resume latest`
# picks up from the most recent checkpoint. The loop exits once the target
# step count is reached. Checkpoints auto-push to Git LFS every --save_every
# (handled inside train.py) so progress survives an instance teardown.
#
# Tunable env vars (with defaults):
#   PRESET=large STEPS=200000 BATCH=4 GRAD_ACCUM=4 OPT=adamw
#   SAVE_EVERY=1000 LOG_EVERY=20 MODE=mix CHAT_RATIO=0.6
#   CKPT_DIR=$REPO_DIR/lfs_checkpoints  EXTRA_ARGS=""
#
# Usage:
#   cd /workspace/brian
#   PRESET=large STEPS=200000 bash scripts/vast_train_loop.sh
#   # background + logfile:
#   PRESET=xl STEPS=300000 nohup bash scripts/vast_train_loop.sh > train.log 2>&1 &
#   tail -f train.log
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
cd "$REPO_DIR"

PRESET="${PRESET:-large}"
STEPS="${STEPS:-200000}"
BATCH="${BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
OPT="${OPT:-adamw}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-20}"
MODE="${MODE:-mix}"
CHAT_RATIO="${CHAT_RATIO:-0.6}"
CKPT_DIR="${CKPT_DIR:-$REPO_DIR/lfs_checkpoints}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MAX_RESTARTS="${MAX_RESTARTS:-1000}"

mkdir -p "$CKPT_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  NeuroSLM training loop"
echo "  preset=$PRESET steps=$STEPS batch=$BATCH grad_accum=$GRAD_ACCUM"
echo "  optimizer=$OPT  ckpt_dir=$CKPT_DIR"
echo "  effective tokens/step ≈ batch*grad_accum*ctx"
echo "════════════════════════════════════════════════════════════════"

restart=0
while [ "$restart" -lt "$MAX_RESTARTS" ]; do
  echo ""
  echo "▶ launch attempt $((restart+1)) @ $(date -u +%H:%M:%SZ)"
  python -u -m neuroslm.train \
    --preset "$PRESET" --steps "$STEPS" \
    --batch_size "$BATCH" --grad_accum "$GRAD_ACCUM" \
    --optimizer "$OPT" \
    --ckpt_dir "$CKPT_DIR" --device cuda \
    --mode "$MODE" --chat_ratio "$CHAT_RATIO" \
    --save_every "$SAVE_EVERY" --log_every "$LOG_EVERY" \
    --resume latest $EXTRA_ARGS
  rc=$?

  if [ "$rc" -eq 0 ]; then
    echo "✓ training reached target ($STEPS steps). Done."
    break
  fi

  echo "⚠ training exited with code $rc — restarting in 15s (resume latest)..."
  restart=$((restart+1))
  sleep 15
done

if [ "$restart" -ge "$MAX_RESTARTS" ]; then
  echo "✗ hit MAX_RESTARTS=$MAX_RESTARTS; giving up." >&2
  exit 1
fi
