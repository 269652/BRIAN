#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# vast_bootstrap.sh — one-time setup of NeuroSLM/BRIAN on a Vast.ai instance
#
# Run this ONCE after SSHing into a fresh Vast.ai GPU instance (use a
# PyTorch/CUDA template, e.g. "pytorch/pytorch:2.x-cuda12.1-cudnn8-runtime").
#
# Required env vars (export before running, or edit the defaults below):
#   GITHUB        GitHub personal-access-token with repo write (for LFS push)
# Optional:
#   HF_TOKEN      HuggingFace token (faster dataset downloads, higher limits)
#   REPO_URL      defaults to the BRIAN repo
#   REPO_DIR      defaults to /workspace/brian
#   GIT_NAME / GIT_EMAIL  commit identity for checkpoint pushes
#
# Usage:
#   export GITHUB=ghp_xxx
#   export HF_TOKEN=hf_xxx          # optional
#   bash scripts/vast_bootstrap.sh
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/269652/BRIAN.git}"
REPO_DIR="${REPO_DIR:-/workspace/brian}"
GIT_NAME="${GIT_NAME:-NeuroSLM Train}"
GIT_EMAIL="${GIT_EMAIL:-train@neuroslm}"

echo "── 1. System packages (git, git-lfs) ─────────────────────────────"
if ! command -v git-lfs >/dev/null 2>&1; then
  apt-get update -y && apt-get install -y git git-lfs
fi
git lfs install --skip-smudge       # don't pull every old checkpoint blob

echo "── 2. GitHub credentials ─────────────────────────────────────────"
if [ -z "${GITHUB:-}" ]; then
  echo "✗ GITHUB token not set. export GITHUB=ghp_... and rerun." >&2
  exit 1
fi
# Persisted credential store — train.py's auto-push reads ~/.git-credentials
git config --global credential.helper store
printf 'https://x-access-token:%s@github.com\n' "$GITHUB" > ~/.git-credentials
chmod 600 ~/.git-credentials
git config --global user.name  "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"

echo "── 3. Clone repo (LFS pointers only) ─────────────────────────────"
if [ ! -d "$REPO_DIR/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git config lfs.fetchexclude '*'     # never auto-download checkpoint blobs

echo "── 4. Python dependencies ────────────────────────────────────────"
# torch should already be in the Vast PyTorch image; only install if missing.
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "transformers>=4.40"    # for Adafactor; harmless if --optimizer adamw

echo "── 5. HuggingFace token (optional) ───────────────────────────────"
if [ -n "${HF_TOKEN:-}" ]; then
  mkdir -p ~/.huggingface
  printf '%s' "$HF_TOKEN" > ~/.huggingface/token
  export HF_TOKEN
  echo "  HF token written."
else
  echo "  (no HF_TOKEN — dataset downloads are unauthenticated/slower)"
fi

echo "── 6. Pull the checkpoint we resume from (if any) ────────────────"
mkdir -p "$REPO_DIR/lfs_checkpoints"
# Materialise ONLY the large adamw checkpoints we resume from (not all blobs).
git lfs pull --include="lfs_checkpoints/neuroslm_large_*adamw*" 2>/dev/null \
  || echo "  (no matching LFS objects yet — will train from scratch)"

echo ""
echo "✓ Bootstrap complete. Repo at: $REPO_DIR"
echo "  Next: bash scripts/vast_train_loop.sh"
