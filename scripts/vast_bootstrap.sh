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
# NOTE: do NOT set `git config lfs.fetchexclude '*'` here. git-lfs combines
# fetchexclude with any `--include` as (include AND NOT exclude), so a global
# '*' exclude silently wins and `git lfs pull --include=...` fetches NOTHING —
# the resume checkpoint then stays an unfetched pointer and training restarts
# from scratch. `git lfs install --skip-smudge` (above) + GIT_LFS_SKIP_SMUDGE=1
# on clone already prevent auto-downloading every blob, so the explicit pull
# below is the only fetch.

echo "── 4. Python dependencies ────────────────────────────────────────"
# torch should already be in the Vast PyTorch image; only install if missing.
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# transformers ONLY provides Adafactor (lazy import; not needed for adamw).
# Pin <5 so pip doesn't pull transformers 5.x, which requires torch>=2.4 and
# disables itself on the image's torch 2.3 (noisy + breaks Adafactor runs).
pip install "transformers>=4.40,<5"

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
# --exclude="" explicitly clears any inherited fetchexclude so --include wins.
git lfs pull --include="lfs_checkpoints/neuroslm_large_*adamw*" --exclude="" \
  || echo "  (no matching LFS objects yet — will train from scratch)"
# Verify the resume target is a real binary, not a leftover pointer.
for _pt in "$REPO_DIR"/lfs_checkpoints/neuroslm_large_*adamw*.pt; do
  [ -e "$_pt" ] || continue
  if head -c 48 "$_pt" 2>/dev/null | grep -q "version https://git-lfs"; then
    echo "  ⚠ still a pointer: $(basename "$_pt") — retrying targeted pull"
    git lfs pull --include="lfs_checkpoints/$(basename "$_pt")" --exclude="" || true
  fi
done
echo "  resume checkpoints present:"
ls -la "$REPO_DIR"/lfs_checkpoints/neuroslm_large_*adamw*.pt 2>/dev/null \
  | awk '{print "    " $5, $NF}' || echo "    (none)"

echo ""
echo "✓ Bootstrap complete. Repo at: $REPO_DIR"
echo "  Next: bash scripts/vast_train_loop.sh"
