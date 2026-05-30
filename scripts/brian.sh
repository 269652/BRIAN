#!/usr/bin/env bash
# brian — unified CLI for the NeuroSLM project.
#
# Thin wrapper: forwards every arg to `py -3 -m neuroslm.cli` (Windows) /
# `python -m neuroslm.cli` (Linux/macOS). Run from anywhere — the cli
# resolves the repo root automatically.
#
# Examples:
#   bash scripts/brian.sh status
#   bash scripts/brian.sh compile architectures/rcc_bowtie --out arch.py
#   bash scripts/brian.sh wolfram architectures/rcc_bowtie --full --out arch.m
#   bash scripts/brian.sh analyze architectures/rcc_bowtie --all
#   bash scripts/brian.sh deploy --steps 10000
#   bash scripts/brian.sh deploy-100k
#   bash scripts/brian.sh logs 38569395
#   bash scripts/brian.sh destroy --all
#   bash scripts/brian.sh ood lfs_checkpoints/dsl_arch_step10000.pt
#   bash scripts/brian.sh test
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

# Pick the right python — prefer `py -3` on Windows where the launcher
# routes to the active 3.x install, else `python3` / `python`.
if command -v py >/dev/null 2>&1 && py -3 -c "import sys" >/dev/null 2>&1; then
    PY=(py -3)
elif command -v python3 >/dev/null 2>&1; then
    PY=(python3)
elif command -v python >/dev/null 2>&1; then
    PY=(python)
else
    echo "✗ no python interpreter found on PATH" >&2
    exit 1
fi

exec "${PY[@]}" -m neuroslm.cli "$@"
