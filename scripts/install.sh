#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# install.sh — one-shot setup for the BRIAN repo.
#
# 1. Creates ./.venv (if missing) using the best available Python 3.
# 2. Activates it and upgrades pip.
# 3. pip install -e .       → puts the `brian` CLI on PATH
# 4. pip install -e ".[ml]" → adds torch + transformers + datasets for
#                              actual training/eval (skip with --no-ml).
#
# Usage:
#   bash scripts/install.sh            # full install incl. ML deps
#   bash scripts/install.sh --no-ml    # CLI-only (lighter, no torch)
#
# After install, activate the venv:
#   source .venv/bin/activate          # Linux / macOS / git-bash
#   .venv\\Scripts\\Activate.ps1       # PowerShell
# Then:
#   brian --help
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

NO_ML=0
for arg in "$@"; do
    case "$arg" in
        --no-ml) NO_ML=1 ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# //' | head -25
            exit 0 ;;
    esac
done

# ── 1. Pick a Python 3 interpreter ──────────────────────────────────────
_pick_py() {
    # Prefer py launcher on Windows (handles the 'python' = py2.7 mess)
    if command -v py >/dev/null 2>&1; then
        if py -3 -c "import sys; assert sys.version_info >= (3,10)" 2>/dev/null; then
            echo "py -3"; return
        fi
    fi
    for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
        if command -v "$c" >/dev/null 2>&1; then
            if "$c" -c "import sys; assert sys.version_info >= (3,10)" 2>/dev/null; then
                echo "$c"; return
            fi
        fi
    done
    echo ""
}
PY_CMD="$(_pick_py)"
if [ -z "$PY_CMD" ]; then
    echo "✗ no Python 3.10+ found on PATH" >&2
    echo "  install python.org Python 3.11+ then re-run" >&2
    exit 1
fi
echo "── using python: $PY_CMD ──"

# ── 2. Create venv if missing ────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "── creating venv at $HERE/.venv ──"
    $PY_CMD -m venv .venv
else
    echo "── reusing existing venv at $HERE/.venv ──"
fi

# Resolve venv python (Windows uses Scripts/, *nix uses bin/)
if [ -f ".venv/Scripts/python.exe" ]; then
    VPY=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    VPY=".venv/bin/python"
else
    echo "✗ venv created but python missing — corrupt install" >&2; exit 1
fi

# ── 3. Upgrade pip + install ────────────────────────────────────────────
echo "── upgrading pip ──"
$VPY -m pip install -q --upgrade pip

echo "── installing brian + CLI deps (pyproject.toml) ──"
$VPY -m pip install -q -e .

if [ "$NO_ML" -eq 0 ]; then
    echo "── installing heavy ML deps (torch, transformers, datasets) ──"
    echo "    this can take 5-10 min — pass --no-ml to skip"
    $VPY -m pip install -q -e ".[ml]"
fi

# ── 4. Smoke test the brian CLI ─────────────────────────────────────────
echo ""
echo "── verifying install ──"
if $VPY -m neuroslm.cli --help >/dev/null 2>&1; then
    echo "✓ brian CLI importable"
else
    echo "✗ brian CLI not importable — install failed" >&2
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✓ install complete"
echo ""
echo "  Activate the venv:"
echo "    bash:        source .venv/bin/activate"
echo "    git-bash:    source .venv/Scripts/activate"
echo "    PowerShell:  .venv\\Scripts\\Activate.ps1"
echo ""
echo "  Then run:    brian --help"
echo "               brian ps          # list active vast instances"
echo "               brian deploy      # launch a training run"
echo "════════════════════════════════════════════════════════════════"
