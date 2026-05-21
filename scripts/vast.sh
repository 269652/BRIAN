#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# vast.sh — thin, cross-platform wrapper around the vastai CLI.
#
# Resolves a python that has the vastai package (preferring your active venv)
# and invokes the CLI via its import entry point — so it works in Windows
# git-bash, Colab, and Linux without PATH/.exe headaches.
#
# Usage (always via bash, even from PowerShell):
#   bash scripts/vast.sh show instances
#   bash scripts/vast.sh logs 37236492
#   bash scripts/vast.sh destroy instance 37236489
#   bash scripts/vast.sh search offers "gpu_name in [A100_SXM4,A100_PCIE] num_gpus=1 rentable=true verified=true" -o dph+ --raw
#
# Reads VAST_API_KEY / VAST_AI from .env (so the key is already set); pass
# --no-auth to skip that if you've set the key another way.
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$HERE/.env}"
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }
VAST_API_KEY="${VAST_API_KEY:-${VAST_AI:-}}"

_norm_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -u "$1" 2>/dev/null || printf '%s' "${1//\\//}"
  else printf '%s' "${1//\\//}"; fi
}
_pick_python() {
  local c ve
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    ve="$(_norm_path "$VIRTUAL_ENV")"
    for c in "$ve/Scripts/python.exe" "$ve/Scripts/python" "$ve/bin/python"; do
      [ -x "$c" ] && "$c" -c 'import vastai.cli.main' >/dev/null 2>&1 && { printf '%s' "$c"; return; }
    done
  fi
  # any python that can import vastai
  for c in python python.exe python3 py; do
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" -c 'import vastai.cli.main' >/dev/null 2>&1 && { printf '%s' "$c"; return; }
  done
  # fall back to one with pip (so we can install)
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    ve="$(_norm_path "$VIRTUAL_ENV")"
    for c in "$ve/Scripts/python.exe" "$ve/bin/python"; do
      [ -x "$c" ] && "$c" -m pip --version >/dev/null 2>&1 && { printf '%s' "$c"; return; }
    done
  fi
  for c in python python.exe python3 py; do
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" -m pip --version >/dev/null 2>&1 && { printf '%s' "$c"; return; }
  done
  printf ''
}

PYTHON="$(_pick_python)"
[ -z "$PYTHON" ] && { echo "✗ no python with vastai/pip found." >&2; exit 1; }

if ! "$PYTHON" -c 'import vastai.cli.main' >/dev/null 2>&1; then
  echo "── installing vastai (via $PYTHON -m pip) ──" >&2
  "$PYTHON" -m pip install -q --upgrade vastai
fi

vastai() { "$PYTHON" -c 'import sys; from vastai.cli.main import main; sys.exit(main())' "$@"; }

# Auth from .env unless told not to.
if [ "${1:-}" = "--no-auth" ]; then
  shift
elif [ -n "$VAST_API_KEY" ]; then
  vastai set api-key "$VAST_API_KEY" >/dev/null 2>&1 || true
fi

vastai "$@"
