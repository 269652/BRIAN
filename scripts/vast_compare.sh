#!/usr/bin/env bash
# vast_compare.sh — one-liner training-progress check for neuroslm-* instances.
# Tails the recent step/loss lines from each.
#
# Usage:  bash scripts/vast_compare.sh         # last 50 lines per instance
#         bash scripts/vast_compare.sh 100     # last 100 lines per instance
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VAST="$HERE/scripts/vast.sh"
N="${1:-50}"

_norm() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -u "$1" 2>/dev/null || printf '%s' "${1//\\//}"
  else printf '%s' "${1//\\//}"; fi
}
_pick_py() {
  local c ve
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    ve="$(_norm "$VIRTUAL_ENV")"
    for c in "$ve/Scripts/python.exe" "$ve/Scripts/python" "$ve/bin/python"; do
      [ -x "$c" ] && "$c" -m pip --version >/dev/null 2>&1 && { printf '%s' "$c"; return; }
    done
  fi
  for ve in "$HERE"/.venv*; do
    [ -d "$ve" ] || continue
    for c in "$ve/Scripts/python.exe" "$ve/bin/python"; do
      [ -x "$c" ] && "$c" -m pip --version >/dev/null 2>&1 && { printf '%s' "$c"; return; }
    done
  done
  for c in python python.exe python3 py; do
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" -m pip --version >/dev/null 2>&1 && { printf '%s' "$c"; return; }
  done
  printf ''
}
PY="$(_pick_py)"
[ -z "$PY" ] && { echo "✗ no python with pip found" >&2; exit 1; }

# Fetch instances JSON via the vast.sh wrapper (it handles auth + CLI path).
JSON="$("$VAST" show instances --raw 2>/dev/null || true)"
if [ -z "$JSON" ]; then
  echo "✗ couldn't fetch instances. Check: bash $VAST show instances" >&2
  exit 1
fi

# Parse neuroslm-* rows as label<TAB>id<TAB>status.
ROWS="$(printf '%s' "$JSON" | "$PY" -c "
import sys, json
try: data = json.load(sys.stdin)
except Exception: data = []
for i in data or []:
    lbl = i.get('label') or ''
    if not lbl.startswith('neuroslm-'): continue
    st = i.get('actual_status') or i.get('cur_state') or 'unknown'
    print('%s\t%s\t%s' % (lbl, i.get('id',''), st))
")"
[ -z "$ROWS" ] && { echo "no neuroslm-* instances found." >&2; exit 1; }

# Show recent training-relevant lines for each instance.
while IFS=$'\t' read -r label id status; do
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  $label  (id=$id, status=$status)"
  echo "════════════════════════════════════════════════════════════════"
  ROLE="${label#neuroslm-}"
  # New onstart streams training via tee → `vastai logs` has the step lines.
  # Older onstart redirected to file → grab via `vastai execute`.
  OUT="$("$VAST" logs "$id" 2>&1 || true)"
  if ! printf '%s' "$OUT" | grep -qE 'step +[0-9]+ *\|'; then
    OUT="$("$VAST" execute "$id" "tail -n $N /workspace/train_${ROLE}.log" 2>&1 || true)"
  fi
  printf '%s\n' "$OUT" | tail -n "$N" \
    | grep -E 'step +[0-9]+|loss|Brain topology|TOTAL|launching|using python|using vastai|awakening|✓|⚠' \
    | tail -n "$N"
done <<< "$ROWS"
