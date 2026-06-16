#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# hooks/scripts/pre-deploy.sh — Linux / macOS impl of the pre-deploy hook.
#
# Symmetric with hooks/scripts/pre-deploy.ps1. Picked by the runner
# when platform.system() != "Windows".
#
# Pipeline (4 steps, fail-fast):
#   1. Refuse if the working tree is dirty (caller must commit / stash
#      first — otherwise the roundtrip commit would smuggle their WIP
#      under our chore message).
#   2. Compile the brian.toml [current].arch → dna/<leaf>/arch.dna
#   3. git add <dna> + commit "chore: recompile <arch> architecture"
#      (no-op if compile produced byte-identical output).
#   4. git push.    Only on a successful push does the hook return 0
#      → cmd_deploy proceeds to provision vast.ai.   Any non-zero exit
#      from steps 1–4 aborts the deploy with the same exit code.
#
# Invoked by neuroslm.hooks.run_hook("pre-deploy", REPO_ROOT). The
# runner sets BRIAN_HOOK_NAME, BRIAN_REPO_ROOT and streams output live.
# ──────────────────────────────────────────────────────────────────

set -euo pipefail

cd "${BRIAN_REPO_ROOT:-.}"

echo "[pre-deploy] cwd=$(pwd)"

# ── Step 1/4: Working tree must be clean ─────────────────────────
echo "[pre-deploy] step 1/4: checking working tree is clean"
dirty=$(git status --porcelain)
if [ -n "$dirty" ]; then
    echo "[pre-deploy] ERROR: working tree is dirty. Commit or stash first:" >&2
    echo "$dirty" >&2
    exit 1
fi

# Resolve python. CLAUDE.md §13 pins everything to ./.venv.
if [ -x .venv/bin/python ]; then
    python=".venv/bin/python"
else
    python="python"
fi

# ── Resolve arch from brian.toml [current].arch ──────────────────
# Read the arch path from brian.toml so the hook always compiles
# whatever the workspace is configured to deploy — not a hardcoded
# folder. Falls back to "architectures/master" for back-compat.
arch_path=$("$python" -c "
from neuroslm.project_config import load_project_config
cfg = load_project_config()
print(cfg.arch)
" 2>/dev/null || echo "architectures/master")
arch_leaf="${arch_path##*/}"
dna_path="dna/${arch_leaf}/arch.dna"

# ── Step 2/4: Compile arch → canonical DNA ───────────────────────
echo "[pre-deploy] step 2/4: brian dna compile ($arch_path → $dna_path)"
"$python" -m neuroslm.cli dna compile "$arch_path" --output "$dna_path"

# ── Step 3/4: Stage + commit the DNA ────────────────────────────
echo "[pre-deploy] step 3/4: staging + committing recompiled DNA"
git add "$dna_path"
if git diff --quiet --cached; then
    echo "[pre-deploy] no DNA diff — already up to date."
else
    git -c user.name="${BRIAN_GIT_NAME:-BRIAN pre-deploy hook}" \
        -c user.email="${BRIAN_GIT_EMAIL:-brian-pre-deploy@local}" \
        commit -m "chore: recompile $arch_leaf architecture"
fi

# ── Step 4/4: Push — only success here unblocks the deploy ───────
echo "[pre-deploy] step 4/4: pushing to origin"
git push

echo "[pre-deploy] ok — $arch_leaf → $dna_path compiled, committed, pushed"
