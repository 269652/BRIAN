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
#   2. Compile  architectures/master → dna/master/arch.dna
#   3. git add <dna> + commit "chore: recompile master architecture"
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

# ── Step 2/4: Compile master arch → canonical DNA ────────────────
echo "[pre-deploy] step 2/4: brian dna compile (architectures/master → dna/master/arch.dna)"
"$python" -m neuroslm.cli dna compile architectures/master --output dna/master/arch.dna

# ── Step 3/4: Stage + commit the DNA ────────────────────────────
echo "[pre-deploy] step 3/4: staging + committing recompiled DNA"
git add dna/master/arch.dna
if git diff --quiet --cached; then
    echo "[pre-deploy] no DNA diff — already up to date."
else
    git -c user.name="${BRIAN_GIT_NAME:-BRIAN pre-deploy hook}" \
        -c user.email="${BRIAN_GIT_EMAIL:-brian-pre-deploy@local}" \
        commit -m "chore: recompile master architecture"
fi

# ── Step 4/4: Push — only success here unblocks the deploy ───────
echo "[pre-deploy] step 4/4: pushing to origin"
git push

echo "[pre-deploy] ok — master → dna/master/arch.dna compiled, committed, pushed"
