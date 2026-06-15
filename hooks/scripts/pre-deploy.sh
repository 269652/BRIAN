#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# hooks/scripts/pre-deploy.sh — Linux / macOS impl of the pre-deploy hook.
#
# Symmetric with hooks/scripts/pre-deploy.ps1. Picked by the runner
# when platform.system() != "Windows".
#
# Pipeline (5 steps, fail-fast):
#   1. Refuse if the working tree is dirty (caller must commit / stash
#      first — otherwise the roundtrip commit would smuggle their WIP
#      under our chore message).
#   2. Compile  architectures/master → dna/master/arch.dna
#   3. Unfold   dna/master/arch.dna  → architectures/current/arch.neuro
#   4. git add -A + commit "chore: roundtrip recompile of current
#      architecture"  (no-op if compile/unfold produced byte-identical
#      output, e.g. you ran the hook twice in a row).
#   5. git push.    Only on a successful push does the hook return 0
#      → cmd_deploy proceeds to provision vast.ai.   Any non-zero exit
#      from steps 1–5 aborts the deploy with the same exit code.
#
# Invoked by neuroslm.hooks.run_hook("pre-deploy", REPO_ROOT). The
# runner sets BRIAN_HOOK_NAME, BRIAN_REPO_ROOT and streams output live.
# ──────────────────────────────────────────────────────────────────

set -euo pipefail

cd "${BRIAN_REPO_ROOT:-.}"

echo "[pre-deploy] cwd=$(pwd)"

# ── Step 1/5: Working tree must be clean ─────────────────────────
# `git status --porcelain` outputs one line per modified / untracked
# / staged path. Empty output ⇒ clean. We bail BEFORE compiling so a
# user with WIP edits doesn't have them quietly bundled into the
# roundtrip commit.
echo "[pre-deploy] step 1/5: checking working tree is clean"
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

# ── Step 2/5: Compile master arch → canonical DNA ────────────────
# Explicit paths (not brian.toml defaults) because the user's
# convention is:
#   architectures/master       = source-of-truth template (edit here)
#   dna/<branch>/arch.dna      = canonical compiled DNA the deploy
#                                trains on. Path mirrors the active
#                                git branch so a fork on feature/X
#                                never clobbers master's DNA.
#   architectures/current/     = live working copy (rewritten in step 3)
# Using `python -m neuroslm.cli` instead of the `brian` entry point so
# the hook works even when the package is checked out but not
# pip-installed.
echo "[pre-deploy] step 2/5: brian dna compile (architectures/master → dna/master/arch.dna)"
"$python" -m neuroslm.cli dna compile architectures/master --output dna/master/arch.dna

# ── Step 3/5: Unfold DNA back to working copy ────────────────────
# Keeps architectures/current/arch.neuro byte-identical to the DNA
# that the deploy will actually train on. Catches "I edited current/
# but forgot to compile" AND "I edited master/ but forgot to
# propagate to current/" in one shot.
echo "[pre-deploy] step 3/5: brian dna unfold (dna/master/arch.dna → architectures/current/arch.neuro)"
"$python" -m neuroslm.cli dna unfold dna/master/arch.dna --output architectures/current/arch.neuro

# ── Step 4/5: Stage + commit the roundtrip artefacts ─────────────
# `git diff --quiet --cached` after `git add -A` returns 0 when there
# are no staged changes (idempotent roundtrip → nothing to commit).
echo "[pre-deploy] step 4/5: staging + committing roundtrip artefacts"
git add -A
if git diff --quiet --cached; then
    echo "[pre-deploy] no roundtrip diff — DNA + current/ already in sync."
else
    git -c user.name="${BRIAN_GIT_NAME:-BRIAN pre-deploy hook}" \
        -c user.email="${BRIAN_GIT_EMAIL:-brian-pre-deploy@local}" \
        commit -m "chore: roundtrip recompile of current architecture"
fi

# ── Step 5/5: Push — only success here unblocks the deploy ───────
# If there was nothing to commit we still push to make sure the
# remote is at HEAD before the deploy box clones it. Push failure
# (network, hook, rebase conflict, …) propagates and aborts the
# deploy.
echo "[pre-deploy] step 5/5: pushing to origin"
git push

echo "[pre-deploy] ok — master → DNA → current synced, committed, pushed"
