# ──────────────────────────────────────────────────────────────────
# hooks/scripts/pre-deploy.ps1 — Windows PowerShell impl.
#
# Symmetric with hooks/scripts/pre-deploy.sh. Picked by the runner
# when platform.system() == "Windows".
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
#      -> cmd_deploy proceeds to provision vast.ai.  Any non-zero exit
#      from steps 1-5 aborts the deploy with the same exit code.
#
# Invoked by neuroslm.hooks.run_hook("pre-deploy", REPO_ROOT). The
# runner sets BRIAN_HOOK_NAME, BRIAN_REPO_ROOT and streams output live.
# ──────────────────────────────────────────────────────────────────

$ErrorActionPreference = 'Stop'

$repoRoot = if ($env:BRIAN_REPO_ROOT) { $env:BRIAN_REPO_ROOT } else { (Get-Location).Path }
Set-Location -Path $repoRoot

Write-Host "[pre-deploy] cwd=$($PWD.Path)"

# Resolve the python interpreter. CLAUDE.md §13 pins everything to
# the single ./.venv at the repo root.
$python = if (Test-Path '.\.venv\Scripts\python.exe') {
    '.\.venv\Scripts\python.exe'
} else {
    'python'
}

# ── Step 1/5: Working tree must be clean ─────────────────────────
# `git status --porcelain` outputs one line per modified / untracked
# / staged path. Empty output -> clean. We bail BEFORE compiling so a
# user with WIP edits doesn't have them quietly bundled into the
# roundtrip commit.
Write-Host "[pre-deploy] step 1/5: checking working tree is clean"
$dirty = git status --porcelain
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($dirty) {
    Write-Host "[pre-deploy] ERROR: working tree is dirty. Commit or stash first:" -ForegroundColor Red
    Write-Host $dirty
    exit 1
}

# ── Step 2/5: Compile master arch -> canonical DNA ───────────────
# Explicit paths (not brian.toml defaults) because the user's
# convention is:
#   architectures/master       = source-of-truth template (edit here)
#   dna/<branch>/arch.dna      = canonical compiled DNA the deploy
#                                trains on. Path mirrors the active
#                                git branch so a fork on feature/X
#                                never clobbers master's DNA.
#   architectures/current/     = live working copy (rewritten in step 3)
# Using -m neuroslm.cli for the checkout-not-installed case.
Write-Host "[pre-deploy] step 2/5: brian dna compile (architectures/master -> dna/master/arch.dna)"
& $python -m neuroslm.cli dna compile architectures/master --output dna/master/arch.dna
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ── Step 3/5: Unfold DNA back to working copy ────────────────────
# Keeps architectures/current/arch.neuro byte-identical to the DNA
# that the deploy will actually train on. Catches "I edited current/
# but forgot to compile" AND "I edited master/ but forgot to
# propagate to current/" in one shot.
Write-Host "[pre-deploy] step 3/5: brian dna unfold (dna/master/arch.dna -> architectures/current/arch.neuro)"
& $python -m neuroslm.cli dna unfold dna/master/arch.dna --output architectures/current/arch.neuro
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ── Step 4/5: Stage + commit the roundtrip artefacts ─────────────
# `git diff --quiet --cached` after `git add -A` returns 0 when there
# are no staged changes (idempotent roundtrip -> nothing to commit),
# 1 when there ARE changes. Any other exit code is a real error.
Write-Host "[pre-deploy] step 4/5: staging + committing roundtrip artefacts"
git add -A
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

git diff --quiet --cached
$diffRc = $LASTEXITCODE
if ($diffRc -eq 0) {
    Write-Host "[pre-deploy] no roundtrip diff - DNA + current/ already in sync."
} elseif ($diffRc -eq 1) {
    $gitName  = if ($env:BRIAN_GIT_NAME)  { $env:BRIAN_GIT_NAME }  else { 'BRIAN pre-deploy hook' }
    $gitEmail = if ($env:BRIAN_GIT_EMAIL) { $env:BRIAN_GIT_EMAIL } else { 'brian-pre-deploy@local' }
    git -c "user.name=$gitName" -c "user.email=$gitEmail" `
        commit -m "chore: roundtrip recompile of current architecture"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    exit $diffRc
}

# ── Step 5/5: Push - only success here unblocks the deploy ───────
# If there was nothing to commit we still push to make sure the
# remote is at HEAD before the deploy box clones it. Push failure
# (network, hook, rebase conflict, ...) propagates and aborts the
# deploy.
Write-Host "[pre-deploy] step 5/5: pushing to origin"
git push
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[pre-deploy] ok - master -> DNA -> current synced, committed, pushed"
exit 0
