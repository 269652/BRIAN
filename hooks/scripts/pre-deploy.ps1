# ──────────────────────────────────────────────────────────────────
# hooks/scripts/pre-deploy.ps1 — Windows PowerShell impl.
#
# Symmetric with hooks/scripts/pre-deploy.sh. Picked by the runner
# when platform.system() == "Windows".
#
# Pipeline (4 steps, fail-fast):
#   1. Refuse if the working tree is dirty (caller must commit / stash
#      first — otherwise the roundtrip commit would smuggle their WIP
#      under our chore message).
#   2. Compile the brian.toml [current].arch → dna/<leaf>/arch.dna
#   3. git add <dna> + commit "chore: recompile <arch> architecture"
#      (no-op if compile produced byte-identical output).
#   4. git push.    Only on a successful push does the hook return 0
#      -> cmd_deploy proceeds to provision vast.ai.  Any non-zero exit
#      from steps 1-4 aborts the deploy with the same exit code.
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

# ── Resolve arch from brian.toml [current].arch ──────────────────
# Read the arch path from brian.toml so the hook always compiles
# whatever the workspace is configured to deploy — not a hardcoded
# folder. Falls back to "architectures/master" for back-compat.
$archPath = & $python -c "
from neuroslm.project_config import load_project_config
cfg = load_project_config()
print(cfg.arch)
" 2>$null
if (-not $archPath -or $LASTEXITCODE -ne 0) {
    $archPath = "architectures/master"
}
$archLeaf = ($archPath -split '/')[-1]
$dnaPath = "dna/$archLeaf/arch.dna"

# ── Step 1/4: Working tree must be clean ─────────────────────────
Write-Host "[pre-deploy] step 1/4: checking working tree is clean"
$dirty = git status --porcelain
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($dirty) {
    Write-Host "[pre-deploy] ERROR: working tree is dirty. Commit or stash first:" -ForegroundColor Red
    Write-Host $dirty
    exit 1
}

# ── Step 2/4: Compile arch -> canonical DNA ──────────────────────
Write-Host "[pre-deploy] step 2/4: brian dna compile ($archPath -> $dnaPath)"
& $python -m neuroslm.cli dna compile $archPath --output $dnaPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ── Step 3/4: Stage + commit the DNA ─────────────────────────────
Write-Host "[pre-deploy] step 3/4: staging + committing recompiled DNA"
git add $dnaPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

git diff --quiet --cached
$diffRc = $LASTEXITCODE
if ($diffRc -eq 0) {
    Write-Host "[pre-deploy] no DNA diff - already up to date."
} elseif ($diffRc -eq 1) {
    $gitName  = if ($env:BRIAN_GIT_NAME)  { $env:BRIAN_GIT_NAME }  else { 'BRIAN pre-deploy hook' }
    $gitEmail = if ($env:BRIAN_GIT_EMAIL) { $env:BRIAN_GIT_EMAIL } else { 'brian-pre-deploy@local' }
    git -c "user.name=$gitName" -c "user.email=$gitEmail" `
        commit -m "chore: recompile $archLeaf architecture"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    exit $diffRc
}

# ── Step 4/4: Push - only success here unblocks the deploy ───────
Write-Host "[pre-deploy] step 4/4: pushing to origin"
git push
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[pre-deploy] ok - $archLeaf -> $dnaPath compiled, committed, pushed"
exit 0
