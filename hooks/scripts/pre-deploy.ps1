# ──────────────────────────────────────────────────────────────────
# hooks/scripts/pre-deploy.ps1 — Windows PowerShell impl.
#
# Symmetric with hooks/scripts/pre-deploy.sh. Picked by the runner
# when platform.system() == "Windows".
#
# Invoked by neuroslm.hooks.run_hook("pre-deploy", REPO_ROOT). The
# runner sets:
#   BRIAN_HOOK_NAME=pre-deploy
#   BRIAN_REPO_ROOT=<absolute repo path>
# Stdout / stderr stream live to the user's terminal (no PIPE).
# ──────────────────────────────────────────────────────────────────

$ErrorActionPreference = 'Stop'

$repoRoot = if ($env:BRIAN_REPO_ROOT) { $env:BRIAN_REPO_ROOT } else { (Get-Location).Path }
Set-Location -Path $repoRoot

Write-Host "[pre-deploy] cwd=$($PWD.Path)"

# Resolve the python interpreter. The CLAUDE.md §13 contract pins
# everything to the single ./.venv at the repo root.
$python = if (Test-Path '.\.venv\Scripts\python.exe') {
    '.\.venv\Scripts\python.exe'
} else {
    'python'
}

Write-Host "[pre-deploy] step 1/2: brian dna compile (architectures/master -> dna/master/arch.dna)"
# Explicit paths (not the brian.toml defaults) because the user's
# convention is:
#   architectures/master       = source-of-truth template (edit here)
#   dna/<branch>/arch.dna      = canonical compiled DNA the deploy trains on.
#                                Path mirrors the active git branch from
#                                brian.toml [defaults].branch so a fork on a
#                                feature branch never clobbers master's DNA.
#   architectures/current/     = live working copy (auto-overwritten below)
# Using -m neuroslm.cli for the checkout-not-installed case.
& $python -m neuroslm.cli dna compile architectures/master --output dna/master/arch.dna
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[pre-deploy] step 2/2: brian dna unfold (dna/master/arch.dna -> architectures/current/arch.neuro)"
# Unfolding back into architectures/current keeps the on-disk working
# copy bit-identical to the bytes the deploy will actually train on.
# Catches "I edited current/ but forgot to compile" foot-guns AND
# "I edited master/ but forgot to propagate to current/" both.
& $python -m neuroslm.cli dna unfold dna/master/arch.dna --output architectures/current/arch.neuro
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[pre-deploy] ok - master -> DNA -> current are all in sync"
exit 0
