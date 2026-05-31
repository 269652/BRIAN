# install.ps1 — one-shot setup for the BRIAN repo (PowerShell).
#
# 1. Creates .\.venv (if missing) using py -3.
# 2. Activates it and upgrades pip.
# 3. pip install -e .       -> puts the `brian` CLI on PATH
# 4. pip install -e ".[ml]" -> adds torch + transformers + datasets
#                              (skip with -NoMl)
#
# Usage:
#   .\scripts\install.ps1            # full install
#   .\scripts\install.ps1 -NoMl      # CLI-only (skips torch)
#
# Then:
#   .\.venv\Scripts\Activate.ps1
#   brian --help

param([switch]$NoMl)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $here

# Force UTF-8 so the CLI's metric chars (Phi, lambda) render
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Pick a Python 3.10+
$py = $null
try {
    $v = & py -3 -c "import sys; print(sys.version_info >= (3,10))" 2>$null
    if ($v -eq "True") { $py = "py -3" }
} catch { }
if (-not $py) {
    foreach ($cand in @("python3.13","python3.12","python3.11","python3.10","python3","python")) {
        $cmd = Get-Command $cand -ErrorAction SilentlyContinue
        if ($cmd) {
            $v = & $cand -c "import sys; print(sys.version_info >= (3,10))" 2>$null
            if ($v -eq "True") { $py = $cand; break }
        }
    }
}
if (-not $py) {
    Write-Host "X no Python 3.10+ found on PATH" -ForegroundColor Red
    Write-Host "  install Python 3.11+ from python.org then re-run"
    exit 1
}
Write-Host "-- using python: $py" -ForegroundColor Cyan

# Create venv if missing
if (-not (Test-Path ".venv")) {
    Write-Host "-- creating venv at .\.venv" -ForegroundColor Cyan
    Invoke-Expression "$py -m venv .venv"
} else {
    Write-Host "-- reusing existing venv at .\.venv" -ForegroundColor Cyan
}

$vpy = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    Write-Host "X venv created but python.exe missing" -ForegroundColor Red
    exit 1
}

Write-Host "-- upgrading pip" -ForegroundColor Cyan
& $vpy -m pip install -q --upgrade pip

Write-Host "-- installing brian + CLI deps (pyproject.toml)" -ForegroundColor Cyan
& $vpy -m pip install -q -e .

if (-not $NoMl) {
    Write-Host "-- installing heavy ML deps (torch, transformers, datasets)" -ForegroundColor Cyan
    Write-Host "   this can take 5-10 min - re-run with -NoMl to skip"
    & $vpy -m pip install -q -e ".[ml]"
}

Write-Host ""
Write-Host "-- verifying install" -ForegroundColor Cyan
& $vpy -m neuroslm.cli --help | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "+ brian CLI importable" -ForegroundColor Green
} else {
    Write-Host "X brian CLI not importable - install failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================================"
Write-Host "  + install complete" -ForegroundColor Green
Write-Host ""
Write-Host "  Activate the venv:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "  Then run:"
Write-Host "    brian --help"
Write-Host "    brian ps          # list active vast instances"
Write-Host "    brian deploy      # launch a training run"
Write-Host "================================================================"
