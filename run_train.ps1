# Helper to run training with the project's virtualenv Python
# Usage: .\run_train.ps1 -m neuroslm.train --preset small --steps 2000 --batch_size 8 --device cuda

$venv = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venv)) {
    Write-Error "Virtualenv python not found at $venv. Activate your venv or run 'py -3 -m neuroslm.train ...' instead."
    exit 1
}

# Forward all args to the venv python
& $venv @args
