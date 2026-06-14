# brian — unified CLI for the NeuroSLM project (PowerShell wrapper).
#
# Examples:
#   .\scripts\brian.ps1 status
#   .\scripts\brian.ps1 compile architectures/current --out arch.py
#   .\scripts\brian.ps1 wolfram architectures/current --full --out arch.m
#   .\scripts\brian.ps1 analyze architectures/current --all
#   .\scripts\brian.ps1 deploy --steps 10000
#   .\scripts\brian.ps1 logs 38569395
#   .\scripts\brian.ps1 destroy --all

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $here

# Make output UTF-8 so the training metric chars (Φ λ θ) render properly.
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

& py -3 -m neuroslm.cli @args
exit $LASTEXITCODE
