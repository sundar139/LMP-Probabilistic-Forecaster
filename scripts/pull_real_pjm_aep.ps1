#!/usr/bin/env pwsh

param(
  [switch]$Write,
  [switch]$OneYear
)

$env:PYTHONHOME = ""
$env:PYTHONPATH = ""

Write-Host "PJM real ingestion helper"
if ($env:PJM_API_KEY) {
  Write-Host "PJM_API_KEY: configured"
} else {
  Write-Host "PJM_API_KEY: missing"
}

$inspectCmd = "uv run python -m lmp_forecaster.cli inspect-pjm-api --zone AEP --start-date 2024-01-01 --end-date 2024-01-02"
$dryRunCmd = "uv run python -m lmp_forecaster.cli pull-real-pjm-lmp --zone AEP --start-date 2024-01-01 --end-date 2024-01-31"
$write30Cmd = "uv run python -m lmp_forecaster.cli pull-real-pjm-lmp --zone AEP --start-date 2024-01-01 --end-date 2024-01-31 --write"
$writeYearCmd = "uv run python -m lmp_forecaster.cli pull-real-pjm-lmp --zone AEP --one-year --write"

Write-Host "Running inspect dry probe..."
Invoke-Expression $inspectCmd

Write-Host "Running dry-run monthly plan..."
Invoke-Expression $dryRunCmd

if ($Write) {
  Write-Host "Running 30-day real pull..."
  Invoke-Expression $write30Cmd
}

if ($OneYear) {
  Write-Host "Running one-year real pull..."
  Invoke-Expression $writeYearCmd
}

Write-Host "Output roots: data/raw/pjm/da_hrl_lmps and data/cache/pjm/da_hrl_lmps"
