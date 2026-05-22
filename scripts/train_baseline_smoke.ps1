#!/usr/bin/env pwsh

param(
  [switch]$Write
)

$env:PYTHONHOME = ""
$env:PYTHONPATH = ""

$dryDefault = "uv run python -m lmp_forecaster.cli train-single-zone-baselines --zone AEP"
$drySynthetic = "uv run python -m lmp_forecaster.cli train-single-zone-baselines --zone AEP --allow-synthetic-panel --build-panel-if-missing"
$writeSynthetic = "uv run python -m lmp_forecaster.cli train-single-zone-baselines --zone AEP --allow-synthetic-panel --build-panel-if-missing --write"

Write-Host "> $dryDefault"
iex $dryDefault
Write-Host ""

Write-Host "> $drySynthetic"
iex $drySynthetic
Write-Host ""

Write-Host "Write command:"
Write-Host "  $writeSynthetic"

if ($Write) {
  Write-Host ""
  Write-Host "> $writeSynthetic"
  iex $writeSynthetic
}
