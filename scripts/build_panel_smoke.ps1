#!/usr/bin/env pwsh

param(
  [switch]$Write
)

$dryReal = "uv run python -m lmp_forecaster.cli build-single-zone-panel --zone AEP"
$drySynth = "uv run python -m lmp_forecaster.cli build-single-zone-panel --zone AEP --allow-synthetic-lmp --allow-synthetic-weather"
$writeSynth = "uv run python -m lmp_forecaster.cli build-single-zone-panel --zone AEP --allow-synthetic-lmp --allow-synthetic-weather --write --summary"

Write-Host "> $dryReal"
iex $dryReal
Write-Host ""

Write-Host "> $drySynth"
iex $drySynth
Write-Host ""

Write-Host "Write command:"
Write-Host "  $writeSynth"

if ($Write) {
  Write-Host ""
  Write-Host "> $writeSynth"
  iex $writeSynth
}
