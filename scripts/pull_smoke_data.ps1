#!/usr/bin/env pwsh

Write-Host "Running dry-run smoke ingestion commands..."
Write-Host ""

$pjm = "uv run python -m lmp_forecaster.cli pull-pjm-smoke --zone AEP"
$weather = "uv run python -m lmp_forecaster.cli pull-weather-smoke --zone AEP"
$forecast = "uv run python -m lmp_forecaster.cli pull-historical-forecast-smoke --zone AEP"

Write-Host "> $pjm"
iex $pjm
Write-Host ""

Write-Host "> $weather"
iex $weather
Write-Host ""

Write-Host "> $forecast"
iex $forecast
Write-Host ""

Write-Host "To write local cache files explicitly, run:"
Write-Host "  $pjm --write"
Write-Host "  $weather --write"
Write-Host "  $forecast --write"
Write-Host ""
Write-Host "No files are written by this script unless you run the commands above with --write."
