Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

param(
  [string]$Zone = 'AEP',
  [ValidateSet('local_safe', 'cloud_16gb', 'cloud_24gb')]
  [string]$ResourceProfile = 'cloud_16gb',
  [string]$Models = 'TFT,DeepAR',
  [int]$MaxTrials = 12,
  [int]$Folds = 2,
  [int]$MaxStepsCap = 50,
  [switch]$Write
)

Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:UV_INTERNAL__PYTHONHOME -ErrorAction SilentlyContinue

$baseArgs = @(
  'run', 'python', '-m', 'lmp_forecaster.cli', 'export-tuning-package',
  '--zone', $Zone,
  '--resource-profile', $ResourceProfile,
  '--models', $Models,
  '--max-trials', $MaxTrials,
  '--folds', $Folds,
  '--max-steps-cap', $MaxStepsCap
)

Write-Host 'Running export-tuning-package dry-run...' -ForegroundColor Cyan
& uv @baseArgs
if ($LASTEXITCODE -ne 0) {
  throw "Dry-run export failed with exit code $LASTEXITCODE"
}

if (-not $Write.IsPresent) {
  Write-Host 'Dry-run completed. Re-run with -Write to persist package manifest.' -ForegroundColor Yellow
  exit 0
}

Write-Host 'Running export-tuning-package write mode...' -ForegroundColor Cyan
& uv @($baseArgs + '--write')
if ($LASTEXITCODE -ne 0) {
  throw "Write-mode export failed with exit code $LASTEXITCODE"
}

Write-Host 'Export write mode completed.' -ForegroundColor Green
Write-Host 'Generated files remain under ignored data/cache paths and are not staged by this script.' -ForegroundColor Green
