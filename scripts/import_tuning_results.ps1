Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

param(
  [Parameter(Mandatory = $true)]
  [string]$RankedCandidatesPath,
  [string]$SummaryPath,
  [string]$BaselineMetricsPath,
  [string]$Zone = 'AEP',
  [switch]$Write
)

Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:UV_INTERNAL__PYTHONHOME -ErrorAction SilentlyContinue

$args = @(
  'run', 'python', '-m', 'lmp_forecaster.cli', 'import-tuning-results',
  '--zone', $Zone,
  '--ranked-candidates-path', $RankedCandidatesPath
)

if ($SummaryPath) {
  $args += @('--summary-path', $SummaryPath)
}

if ($BaselineMetricsPath) {
  $args += @('--baseline-metrics-path', $BaselineMetricsPath)
}

Write-Host 'Running import-tuning-results validation dry-run...' -ForegroundColor Cyan
& uv @args
if ($LASTEXITCODE -ne 0) {
  throw "Dry-run import validation failed with exit code $LASTEXITCODE"
}

if (-not $Write.IsPresent) {
  Write-Host 'Dry-run completed. Re-run with -Write to persist import validation report.' -ForegroundColor Yellow
  exit 0
}

Write-Host 'Running import-tuning-results write mode...' -ForegroundColor Cyan
& uv @($args + '--write')
if ($LASTEXITCODE -ne 0) {
  throw "Write-mode import validation failed with exit code $LASTEXITCODE"
}

Write-Host 'Import write mode completed.' -ForegroundColor Green
Write-Host 'Generated report files remain under ignored data/cache paths and are not staged by this script.' -ForegroundColor Green
