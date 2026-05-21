Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
  Write-Host "uv is not installed. Install it with:" -ForegroundColor Yellow
  Write-Host 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"' -ForegroundColor Cyan
  exit 1
}

Write-Host "Using uv: $(uv --version)"

Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:UV_INTERNAL__PYTHONHOME -ErrorAction SilentlyContinue

uv python install 3.12
uv python pin 3.12
uv sync

Write-Host "Bootstrap complete." -ForegroundColor Green
Write-Host "Next commands:"
Write-Host "  ./scripts/check_environment.ps1"
Write-Host "  ./scripts/run_tests.ps1"
