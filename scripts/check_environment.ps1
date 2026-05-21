Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:UV_INTERNAL__PYTHONHOME -ErrorAction SilentlyContinue

uv run python -c "import platform,torch; print('Python:', platform.python_version()); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

try {
  $response = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:11434" -Method GET -TimeoutSec 2
  Write-Host "Ollama reachable at http://localhost:11434 (HTTP $($response.StatusCode))"
}
catch {
  Write-Host "Ollama not reachable at http://localhost:11434 (this is non-blocking)." -ForegroundColor Yellow
}
