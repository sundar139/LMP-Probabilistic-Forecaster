Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:UV_INTERNAL__PYTHONHOME -ErrorAction SilentlyContinue

uv run ruff check .
uv run mypy src
uv run pytest -q
