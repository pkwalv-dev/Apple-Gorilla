# Apple-Gorilla one-command installer (Windows PowerShell).
# Usage:  iwr https://raw.githubusercontent.com/pkwalv-dev/Apple-Gorilla/main/scripts/install.ps1 | iex
# or:     powershell -ExecutionPolicy Bypass -File scripts\install.ps1
param([string]$Dir = "$HOME\Apple-Gorilla")

$ErrorActionPreference = "Stop"
Write-Host "== Apple-Gorilla installer ==" -ForegroundColor Cyan

# 1. Get the code (clone or update).
if (Test-Path (Join-Path $Dir ".git")) {
  Write-Host "updating existing checkout in $Dir"
  git -C $Dir pull --ff-only
} else {
  Write-Host "cloning into $Dir"
  git clone https://github.com/pkwalv-dev/Apple-Gorilla.git $Dir
}
Set-Location $Dir

# 2. Verify AG imports and runs (backend-independent).
python -m ag --dry-run run "install check" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "AG failed to run - check Python 3.10+ is installed" }
Write-Host "AG installed OK" -ForegroundColor Green

# 2b. Enable the self-improvement test gate (best-effort; ag evolve needs pytest).
python -c "import pytest" 2>$null
if ($LASTEXITCODE -eq 0) {
  Write-Host "evolve gate: pytest present" -ForegroundColor Green
} else {
  Write-Host "installing pytest (self-improvement test gate)..."
  python -m pip install -q pytest
  if ($LASTEXITCODE -ne 0) {
    Write-Host "could not install pytest - 'ag evolve' stays disabled until you: pip install pytest" -ForegroundColor Yellow
  }
}

# 3. Point at Ollama and report readiness (pull the model if Ollama is present).
python -m ag setup-ollama --model qwen2.5:7b | Out-Null
if (Get-Command ollama -ErrorAction SilentlyContinue) {
  ollama pull qwen2.5:7b
} else {
  Write-Host "Ollama not found - install from https://ollama.com then re-run" -ForegroundColor Yellow
}
python -m ag doctor

# 4. Launch the web app (open in browser).
Write-Host "starting web app - opening your browser..." -ForegroundColor Cyan
python -m ag serve --open
