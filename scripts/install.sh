#!/usr/bin/env bash
# Apple-Gorilla one-command installer (macOS / Linux).
# Usage:  curl -fsSL https://raw.githubusercontent.com/pkwalv-dev/Apple-Gorilla/main/scripts/install.sh | bash
set -euo pipefail
DIR="${1:-$HOME/Apple-Gorilla}"
echo "== Apple-Gorilla installer =="

# 1. Get the code.
if [ -d "$DIR/.git" ]; then
  echo "updating existing checkout in $DIR"; git -C "$DIR" pull --ff-only
else
  echo "cloning into $DIR"; git clone https://github.com/pkwalv-dev/Apple-Gorilla.git "$DIR"
fi
cd "$DIR"

PY=python3; command -v "$PY" >/dev/null 2>&1 || PY=python

# 2. Verify AG runs (backend-independent).
"$PY" -m ag --dry-run run "install check" >/dev/null || { echo "AG failed - need Python 3.10+"; exit 1; }
echo "AG installed OK"

# 2b. Enable the self-improvement test gate (best-effort; ag evolve needs pytest).
if "$PY" -c "import pytest" >/dev/null 2>&1; then
  echo "evolve gate: pytest present"
else
  echo "installing pytest (self-improvement test gate)..."
  "$PY" -m pip install -q pytest || echo "could not install pytest - 'ag evolve' will stay disabled until you: pip install pytest"
fi

# 3. Point at Ollama; pull model if present.
"$PY" -m ag setup-ollama --model qwen2.5:7b >/dev/null
if command -v ollama >/dev/null 2>&1; then
  ollama pull qwen2.5:7b
else
  echo "Ollama not found - install from https://ollama.com then re-run"
fi
"$PY" -m ag doctor

# 4. Launch the web app.
echo "starting web app..."
"$PY" -m ag serve --open
