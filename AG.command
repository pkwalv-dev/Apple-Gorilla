#!/bin/bash
# Apple-Gorilla one-click launcher (macOS) — double-click in Finder to open the web app.
# Copyable anywhere: it looks for the AG folder beside itself, then remembers it.
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

# 1) launcher inside the repo, 2) a clone beside it, 3) a remembered path, 4) ask once.
if [ -f ag/__init__.py ]; then AGDIR="$here"
elif [ -f "$here/Apple-Gorilla/ag/__init__.py" ]; then AGDIR="$here/Apple-Gorilla"
elif [ -f "$here/AG_path.txt" ]; then AGDIR="$(cat "$here/AG_path.txt")"
else
  read -r -p "Paste the full path to your Apple-Gorilla folder: " AGDIR
fi

if [ ! -f "$AGDIR/ag/__init__.py" ]; then
  echo "That folder does not contain AG (no ag/__init__.py): $AGDIR"
  echo "Delete AG_path.txt beside this launcher and try again."
  read -r -p "Press Enter to close."
  exit 1
fi
echo "$AGDIR" > "$here/AG_path.txt"
cd "$AGDIR"

PY=python3; command -v python3 >/dev/null 2>&1 || PY=python
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python 3 was not found. Install it from https://www.python.org/downloads/ and retry."
  read -r -p "Press Enter to close."
  exit 1
fi

echo "Checking dependencies (first run may take a moment)..."
"$PY" -m pip install -q -r requirements.txt >/dev/null 2>&1

echo "Apple-Gorilla is starting - a browser tab will open. Close this window to stop."
"$PY" -m ag serve --open
