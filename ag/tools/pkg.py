"""Gated dependency installation for acquired skills.

Installing software is powerful, so it is default-deny: `install()` requires the
`install_package` grant AND `allow_external_tools`. Installs go into the current
interpreter's environment via pip, are logged (so they are auditable and reversible),
and only touch PyPI — never an arbitrary URL. System-level binaries (ffmpeg, etc.) are
NOT installed here; those are surfaced to the user for explicit, per-item approval.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from typing import List, Optional

from ..config import STATE_DIR
from ..permissions import PermissionBroker

INSTALL_LOG = STATE_DIR / "skills" / "installs.jsonl"
MAX_SECONDS = 300

# A conservative package-spec pattern: name with optional extras and a pinned version.
_SPEC = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?"
                   r"([=<>!~]=?[A-Za-z0-9.*+!-]+)?$")


def valid_spec(spec: str) -> bool:
    return bool(_SPEC.match((spec or "").strip()))


def is_installed(pkg: str) -> bool:
    """True if the base distribution name is importable/installed."""
    import importlib.util
    base = re.split(r"[\[=<>!~ ]", (pkg or "").strip(), 1)[0].replace("-", "_")
    if not base:
        return False
    try:
        from importlib.metadata import distributions
        names = {d.metadata["Name"].lower().replace("-", "_")
                 for d in distributions() if d.metadata and d.metadata["Name"]}
        if base.lower() in names:
            return True
    except Exception:
        pass
    return importlib.util.find_spec(base) is not None


def _log(record: dict) -> None:
    try:
        INSTALL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with INSTALL_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


def install(specs: List[str], *, broker: PermissionBroker,
            timeout: int = MAX_SECONDS) -> str:
    """Install one or more PyPI packages (pinned specs). Gated by 'install_package'.

    Returns a human-readable summary. Rejects malformed specs before shelling out; a
    package already present is skipped. Every attempt is logged to installs.jsonl.
    """
    broker.require("install_package")
    specs = [s.strip() for s in (specs or []) if s and s.strip()]
    if not specs:
        return "install: nothing to install"
    bad = [s for s in specs if not valid_spec(s)]
    if bad:
        return f"install refused: malformed spec(s): {', '.join(bad)}"

    todo = [s for s in specs if not is_installed(s)]
    skipped = [s for s in specs if s not in todo]
    if not todo:
        return f"install: already present ({', '.join(skipped)})"

    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *todo]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        ok = r.returncode == 0
        out = (r.stdout + r.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        ok, out = False, f"timed out after {timeout}s"
    except Exception as e:  # pragma: no cover - defensive
        ok, out = False, str(e)
    _log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "specs": todo,
          "ok": ok, "output": out[-500:]})
    status = "installed" if ok else "FAILED"
    tail = f"; skipped already-present: {', '.join(skipped)}" if skipped else ""
    return f"{status}: {', '.join(todo)}{tail}\n{out}" if not ok else \
           f"{status}: {', '.join(todo)}{tail}"
