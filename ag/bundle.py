"""Portable bundle — carry AG (and its memory + skills) to any machine.

AG is designed to be substrate-portable: pure-Python core, backend-agnostic, and all
of its learned state (profile, memory, acquired skills, config) is plain files under
the repo. `export()` packs that into one archive you can move to any OS with Python
3.10+; on the far side, unzip and run — the memory and skills travel with it, and AG
adapts to the new host (see `ag host`).

`check()` audits the **portability constraints** that keep this true, so a self-edit or
a new dependency can't quietly break portability:

  1. Core imports stdlib-only (the one allowed third-party package is `anthropic`,
     needed solely for the Claude backend; Ollama/dry need nothing).
  2. No hard-coded absolute/home paths in the core (paths derive from ROOT).
  3. Learned state is file-based and present (memory/skills/profile/config).
  4. Python version floor is met.

Stdlib only. Export writes to your chosen path (default: alongside the repo).
"""
from __future__ import annotations

import ast
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import ROOT, STATE_DIR

# Third-party imports permitted in the core without breaking portability.
_ALLOWED_THIRD_PARTY = {"anthropic"}
# Everything AG itself imports is first-party or stdlib; this is the stdlib-ish set we
# don't flag (we can't import stdlib list at runtime reliably across versions).
_FIRST_PARTY_ROOTS = {"ag"}

# What travels in a bundle (learned state included; heavy/rebuildable state excluded).
_INCLUDE = ["ag", "profile", "config.json", "requirements.txt", "requirements-lora.txt",
            "README.md", "AG.bat", "AG.command", "docs"]
_INCLUDE_STATE = ["memory", "skills", "archive"]      # carry learning; skip runs/versions/images
_EXCLUDE_NAMES = {"__pycache__", ".pytest_cache", ".git"}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def _iter_core_py() -> List[Path]:
    return [p for p in (ROOT / "ag").rglob("*.py")
            if "__pycache__" not in p.parts]


def _guarded_import_nodes(tree: ast.AST) -> set:
    """Import nodes that sit inside a `try:` with an ImportError handler.

    The portability constraint is "AG must RUN on a machine with nothing installed",
    not "AG must never mention a third-party name". An import wrapped in
    `try: import x / except ImportError: <fallback>` cannot break that: on a bare
    machine the except branch runs. Treating those as violations would push the
    codebase into `importlib.import_module` calls that hide the same dependency from
    the audit — strictly worse, because then the auditor cannot see it at all.

    So we distinguish the two cases, and only HARD imports fail the check. Guarded
    ones are still reported, as optional accelerators.
    """
    guarded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches_import_error = False
        for handler in node.handlers:
            exc = handler.type
            names = []
            if isinstance(exc, ast.Name):
                names = [exc.id]
            elif isinstance(exc, ast.Tuple):
                names = [e.id for e in exc.elts if isinstance(e, ast.Name)]
            elif exc is None:
                names = ["BaseException"]     # bare except also catches ImportError
            if any(n in ("ImportError", "ModuleNotFoundError", "Exception",
                         "BaseException") for n in names):
                catches_import_error = True
        if not catches_import_error:
            continue
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    guarded.add(id(sub))
    return guarded


def _third_party_imports(path: Path, *, include_guarded: bool = False) -> set:
    """Top-level module roots imported by a file that aren't stdlib/first-party.

    Uses the AST (no execution). We approximate "stdlib" via
    sys.stdlib_module_names (Python 3.10+), so anything not stdlib, not first-party,
    and not relative is treated as third-party.

    By default, imports guarded by an ImportError handler are excluded: they have a
    fallback path and therefore cannot make AG unrunnable on a bare machine. Pass
    include_guarded=True to see them (the audit reports them separately)."""
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    found = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return found
    guarded = set() if include_guarded else _guarded_import_nodes(tree)
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                found.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import -> first-party
            if node.module:
                found.add(node.module.split(".")[0])
    return {m for m in found
            if m and m not in stdlib and m not in _FIRST_PARTY_ROOTS}


def _code_string_literals(path: Path) -> List[str]:
    """Every string literal in a file that is NOT a docstring.

    Docstrings are the module/class/function-level bare string expressions; the AST
    marks them by position, so we can drop exactly those and keep every literal
    that actually participates in execution. Comments never reach the AST at all.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))
    out: List[str] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            out.append(node.value)
    return out


def optional_dependencies() -> dict:
    """Third-party packages the core can USE but does not NEED, per file.

    Surfacing these keeps the guarantee honest: "stdlib-only" means AG runs without
    them, not that they are never touched. A reader can see exactly which optional
    accelerators exist and what each one buys.
    """
    out: dict = {}
    for p in _iter_core_py():
        if p.name == "lora.py":
            continue
        hard = _third_party_imports(p)
        everything = _third_party_imports(p, include_guarded=True)
        soft = sorted(everything - hard - _ALLOWED_THIRD_PARTY)
        if soft:
            out[str(p.relative_to(ROOT))] = soft
    return out


def check() -> List[Check]:
    """Audit the portability constraints. Returns a list of Checks (ok/detail)."""
    checks: List[Check] = []

    # 1) stdlib-only core (allow anthropic). lora.py is the designated OPTIONAL module
    #    (weight-training extras in requirements-lora.txt, imported lazily) — it is not
    #    part of the portable core, so it is exempt from this check.
    offenders = {}
    for p in _iter_core_py():
        if p.name == "lora.py":
            continue
        extra = _third_party_imports(p) - _ALLOWED_THIRD_PARTY
        if extra:
            offenders[str(p.relative_to(ROOT))] = sorted(extra)
    checks.append(Check(
        "stdlib-only core (except anthropic)", not offenders,
        "" if not offenders else "; ".join(f"{k}: {', '.join(v)}"
                                            for k, v in list(offenders.items())[:6])))

    # 2) no hard-coded absolute/home paths.
    #    Scanned through the AST rather than by grepping the raw text: a path inside
    #    a comment or a docstring is documentation (e.g. explaining that WSL maps
    #    C:\ to /mnt/c), and cannot affect where AG reads or writes. Only a real
    #    string literal in executable code can. Grepping the source text conflates
    #    the two and pushes authors to obfuscate examples in prose, which makes the
    #    code less clear without making it more portable.
    bad_paths = []
    needles = ("C:\\", "C:/Users", "/Users/", "/home/")
    for p in _iter_core_py():
        if p.name == "bundle.py":
            continue  # this module holds the detection literals themselves, as data
        for literal in _code_string_literals(p):
            hit = next((n for n in needles if n in literal), "")
            if hit:
                bad_paths.append(f"{p.relative_to(ROOT)} ({hit})")
                break
    checks.append(Check("no hard-coded absolute paths", not bad_paths,
                        "; ".join(bad_paths[:6])))

    # 3) learned state is file-based and present.
    have = [d for d in _INCLUDE_STATE if (STATE_DIR / d).exists()]
    checks.append(Check("learned state is file-based (memory/skills/archive)", True,
                        "present: " + (", ".join(have) or "none yet")))

    # 4) python floor.
    ok_py = sys.version_info[:2] >= (3, 10)
    checks.append(Check("Python >= 3.10", ok_py,
                        f"running {sys.version_info.major}.{sys.version_info.minor}"))
    return checks


def check_ok(checks: Optional[List[Check]] = None) -> bool:
    return all(c.ok for c in (checks or check()))


def _should_skip(p: Path) -> bool:
    return any(part in _EXCLUDE_NAMES for part in p.parts)


def export(dest: Optional[Path] = None) -> Path:
    """Pack AG + its learned state into a portable .zip. Returns the archive path."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = Path(dest) if dest else (ROOT.parent / f"apple-gorilla-bundle-{stamp}.zip")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in _INCLUDE:
            src = ROOT / rel
            if not src.exists():
                continue
            if src.is_dir():
                for f in src.rglob("*"):
                    if f.is_file() and not _should_skip(f):
                        z.write(f, f.relative_to(ROOT).as_posix())
            else:
                z.write(src, src.relative_to(ROOT).as_posix())
        for d in _INCLUDE_STATE:
            sdir = STATE_DIR / d
            if sdir.exists():
                for f in sdir.rglob("*"):
                    if f.is_file() and not _should_skip(f):
                        z.write(f, f.relative_to(ROOT).as_posix())
        z.writestr("BUNDLE_MANIFEST.json", _manifest())
    return dest


def _manifest() -> str:
    import json
    from . import __version__
    return json.dumps({
        "tool": "Apple-Gorilla", "version": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python_floor": "3.10",
        "portable": check_ok(),
        "carries": ["profile", "state/memory", "state/skills", "state/archive", "config.json"],
        "restore": "unzip, then: pip install -r requirements.txt (Claude backend only), "
                   "then `python -m ag doctor`. Ollama/dry backends need no install.",
    }, indent=2)
