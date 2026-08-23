"""Self-improvement loop with test-gated adoption and instant rollback.

Flow:
  1. Snapshot all evolvable files (fast, local).
  2. Ask the model for small patches to those files, given recent telemetry.
  3. Apply patches to the working tree.
  4. Run the test suite in a subprocess against the candidate.
  5. If tests pass -> keep + git-commit. If they fail (or anything errors) ->
     restore the snapshot immediately. AG never runs on unverified code.

Autonomy levels (config.autonomy_level):
  - "never":   evolution disabled.
  - "manual":  propose + validate, but require --apply to keep (still auto-rolls
               back a failed candidate; a *passing* candidate is left for review).
  - "guarded": auto-adopt only if the test suite passes; else auto-rollback (default).
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from . import backup, prompts
from .config import ROOT, Config
from .model import extract_json
from .pipeline import recent_runs


@dataclass
class EvolveResult:
    attempted: bool
    adopted: bool
    rolled_back: bool
    rationale: str = ""
    changed: List[str] = field(default_factory=list)
    test_output: str = ""
    snapshot_id: str = ""
    commit: Optional[str] = None
    reason: str = ""


def _read_evolvable(cfg: Config) -> dict:
    files = {}
    for rel in cfg.evolvable_paths:
        p = ROOT / rel
        files[rel] = p.read_text() if p.exists() else ""
    return files


GATE_MISSING_MSG = (
    "pytest not installed — the self-improvement gate cannot verify changes. "
    "Install it: pip install pytest  (or pip install -r requirements.txt)"
)


def gate_available() -> bool:
    """True if the test gate can actually run.

    The suite relies on pytest fixtures (`tmp_path`, `monkeypatch`) and
    `pytest.raises`, so plain `unittest` cannot execute it — pytest is a hard
    requirement of the gate, not an optional accelerator. We detect it up front
    so the gate can fail CLOSED with an actionable message instead of silently
    rolling every candidate back on a cryptic "No module named pytest".
    """
    import importlib.util
    return importlib.util.find_spec("pytest") is not None


def run_tests() -> tuple[bool, str]:
    """Run the test suite in a subprocess. Passing tests gate adoption.

    Fails closed: if the gate can't run (pytest missing, no tests collected,
    timeout, crash), it reports failure so AG never adopts unverified code.
    """
    if not gate_available():
        return False, GATE_MISSING_MSG
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header"],
            cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
        # returncode 5 = "no tests collected"; treat as a failed gate, not a pass.
        if r.returncode == 5:
            return False, "no tests collected — refusing to adopt unverified change"
        return r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return False, "tests timed out"


def _validate_patch(rel: str, content: str) -> Optional[str]:
    """Cheap static validation before writing to disk. Returns error or None."""
    if rel.endswith(".json"):
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            return f"invalid JSON: {e}"
    if rel.endswith(".py"):
        try:
            compile(content, rel, "exec")
        except SyntaxError as e:
            return f"syntax error: {e}"
    return None


def evolve(client, cfg: Config, *, apply: bool = False,
           dry_run: bool = False) -> EvolveResult:
    if cfg.autonomy_level == "never":
        return EvolveResult(False, False, False, reason="autonomy_level=never")

    # No usable test gate means no safe way to verify a self-edit. Bail out BEFORE
    # spending a (possibly billed) model call, snapshotting, or touching the tree.
    if not gate_available():
        return EvolveResult(False, False, False,
                            reason="gate unavailable: " + GATE_MISSING_MSG)

    evolvable = _read_evolvable(cfg)
    telemetry = recent_runs(limit=5)

    user = (
        "# Recent run telemetry\n"
        + json.dumps(telemetry, indent=2)[:8000]
        + "\n\n# Current evolvable files\n"
        + json.dumps(evolvable, indent=2)[:12000]
        + "\n\nPropose small, safe improvements per your rules."
    )
    res = client.complete(system=prompts.EVOLVER_SYSTEM, user=user, cfg=cfg,
                          max_tokens=cfg.meta_output_tokens)
    data = extract_json(res.text) or {}
    patches = data.get("patches", []) or []
    rationale = str(data.get("rationale", ""))

    if not patches:
        return EvolveResult(True, False, False, rationale=rationale,
                            reason="no patches proposed")

    # Keep only patches to declared evolvable paths, that pass static validation.
    valid = []
    for patch in patches:
        rel = patch.get("path", "")
        content = patch.get("new_content", "")
        if rel not in cfg.evolvable_paths:
            continue
        err = _validate_patch(rel, content)
        if err:
            return EvolveResult(True, False, False, rationale=rationale,
                                reason=f"rejected {rel}: {err}")
        valid.append((rel, content))

    if not valid:
        return EvolveResult(True, False, False, rationale=rationale,
                            reason="no valid patches to evolvable paths")

    # 1) Snapshot BEFORE touching anything, then bound how many we keep.
    snap = backup.snapshot(cfg.evolvable_paths, note=f"pre-evolve: {rationale[:80]}")
    backup.prune_snapshots(cfg.max_snapshots)

    # 2) Apply candidate patches.
    changed = []
    for rel, content in valid:
        (ROOT / rel).write_text(content)
        changed.append(rel)

    # 3) Gate on the test suite.
    passed, output = run_tests()

    if not passed:
        backup.restore(snap)  # instant rollback
        return EvolveResult(True, False, True, rationale=rationale, changed=changed,
                            test_output=output, snapshot_id=snap.id,
                            reason="tests failed -> rolled back")

    # Passing candidate.
    if cfg.autonomy_level == "manual" and not apply:
        # Leave the passing change in place for human review, but do not commit.
        return EvolveResult(True, False, False, rationale=rationale, changed=changed,
                            test_output=output, snapshot_id=snap.id,
                            reason="manual mode: passing candidate left for review "
                                   "(use rollback to discard)")

    # AG commits to its own branch, never main — advancing main stays a human action.
    commit = None if dry_run else backup.git_commit_evolve(
        f"AG self-improve: {rationale[:60]}", branch=cfg.evolve_branch
    )
    return EvolveResult(True, True, False, rationale=rationale, changed=changed,
                        test_output=output, snapshot_id=snap.id, commit=commit,
                        reason="tests passed -> adopted")
