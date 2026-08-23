"""Backup and rollback: the safety net for self-modification.

Every self-edit is preceded by a full snapshot of the evolvable files into
state/versions/<timestamp>/. Restoring is a fast file copy, so a failed edit is
undone in milliseconds. Git is used as a secondary record when available.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import ROOT, VERSIONS_DIR, ensure_dirs


@dataclass
class Snapshot:
    id: str
    path: Path
    files: List[str]

    @property
    def label(self) -> str:
        return self.id


def snapshot(paths: List[str], note: str = "") -> Snapshot:
    """Copy each evolvable file into a timestamped version directory."""
    ensure_dirs()
    sid = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    dest = VERSIONS_DIR / sid
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for rel in paths:
        src = ROOT / rel
        if not src.exists():
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        saved.append(rel)
    (dest / "MANIFEST.txt").write_text(
        f"note: {note}\n" + "\n".join(saved) + "\n"
    )
    return Snapshot(id=sid, path=dest, files=saved)


def restore(snap: Snapshot) -> List[str]:
    """Restore every file recorded in a snapshot back into the repo."""
    restored = []
    for rel in snap.files:
        src = snap.path / rel
        if not src.exists():
            continue
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        restored.append(rel)
    return restored


def restore_by_id(sid: str) -> List[str]:
    dest = VERSIONS_DIR / sid
    manifest = dest / "MANIFEST.txt"
    if not manifest.exists():
        raise FileNotFoundError(f"No snapshot '{sid}'")
    files = [ln for ln in manifest.read_text().splitlines()
             if ln and not ln.startswith("note:")]
    return restore(Snapshot(id=sid, path=dest, files=files))


def list_snapshots() -> List[str]:
    ensure_dirs()
    return sorted(
        p.name for p in VERSIONS_DIR.iterdir()
        if p.is_dir() and (p / "MANIFEST.txt").exists()
    )


def prune_snapshots(keep: int) -> int:
    """Keep only the newest `keep` snapshots; delete the rest. Bounds disk use."""
    if keep <= 0:
        return 0
    ids = list_snapshots()  # ascending (oldest first)
    remove = ids[:-keep] if len(ids) > keep else []
    for sid in remove:
        shutil.rmtree(VERSIONS_DIR / sid, ignore_errors=True)
    return len(remove)


def git_commit_evolve(message: str, branch: str = "ag/evolve") -> Optional[str]:
    """Record AG's self-edits on `branch` WITHOUT moving HEAD/main or switching the
    checkout. Advancing `main` therefore stays a deliberate human action (review the
    branch, then merge). Best-effort; never raises.

    Uses plumbing: stage -> write-tree -> commit-tree onto the branch's tip -> move
    only the branch ref -> unstage. The working tree is left exactly as adopted.
    """
    def _git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    try:
        if _git("rev-parse", "--git-dir").returncode != 0:
            return None
        _git("add", "-A")
        tree = _git("write-tree").stdout.strip()
        if not tree:
            return None
        parent = _git("rev-parse", "--verify", "-q",
                      f"refs/heads/{branch}").stdout.strip()
        if not parent:  # branch doesn't exist yet -> start it from current HEAD (main)
            parent = _git("rev-parse", "--verify", "-q", "HEAD").stdout.strip()
        args = ["commit-tree", tree, "-m", message]
        if parent:
            args += ["-p", parent]
        commit = _git(*args).stdout.strip()
        if not commit:
            return None
        _git("update-ref", f"refs/heads/{branch}", commit)
        _git("reset", "-q")  # unstage; leave working tree as AG adopted it
        return commit
    except Exception:
        return None
