"""The evolution archive — AG's persistent memory of its own improvement.

Every self-edit attempt is recorded here with its *measured* fitness delta, so AG
accumulates a durable, inspectable lineage of what it tried, what it kept, and why.
This is the Darwin Gödel Machine's central data structure (arXiv:2505.22954): an
archive of variants scored by an objective evaluator. Two things it buys us:

  - **Provenance & honesty.** Every adopted change has a number attached — the
    benchmark score before and after — instead of a hope. `ag evolve --history`
    (and the web UI) can show the real fitness trajectory over time.
  - **A fitness cache.** Re-measuring the incumbent on every evolve is wasteful, so
    we cache fitness keyed by a hash of the evolvable files' contents. A hash we've
    scored before is free.

Stdlib only; append-only JSONL plus a small bounded JSON cache, under state/archive/
(git-ignored). Nothing here ever raises into the evolve loop.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

from .config import STATE_DIR, ensure_dirs

ARCHIVE_DIR = STATE_DIR / "archive"
LINEAGE_FILE = ARCHIVE_DIR / "lineage.jsonl"
CACHE_FILE = ARCHIVE_DIR / "fitness_cache.json"


@dataclass
class Entry:
    """One recorded evolve attempt."""

    ts: str
    parent_hash: str
    candidate_hash: str
    incumbent_fitness: Optional[float]
    candidate_fitness: Optional[float]
    delta: Optional[float]
    tests_passed: bool
    adopted: bool
    verdict: str                     # improved | neutral | regressed | unknown
    rationale: str = ""
    changed: List[str] = field(default_factory=list)
    snapshot_id: str = ""
    candidate_stdev: Optional[float] = None   # benchmark spread (nondeterminism)
    samples: int = 0                          # fitness measurements taken
    margin: Optional[float] = None            # significance threshold used

    def as_dict(self) -> dict:
        return asdict(self)


def _ensure() -> None:
    ensure_dirs()
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def record(entry: Entry) -> None:
    """Append one attempt to the lineage. Best-effort; never raises."""
    try:
        _ensure()
        with LINEAGE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.as_dict()) + "\n")
    except OSError:
        pass


def history(limit: int = 20) -> List[dict]:
    """Most-recent-first list of past attempts (for `--history` / the UI)."""
    try:
        if not LINEAGE_FILE.exists():
            return []
        rows = []
        for line in LINEAGE_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows[::-1][:limit]
    except OSError:
        return []


def adopted_history(limit: int = 50) -> List[dict]:
    return [r for r in history(limit=10_000) if r.get("adopted")][:limit]


# --- fitness cache ---------------------------------------------------------

def _load_cache() -> Dict[str, float]:
    try:
        if CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items()}
    except (OSError, ValueError):
        pass
    return {}


def get_cached_fitness(files_hash: str) -> Optional[float]:
    return _load_cache().get(files_hash)


def set_cached_fitness(files_hash: str, fitness: float, *, cap: int = 200) -> None:
    """Remember a measured fitness for a given evolvable-files hash (bounded)."""
    try:
        _ensure()
        cache = _load_cache()
        cache[files_hash] = round(float(fitness), 3)
        if len(cache) > cap:  # drop oldest-inserted keys (dicts preserve order)
            for k in list(cache.keys())[: len(cache) - cap]:
                cache.pop(k, None)
        CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except (OSError, ValueError):
        pass


def evolvable_hash(files: Dict[str, str]) -> str:
    """Content hash of the evolvable file set — the cache/lineage key."""
    from hashlib import sha256
    blob = json.dumps(files, sort_keys=True)
    return sha256(blob.encode("utf-8")).hexdigest()[:16]


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
