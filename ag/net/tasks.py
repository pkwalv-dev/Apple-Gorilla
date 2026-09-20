"""A tiny persistent task log for distributed jobs.

The water18 proposal keeps tasks in Postgres; AG keeps them in a JSONL file (same
last-write-wins pattern as fleet.py), so a fanout job's parent + shard results survive
a restart and the dashboard can show history. Stdlib only, best-effort.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Dict, List, Optional

from . import CLUSTER_DIR

TASKS_FILE = CLUSTER_DIR / "tasks.jsonl"
MAX_TASKS = 200


def _now() -> float:
    return time.time()


def _load() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not TASKS_FILE.exists():
        return out
    try:
        for line in TASKS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out[d["id"]] = d
            except Exception:
                continue
    except OSError:
        pass
    return out


def _save(tasks: Dict[str, dict]) -> None:
    # Keep only the most recent MAX_TASKS by created time.
    items = sorted(tasks.values(), key=lambda t: t.get("created", 0))[-MAX_TASKS:]
    try:
        CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
        TASKS_FILE.write_text("".join(json.dumps(t) + "\n" for t in items),
                              encoding="utf-8")
    except OSError:
        pass


def create(*, title: str, kind: str, status: str = "queued", node_id: str = "",
           parent_id: str = "", shard_index: int = -1, payload: Optional[dict] = None) -> dict:
    task = {"id": uuid.uuid4().hex[:12], "title": title, "kind": kind, "status": status,
            "node_id": node_id, "parent_id": parent_id, "shard_index": shard_index,
            "payload": payload or {}, "result": {}, "error": "",
            "created": _now(), "finished": 0.0}
    tasks = _load()
    tasks[task["id"]] = task
    _save(tasks)
    return task


def update(task_id: str, **fields) -> Optional[dict]:
    tasks = _load()
    t = tasks.get(task_id)
    if not t:
        return None
    t.update(fields)
    if fields.get("status") in ("done", "failed", "cancelled") and not t.get("finished"):
        t["finished"] = _now()
    tasks[task_id] = t
    _save(tasks)
    return t


def get(task_id: str) -> Optional[dict]:
    return _load().get(task_id)


def children(parent_id: str) -> List[dict]:
    return sorted((t for t in _load().values() if t.get("parent_id") == parent_id),
                  key=lambda t: t.get("shard_index", 0))


def recent(limit: int = 25) -> List[dict]:
    return sorted(_load().values(), key=lambda t: t.get("created", 0),
                  reverse=True)[:limit]


def clear() -> int:
    n = len(_load())
    _save({})
    return n
