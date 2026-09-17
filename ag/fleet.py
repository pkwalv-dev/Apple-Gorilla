"""Fleet control — master oversight of the agent swarm.

Sub-agents are spawned by name (derived from the role they spawned as) and inherit
their lineage's memory + skills. This module gives you, the operator, a control plane
over that swarm:

- a **registry** of every agent that has been spawned (name, role, parent, depth,
  when, status, how many skills it acquired),
- **master control**: disable an agent (it may not be spawned/used) or re-enable it,
- a **kill switch**: one flag that halts all spawning and signals running loops to
  stop — the emergency brake for the whole lineage.

Stdlib only; state under state/fleet/ (git-ignored). Nothing here raises into a run.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

from .config import STATE_DIR

FLEET_DIR = STATE_DIR / "fleet"
AGENTS_FILE = FLEET_DIR / "agents.jsonl"
STOP_FILE = FLEET_DIR / "STOP"          # presence = kill switch engaged


@dataclass
class AgentRecord:
    agent: str                          # unique namespace/name
    role: str
    parent: str = "root"
    depth: int = 0
    created: str = ""
    last_active: str = ""
    status: str = "active"              # active | done | disabled
    spawns: int = 0                     # sub-agents this one spawned
    skills_acquired: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _ensure() -> None:
    FLEET_DIR.mkdir(parents=True, exist_ok=True)


def _load() -> Dict[str, AgentRecord]:
    out: Dict[str, AgentRecord] = {}
    if not AGENTS_FILE.exists():
        return out
    try:
        for line in AGENTS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out[d["agent"]] = AgentRecord(**{k: d.get(k) for k in
                                                 AgentRecord.__dataclass_fields__ if k in d})
            except Exception:
                continue
    except OSError:
        pass
    return out  # last write per agent wins (file is append-ordered)


def _save(records: Dict[str, AgentRecord]) -> None:
    try:
        _ensure()
        AGENTS_FILE.write_text(
            "".join(json.dumps(r.as_dict()) + "\n" for r in records.values()),
            encoding="utf-8")
    except OSError:
        pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# --- registry --------------------------------------------------------------
def record_spawn(agent: str, *, role: str, parent: str = "root", depth: int = 0) -> AgentRecord:
    recs = _load()
    rec = recs.get(agent) or AgentRecord(agent=agent, role=role, parent=parent,
                                         depth=depth, created=_now())
    rec.role, rec.parent, rec.depth = role, parent, depth
    rec.status = "active"
    rec.last_active = _now()
    recs[agent] = rec
    # bump the parent's spawn count
    if parent in recs:
        recs[parent].spawns += 1
    _save(recs)
    return rec


def set_status(agent: str, status: str) -> bool:
    recs = _load()
    if agent not in recs:
        return False
    recs[agent].status = status
    recs[agent].last_active = _now()
    _save(recs)
    return True


def note_skill_acquired(agent: str, n: int = 1) -> None:
    recs = _load()
    if agent in recs:
        recs[agent].skills_acquired += n
        _save(recs)


def is_disabled(agent: str) -> bool:
    rec = _load().get(agent)
    return bool(rec and rec.status == "disabled")


def list_agents() -> List[AgentRecord]:
    return sorted(_load().values(), key=lambda r: r.created)


def get(agent: str) -> Optional[AgentRecord]:
    return _load().get(agent)


def clear() -> int:
    recs = _load()
    n = len(recs)
    _save({})
    return n


# --- kill switch -----------------------------------------------------------
def engage_kill() -> None:
    """Halt all spawning and signal running loops to stop."""
    _ensure()
    try:
        STOP_FILE.write_text(_now(), encoding="utf-8")
    except OSError:
        pass


def clear_kill() -> None:
    try:
        STOP_FILE.unlink()
    except OSError:
        pass


def kill_active() -> bool:
    return STOP_FILE.exists()
