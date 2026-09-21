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

from .config import STATE_DIR, Config

FLEET_DIR = STATE_DIR / "fleet"
AGENTS_FILE = FLEET_DIR / "agents.jsonl"
STOP_FILE = FLEET_DIR / "STOP"          # presence = kill switch engaged
KILL_DIR = FLEET_DIR / "kill"           # one marker file per individually-killed agent

# agents.jsonl is load-modify-write shared state. Serial spawns made that safe by
# accident; parallel spawn_many makes it a real lost-update race, so every mutation
# below holds this lock. Reads stay lock-free (last writer wins, self-healing).
_LOCK = __import__("threading").Lock()


@dataclass
class AgentRecord:
    agent: str                          # unique namespace/name
    role: str
    parent: str = "root"
    depth: int = 0
    created: str = ""
    last_active: str = ""
    status: str = "active"              # active | done | disabled | killed
    spawns: int = 0                     # sub-agents this one spawned
    skills_acquired: int = 0
    # --- where this agent runs (populated for both local and remote sub-agents) ---
    node: str = "local"                 # node id, or "local" for this machine
    location: str = ""                  # human-readable host ("desktop @ 192.168.1.5")
    pid: int = 0                        # OS pid of the worker running it (0 if unknown)
    heartbeat: float = 0.0             # epoch seconds of the agent's last heartbeat

    def as_dict(self) -> dict:
        d = asdict(self)
        d["stale"] = self.is_stale()
        return d

    def is_stale(self, stale_s: Optional[float] = None) -> bool:
        """A still-'active' agent whose heartbeat has gone quiet is stale — a candidate
        for reaping. done/disabled/killed agents are never 'stale'."""
        if self.status != "active" or not self.heartbeat:
            return False
        if stale_s is None:
            stale_s = float(getattr(Config.load(), "net_agent_stale_s", 120.0) or 120.0)
        return (time.time() - self.heartbeat) > stale_s


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
def record_spawn(agent: str, *, role: str, parent: str = "root", depth: int = 0,
                 node: str = "local", location: str = "", pid: int = 0) -> AgentRecord:
    with _LOCK:
        recs = _load()
        rec = recs.get(agent) or AgentRecord(agent=agent, role=role, parent=parent,
                                             depth=depth, created=_now())
        rec.role, rec.parent, rec.depth = role, parent, depth
        rec.node, rec.location, rec.pid = node, location, pid
        rec.status = "active"
        rec.last_active = _now()
        rec.heartbeat = time.time()
        recs[agent] = rec
        # bump the parent's spawn count
        if parent in recs:
            recs[parent].spawns += 1
        _save(recs)
        return rec


def heartbeat(agent: str, *, node: str = "", location: str = "", pid: int = 0) -> None:
    """A running agent (local or remote) signals it is still alive. Also refreshes the
    where-it-runs fields so a remote node can report its own host/pid back to the fleet."""
    recs = _load()
    rec = recs.get(agent)
    if not rec:
        return
    rec.heartbeat = time.time()
    rec.last_active = _now()
    if node:
        rec.node = node
    if location:
        rec.location = location
    if pid:
        rec.pid = pid
    _save(recs)


def set_status(agent: str, status: str) -> bool:
    with _LOCK:
        recs = _load()
        if agent not in recs:
            return False
        recs[agent].status = status
        recs[agent].last_active = _now()
        _save(recs)
        return True


def note_skill_acquired(agent: str, n: int = 1) -> None:
    with _LOCK:
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
    try:  # drop any lingering per-agent kill markers too
        for m in KILL_DIR.glob("*"):
            m.unlink()
    except OSError:
        pass
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


# --- per-agent kill + stale reaping ----------------------------------------
def _kill_marker(agent: str):
    # A per-agent stop marker, named so it is filesystem-safe on every OS.
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in agent)
    return KILL_DIR / safe


def kill_agent(agent: str) -> bool:
    """Signal one specific agent (and, by prefix, its descendants) to stop, and mark it
    killed in the registry. A running reason loop checks `should_stop` between steps; a
    remote node polls the same for the runs it hosts."""
    recs = _load()
    if agent not in recs:
        return False
    try:
        KILL_DIR.mkdir(parents=True, exist_ok=True)
        _kill_marker(agent).write_text(_now(), encoding="utf-8")
    except OSError:
        pass
    recs[agent].status = "killed"
    recs[agent].last_active = _now()
    _save(recs)
    return True


def is_killed(agent: str) -> bool:
    if _kill_marker(agent).exists():
        return True
    # A child inherits an ancestor's kill: root.a.b is killed if root.a was.
    parts = agent.split(".")
    for i in range(1, len(parts)):
        if _kill_marker(".".join(parts[:i])).exists():
            return True
    return False


def clear_agent_kill(agent: str) -> None:
    try:
        _kill_marker(agent).unlink()
    except OSError:
        pass


def should_stop(agent: str) -> bool:
    """The single check a running loop makes between steps: global kill switch, this
    agent individually killed, or this agent disabled by the operator."""
    if kill_active() or is_killed(agent):
        return True
    rec = _load().get(agent)
    return bool(rec and rec.status == "disabled")


def reap_stale(stale_s: Optional[float] = None) -> List[str]:
    """Mark every stale 'active' agent as killed and signal it to stop. Returns the ids
    reaped. This is the 'kill stale or superfluous sub-agents' broom for the swarm UI."""
    reaped = []
    recs = _load()
    for name, rec in recs.items():
        if rec.is_stale(stale_s):
            kill_agent(name)
            reaped.append(name)
    return reaped
