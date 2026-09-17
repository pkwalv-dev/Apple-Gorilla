"""Sub-agents — bounded, capability-capped, and able to acquire skills.

A sub-agent is a fresh role-scoped run of the reason→act loop. It:
- runs with a broker whose grants are a SUBSET of its parent's (no privilege
  escalation — a child can never do more than the agent that spawned it),
- has its own skill/memory namespace but INHERITS the parent's + the shared tier,
- can itself acquire new skills when a task needs one; anything it authors is
  promoted to the shared registry so the parent and siblings inherit it,
- is depth-bounded (`cfg.max_subagent_depth`): a child at the limit is not granted
  `spawn_agent`, so recursion cannot run away.

Spawning is still gated by `spawn_agent`; the master kill switch (revoking grants /
stopping the run) halts the whole lineage.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from .config import Config
from .permissions import PermissionBroker


@dataclass
class SubAgentResult:
    role: str
    task: str
    output: str
    agent: str = ""


def _child_broker(parent: PermissionBroker, *, can_spawn: bool) -> PermissionBroker:
    """A broker for the child: parent's grants, minus spawn_agent at the depth limit."""
    child = PermissionBroker(allow_external_tools=parent.allow_external_tools,
                             grants=set(parent.grants))
    if not can_spawn:
        child.revoke("spawn_agent")
    return child


def spawn(client, cfg: Config, broker: PermissionBroker, *, role: str, task: str,
          max_tokens: int = 8000, parent_agent: str = "root", depth: int = 0,
          emit=None) -> SubAgentResult:
    broker.require("spawn_agent")
    from . import fleet, reason

    # Master kill switch: refuse to spawn if the operator has engaged it.
    if fleet.kill_active():
        return SubAgentResult(role=role, task=task,
                              output="(fleet kill switch engaged — spawning halted)")

    child_depth = depth + 1
    max_depth = int(getattr(cfg, "max_subagent_depth", 2) or 2)
    child_agent = f"{parent_agent}.{_slug(role)}-{int(time.time() * 1000) % 100000}"
    child_broker = _child_broker(broker, can_spawn=child_depth < max_depth)
    fleet.record_spawn(child_agent, role=role, parent=parent_agent, depth=child_depth)

    system = (
        f"You are a focused sub-agent with the single role: {role}. "
        "Do only this task. If you lack a capability, you may acquire_skill to gain it. "
        "Be concise and return just the result."
    )
    res = reason.solve(client, cfg, system=system, user=task, broker=child_broker,
                       emit=emit, agent=child_agent, parents=[parent_agent],
                       depth=child_depth)

    # Heredity: promote skills the child authored up to the shared tier so the parent
    # and siblings inherit the new capability.
    n_promoted = _promote_new_skills(child_agent)
    from . import fleet
    if n_promoted:
        fleet.note_skill_acquired(child_agent, n_promoted)
    fleet.set_status(child_agent, "done")
    return SubAgentResult(role=role, task=task, output=res.answer, agent=child_agent)


def _promote_new_skills(child_agent: str) -> int:
    n = 0
    try:
        from . import skills
        reg = skills.get_registry(child_agent)
        for sk in reg.list(include_disabled=False):
            if sk.agent == child_agent:            # authored by this child
                reg.promote(sk.name, to=skills.SHARED_AGENT)
                n += 1
    except Exception:
        pass
    return n


def spawn_many(client, cfg: Config, broker: PermissionBroker,
               tasks: List[dict], *, max_agents: int = 3,
               parent_agent: str = "root", depth: int = 0) -> List[SubAgentResult]:
    broker.require("spawn_agent")
    results = []
    for t in tasks[:max_agents]:
        results.append(spawn(client, cfg, broker,
                             role=t.get("role", "worker"), task=t.get("task", ""),
                             parent_agent=parent_agent, depth=depth))
    return results


def _slug(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (name or "worker"))
    return keep or "worker"
