"""Optional sub-agents.

AG can delegate a bounded sub-task to a fresh role-scoped model call. Spawning is
permission-gated ('spawn_agent') and bounded by `max_agents` so autonomy stays
under control. Sub-agents cannot themselves spawn (no recursion by default).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import Config
from .permissions import PermissionBroker


@dataclass
class SubAgentResult:
    role: str
    task: str
    output: str


def spawn(client, cfg: Config, broker: PermissionBroker, *, role: str, task: str,
          max_tokens: int = 8000) -> SubAgentResult:
    broker.require("spawn_agent")
    system = (
        f"You are a focused sub-agent with the single role: {role}. "
        "Do only this task. Be concise and return just the result."
    )
    res = client.complete(system=system, user=task, cfg=cfg, max_tokens=max_tokens)
    return SubAgentResult(role=role, task=task, output=res.text)


def spawn_many(client, cfg: Config, broker: PermissionBroker,
               tasks: List[dict], *, max_agents: int = 3) -> List[SubAgentResult]:
    broker.require("spawn_agent")
    results = []
    for t in tasks[:max_agents]:
        results.append(spawn(client, cfg, broker,
                             role=t.get("role", "worker"), task=t.get("task", "")))
    return results
