"""AG's skill system — the substrate of directed capability acquisition.

A *skill* is a small, tested, self-contained tool AG authored (or was given) to
provide a capability it did not previously have. Skills are the heritable unit of
AG's evolution: they are gated on a test before they exist, persist in a registry,
are recalled and reused instead of re-authored, and are inherited by sub-agents.

Public surface:
- Skill, SkillRegistry            — the data model and the store
- get_registry(agent, parents)    — a namespaced registry (own + inherited skills)
- load_tools(broker, agent)       — acquired skills as reason-loop Tools (gated)
- contract                        — what a skill's run() actually reads from args
"""
from __future__ import annotations

from . import contract
from .registry import (Skill, SkillRegistry, SHARED_AGENT, get_registry,
                       load_tools)

__all__ = ["Skill", "SkillRegistry", "SHARED_AGENT", "get_registry", "load_tools",
           "contract"]
