"""The skill registry — where acquired capabilities live and are inherited.

Storage (git-ignored, portable, per-agent so a lineage can inherit):

    state/skills/<agent>/<name>/manifest.json   # metadata + grants + deps
    state/skills/<agent>/<name>/skill.py         # defines run(args, broker) -> str
    state/skills/<agent>/<name>/test_skill.py    # the gate a skill must pass to exist

A skill is loaded by importing its `skill.py` in isolation and binding its `run`.
`load_tools()` turns registered skills into reason-loop Tool objects, each requiring
its declared broker capabilities — so a skill that needs the network is only offered
when that grant is held, exactly like the built-in tools.

Namespacing mirrors memory: an agent sees its own skills plus its parents' and the
shared tier, so a spawned sub-agent inherits the lineage's capabilities and can
`promote` a newly-authored one upward for siblings to inherit.
"""
from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

SHARED_AGENT = "shared"


@dataclass
class Skill:
    name: str
    description: str
    arg: str = "input"                     # primary args key run() reads
    capabilities: List[str] = field(default_factory=list)  # broker grants run() needs
    deps: List[str] = field(default_factory=list)          # pip deps
    agent: str = "root"                    # owning namespace
    source: str = "authored"               # authored | given | promoted
    created: str = ""
    enabled: bool = True
    version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        return cls(
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
            arg=str(d.get("arg", "input")) or "input",
            capabilities=[str(c) for c in (d.get("capabilities") or [])],
            deps=[str(x) for x in (d.get("deps") or [])],
            agent=str(d.get("agent", "root")),
            source=str(d.get("source", "authored")),
            created=str(d.get("created", "")),
            enabled=bool(d.get("enabled", True)),
            version=int(d.get("version", 1) or 1),
        )


class SkillRegistry:
    """A namespaced registry: this agent's skills plus inherited (parents + shared)."""

    def __init__(self, agent: str = "root", *, parents: Sequence[str] = (),
                 base_dir: Optional[Path] = None):
        from ..config import STATE_DIR
        self.agent = agent or "root"
        chain = [self.agent, *[p for p in parents if p], SHARED_AGENT]
        seen: set = set()
        self.lineage = [a for a in chain if not (a in seen or seen.add(a))]
        self.base = Path(base_dir) if base_dir else (STATE_DIR / "skills")

    # --- paths -------------------------------------------------------------
    def _dir(self, agent: str, name: str) -> Path:
        return self.base / _safe(agent) / _safe(name)

    def _manifest_path(self, agent: str, name: str) -> Path:
        return self._dir(agent, name) / "manifest.json"

    # --- write -------------------------------------------------------------
    def register(self, skill: Skill, code: str, test: str) -> Skill:
        """Persist a skill's manifest, code, and test. Additive: never touches core."""
        skill.created = skill.created or _now()
        skill.agent = skill.agent or self.agent
        d = self._dir(skill.agent, skill.name)
        d.mkdir(parents=True, exist_ok=True)
        (d / "skill.py").write_text(code, encoding="utf-8")
        (d / "test_skill.py").write_text(test, encoding="utf-8")
        self._manifest_path(skill.agent, skill.name).write_text(
            json.dumps(skill.to_dict(), indent=2), encoding="utf-8")
        return skill

    def set_enabled(self, name: str, enabled: bool, *, agent: Optional[str] = None) -> bool:
        agent = agent or self._owner(name)
        if agent is None:
            return False
        sk = self.get(name, agent=agent)
        if sk is None:
            return False
        sk.enabled = enabled
        self._manifest_path(agent, name).write_text(
            json.dumps(sk.to_dict(), indent=2), encoding="utf-8")
        return True

    def remove(self, name: str, *, agent: Optional[str] = None) -> bool:
        agent = agent or self._owner(name)
        if agent is None:
            return False
        import shutil
        d = self._dir(agent, name)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            return True
        return False

    def promote(self, name: str, *, to: str = SHARED_AGENT) -> Optional[Skill]:
        """Copy a skill up to a shared/parent namespace so the lineage inherits it."""
        owner = self._owner(name)
        if owner is None:
            return None
        sk = self.get(name, agent=owner)
        d = self._dir(owner, name)
        code = (d / "skill.py").read_text(encoding="utf-8")
        test = (d / "test_skill.py").read_text(encoding="utf-8")
        promoted = Skill.from_dict({**sk.to_dict(), "agent": to, "source": "promoted"})
        return self.register(promoted, code, test)

    # --- read --------------------------------------------------------------
    def get(self, name: str, *, agent: Optional[str] = None) -> Optional[Skill]:
        agents = [agent] if agent else self.lineage
        for a in agents:
            p = self._manifest_path(a, name)
            if p.exists():
                try:
                    return Skill.from_dict(json.loads(p.read_text(encoding="utf-8")))
                except Exception:
                    continue
        return None

    def _owner(self, name: str) -> Optional[str]:
        for a in self.lineage:
            if self._manifest_path(a, name).exists():
                return a
        return None

    def list(self, *, include_disabled: bool = True) -> List[Skill]:
        """All skills visible to this agent (own + inherited), nearer namespace wins."""
        out: Dict[str, Skill] = {}
        for a in self.lineage:  # lineage[0] is self -> its skills take precedence
            adir = self.base / _safe(a)
            if not adir.exists():
                continue
            for sd in sorted(adir.iterdir()):
                mp = sd / "manifest.json"
                if not mp.exists():
                    continue
                try:
                    sk = Skill.from_dict(json.loads(mp.read_text(encoding="utf-8")))
                except Exception:
                    continue
                if sk.name in out:
                    continue  # a nearer namespace already provided this name
                if include_disabled or sk.enabled:
                    out[sk.name] = sk
        return list(out.values())

    def load_run(self, skill: Skill) -> Optional[Callable]:
        """Import a skill's module in isolation and return its run(args, broker)."""
        owner = self._owner(skill.name) or skill.agent
        path = self._dir(owner, skill.name) / "skill.py"
        if not path.exists():
            return None
        try:
            mod_name = f"ag_skill_{_safe(owner)}_{_safe(skill.name)}"
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            run = getattr(module, "run", None)
            return run if callable(run) else None
        except Exception:
            return None


def get_registry(agent: str = "root", *, parents: Sequence[str] = ()) -> SkillRegistry:
    return SkillRegistry(agent, parents=parents)


def load_tools(broker, *, agent: str = "root", parents: Sequence[str] = (),
               registry: Optional[SkillRegistry] = None) -> list:
    """Return enabled acquired skills as reason-loop Tool objects.

    A skill is offered only when every capability it declares is granted on `broker`
    (or it needs none) — the same default-deny posture as the built-in tools. A skill
    whose module fails to import is silently skipped rather than breaking the loop.
    """
    from ..reason import Tool
    reg = registry or get_registry(agent, parents=parents)
    tools = []
    for sk in reg.list(include_disabled=False):
        if sk.capabilities and not all(
                (broker is not None and broker.check(c)) for c in sk.capabilities):
            continue
        run = reg.load_run(sk)
        if run is None:
            continue
        tools.append(_to_tool(sk, run))
    return tools


def _to_tool(skill: Skill, run: Callable):
    from ..reason import Tool
    # A skill declares its grants internally (run() calls broker.require); the Tool's
    # own `grant` field stays None so availability is decided by the capability check
    # in load_tools, which can require several grants, not just one.
    def _invoke(args, broker):
        try:
            return str(run(args, broker))
        except Exception as e:
            return f"{skill.name} error: {e}"
    desc = (f'{skill.description} '
            f'e.g. {{"tool":"{skill.name}","args":{{"{skill.arg}":"..."}}}}')
    return Tool(skill.name, skill.arg, None, desc, _invoke)


def _safe(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in (name or "x"))
    return keep or "x"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
