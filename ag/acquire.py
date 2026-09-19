"""Directed capability acquisition — AG extends itself to finish a task.

When a prompt needs a capability AG lacks, it AUTHORS a new skill: a model writes the
tool code plus a test, AG installs any declared dependencies (gated), runs the test in
an isolated subprocess (the SELECTION gate — a skill that fails its own test is
discarded), registers the surviving skill, and records a procedural memory so the
capability is reused and inherited rather than re-authored. This is the whole loop
behind "ask in the prompt → AG acquires what it needs → the task gets done."

Autonomy (config.acquisition_autonomy):
- "ask"  — plan only unless the caller approves (install + run are the side effects a
           human should sign off on); returns a proposed plan otherwise.
- "auto" — proceed end-to-end within the session's grants.

Safety: authoring requires the `write_skill` grant; the skill's test runs with NO
capability grants, so authored code cannot touch the network or filesystem while being
gated. Skills are additive files under state/ — they never modify AG's core.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from . import prompts
from .config import Config
from .model import extract_json
from .permissions import PermissionBroker
from .skills import Skill, get_registry

TEST_TIMEOUT = 60


@dataclass
class AcquireResult:
    attempted: bool
    acquired: bool
    name: str = ""
    reason: str = ""
    plan: dict = field(default_factory=dict)
    test_output: str = ""
    needs_approval: bool = False

    def as_dict(self) -> dict:
        return {"attempted": self.attempted, "acquired": self.acquired,
                "name": self.name, "reason": self.reason, "plan": self.plan,
                "needs_approval": self.needs_approval}


def _authorized_auto(cfg: Config, broker: Optional[PermissionBroker], approve: bool) -> bool:
    if approve:
        return True
    if getattr(cfg, "acquisition_autonomy", "ask") == "auto":
        return True
    # A session can opt into auto by granting a sentinel capability.
    return bool(broker is not None and "acquire_auto" in getattr(broker, "grants", set()))


def author_skill(client, cfg: Config, spec: str, *, broker: PermissionBroker,
                 agent: str = "root", parents=(), approve: bool = False,
                 emit=None) -> AcquireResult:
    """Author, gate, and register a skill that provides the capability `spec` asks for."""
    from .pipeline import _emit
    if not getattr(cfg, "allow_acquire", True):
        return AcquireResult(False, False, reason="acquisition disabled (allow_acquire=false)")
    if broker is None or not broker.check("write_skill"):
        return AcquireResult(False, False,
                             reason="write_skill not granted — cannot author a skill")

    reg = get_registry(agent, parents=parents)
    _emit(emit, "acquire", f"authoring a skill for: {spec[:100]}", level="tool")

    # 1) Ask the model for a skill (code + test) targeting the requested capability.
    res = client.complete(system=prompts.SKILL_AUTHOR_SYSTEM,
                          user=f"# Capability needed\n{spec}\n", cfg=cfg,
                          max_tokens=cfg.meta_output_tokens)
    data = extract_json(res.text) or {}
    name = _slug(str(data.get("name", "")))
    code = data.get("code", "")
    test = data.get("test", "")
    if not (name and isinstance(code, str) and isinstance(test, str) and code and test):
        return AcquireResult(True, False, reason="author returned no usable skill")

    skill = Skill(
        name=name, description=str(data.get("description", "")) or name,
        arg=str(data.get("arg", "input")) or "input",
        capabilities=[str(c) for c in (data.get("capabilities") or [])],
        deps=[str(d) for d in (data.get("deps") or [])], agent=agent, source="authored",
    )
    plan = {"name": name, "description": skill.description, "arg": skill.arg,
            "capabilities": skill.capabilities, "deps": skill.deps,
            "code_preview": code[:400]}

    # 2) Static validation before anything executes.
    err = _compile_error(code) or _compile_error(test)
    if err:
        return AcquireResult(True, False, name=name, plan=plan,
                             reason=f"authored code did not compile: {err}")

    # 3) Autonomy gate: installing deps and running code are the side effects a human
    #    signs off on. In "ask" mode without approval, return the plan and stop.
    if (skill.deps or True) and not _authorized_auto(cfg, broker, approve):
        return AcquireResult(True, False, name=name, plan=plan, needs_approval=True,
                             reason="approval required (acquisition_autonomy=ask): "
                                    "review plan, then approve to install + test")

    # 4) Install declared dependencies (gated).
    if skill.deps:
        from .tools import pkg
        if not broker.check("install_package"):
            return AcquireResult(True, False, name=name, plan=plan,
                                 reason="skill needs deps but install_package not granted")
        out = pkg.install(skill.deps, broker=broker)
        _emit(emit, "acquire", out.splitlines()[0] if out else "install done", level="tool")
        if out.startswith("install refused") or out.startswith("FAILED"):
            return AcquireResult(True, False, name=name, plan=plan,
                                 reason=f"dependency install failed: {out[:200]}")

    # 5) SELECTION gate: the skill must pass its own test, run in isolation with no grants.
    if getattr(cfg, "skill_test_gate", True):
        passed, output = _run_skill_test(code, test)
        if not passed:
            _emit(emit, "acquire", f"skill '{name}' failed its test — discarded", level="error")
            return AcquireResult(True, False, name=name, plan=plan, test_output=output,
                                 reason="skill failed its own test — discarded")

    # 6) Register the surviving skill.
    reg.register(skill, code, test)
    _emit(emit, "acquire", f"acquired skill '{name}' ({skill.description})", level="result")

    # 7) Optional: confirm it doesn't break AG's full suite (slow; off by default).
    if getattr(cfg, "skill_full_suite_gate", False):
        ok, out = _run_full_suite()
        if not ok:
            reg.remove(name, agent=agent)
            return AcquireResult(True, False, name=name, plan=plan, test_output=out,
                                 reason="removed: broke AG's full test suite")

    # 8) Remember it (procedural memory) so it is reused and inherited, not re-authored.
    _remember_skill(cfg, skill, agent)
    return AcquireResult(True, True, name=name, plan=plan,
                         reason=f"acquired and registered skill '{name}'")


def _run_skill_test(code: str, test: str) -> tuple[bool, str]:
    """Run the skill's test in a throwaway dir via pytest, in a subprocess. The skill
    executes here with NO broker grants (its test must be offline/deterministic), so
    gating never grants authored code network or filesystem authority."""
    try:
        d = Path(tempfile.mkdtemp(prefix="ag_skill_"))
        (d / "skill.py").write_text(code, encoding="utf-8")
        (d / "test_skill.py").write_text(test, encoding="utf-8")
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "test_skill.py"],
            cwd=d, capture_output=True, text=True, timeout=TEST_TIMEOUT)
        return r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        return False, f"skill test timed out after {TEST_TIMEOUT}s"
    except Exception as e:  # pragma: no cover - defensive
        return False, str(e)


def _run_full_suite() -> tuple[bool, str]:
    from .config import ROOT
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header"],
                           cwd=ROOT, capture_output=True, text=True, timeout=300)
        return r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    except Exception as e:  # pragma: no cover
        return False, str(e)


def _remember_skill(cfg: Config, skill: Skill, agent: str) -> None:
    try:
        from . import memory
        mgr = memory.get_manager(agent, cfg=cfg)
        mgr.remember(
            f"Acquired skill '{skill.name}': {skill.description}. "
            f"Use the '{skill.name}' tool when this capability is needed.",
            kind=memory.MemoryKind.PROCEDURAL, source="acquire", importance=0.75,
            tags=["skill", skill.name],
            # Self-knowledge AG earned by observation: the skill exists because it was
            # authored here and passed its own test gate. That is the one kind of claim
            # about AG that AG is entitled to record.
            subject=memory.Subject.SELF, volatile=False)
    except Exception:
        pass


def _compile_error(src: str) -> Optional[str]:
    try:
        compile(src, "<skill>", "exec")
        return None
    except SyntaxError as e:
        return f"syntax error: {e}"


def _slug(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c == "_") else "_"
                   for c in (name or "").strip().lower())
    keep = keep.strip("_")
    return keep or ""
