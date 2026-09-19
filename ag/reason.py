"""A bounded reasoning + tool-use loop (the 'reason and act' capability).

Given a task, AG can think, call a tool, observe the result, and continue — up to
`max_tool_steps` — before answering. This is what lets it *do* things (compute,
read files, run code, use memory) instead of only rewriting text. It degrades
gracefully: if the model calls no tool, the loop just returns its answer, so a weak
local model is never worse off than the plain single-shot path.

Protocol (kept simple and forgiving for small models): each turn the model returns
EITHER a JSON action `{"tool": "<name>", "args": {...}}` OR a final answer. We parse
the first JSON object that carries a "tool" key; anything else is treated as final.

Every tool call goes through the PermissionBroker, so a tool the user hasn't granted
simply isn't offered. Tools are described to the model based on what's actually
available this run.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .config import Config
from .model import extract_json
from .tools import local


@dataclass
class Tool:
    name: str
    arg: str                      # the single primary argument key
    grant: Optional[str]          # broker capability required, or None
    desc: str
    run: Callable[..., str]


def _registry(client=None, cfg=None, agent: str = "root", parents=(),
              depth: int = 0) -> List[Tool]:
    def _delegate(args, broker):
        from . import agents
        role = str(args.get("role", "worker"))
        task = str(args.get("task", "")).strip()
        if not task:
            return "delegate error: provide a 'task' for the sub-agent"
        return agents.spawn(client, cfg, broker, role=role, task=task,
                            parent_agent=agent, depth=depth).output

    def _acquire(args, broker):
        spec = str(args.get("spec", "")).strip()
        if not spec:
            return "acquire_skill error: provide a 'spec' describing the capability"
        from . import acquire
        r = acquire.author_skill(client, cfg, spec, broker=broker, agent=agent,
                                 parents=parents)
        if r.needs_approval:
            return f"{r.reason}\nPLAN: {json.dumps(r.plan)[:400]}"
        return r.reason

    def _github(args, broker):
        from .tools import github
        allow = getattr(cfg, "github_allowlist", []) if cfg else []
        return github.fetch(str(args.get("repo", "")), str(args.get("path", "")),
                            broker=broker, allowlist=allow,
                            ref=str(args.get("ref", "main")))

    def _generate_image(args, broker):
        from . import images
        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            return "generate_image error: provide a 'prompt'"
        try:
            res = images.generate(prompt, cfg,
                                  negative_prompt=str(args.get("negative", "")))
        except Exception as e:
            return f"generate_image error: {e}"
        return f"image generated and saved to {res.path}"

    reg = [
        Tool("calc", "expr", None,
             'exact arithmetic, e.g. {"tool":"calc","args":{"expr":"(17*23)-4"}}',
             lambda args, broker: local.calc(str(args.get("expr", "")))),
        Tool("delegate", "task", "spawn_agent",
             'hand a focused subtask to a fresh sub-agent and get its result back, '
             'e.g. {"tool":"delegate","args":{"role":"researcher",'
             '"task":"list the tradeoffs of X"}}',
             _delegate),
        Tool("recall", "query", None,
             'search AG memory, e.g. {"tool":"recall","args":{"query":"my timezone"}}',
             lambda args, broker: local.memory_recall(str(args.get("query", "")))),
        Tool("remember", "text", None,
             'save a durable fact, e.g. {"tool":"remember","args":{"text":"User prefers metric units"}}',
             lambda args, broker: local.memory_remember(str(args.get("text", "")))),
        Tool("read_file", "path", "filesystem_read",
             'read a local file, e.g. {"tool":"read_file","args":{"path":"./README.md"}}',
             lambda args, broker: local.read_file(str(args.get("path", "")), broker=broker)),
        Tool("list_dir", "path", "filesystem_read",
             'list a directory, e.g. {"tool":"list_dir","args":{"path":"."}}',
             lambda args, broker: local.list_dir(str(args.get("path", ".")), broker=broker)),
        Tool("python_exec", "code", "code_exec",
             'run a short Python snippet (print results), '
             'e.g. {"tool":"python_exec","args":{"code":"print(sum(range(10)))"}}',
             lambda args, broker: local.python_exec(str(args.get("code", "")), broker=broker)),
        Tool("acquire_skill", "spec", "write_skill",
             'ACQUIRE a new capability you lack: describe it and AG will author, test, '
             'and register a reusable tool, then you can call it. '
             'e.g. {"tool":"acquire_skill","args":{"spec":"convert a CSV file to JSON"}}',
             _acquire),
        Tool("github_fetch", "path", "github_fetch",
             'read a file from an allowlisted repo (reference code), '
             'e.g. {"tool":"github_fetch","args":{"repo":"ollama/ollama","path":"README.md"}}',
             _github),
    ]
    # Local image generation, offered only when enabled in config.
    if cfg is None or getattr(cfg, "allow_image_gen", False):
        reg.append(Tool(
            "generate_image", "prompt", None,
            'create an image from a text prompt via the local Stable Diffusion server, '
            'e.g. {"tool":"generate_image","args":{"prompt":"a red bicycle at sunset"}}',
            _generate_image))
    return reg


def available_tools(broker, client=None, cfg=None, *, agent: str = "root",
                    parents=(), depth: int = 0) -> List[Tool]:
    """Only tools whose grant is held (or that need none) are offered this run.

    Built-in tools plus every acquired skill this agent has inherited whose declared
    capabilities are all granted — so AG's toolset grows as it acquires skills."""
    out = []
    for t in _registry(client, cfg, agent, parents, depth):
        if t.grant is None or (broker is not None and broker.check(t.grant)):
            out.append(t)
    try:
        from . import skills
        out.extend(skills.load_tools(broker, agent=agent, parents=parents))
    except Exception:
        pass  # a broken skill must never remove access to the built-ins
    return out


def _tools_system(base_system: str, tools: List[Tool]) -> str:
    catalog = "\n".join(f"- {t.name}: {t.desc}" for t in tools)
    return (
        f"{base_system}\n\n"
        "# Tools\n"
        "You can use tools to compute or look things up before answering. To call a "
        "tool, reply with ONLY a JSON object:\n"
        '  {"tool": "<name>", "args": {...}}\n'
        "You will then receive an OBSERVATION and may call another tool or answer. "
        "When you have enough to respond, reply with your final answer as plain text "
        "(no JSON). Prefer a tool over guessing at arithmetic or file contents.\n\n"
        f"Available tools:\n{catalog}"
    )


def _parse_action(text: str, tools: List[Tool]) -> Optional[dict]:
    """Return {'tool','args'} if the model emitted a valid tool call, else None."""
    names = {t.name for t in tools}
    # Scan every JSON-ish object; take the first that names a known tool.
    for m in re.finditer(r"\{.*?\}", text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("tool") in names:
            return {"tool": obj["tool"], "args": obj.get("args", {}) or {}}
    # Fallback: a single fenced/bare object via extract_json.
    obj = extract_json(text)
    if isinstance(obj, dict) and obj.get("tool") in names:
        return {"tool": obj["tool"], "args": obj.get("args", {}) or {}}
    return None


@dataclass
class ReasonResult:
    answer: str
    steps: List[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def _is_not_an_answer(text: str, names) -> bool:
    """True when the model's "final answer" is plainly not one.

    Small models sometimes end a turn with nothing, or with a bare tool name — a tool
    call whose JSON never got written. Returned as-is that becomes the user's answer,
    and the observations the loop just gathered are thrown away. Kept deliberately
    narrow: only empty output and a naked tool name qualify, so a legitimately terse
    answer ("10063", "Paris") is never second-guessed.
    """
    t = (text or "").strip().strip("`\"'.,: \n\t")
    return not t or t in names


def solve(client, cfg: Config, *, system: str, user: str, broker=None,
          emit=None, max_steps: Optional[int] = None, on_delta=None,
          cancel=None, agent: str = "root", parents=(), depth: int = 0) -> ReasonResult:
    """Run the reason→act→observe loop and return the final answer + trace.

    `on_delta(text)` streams each model call's output live; `cancel` (a Canceller) lets
    a run be stopped at any point. Both are optional. `agent`/`parents`/`depth` scope
    the skill namespace and bound sub-agent recursion.
    """
    from .pipeline import _emit  # reuse the pipeline's safe emitter
    # Only forward these when set, so stub clients that don't accept them still work.
    dkw = {}
    if on_delta is not None:
        dkw["on_delta"] = on_delta
    if cancel is not None:
        dkw["cancel"] = cancel
    tools = available_tools(broker, client, cfg, agent=agent, parents=parents, depth=depth)
    if not tools:  # nothing to use — behave like a normal single call
        res = client.complete(system=system, user=user, cfg=cfg, **dkw)
        return ReasonResult(res.text, [], res.input_tokens, res.output_tokens)

    max_steps = cfg.max_tool_steps if max_steps is None else max_steps
    by_name = {t.name: t for t in tools}
    sys_p = _tools_system(system, tools)
    transcript = f"# Task\n{user}\n"
    steps: List[dict] = []
    tin = tout = 0

    for _ in range(max(1, max_steps)):
        try:
            from . import fleet
            if fleet.kill_active():
                return ReasonResult("(halted: fleet kill switch engaged)", steps, tin, tout)
        except Exception:
            pass
        res = client.complete(system=sys_p, user=transcript + "\nYour move:", cfg=cfg,
                              **dkw)
        tin += res.input_tokens
        tout += res.output_tokens
        action = _parse_action(res.text, tools)
        if not action:
            if steps and _is_not_an_answer(res.text, by_name):
                # It gathered what it needed and then said nothing usable. Ask once
                # more from the same transcript rather than handing that on.
                _emit(emit, "reason", "no usable answer — asking once more",
                      level="tool")
                retry = client.complete(
                    system=system,
                    user=transcript + "\nUsing the observations above, give your "
                                      "final answer as plain text.", cfg=cfg, **dkw)
                tin += retry.input_tokens
                tout += retry.output_tokens
                if not _is_not_an_answer(retry.text, by_name):
                    return ReasonResult(retry.text.strip(), steps, tin, tout)
            return ReasonResult(res.text.strip(), steps, tin, tout)  # final answer
        tool = by_name[action["tool"]]
        _emit(emit, "reason", f"tool: {tool.name}({_short(action['args'])})",
              level="tool")
        try:
            observation = tool.run(action["args"], broker)
        except Exception as e:  # a denied/failed tool must not kill the loop
            observation = f"{tool.name} error: {e}"
        _emit(emit, "reason", f"observation: {_short(observation)}", level="result")
        steps.append({"tool": tool.name, "args": action["args"],
                      "observation": observation[:500]})
        transcript += (f"\nACTION: {json.dumps(action)}\nOBSERVATION: {observation}\n")
        # A freshly-acquired skill should be usable immediately: reload the toolset so
        # the new tool is offered on the next step of THIS run.
        if tool.name == "acquire_skill" and "registered skill" in observation:
            tools = available_tools(broker, client, cfg, agent=agent,
                                    parents=parents, depth=depth)
            by_name = {t.name: t for t in tools}
            sys_p = _tools_system(system, tools)

    # Steps exhausted — force a final answer from what we've gathered.
    res = client.complete(
        system=system,
        user=transcript + "\nUsing the observations above, give your final answer.",
        cfg=cfg, **dkw)
    return ReasonResult(res.text.strip(), steps, tin + res.input_tokens,
                        tout + res.output_tokens)


def _short(x, n: int = 80) -> str:
    s = x if isinstance(x, str) else json.dumps(x)
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"
