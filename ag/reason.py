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
              depth: int = 0, task: str = "") -> List[Tool]:
    def _delegate(args, broker):
        from . import agents
        role = str(args.get("role", "worker"))
        task = str(args.get("task", "")).strip()
        if not task:
            return "delegate error: provide a 'task' for the sub-agent"
        # Optional model override: "specialist" runs this sub-agent on the
        # abliterated coder (long code, security tooling), anything else a model name.
        # A model choice changes WHO thinks, never WHAT is permitted — the child
        # broker is derived from the parent's grants either way.
        model = str(args.get("model", "")).strip()
        return agents.spawn(client, cfg, broker, role=role, task=task,
                            parent_agent=agent, depth=depth, model=model).output

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

    def _ask_guidance(args, broker):
        """Escalate to the human — the third option besides guessing and refusing.

        Deliberately non-blocking. The request is queued for the operator and the
        loop is told to continue with its best option and to SAY that it assumed.
        A blocking ask would hang every unattended run; an unmarked guess is how
        agents mislead people. This is the honest middle.
        """
        question = str(args.get("question", "")).strip()
        if not question:
            return "ask_guidance error: provide a 'question'"
        raw_opts = args.get("options") or []
        if isinstance(raw_opts, str):
            raw_opts = [o.strip() for o in raw_opts.split("|") if o.strip()]
        tried = args.get("tried") or []
        if isinstance(tried, str):
            tried = [tried]
        from . import guidance
        req = guidance.ask(
            question,
            blocked_on=str(args.get("blocked_on", "")),
            tried=tried, options=raw_opts,
            recommendation=str(args.get("recommendation", "")),
            urgency=str(args.get("urgency", "soon")),
            confidence=args.get("confidence", 0.5),
            context=task[:600], agent=agent)
        return req.fallback_note()

    def _os_info(args, broker):
        """What machine am I on, and what commands are valid here."""
        from . import osadapt
        return osadapt.report()

    def _consult_specialist(args, broker):
        """Delegate a subtask to the specialist model (the abliterated coder) — for code,
        strict-format output, or a request the primary would refuse. Returns its reply.

        The specialist's known weakness is discipline, not knowledge — so its answer
        is checked for degenerate output (empty / repetition loop) and retried once
        under a tighter contract before anything reaches the primary's transcript.
        An unusable reply is never passed off as a usable one.
        """
        task = str(args.get("task", "")).strip()
        if not task:
            return "consult_specialist error: provide a 'task'"
        spec = getattr(cfg, "specialist_model", "") if cfg else ""
        if not spec:
            return "consult_specialist error: no specialist model configured"
        import dataclasses
        from .model import looks_degenerate, make_client, ollama_has_model
        if not ollama_has_model(cfg, spec):
            return f"consult_specialist error: {spec} is not available locally"
        scfg = dataclasses.replace(cfg, ollama_model=spec)
        system = ("You are a specialist model assisting the primary model. Do the "
                  "task directly and completely; return only the result.")
        try:
            sclient = make_client(scfg, backend="ollama")
            r = sclient.complete(system=system, user=task, cfg=scfg)
            text = (r.text or "").strip()
            deg = looks_degenerate(text)
            if deg:
                r = sclient.complete(
                    system=system,
                    user=task + "\n\nAnswer in under 150 words. No repetition, no "
                                "scaffolding — start with the answer itself.",
                    cfg=scfg)
                retry_text = (r.text or "").strip()
                if not looks_degenerate(retry_text):
                    return retry_text
                # Still degenerate: hand back the salvageable head with the caveat
                # FIRST, so the primary reads the warning before the content (and the
                # caveat survives being truncated in a run trace).
                return (f"[specialist output truncated: {deg}; use with care]\n"
                        + text[:700])
            return text or "(no output)"
        except Exception as e:
            return f"consult_specialist error: {e}"

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

    def _generate_video(args, broker):
        from . import video
        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            return "generate_video error: provide a 'prompt'"
        try:
            secs = float(args.get("seconds") or 0) or None
        except (TypeError, ValueError):
            secs = None
        try:
            res = video.generate(prompt, cfg, seconds=secs,
                                 negative_prompt=str(args.get("negative", "")))
        except Exception as e:
            return f"generate_video error: {e}"
        return (f"video generated ({res.seconds}s, {res.width}x{res.height}) and "
                f"saved to {res.path}")

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
             'read a local TEXT file, e.g. {"tool":"read_file","args":{"path":"./README.md"}}',
             lambda args, broker: local.read_file(str(args.get("path", "")), broker=broker)),
        Tool("read_any", "path", "filesystem_read",
             'read a file of ANY format — PDF, Word/Excel/PowerPoint, SQLite, zip, '
             'CSV, JSON, image, binary. Use this instead of read_file whenever the '
             'file is not plain text. '
             'e.g. {"tool":"read_any","args":{"path":"./report.pdf"}}',
             lambda args, broker: local.read_any(str(args.get("path", "")),
                                                 broker=broker)),
        Tool("inspect_format", "path", "filesystem_read",
             'identify what a file IS (format, structure, confidence) without '
             'reading all of it — cheap, use it before read_any on a large or '
             'unknown file. '
             'e.g. {"tool":"inspect_format","args":{"path":"./mystery.bin"}}',
             lambda args, broker: local.inspect_format(str(args.get("path", "")),
                                                       broker=broker)),
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
        Tool("ask_guidance", "question", None,
             'ASK THE USER when you genuinely cannot decide — an ambiguous '
             'instruction, a destructive action needing sign-off, or a choice only '
             'they can make. You will NOT be blocked: continue with your best '
             'option and say that you assumed it. '
             'e.g. {"tool":"ask_guidance","args":{"question":"Which database '
             'should I export to?","options":["prod postgres","local sqlite"],'
             '"recommendation":"local sqlite","urgency":"blocking",'
             '"confidence":0.3}}',
             _ask_guidance),
        Tool("os_info", "", None,
             'report THIS machine: OS, shell, package manager, paths, privileges — '
             'check before suggesting or running any system command. '
             'e.g. {"tool":"os_info","args":{}}',
             _os_info),
    ]
    # The specialist model, offered when routing is on and it's actually available.
    if cfg is not None and getattr(cfg, "model_routing", True):
        spec = getattr(cfg, "specialist_model", "")
        if spec and spec != getattr(cfg, "ollama_model", ""):
            reg.append(Tool(
                "consult_specialist", "task", None,
                "delegate a subtask to the specialist model (strong at code, strict "
                "formats, and won't refuse), e.g. {\"tool\":\"consult_specialist\","
                "\"args\":{\"task\":\"write a Python function that ...\"}}",
                _consult_specialist))
    # Local media generation, each offered only when enabled in config.
    if cfg is None or getattr(cfg, "allow_image_gen", False):
        reg.append(Tool(
            "generate_image", "prompt", None,
            "create an image from a text prompt on this machine's GPU, "
            'e.g. {"tool":"generate_image","args":{"prompt":"a red bicycle at sunset"}}',
            _generate_image))
    if cfg is None or getattr(cfg, "allow_video_gen", False):
        reg.append(Tool(
            "generate_video", "prompt", None,
            "create a short video clip from a text prompt on this machine's GPU "
            '(slow — minutes, not seconds), e.g. {"tool":"generate_video","args":'
            '{"prompt":"a red bicycle rolling down a hill at sunset","seconds":3}}',
            _generate_video))
    return reg


def available_tools(broker, client=None, cfg=None, *, agent: str = "root",
                    parents=(), depth: int = 0, task: str = "") -> List[Tool]:
    """Only tools whose grant is held (or that need none) are offered this run.

    Built-in tools plus every acquired skill this agent has inherited whose declared
    capabilities are all granted — so AG's toolset grows as it acquires skills."""
    out = []
    for t in _registry(client, cfg, agent, parents, depth, task):
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


_ATTEMPT_HINT = re.compile(r'\{|"tool"|\btool\s*:', re.I)


def _attempted_tool_call(text: str, names) -> bool:
    """True when the reply LOOKS like a tool call that failed to parse.

    This is the compensation for a model with weak instruction-following (the
    specialist coder's known failure mode): it tries to call a tool but wraps the
    JSON in prose, breaks the quoting, or writes `tool: calc` instead of JSON.
    Treating that as a final answer would discard the whole loop's observations and
    hand the user protocol garbage — so the loop repairs instead of accepting.
    Kept conservative: a prose answer that merely mentions the word 'tool' in a
    sentence does NOT trigger it (requires braces or the protocol marker).
    """
    t = (text or "").strip()
    if not t:
        return False
    if "{" not in t and '"tool"' not in t and "tool:" not in t.lower():
        return False
    if not _ATTEMPT_HINT.search(t):
        return False
    # A brace alone in a prose answer is not an attempt; the protocol marker or a
    # known tool name must also be present.
    if '"tool"' in t or "tool:" in t.lower():
        return True
    return any(re.search(rf"\b{re.escape(n)}\b", t) for n in names)


_MAX_CONSECUTIVE_REPAIRS = 2


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
    t = (text or "").strip().strip("`\"'.,:  \n\t")
    return not t or t in names


def solve(client, cfg: Config, *, system: str, user: str, broker=None,
          emit=None, max_steps: Optional[int] = None, on_delta=None,
          cancel=None, agent: str = "root", parents=(), depth: int = 0,
          budget=None, stop_check=None, on_step=None) -> ReasonResult:
    """Run the reason→act→observe loop and return the final answer + trace.

    `on_delta(text)` streams each model call's output live; `cancel` (a Canceller) lets
    a run be stopped at any point. Both are optional. `agent`/`parents`/`depth` scope
    the skill namespace and bound sub-agent recursion. `budget` (ag/budget.py) caps
    what the run may spend; when omitted it is built from the cfg.budget_* fields, and
    exhausting it returns the best answer so far WITH the stop declared in the text —
    a truncated result is never presented as a complete one.

    `stop_check()` is consulted between steps and lets the operator halt THIS agent
    specifically (the swarm UI's per-agent kill, or a remote node checking back with the
    coordinator). It defaults to the fleet's own signal for this agent. `on_step()` is
    called each step so a running agent can heartbeat its liveness.
    """
    from . import fleet
    from .budget import Budget, BudgetExceeded
    if stop_check is None:
        def stop_check():
            try:
                return fleet.should_stop(agent)
            except Exception:
                return False
    from .pipeline import _emit  # reuse the pipeline's safe emitter
    if budget is None:
        budget = Budget.from_config(cfg)
    # Only forward these when set, so stub clients that don't accept them still work.
    dkw = {}
    if on_delta is not None:
        dkw["on_delta"] = on_delta
    if cancel is not None:
        dkw["cancel"] = cancel

    def _stopped(reason: str, last_text: str, steps: List[dict]) -> ReasonResult:
        base = (last_text or "").strip()
        note = f"(stopped: {reason})"
        answer = f"{base}\n\n{note}" if base else f"{note} before producing an answer."
        _emit(emit, "budget", note, level="info")
        return ReasonResult(answer, steps, tin, tout)

    tools = available_tools(broker, client, cfg, agent=agent, parents=parents,
                            depth=depth, task=user)
    if not tools:  # nothing to use — behave like a normal single call
        res = client.complete(system=system, user=user, cfg=cfg, **dkw)
        return ReasonResult(res.text, [], res.input_tokens, res.output_tokens)

    max_steps = cfg.max_tool_steps if max_steps is None else max_steps
    by_name = {t.name: t for t in tools}
    sys_p = _tools_system(system, tools)
    transcript = f"# Task\n{user}\n"
    steps: List[dict] = []
    tin = tout = 0
    repairs = 0

    for _ in range(max(1, max_steps)):
        if stop_check():
            return ReasonResult("(halted: agent stopped by operator)", steps, tin, tout)
        if on_step is not None:
            try:
                on_step()
            except Exception:
                pass
        try:
            if budget is not None:
                budget.check_wall()
            res = client.complete(system=sys_p, user=transcript + "\nYour move:",
                                  cfg=cfg, **dkw)
            tin += res.input_tokens
            tout += res.output_tokens
            if budget is not None:
                budget.charge_model_call(res.input_tokens, res.output_tokens)
        except BudgetExceeded as e:
            return _stopped(e.reason, "", steps)
        action = _parse_action(res.text, tools)
        if not action:
            if _attempted_tool_call(res.text, by_name) and repairs < _MAX_CONSECUTIVE_REPAIRS:
                # A weak instruction-follower TRIED to call a tool and produced
                # unparseable output. Replying as-is would hand the user protocol
                # garbage, so the loop names the problem and asks again — same
                # transcript, explicit contract. Capped, so a model that cannot
                # comply still ends the loop with its prose answer.
                repairs += 1
                _emit(emit, "reason", "malformed tool call — asking for it again",
                      level="tool")
                transcript += (
                    f"\nASSISTANT (unusable): {res.text[:300]}\n"
                    "OBSERVATION (system): that looked like a tool call but was not "
                    'valid JSON of the form {"tool": "<name>", "args": {...}}. Reply '
                    "with EXACTLY one such JSON object and nothing else, or give your "
                    "final answer as plain prose with no JSON in it.\n")
                continue
            repairs = 0
            if steps and _is_not_an_answer(res.text, by_name):
                # It gathered what it needed and then said nothing usable. Ask once
                # more from the same transcript rather than handing that on.
                _emit(emit, "reason", "no usable answer — asking once more",
                      level="tool")
                try:
                    retry = client.complete(
                        system=system,
                        user=transcript + "\nUsing the observations above, give your "
                                          "final answer as plain text.", cfg=cfg, **dkw)
                    tin += retry.input_tokens
                    tout += retry.output_tokens
                    if budget is not None:
                        budget.charge_model_call(retry.input_tokens,
                                                 retry.output_tokens)
                except BudgetExceeded as e:
                    return _stopped(e.reason, "", steps)
                if not _is_not_an_answer(retry.text, by_name):
                    return ReasonResult(retry.text.strip(), steps, tin, tout)
            return ReasonResult(res.text.strip(), steps, tin, tout)  # final answer
        repairs = 0
        tool = by_name[action["tool"]]
        _emit(emit, "reason", f"tool: {tool.name}({_short(action['args'])})",
              level="tool")
        try:
            if budget is not None:
                budget.charge_tool_call(tool.name)
            observation = tool.run(action["args"], broker)
        except BudgetExceeded as e:
            return _stopped(e.reason, "", steps)
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
                                    parents=parents, depth=depth, task=user)
            by_name = {t.name: t for t in tools}
            sys_p = _tools_system(system, tools)

    # Steps exhausted — force a final answer from what we've gathered. This call is
    # also charged: a budget that stops the loop must not be bypassed by its cleanup.
    try:
        res = client.complete(
            system=system,
            user=transcript + "\nUsing the observations above, give your final answer.",
            cfg=cfg, **dkw)
        if budget is not None:
            budget.charge_model_call(res.input_tokens, res.output_tokens)
    except BudgetExceeded as e:
        return _stopped(e.reason, "", steps)
    return ReasonResult(res.text.strip(), steps, tin + res.input_tokens,
                        tout + res.output_tokens)


def _short(x, n: int = 80) -> str:
    s = x if isinstance(x, str) else json.dumps(x)
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"
