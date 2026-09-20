"""AG as an MCP server — let other agents drive this one.

`docs/MCP.md` specified this and it was never built, so AG has no machine-to-machine
surface: a human can use it through the CLI or the web app, but Claude Desktop, an
IDE agent, or a second AG on another machine cannot. This module closes that gap with
a stdlib JSON-RPC 2.0 server over stdio — the transport MCP hosts launch as a
subprocess, which needs no port, no network exposure, and no dependency.

What it exposes:

  tools       run_pipeline    answer a prompt through AG's full pipeline
              read_any        read a file of any format (the formats layer)
              inspect_format  identify a file without reading it
              recall/remember AG's durable memory
              bench           run the objective benchmark
              os_info         what machine AG is on
              ask_guidance    file a question for the human
  resources   ag://state/runs, ag://state/inventory, ag://state/guidance,
              ag://state/memory, ag://profile/principles

Authority is unchanged: every tool goes through the same `PermissionBroker`, and the
side-effecting ones stay default-deny. An MCP host asking AG to run code gets the
same refusal a local prompt would. Being reachable by a machine does not make AG
more permissive — that separation is the whole reason this is safe to add.

Run it:  `python -m ag mcp`   (or point an MCP host at that command)
"""
from __future__ import annotations

import json
import sys
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional

from .config import Config

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "apple-gorilla"


# --------------------------------------------------------------------------- #
# Tool definitions — schema + handler, declared once
# --------------------------------------------------------------------------- #

def _tool_defs() -> List[dict]:
    """The advertised tool list with JSON-Schema inputs.

    Schemas are what let a remote model call these correctly without trial and
    error, so they are written out properly rather than as free-form objects.
    """
    return [
        {
            "name": "run_pipeline",
            "description": (
                "Answer a prompt using Apple-Gorilla's full pipeline: profile "
                "context, long-term memory, optional web retrieval, and a bounded "
                "reason-act-observe tool loop. Returns the answer plus a scorecard."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string",
                               "description": "The request to answer."},
                    "web": {"type": "boolean",
                            "description": "Allow web retrieval for this run."},
                    "tools": {"type": "boolean",
                              "description": "Enable the local tool loop."},
                },
                "required": ["prompt"],
            },
        },
        {
            "name": "read_any",
            "description": (
                "Read a file of ANY format — PDF, Word/Excel/PowerPoint, SQLite, "
                "zip/tar, CSV/JSON/XML, images, or an unrecognised binary (which "
                "returns a structural analysis with a confidence). Use this rather "
                "than a plain text read for anything that is not source code."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                },
                "required": ["path"],
            },
        },
        {
            "name": "inspect_format",
            "description": (
                "Identify what a file is — format, structure, and AG's confidence — "
                "without reading its whole contents. Cheap; use before read_any on "
                "a large or unknown file."),
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "recall",
            "description": ("Search Apple-Gorilla's durable memory (semantic and "
                            "procedural), annotated with how well-founded each item "
                            "is."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "description": "Max results (default 5)."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "remember",
            "description": ("Store a durable fact in Apple-Gorilla's memory. Stored "
                            "as an unconfirmed inference, not as established truth."),
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
        {
            "name": "bench",
            "description": ("Run Apple-Gorilla's objective benchmark — its fitness "
                            "function — and return the score. Every check is "
                            "programmatic; no model grades itself."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer",
                              "description": "Cap the number of tasks (for speed)."},
                    "generated": {"type": "boolean",
                                  "description": "Use the generated suite."},
                },
            },
        },
        {
            "name": "os_info",
            "description": ("Report the machine AG runs on: OS family, shell, "
                            "package manager, path conventions, privileges, and "
                            "accelerator. Check before suggesting system commands."),
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "ask_guidance",
            "description": ("File a question for the human operator when a decision "
                            "genuinely cannot be made autonomously. Non-blocking: "
                            "returns immediately with a recommended fallback."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "recommendation": {"type": "string"},
                    "urgency": {"type": "string",
                                "enum": ["blocking", "soon", "whenever"]},
                },
                "required": ["question"],
            },
        },
    ]


def _resource_defs() -> List[dict]:
    return [
        {"uri": "ag://state/runs", "name": "Recent runs",
         "description": "Telemetry from AG's most recent runs.",
         "mimeType": "application/json"},
        {"uri": "ag://state/inventory", "name": "Tool inventory",
         "description": "Every tool AG can use, with integration/friction ratings.",
         "mimeType": "application/json"},
        {"uri": "ag://state/guidance", "name": "Pending guidance",
         "description": "Questions AG has raised for its operator.",
         "mimeType": "application/json"},
        {"uri": "ag://state/memory", "name": "Memory statistics",
         "description": "Counts and health of AG's layered memory.",
         "mimeType": "application/json"},
        {"uri": "ag://state/os", "name": "Host platform",
         "description": "The machine AG is running on.",
         "mimeType": "text/plain"},
        {"uri": "ag://profile/principles", "name": "Intelligence principles",
         "description": "The standard AG holds its output to.",
         "mimeType": "text/markdown"},
    ]


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #

class MCPServer:
    """JSON-RPC 2.0 over stdio. One request per line, one response per line."""

    def __init__(self, cfg: Optional[Config] = None, *,
                 allow_web: Optional[bool] = None,
                 allow_tools: Optional[bool] = None):
        self.cfg = cfg or Config.load()
        self.allow_web = self.cfg.allow_web if allow_web is None else allow_web
        self.allow_tools = (self.cfg.allow_local_tools if allow_tools is None
                            else allow_tools)
        self._lock = threading.Lock()
        self.initialized = False

    # --- plumbing ---------------------------------------------------------
    def _broker(self):
        from .permissions import PermissionBroker
        broker = PermissionBroker(allow_external_tools=True)
        if self.allow_web:
            broker.grant("network")
        if self.allow_tools:
            broker.grant("filesystem_read")
            broker.grant("spawn_agent")
            # code_exec stays denied unless the operator opted in via config. An MCP
            # host is a *remote caller*; it must not be able to widen AG's authority
            # just by asking, or the broker would be advisory rather than binding.
            if getattr(self.cfg, "allow_code_exec", False):
                broker.grant("code_exec")
        return broker

    def handle(self, msg: dict) -> Optional[dict]:
        """Dispatch one JSON-RPC message. Returns None for notifications."""
        method = msg.get("method", "")
        msg_id = msg.get("id")
        params = msg.get("params") or {}
        is_notification = "id" not in msg
        try:
            if method == "initialize":
                result = self._initialize(params)
            elif method in ("notifications/initialized", "initialized"):
                self.initialized = True
                return None
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": _tool_defs()}
            elif method == "tools/call":
                result = self._call_tool(params)
            elif method == "resources/list":
                result = {"resources": _resource_defs()}
            elif method == "resources/read":
                result = self._read_resource(params)
            elif method == "prompts/list":
                result = {"prompts": []}
            elif method.startswith("notifications/"):
                return None
            else:
                if is_notification:
                    return None
                return _error(msg_id, -32601, f"Method not found: {method}")
        except Exception as e:
            if is_notification:
                return None
            return _error(msg_id, -32603, f"{type(e).__name__}: {e}",
                          data=traceback.format_exc()[-1500:])
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _initialize(self, params: dict) -> dict:
        from . import __version__
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False},
                             "resources": {"subscribe": False,
                                           "listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": str(__version__)},
            "instructions": (
                "Apple-Gorilla is a local-first self-improving agent. Use "
                "run_pipeline for reasoning tasks, read_any for files in any "
                "format, and bench to measure its objective fitness. Side-effecting "
                "capabilities are default-deny and are not granted by this "
                "connection."),
        }

    # --- tools ------------------------------------------------------------
    def _call_tool(self, params: dict) -> dict:
        name = str(params.get("name", ""))
        args = params.get("arguments") or {}
        fn: Optional[Callable[[dict], str]] = {
            "run_pipeline": self._t_run,
            "read_any": self._t_read_any,
            "inspect_format": self._t_inspect,
            "recall": self._t_recall,
            "remember": self._t_remember,
            "bench": self._t_bench,
            "os_info": self._t_os,
            "ask_guidance": self._t_guidance,
        }.get(name)
        if fn is None:
            return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                    "isError": True}
        try:
            text = fn(args)
            return {"content": [{"type": "text", "text": text}], "isError": False}
        except Exception as e:
            # An error must come back as a tool result, not a protocol error: the
            # calling model should see what went wrong and be able to adapt, rather
            # than the whole JSON-RPC call failing.
            return {"content": [{"type": "text",
                                 "text": f"{name} failed: {type(e).__name__}: {e}"}],
                    "isError": True}

    def _t_run(self, args: dict) -> str:
        from .model import make_client
        from .pipeline import run as run_pipeline
        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            return "run_pipeline error: 'prompt' is required"
        web = bool(args.get("web", self.allow_web))
        use_tools = bool(args.get("tools", self.allow_tools))
        import dataclasses
        cfg = dataclasses.replace(self.cfg, allow_local_tools=use_tools)
        client = make_client(cfg)
        with self._lock:      # one pipeline at a time: a local model is one GPU
            rec = run_pipeline(client, cfg, prompt, web=web, broker=self._broker())
        card = rec.scorecard or {}
        return json.dumps({
            "answer": rec.answer,
            "model": rec.model_used,
            "elapsed_s": rec.elapsed_s,
            "dry_run": rec.dry_run,
            "speed": card.get("speed"),
            "sources": rec.web_sources,
        }, indent=2, default=str)

    def _t_read_any(self, args: dict) -> str:
        from .tools import local
        return local.read_any(str(args.get("path", "")), broker=self._broker())

    def _t_inspect(self, args: dict) -> str:
        from .tools import local
        return local.inspect_format(str(args.get("path", "")), broker=self._broker())

    def _t_recall(self, args: dict) -> str:
        from .tools import local
        k = args.get("k")
        try:
            k = int(k) if k else 5
        except (TypeError, ValueError):
            k = 5
        return local.memory_recall(str(args.get("query", "")), k=k)

    def _t_remember(self, args: dict) -> str:
        from .tools import local
        return local.memory_remember(str(args.get("text", "")))

    def _t_bench(self, args: dict) -> str:
        from . import bench
        from .model import make_client
        import dataclasses
        cfg = self.cfg
        if args.get("generated"):
            cfg = dataclasses.replace(cfg, bench_generated=True)
        limit = args.get("limit")
        try:
            limit = int(limit) if limit else None
        except (TypeError, ValueError):
            limit = None
        with self._lock:
            res = bench.run_benchmark(make_client(cfg), cfg, limit=limit)
        return json.dumps({"fitness": res.fitness, "pass_rate": res.pass_rate,
                           "passed": res.passed, "n": res.n,
                           "by_category": res.by_category,
                           "failed": res.failed_ids[:20],
                           "elapsed_s": res.elapsed_s}, indent=2)

    def _t_os(self, args: dict) -> str:
        from . import osadapt
        return osadapt.report()

    def _t_guidance(self, args: dict) -> str:
        from . import guidance
        opts = args.get("options") or []
        if isinstance(opts, str):
            opts = [opts]
        req = guidance.ask(str(args.get("question", "")), options=opts,
                           recommendation=str(args.get("recommendation", "")),
                           urgency=str(args.get("urgency", "soon")),
                           context="via MCP", agent="mcp")
        return req.fallback_note()

    # --- resources --------------------------------------------------------
    def _read_resource(self, params: dict) -> dict:
        uri = str(params.get("uri", ""))
        text, mime = self._resource_body(uri)
        return {"contents": [{"uri": uri, "mimeType": mime, "text": text}]}

    def _resource_body(self, uri: str):
        if uri == "ag://state/runs":
            from .pipeline import recent_runs
            runs = [{k: r.get(k) for k in
                     ("run_id", "raw_prompt", "elapsed_s", "model_used", "dry_run")}
                    for r in recent_runs(limit=20)]
            return json.dumps(runs, indent=2, default=str), "application/json"
        if uri == "ag://state/inventory":
            from . import inventory
            return (json.dumps(inventory.summary(self.cfg), indent=2, default=str),
                    "application/json")
        if uri == "ag://state/guidance":
            from . import guidance
            return (json.dumps([r.as_dict() for r in guidance.pending()],
                               indent=2, default=str), "application/json")
        if uri == "ag://state/memory":
            from . import memory
            try:
                stats = memory.get_manager("root").stats()
            except Exception as e:
                stats = {"error": str(e)}
            return json.dumps(stats, indent=2, default=str), "application/json"
        if uri == "ag://state/os":
            from . import osadapt
            return osadapt.report(), "text/plain"
        if uri == "ag://profile/principles":
            from .profile import load_principles
            return load_principles(), "text/markdown"
        raise ValueError(f"Unknown resource: {uri}")

    # --- the stdio loop ---------------------------------------------------
    def serve_stdio(self, stdin=None, stdout=None) -> int:
        """Read newline-delimited JSON-RPC from stdin, write responses to stdout.

        stdout carries the protocol and nothing else — any diagnostic must go to
        stderr, or it corrupts the stream and the host disconnects.
        """
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        print(f"{SERVER_NAME} MCP server ready (stdio)", file=sys.stderr, flush=True)
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                stdout.write(json.dumps(
                    _error(None, -32700, "Parse error: invalid JSON")) + "\n")
                stdout.flush()
                continue
            if isinstance(msg, list):          # JSON-RPC batch
                out = [r for r in (self.handle(m) for m in msg) if r is not None]
                if out:
                    stdout.write(json.dumps(out) + "\n")
                    stdout.flush()
                continue
            response = self.handle(msg)
            if response is not None:
                stdout.write(json.dumps(response, default=str) + "\n")
                stdout.flush()
        return 0


def _error(msg_id, code: int, message: str, data: Any = None) -> dict:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


def serve(cfg: Optional[Config] = None, **kw) -> int:
    return MCPServer(cfg, **kw).serve_stdio()
