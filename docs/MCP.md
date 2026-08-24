# What is an MCP server? (and how AG would use one)

**MCP — the Model Context Protocol** — is an open standard (introduced by Anthropic in late
2024) for connecting LLM applications to external capabilities through one uniform
interface. Before MCP, every app wired up every tool in its own bespoke way; MCP is the
"USB-C port" for that — write a capability once as an MCP *server*, and any MCP-aware
*client* can use it.

## The three roles

- **Host** — the LLM application the user interacts with (Claude Desktop, Claude Code, an
  IDE assistant). It contains one or more clients.
- **Client** — the connector inside the host that speaks MCP to exactly one server.
- **Server** — a separate program that *exposes capabilities*. This is the piece you build.
  It does not contain a model; it just offers things a model (via the host) can use.

```
[ model in the Host ]  ⇄  Client  ⇄  MCP  ⇄  Server  ⇄  your code / API / files
```

## What a server exposes (its primitives)

- **Tools** — model-callable functions with a JSON-Schema input (e.g. `run_pipeline`,
  `search_web`, `create_ticket`). The model decides when to call them; the host asks the
  user to approve side effects.
- **Resources** — readable data the host can load as context (files, records, a config,
  telemetry) addressed by URI. Read-only, no side effects.
- **Prompts** — reusable, parameterized prompt/workflow templates the user can invoke.

## How it talks

Messages are **JSON-RPC 2.0** over one of two transports:

- **stdio** — the host launches the server as a local subprocess and talks over
  stdin/stdout. Best for local, personal tools (no ports, no network exposure).
- **Streamable HTTP (with SSE)** — the server runs as an HTTP service for remote or
  multi-client use.

The connection opens with a capability **handshake** (each side advertises what it
supports), after which the client can `list` and `call` tools, `read` resources, etc.

A minimal server, conceptually:

```
server = McpServer("apple-gorilla")

@server.tool()                 # advertise a tool + its input schema
def run_pipeline(prompt: str, web: bool = False) -> dict:
    rec = pipeline.run(make_client(cfg), cfg, prompt, web=web, broker=broker)
    return {"answer": rec.answer, "scorecard": rec.scorecard}

server.run(transport="stdio")  # host launches this as a subprocess
```

The host lists `run_pipeline`, the model calls it when useful, the user approves, and the
result flows back as context — no custom integration on the host side.

## Why this matters for Apple-Gorilla — two directions

AG's inventory already frames its tools as *scaffolded* (gates present, drivers absent).
MCP is the natural driver, and it cuts both ways:

1. **AG as an MCP _server_** — expose AG's `optimize → execute → critique → score` pipeline
   as a `run_pipeline` tool (plus `tools_inventory` and `recent_runs` as resources). Then
   Claude Desktop, an IDE agent, or any MCP host can say "run this through AG" and get back
   the answer *and* the accuracy/quality/speed scorecard. This is the recommended next
   modality in [MODALITY.md](MODALITY.md): it reuses the pipeline verbatim and needs no UI.

2. **AG as an MCP _client_** — instead of hand-writing each browser/shell/email driver, AG
   connects to existing MCP servers for those capabilities. The `permissions.py` broker maps
   cleanly onto MCP's approval model: a gated capability becomes "this MCP tool call requires
   a grant." One integration surface (MCP) replaces N bespoke ones.

**Safety fit.** MCP keeps the same guardrails AG already relies on: servers are separate
processes with only the authority you give them, tool calls are explicit and
user-approvable, and resources are read-only. AG's default-deny broker and human-in-the-loop
stay exactly where they are — MCP just standardizes what sits on the other side of the gate.

## If/when we build it

- Start with a **stdio** server exposing one tool, `run_pipeline` — smallest useful surface,
  no network exposure, drop-in for Claude Desktop / Claude Code.
- Add `tools_inventory` and `recent_runs` as **resources** so a host can see AG's state.
- Extend the permission broker with an `mcp` capability so AG-as-client calls are gated like
  every other side-effecting action.
- Keep it its own module (e.g. `ag/mcp_server.py`) in the *orchestration* role — not
  evolvable, like `server.py`.
