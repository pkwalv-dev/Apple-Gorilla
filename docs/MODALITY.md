# What form should Apple-Gorilla take? — Web app vs. other modalities

AG currently ships two surfaces over one pipeline: a **CLI** (`python -m ag …`) and a
**built-in web app** (`ag serve`). This note evaluates that choice against the
alternatives, judged on the constraints that actually matter for *this* project:

- **Reach** — how many of {laptop, phone, another desktop} it works on with least effort.
- **Zero-packaging** — AG is stdlib-first and maintained by one person; a modality that
  needs a build toolchain, code signing, or an app-store review is expensive forever.
- **Local-first & private** — AG runs a local Ollama model and keeps `state/`/`profile/`
  on the machine; the surface should not force data off-box.
- **Realtime introspection** — the new live log (thoughts, tool/web calls, errors,
  scorecard) needs a streaming-capable UI.
- **Self-improvement fit** — the surface shouldn't obstruct the snapshot→test→rollback loop.

## The current choice: built-in web app

**Pros**
- **Universal reach for near-zero cost.** One stdlib HTTP server, and every OS with a
  browser — including phones on the same wifi (`--host 0.0.0.0`) — is a client. No
  packaging, no install on the viewing device. This is the single biggest win and the
  reason a "native installer for every OS" is the wrong goal (mobile can't run a Python
  CLI at all).
- **Streaming is native.** The realtime log is just a streamed response the browser reads
  incrementally — no extra infrastructure.
- **Rich rendering for free** — scorecard bars, tables, colour-coded logs, and eventually
  markdown/plots, all with HTML/CSS the runtime already has.
- **Decoupled from the engine.** The UI only speaks `POST /run` + `GET /tools`; the
  pipeline, profile, and evolve loop are identical to the CLI.

**Cons**
- **It's a listening socket.** The CLI is egress-only; `serve` opens an inbound port. It's
  bound to localhost by default and does no auth — fine for a personal tool on a trusted
  machine, but `--host 0.0.0.0` on an untrusted network exposes an unauthenticated
  endpoint that runs your model. (Mitigation below.)
- **Not an "app" your OS knows about** — no dock/taskbar icon, no notifications, no global
  hotkey; you start a server and open a tab.
- **Browser sandbox limits** — no native filesystem, no local hardware access beyond what
  the page is handed by the server.

## The alternatives

### CLI (already shipped)
**Keep it.** It's the automation and power-user surface: scriptable, pipeable, egress-only,
and the natural host for `evolve`, `ingest`, `doctor`, `tools`. It is a poor *interactive*
surface (no live panels, no phone) — which is exactly why the web app complements rather
than replaces it. **Verdict: permanent companion, not a replacement.**

### Terminal UI (TUI — curses/Textual)
Live panels in the terminal; good introspection, no inbound port, no browser.
**But:** desktop-only (no phone), adds a dependency for a real TUI framework, and
duplicates rendering the web app already does better. **Verdict: low priority; the web
app covers the same need with more reach.**

### Native desktop app (Electron / Tauri / PyQt)
Gives a real app icon, notifications, a global hotkey, offline packaging.
**But:** every one is a heavy tax for a solo, stdlib-first project — a JS/Rust build
chain and per-OS bundles (Electron), Rust + platform webviews (Tauri), or large GUI
deps and fragile packaging (PyQt). Tauri is the least-bad if a true desktop app is ever
wanted (small binaries, reuses the existing web UI as its frontend). **Verdict: defer;
adopt Tauri-over-the-existing-web-UI only if a installable desktop app becomes a real
requirement.**

### Menubar / system-tray wrapper
A thin tray icon that starts/stops `ag serve` and opens the tab — keeps the web app's
reach while fixing its worst ergonomic gap (no OS presence). Small, optional, per-OS.
**Verdict: cheap, high-comfort add-on *on top of* the web app — a good near-term nicety.**

### Mobile-native app (iOS/Android)
Best-in-class phone UX, push notifications, home-screen presence.
**But:** two more codebases, two app-store review pipelines, signing, and a network path
back to the machine that runs the model — enormous cost for one maintainer. The web app
already delivers "use it from my phone" with none of this. A **PWA** (add the existing web
page to the home screen, add a manifest + service worker) captures ~80% of the benefit for
a few lines. **Verdict: PWA polish yes; native apps no.**

### Editor extension (VS Code)
Great if AG's main job were in-editor coding assistance. It isn't — AG is a general
prompt executor with self-improvement. **Verdict: out of scope unless the product pivots.**

### MCP server (expose AG's pipeline as a tool to *other* agents)
Instead of only being something a human drives, AG becomes a capability other clients
(Claude Desktop, IDE agents, other MCP hosts) can call: "run this through AG's
optimize→critique→score pipeline." This is the natural counterpart to AG's own
*scaffolded* tool integrations (browser/shell/etc. in the inventory) — the same MCP plumbing
that would let AG *call* tools lets AG *be* a tool.
**Pros:** high leverage, reuses the pipeline verbatim, no UI to maintain, aligns with where
the tool ecosystem is going. **Cons:** not a human surface (complements, doesn't replace the
web app); needs the permission broker extended to MCP-initiated calls. **Verdict: the
highest-value *next* modality after the web app — recommended as the next build.**

### Chat-platform bot (Slack/Discord/Telegram)
Frictionless "message AG from anywhere," notifications for free.
**But:** routes your prompts and answers through a third-party platform (against
local-first/private), and needs a hosted, always-on ingress. **Verdict: only if a
specific "team access" need appears; conflicts with the privacy stance otherwise.**

## Recommendation

1. **Keep the web app as the primary human surface, and the CLI as the power/automation
   surface.** Together they already cover laptop + phone + scripting at near-zero
   maintenance — no other single modality beats that trade for a solo, stdlib-first,
   local-first tool.
2. **Harden the web app** rather than replace it: bind localhost by default (done); add an
   optional shared-secret/token when `--host 0.0.0.0` is used; make it a **PWA** (manifest +
   service worker) so "install to home screen" works without a native app.
3. **Add small OS-presence wrappers opportunistically** — a menubar/tray launcher — without
   taking on a full native-app build chain (Tauri-over-the-web-UI is the escape hatch if a
   real installable desktop app is ever required).
4. **Build an MCP server as the next new modality.** It turns AG from a thing-you-drive into
   a capability-other-agents-call, reuses the pipeline and scorecard as-is, and is the
   natural pair to AG's own (currently scaffolded) tool integrations.

**One-line answer:** the web app is the right universal surface and should stay; the next
step is not a different human UI but exposing the same pipeline over **MCP**, plus light
PWA/tray polish to close the web app's only real gaps (OS presence and remote-access auth).
