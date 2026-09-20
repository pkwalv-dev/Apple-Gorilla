# Apple-Gorilla — The Ceiling Plan

*A full-system improvement plan: what AG is, where its ceiling actually sits, and the
ordered set of changes that raises it — without pretending a local 8B model is AGI.*

---

## 0. What AG is today (read of the codebase as of `208bce2`)

AG is a **local-first, self-improving agent harness**: ~14k lines of stdlib-heavy
Python, 349 hermetic tests, no required third-party dependency outside the optional
`anthropic` SDK and `pytest`.

| Layer | Modules | State |
|---|---|---|
| Entry | `cli.py` (915), `__main__.py`, `AG.bat`/`AG.command`, `scripts/install.*` | 20 subcommands, one-click launchers |
| Serving | `server.py` (2328), `controls.py`, `theme.py` | stdlib `ThreadingHTTPServer`, NDJSON streaming, one inline HTML page |
| Execution | `pipeline.py` (658), `reason.py` (340), `model.py` (681), `routing.py` | single-call answer + bounded reason→act→observe loop; Ollama/Anthropic/dry backends; primary↔specialist routing with failure fallback |
| Memory | `memory/` (2100) | episodic/semantic/procedural, embeddings w/ hashing fallback, veracity & belief, contradiction detection, reflection loop, per-agent namespaces |
| Self-change | `evolve.py` (742), `bench.py` (330), `archive.py`, `backup.py` | snapshot → propose → static-validate → pytest gate → statistical fitness gate → adopt/rollback, commits to `ag/evolve` never `main` |
| Self-extension | `acquire.py`, `skills/` | model authors a skill + test, test runs with zero grants, survivors registered & inherited |
| Agents | `agents.py`, `fleet.py` | depth-bounded sub-agents, grant subsetting, registry + kill switch |
| Weights | `lora.py` (1059) | QLoRA on a local base, dataset distilled from memory + teacher set, GGUF export back into Ollama |
| Media | `comfy.py`, `images.py`, `video.py` | ComfyUI/A1111 image + Wan 2.2 video, workflows as swappable JSON data |
| Safety | `permissions.py` (65) | default-deny capability broker, `GATED` set, audit log |

**The architecture is genuinely good.** Four properties are load-bearing and must
survive every change below:

1. **Keep-if-better selection** against a programmatic fitness function (no model
   grades itself).
2. **Default-deny authority** — broad capability, narrow standing permission.
3. **Data-not-code extension points** — bench tasks, ComfyUI workflows, skills,
   memory, routing profiles are all files, not patches.
4. **Interface-behind-engine** — `MemoryStore`, `Embedder`, `Tool`, backends are all
   swappable without touching callers.

---

## 1. Where the ceiling actually is

Honest diagnosis, ranked by how much each one caps the system:

| # | Ceiling | Evidence in code | Severity |
|---|---|---|---|
| C1 | **The fitness function is 14 hand-written tasks.** Evolution can only climb what it can measure, so the entire self-improvement loop asymptotes at "answers 14 trivia/arith prompts." | `bench.DEFAULT_TASKS` | **critical** |
| C2 | **Only 5 files are evolvable** (`prompts.py`, `tools/web.py`, `theme.py`, `principles.md`, `config.json`). Everything that decides *behaviour* is out of reach. | `config.evolvable_paths` | **critical** |
| C3 | **AG can only read what `read_file` can `decode('utf-8')`.** A PDF, a zip, a sqlite db, a docx, a parquet, an unknown binary → garbage. This is the single biggest *capability* gap versus the "any format" goal. | `tools/local.read_file` | **critical** |
| C4 | **No way to ask the human a question mid-task.** The agent either guesses or refuses; there is no structured "I need guidance" channel. Directly contradicts the design goal of staying in its lane. | none | **high** |
| C5 | **OS adaptation is a read-only report.** `host.py` describes the machine; nothing *adapts* to it (no shell/pkg-manager/path abstraction, no WSL/container/Android awareness). | `host.py` | high |
| C6 | **Serial everything.** `bench` runs N tasks × M samples one at a time; evolve therefore costs `bench_samples × tasks` sequential model calls. Memory `get()` calls `all()` which re-reads and re-parses every JSONL line from disk. Embeddings are recomputed per call. | `bench.run_benchmark`, `memory/store.py`, `memory/manager.py` | high |
| C7 | **No machine-to-machine surface.** `docs/MCP.md` specifies an MCP server in detail; `ag/mcp_server.py` does not exist. AG cannot be used *by* other agents. | docs only | high |
| C8 | **`ag serve --host 0.0.0.0` is unauthenticated.** Their own `MODALITY.md` flags it: an open port that runs your model, executes tools, and can trigger `evolve`. | `server.serve` | high (security) |
| C9 | **GUI is one 2328-line module with an inline HTML string.** Hard to evolve, and `theme.py` is the only part AG may touch. | `server.py` | medium |
| C10 | **Skill acquisition has no reuse-first step.** Every request that needs a capability can author a *new* skill rather than searching for an existing one; and skills are never re-tested after registration (silent rot). | `acquire.py` | medium |
| C11 | **No cost/latency budget.** `max_tool_steps: 4` is the only bound. A run cannot be told "you have 20 seconds and 3 model calls." | config | medium |
| C12 | **Sub-agents are sequential and shallow.** `spawn_many` loops; no parallelism, no shared scratchpad, no result aggregation/voting. | `agents.spawn_many` | medium |

---

## 2. The stance: a *bounded* everything-agent

The request is "an everything AGI that understands it can't be true AGI on limited
local compute." That is a design constraint, not a disclaimer. It resolves into four
rules the implementation obeys:

1. **Reach is universal; authority is not.** AG should be *able* to open any file on
   any OS and talk to any host — and should do none of it without a grant.
2. **Competence is declared, not assumed.** Every subsystem reports its own
   confidence. Below threshold, AG escalates instead of hallucinating.
3. **Escalation is a first-class action, not a failure.** "Ask the human" ranks
   alongside "call a tool" in the reason loop, with a structured question, the options
   considered, and what it already tried.
4. **The ceiling is raised by improving the *measurement*, not the *claims*.** Every
   capability added must come with a way to score whether it works, or it cannot be
   evolved and will rot.

---

## 3. The plan

Six phases. Phase 1 is implemented in this change set; 2–4 are specified precisely
enough to execute directly; 5–6 are the long-horizon ceiling.

### Phase 1 — Universal I/O, adaptability, escalation, and an honest speedup  ✅ *shipped here*

| Ceiling | Change | Module |
|---|---|---|
| C3 | **Universal format layer.** A sniff→handler registry covering text, JSON/JSONL/NDJSON, delimited (auto-detected delimiter), XML/HTML, INI/TOML-ish, YAML-ish, Markdown, ZIP/TAR/GZIP, OOXML (`.docx`/`.xlsx`/`.pptx` — parsed as zip+XML, stdlib only), SQLite, PDF (stream text extraction), PNG/JPEG/GIF/BMP/WEBP (dimensions from headers), WAV/MP3/MP4 (container probes), ELF/PE/Mach-O, plus base64 and hexdump fallbacks. | `ag/formats.py` |
| C3 | **Inference for formats that have no handler.** Structural forensics on unknown bytes: BOM/encoding sniff, printable ratio, Shannon entropy, magic-prefix extraction, line-structure and delimiter histograms, fixed-record-size detection by divisor scoring, token-dictionary detection. Emits a *hypothesis with confidence* that the model can reason about — and which routes to `acquire_skill` or a guidance request when confidence is low. This is the "interpret new formats" mechanism. | `ag/formats.py` |
| C5 | **OS adaptation layer.** Detects Linux/macOS/Windows/BSD + WSL1/2, Docker/Podman/k8s, Termux/Android, CI; resolves shell, package manager (apt/dnf/pacman/apk/zypper/brew/winget/choco/scoop/pkg), open-file command, path conventions, case sensitivity, line endings, and the *unknown-OS* fallback with a POSIX-assumption note. Everything is a capability record with confidence, never a hard-coded branch. | `ag/osadapt.py` |
| C4 | **Guidance protocol.** A structured `ask_guidance` tool + persistent queue: question, why it's blocked, options with tradeoffs, what was already tried, a recommendation, and an urgency. CLI (`ag guidance`) and the reason loop both read/answer it. This is the "stays in its lane, asks when needed" mechanism made mechanical. | `ag/guidance.py` |
| C6 | **Caching + parallelism.** mtime-keyed store cache (kills the re-parse-everything-on-every-recall cost), LRU embedding cache, and a thread-pooled benchmark (`bench_workers`) that cuts evolve wall-clock by ~N×. | `ag/cache.py`, `ag/bench.py` |
| C1 | **Self-growing fitness function.** Deterministic, seeded *generators* of objectively-checkable tasks (arithmetic, base conversion, date math, string/text ops, JSON transforms, sorting, regex, unit conversion, logic) with difficulty tiers and a held-out validation split. The benchmark stops being 14 fixed items and becomes an unbounded, non-memorizable suite — which is precisely what raises the evolution ceiling. | `ag/benchgen.py` |
| C7 | **MCP server.** stdio JSON-RPC 2.0, stdlib only, exposing `run_pipeline`, `read_any`, `recall`/`remember`, `bench`, `os_info`, `ask_guidance` as tools plus AG state as resources — so Claude Desktop, an IDE agent, or another AG instance can drive this one. | `ag/mcp_server.py` |
| C8 | **Bind-time auth.** A token is required (and auto-generated, printed once) whenever `serve` binds anything other than loopback. Loopback behaviour is unchanged. | `ag/server.py` |

### Phase 2 — Raise what evolution is allowed to touch

- **C2 — tiered evolvability.** Replace the flat `evolvable_paths` list with three
  tiers: `tune` (prompts/theme/config — today's behaviour), `extend` (routing
  heuristics, format handlers, bench generators, inventory scorers — additive code
  behind stable interfaces), and `core` (permanently frozen: `permissions.py`,
  `backup.py`, `evolve.py`, `model.py`, `bench.py` checkers). Each tier gets a
  stricter gate: `extend` edits must additionally keep a **held-out** validation split
  green, so evolution cannot overfit the tasks it can see.
- **Anti-reward-hacking invariants.** Hash-pin the verifier functions and the
  validation split; refuse adoption if either changed during a cycle. Add a
  `tests/test_evolve_integrity.py` asserting the frozen set and the checker hashes.
- **Multi-candidate evolution.** Propose *k* patches per cycle, benchmark them in
  parallel (Phase 1 gave us the pool), adopt the argmax — a population step instead of
  a single hill-climb trial. Archive all k with their measured deltas so the lineage
  records the search, not just the winner.
- **Regression corpus.** Every failed benchmark task that later passes becomes a
  permanent regression item; every user-reported bad answer becomes a task with a
  human-supplied checker. The benchmark grows from *lived failure*, which is the only
  data source that reliably predicts real use.

### Phase 3 — Harness, agents, and cost discipline

- **C11 — budgets.** A `Budget(wall_s, model_calls, tokens, tool_calls)` threaded
  through `pipeline` → `reason` → `agents`, enforced at every step boundary, reported
  in the scorecard, and *learned*: routing records which task tags need which budget,
  so simple prompts stop paying for deep loops.
- **C12 — parallel sub-agents with aggregation.** `spawn_many` becomes a thread pool
  with a shared read-only scratchpad, plus three aggregation modes: `first-valid`,
  `vote` (majority on a checkable answer), and `synthesize`. Depth and grant subsetting
  stay exactly as they are.
- **C10 — reuse-first acquisition + skill health.** Before authoring, semantic-search
  the registry and the procedural memory layer; author only on a miss. Re-run every
  skill's own test on a schedule and on Python-version change; auto-disable and report
  a rotted skill instead of silently failing at call time.
- **Tool result typing.** `Tool.run` returns a structured `Observation`
  (`ok`, `value`, `text`, `cost_s`, `truncated`) instead of a bare string, so the loop
  can distinguish "tool failed" from "tool said the word error", and so observations
  can be summarised when they blow the context window.
- **Context budgeter.** Today `exec_sys` is assembled by string concatenation with no
  token accounting; a long web fetch can silently evict memory. Add an explicit
  packer with per-section budgets and priority-ordered eviction.

### Phase 4 — GUI

- **C9 — split `server.py`** into `server/app.py` (routing), `server/api.py`
  (handlers), and `server/ui/` (`page.html`, `app.js`, `theme.css` as real files
  served from disk). This is what makes the GUI *evolvable*: AG can rewrite
  `ui/app.js` under the same gate that protects everything else.
- **Workspace view.** A file tree that uses the Phase-1 format layer: click any
  file — pdf, xlsx, sqlite, unknown binary — and see AG's interpretation, structure
  summary, and confidence. This turns C3 into something visible.
- **Guidance inbox.** Pending questions with their options rendered as buttons; the
  answer flows straight back into the blocked run.
- **Evolution dashboard.** Fitness over time, per-category pass rates, the candidate
  population from each cycle, and a diff viewer with accept/reject per hunk.
- **PWA manifest + service worker** so the phone install is a real home-screen app,
  and SSE keepalives so mobile Safari stops dropping the stream.

### Phase 5 — Weights and the local-compute reality

- **Distil the harness, not the chat.** The LoRA dataset should be built from
  *verified* traces: benchmark tasks AG got right after tool use, successful tool-call
  JSON, successful format interpretations. Training the model to be a better
  *operator of this harness* is the highest-value use of 8GB of VRAM — far better than
  general instruction tuning it will never win at.
- **Speculative/draft routing.** Answer with a 1–3B model, verify with the 8B only
  when the checker or a confidence heuristic says it matters. Most turns are easy; pay
  for the big model only on the hard ones.
- **Honest capability card.** Ship a generated `CAPABILITY.md`: measured fitness per
  category, per-model, on *this* machine. This is the artifact that keeps the system
  in its lane — it can point at its own numbers instead of claiming competence.

### Phase 6 — The open-ended ceiling

- **Open-ended task proposal (POET/OMNI-style).** A proposer generates *new* task
  families at the frontier of what AG currently fails, with programmatic checkers;
  tasks that are too easy or unsolvable are culled. The benchmark then co-evolves with
  the agent, which is the only known way to keep keep-if-better from stalling.
- **Federated lineage.** `bundle.py` already exports a portable AG. Make the evolution
  archive mergeable: two AGs on two machines exchange *validated* patches with their
  measured deltas, each re-validating locally before adoption. Improvement compounds
  across machines without a server.
- **Interpreter bootstrapping.** When the format layer meets an unknown container
  repeatedly, it should propose a *handler skill* — authored, tested against the
  samples it has collected, and registered. The system then literally learns to read
  new formats, which is C3's endgame.

---

## 4. Non-goals (kept explicit, kept honest)

- No unbounded self-modification: the two gates and the frozen core stay.
- No network/router/device control, no persistence or anti-shutdown behaviour.
- No pretending a local 8B is Claude: routing escalates and the capability card
  reports the gap rather than hiding it.
- No new hard dependencies in the core: stdlib-first is what makes AG portable to the
  weird OS, and portability *is* the "runs anywhere" feature.

---

## 5. What shipped in this change set

```
ag/formats.py      universal format sniffing, parsing, and unknown-format inference
ag/osadapt.py      OS/platform adaptation with graceful unknown-OS handling
ag/guidance.py     structured "ask the human" protocol + persistent queue
ag/cache.py        mtime-keyed file cache + LRU, used by memory store and embeddings
ag/benchgen.py     deterministic generators for an unbounded fitness suite
ag/mcp_server.py   stdio JSON-RPC MCP server exposing AG to other agents
ag/bench.py        parallel task execution (bench_workers)
ag/memory/store.py cached reads; get() no longer re-parses every layer
ag/memory/embed.py LRU-cached embeddings
ag/reason.py       new tools: read_any, inspect_format, ask_guidance, os_info
ag/tools/local.py  read_any / inspect_format tool bodies
ag/server.py       token auth on non-loopback binds
ag/cli.py          ag read / ag inspect / ag guidance / ag osinfo / ag mcp / ag bench --generated
ag/inventory.py    reports for the new subsystems
tests/             7 new test modules
```

Every change keeps the existing 349 tests green and adds its own.
