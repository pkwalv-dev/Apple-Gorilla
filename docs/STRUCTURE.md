# Apple-Gorilla — File Structure Plan

Organized by *role*, not just folder. The guiding rule: the parts AG may rewrite are
small and isolated; the parts that keep AG safe are separate and immutable; and
everything AG *generates* is bounded and never committed.

```
Apple-Gorilla/
├── config.json            [evolvable] runtime config (backend, limits, autonomy, LoRA)
├── README.md              overview + run-books
├── requirements.txt       anthropic (Claude backend) + pytest (the evolve gate)
├── requirements-lora.txt  heavy, optional training extras (torch/transformers/peft/…)
├── conftest.py            makes `ag` importable in tests
├── AG.bat / AG.command    one-click launchers (Windows / macOS)
├── docs/
│   ├── STRUCTURE.md        this file
│   ├── MODALITY.md         web app vs. other modalities — analysis + recommendation
│   ├── MCP.md              what an MCP server is + how AG would use one
│   ├── OLLAMA.md           efficient Ollama build guide (VRAM sizing, keep-alive, ctx)
│   └── LORA_WSL.md         verified WSL2/CUDA setup for LoRA training
├── ag/                    the package
│   ├── __main__.py / cli.py    entry point + commands
│   ├── config.py               config + paths (Config dataclass ↔ config.json)
│   ├── model.py                backend clients (Claude / Ollama / dry-run) + auth
│   ├── pipeline.py             optimize → execute → critique → iterate (+ live events)
│   ├── prompts.py        [evolvable] the meta-prompts + AG's identity ("brain")
│   ├── theme.py          [evolvable] web app look (fonts/palette) — AG may tune it
│   ├── profile.py              loads the user principles + about-me
│   ├── scoring.py              accuracy/quality/speed scorecard (directs evolution)
│   ├── reason.py               bounded reason→act→observe tool-use loop
│   ├── inventory.py            tool/app inventory + integration & friction ratings
│   ├── update.py               on-demand Ollama model update check + pull (no polling)
│   ├── ingest.py               distills your past chats into the profile
│   ├── evolve.py               self-improvement loop (snapshot→test→fitness→adopt/rollback)
│   ├── bench.py          [evolvable tasks] objective benchmark = the fitness function
│   ├── archive.py              evolution lineage + measured fitness deltas + cache
│   ├── backup.py               snapshot / restore / prune / git record
│   ├── acquire.py              directed capability acquisition (author→test→register a skill)
│   ├── agents.py               bounded, gated sub-agents
│   ├── fleet.py                agent-swarm registry + master kill switch
│   ├── bundle.py               portable bundle export + portability-constraints audit
│   ├── images.py               image generation/handling (gated, optional)
│   ├── host.py                 host introspection + egress-only posture
│   ├── permissions.py          default-deny capability broker
│   ├── server.py               built-in web app: streaming realtime log + scorecard + LoRA tab
│   ├── lora.py                 [extras only] QLoRA fine-tune of a LOCAL model (weights)
│   ├── memory/                 layered persistent memory, graded by how well-founded it is
│   │   ├── types.py            Memory record + layers + veracity (Origin/priors) + self-other (Subject)
│   │   ├── embed.py            meaning-based embeddings (Ollama + stdlib hashing fallback)
│   │   ├── store.py            MemoryStore interface + JSONL engine (per-agent, per-layer)
│   │   ├── manager.py          hybrid recall (meaning+keyword+recency+belief), corroboration,
│   │   │                       contradiction, decay, self-write gate, graph links, namespaces
│   │   └── reflect.py          the learning loop: episodes → facts + procedures (as hypotheses)
│   ├── tools/
│   │   ├── web.py        [evolvable] permission-gated internet (search/fetch/rerank/extract)
│   │   ├── local.py           offline tools: calc / file read / python_exec / memory
│   │   ├── github.py          vetted-GitHub fetch (allowlist, read-only)
│   │   └── pkg.py             gated dependency installer (pinned, logged)
│   └── skills/                the skill registry — acquired capabilities (per-agent, inherited)
├── profile/
│   ├── principles.md     [evolvable] the intelligence bar AG judges against
│   └── about_me.md            user context (from your claude.ai export)
├── scripts/               setup/util scripts (not imported by the package)
├── state/                 GENERATED — bounded, git-ignored (see below)
│   ├── runs/                   run telemetry (feeds evolve)      cap: max_runs
│   ├── versions/               source snapshots (rollback)       cap: max_snapshots
│   └── lora/                   datasets + trained adapters (build artifacts)
└── tests/                 the gate every self-edit must pass (pytest)
```

## Four roles (the real organizing principle)

1. **Evolvable brain** — the only files AG may rewrite (`evolvable_paths` in config):
   `ag/prompts.py`, `ag/tools/web.py`, `ag/theme.py`, `profile/principles.md`,
   `config.json`. Small, high-churn, safe to iterate. Internet code and the GUI look
   live here on purpose so AG can tune its own retrieval and design within the gate.

2. **Immutable safety core** — never in `evolvable_paths`, so AG cannot touch it:
   `permissions.py`, `backup.py`, `evolve.py`, `host.py`, `model.py`.
   These enforce the gate, the rollback, and egress-only. Load-bearing — keep them out.

3. **Orchestration & capabilities** — the wiring and the tools: `pipeline.py`, `cli.py`,
   `server.py`, `agents.py`, `fleet.py`, `scoring.py`, `inventory.py`, `update.py`,
   `ingest.py`, `acquire.py`, `bundle.py`, `images.py`, `reason.py`, `memory/`,
   `tools/`. Not evolvable (they run the loop and provide the capabilities it uses).

4. **Weight-level self-improvement** — `lora.py` is the one subsystem that changes the
   *brain itself* (a local model's weights), not the scaffolding around it. Optional and
   out of the portable core: its heavy deps live in `requirements-lora.txt` and import
   lazily only at train time. Produces adapters under `state/lora/` (git-ignored).

5. **Generated state** — everything AG produces at runtime: `state/`. Bounded and
   git-ignored; see below.

## Keeping it from becoming a file-size burden

The iterative machinery is the only unbounded producer, so it is capped at the source:

| Artifact | Location | Bound | Enforced by |
|----------|----------|-------|-------------|
| Run telemetry | `state/runs/*.json` | newest `max_runs` (100) | `pipeline._prune_runs` after each run |
| Source snapshots | `state/versions/<ts>/` | newest `max_snapshots` (20) | `backup.prune_snapshots` after each evolve |
| LoRA artifacts | `state/lora/` | manual (large binaries) | git-ignored; delete old adapters as needed |
| Git history | repo | `main` = human only; AG self-commits → `ag/evolve` | `git_commit_evolve` |

- **Snapshots copy only the evolvable files** (~tens of KB), not the whole repo.
- **Run/snapshot caps auto-prune oldest** on every write, so `state/` reaches a steady
  size and stays there — no manual cleanup, no growth over time.
- **`state/` is git-ignored** (`.gitkeep` only), so generated artifacts never bloat the
  repo or a `git clone`. The repo stays lean and portable.
- Tune the caps in `config.json` (`max_runs`, `max_snapshots`); `0` disables pruning.

## Commit policy — `main` is human-only

AG's `evolve` records accepted self-edits with `backup.git_commit_evolve`, which commits
onto the **`ag/evolve`** branch using plumbing that never moves `main` or switches your
checkout. **`main` only ever advances when a human reviews `ag/evolve` and merges it.**
Review with `git log ag/evolve`, adopt with `git merge ag/evolve` (or cherry-pick), or
discard the branch — AG's own snapshots still provide rollback independent of git.
