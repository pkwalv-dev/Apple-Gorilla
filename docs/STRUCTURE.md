# Apple-Gorilla — File Structure Plan

Organized by *role*, not just folder. The guiding rule: the parts AG may rewrite are
small and isolated; the parts that keep AG safe are separate and immutable; and
everything AG *generates* is bounded and never committed.

```
Apple-Gorilla/
├── config.json            [evolvable] runtime config (backend, limits, autonomy)
├── README.md              overview + run-books
├── requirements.txt       one dep (anthropic), only for the Claude backend
├── conftest.py            makes `ag` importable in tests
├── docs/
│   ├── STRUCTURE.md        this file
│   ├── MODALITY.md         web app vs. other modalities — analysis + recommendation
│   ├── MCP.md              what an MCP server is + how AG would use one
│   └── OLLAMA.md           efficient Ollama build guide (VRAM sizing, keep-alive, ctx)
├── ag/                    the package
│   ├── __main__.py / cli.py    entry point + commands
│   ├── config.py               config + paths
│   ├── model.py                backend clients (Claude / Ollama / dry-run)
│   ├── pipeline.py             optimize → execute → critique → iterate (+ live events)
│   ├── scoring.py              accuracy/quality/speed scorecard (directs evolution)
│   ├── inventory.py            tool/app inventory + integration & friction ratings
│   ├── update.py               on-demand Ollama model update check + pull (no polling)
│   ├── server.py               built-in web app: streaming realtime log + scorecard
│   ├── prompts.py         [evolvable] the meta-prompts (AG's "brain")
│   ├── profile.py              loads the user principles
│   ├── evolve.py               self-improvement loop (snapshot→test→adopt/rollback)
│   ├── backup.py               snapshot / restore / prune
│   ├── permissions.py          default-deny capability broker
│   ├── host.py                 host introspection + egress-only posture
│   ├── agents.py               bounded, gated sub-agents
│   └── tools/
│       └── web.py         [evolvable] permission-gated internet (search/fetch)
├── profile/
│   ├── principles.md      [evolvable] the intelligence bar AG judges against
│   └── about_me.md             user context (from your claude.ai export)
├── state/                 GENERATED — bounded, git-ignored (see below)
│   ├── runs/                   run telemetry (feeds evolve)      cap: max_runs
│   └── versions/               source snapshots (rollback)       cap: max_snapshots
└── tests/                 the gate every self-edit must pass
```

## Four roles (the real organizing principle)

1. **Evolvable brain** — the only files AG may rewrite (`evolvable_paths` in config):
   `ag/prompts.py`, `ag/tools/web.py`, `profile/principles.md`, `config.json`.
   Small, high-churn, safe to iterate. Internet code lives here on purpose.

2. **Immutable safety core** — never in `evolvable_paths`, so AG cannot touch it:
   `permissions.py`, `backup.py`, `evolve.py`, `host.py`, `model.py`.
   These enforce the gate, the rollback, and egress-only. Load-bearing — keep them out.

3. **Orchestration** — the wiring: `pipeline.py`, `cli.py`, `agents.py`, `server.py`,
   `scoring.py`, `inventory.py`, `update.py`. Not evolvable (they run the loop and score it).

4. **Generated state** — everything AG produces at runtime: `state/`. Bounded and
   git-ignored; see below.

## Keeping it from becoming a file-size burden

The iterative machinery is the only unbounded producer, so it is capped at the source:

| Artifact | Location | Bound | Enforced by |
|----------|----------|-------|-------------|
| Run telemetry | `state/runs/*.json` | newest `max_runs` (100) | `pipeline._prune_runs` after each run |
| Source snapshots | `state/versions/<ts>/` | newest `max_snapshots` (20) | `backup.prune_snapshots` after each evolve |
| Git history | repo | `main` = human only; AG self-commits → `ag/evolve` | `git_commit_evolve` |

- **Snapshots copy only the four evolvable files** (~tens of KB), not the whole repo.
- **Both caps auto-prune oldest** on every write, so `state/` reaches a steady size and
  stays there — no manual cleanup, no growth over time.
- **`state/` is git-ignored** (`.gitkeep` only), so generated artifacts never bloat the
  repo or a `git clone`. The repo stays lean and portable.
- Tune the caps in `config.json` (`max_runs`, `max_snapshots`); `0` disables pruning.

## Commit policy — `main` is human-only

AG's `evolve` records accepted self-edits with `backup.git_commit_evolve`, which commits
onto the **`ag/evolve`** branch using plumbing that never moves `main` or switches your
checkout. **`main` only ever advances when a human reviews `ag/evolve` and merges it.**
Review with `git log ag/evolve`, adopt with `git merge ag/evolve` (or cherry-pick), or
discard the branch — AG's own snapshots still provide rollback independent of git.
```
