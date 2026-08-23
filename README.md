# Apple-Gorilla (AG)

A self-improving prompt executor built on the Claude API. You give it a raw prompt;
it engineers that prompt for the model, executes it, reviews and iterates on the
output (fact-check / vibe-check / formatting), and — separately and safely — evolves
its own tunable source over time.

> **Design stance.** The brief asked AG to "rewrite its source code after every
> output." Unbounded, unsupervised self-modification is how a system bricks itself,
> so AG implements the *disciplined* version of that intent: every self-edit is
> **snapshotted, then validated against AG's own test suite, and adopted only if the
> tests pass — otherwise it is rolled back in milliseconds.** AG never runs on
> unverified code. That guarantee is the whole point; don't remove it.

## What it does

| Stage | Module | Job |
|-------|--------|-----|
| **Optimize** | `ag/pipeline.py` → `optimize` | Rewrites your raw prompt into an efficient, unambiguous prompt for the model **without changing intent**. |
| **Execute** | `ag/model.py` | Calls Claude (adaptive thinking + effort) or an offline stub. |
| **Critique** | `pipeline.py` → `critique` | Scores the answer on correctness, fidelity, "vibe", and formatting, judged against *your* principles. |
| **Iterate** | `pipeline.py` → `revise` | Applies the reviewer's fixes and re-checks, up to `max_iterations`. |
| **Evolve** | `ag/evolve.py` | Proposes small patches to AG's evolvable files, snapshots, **tests**, adopts-or-rolls-back. |

## "You as a standard for intelligence" — principles, not mimicry

AG does **not** imitate your writing. It reads `profile/principles.md` and
`profile/about_me.md` as the *standard its critic holds every output to*, so quality
converges on your bar through underlying principles and iteration — evolutionary, not
imitative. **Fill in `profile/about_me.md`** to calibrate it to you.

## Run it as an app (any OS, phone included)

AG ships a built-in web UI — cross-platform by being a web page, no packaging.

```bash
python -m ag serve --open              # local web app in your browser
python -m ag serve --host 0.0.0.0      # also reachable from your phone on the same wifi
```

The web app streams a **realtime log** of AG's run as it happens — the optimize →
execute → critique → score stages, every web search/fetch, revisions, and errors —
plus live **accuracy / quality / speed** scorecard bars and a **Tools & friction**
inventory panel. See [docs/MODALITY.md](docs/MODALITY.md) for why AG stays a web app
and what modality comes next (MCP).

One-command install + launch (clones, checks, points at Ollama, starts the app):

```powershell
# Windows
iwr https://raw.githubusercontent.com/pkwalv-dev/Apple-Gorilla/main/scripts/install.ps1 | iex
```
```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/pkwalv-dev/Apple-Gorilla/main/scripts/install.sh | bash
```

A truly native installer for every desktop *and* mobile OS isn't one command (mobile
can't run a Python CLI). The web app is the universal answer: run it on one machine,
open it from any browser — laptop or phone. The pipeline, profile principles, and
self-improvement are identical to the CLI.

## Quick start

```bash
pip install -r requirements.txt
python -m ag doctor                 # readiness check
python -m ag --dry-run run "explain why the ocean is salty" --show-prompt
```

With no `ANTHROPIC_API_KEY`, AG runs fully offline in **dry-run** mode so you can see
the machinery. To use the real model:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m ag run "draft a launch email for our beta" --verbose
```

## Commands

```bash
python -m ag run "<prompt>" [--verbose] [--show-prompt] [--model claude-opus-5]
python -m ag ingest <export>   # distill a claude.ai data export into your profile
python -m ag evolve            # attempt a test-gated self-improvement
python -m ag tools [-v|--json] # inventory tools/apps + integration & friction ratings
python -m ag update [--apply]  # check/pull a newer Ollama model build (on demand)
python -m ag versions          # list source snapshots
python -m ag rollback <id>     # restore a snapshot instantly
python -m ag profile           # show the loaded intelligence principles
python -m ag doctor            # environment / readiness check
```

## Backends — run with or without an API key

Set `backend` in `config.json`, or pass `--backend` per command:

| Backend | Key? | Network? | Quality | Use |
|---------|------|----------|---------|-----|
| `auto` (default) | uses Anthropic if `ANTHROPIC_API_KEY` set, else falls back to `dry` | — | — | sensible default |
| `anthropic` | **yes** | yes | best | production answers via Claude |
| `ollama` | **no** | local only | depends on local model | free, offline, keyless |
| `dry` | no | no | **stub only** (no reasoning) | testing the machinery |

```bash
python -m ag doctor                          # shows effective backend + ollama probe
python -m ag --backend dry run "..."         # offline machinery test (stub answers)
python -m ag --backend ollama run "..."      # real answers, keyless, local
```

### Keyless real answers via Ollama

```bash
# one-time
brew install ollama          # macOS; or the Windows/Linux installer from ollama.com
ollama pull llama3.1         # or a bigger model if you have the RAM/VRAM
# run it
ollama serve                 # leave running in a terminal
python -m ag --backend ollama run "your prompt"
```

Ollama needs no account or key and runs fully offline. AG talks to it over
`http://127.0.0.1:11434` (override with `ollama_host`); the whole optimize → critique →
iterate loop is identical — only the model changes. Local quality tracks the model you
pull, and is a weight class below Claude.

**Tuning for your GPU.** VRAM is the binding constraint — a model that fits entirely in the
GPU runs at full speed. AG sends `ollama_keep_alive` (default `30m`, keeps the model
resident across a run's 3–4 calls) and an `ollama_options` passthrough (`num_ctx`,
`num_gpu`, …). See **[docs/OLLAMA.md](docs/OLLAMA.md)** for the efficient-build guide,
including a per-VRAM sizing table and a recommended `config.json`.

**Keeping the model current — on demand, one click.** Model tags get republished; AG checks
whether a newer build exists by comparing manifest digests (no download) and *proposes* an
update — it never polls in the background and never pulls without approval. The web app asks
once after a run with a **Update now? Yes/No** button; the CLI equivalent is `ag update`
(check) / `ag update --apply` (pull).

## Running on another machine (e.g. a desktop with a GPU)

AG is pure Python + stdlib (the `anthropic` SDK is only needed for the `anthropic`
backend). It runs anywhere Python 3.10+ runs — macOS, Windows, Linux. A machine with a
strong GPU and lots of RAM/VRAM runs much larger, much better local models.

Move it with git (recommended) or a plain copy:

```bash
# on this Mac (once):
git add -A && git commit -m "AG"            # then push to a remote you control
# on the desktop:
git clone <your-remote> Apple-Gorilla
cd Apple-Gorilla
pip install -r requirements.txt             # only needed for the anthropic backend
# install + start Ollama (ollama.com), then:
ollama pull llama3.1                         # or e.g. qwen2.5:32b on a big-VRAM GPU
python -m ag --backend ollama run "your prompt"
```

`state/` (run logs + snapshots) and your `profile/` travel with the repo. On a GPU
desktop, pick a larger model in `config.json` (`ollama_model`) — that's where local AG
gets genuinely useful.

## Internet access — permanent, but egress-only

AG has standing outbound internet (`allow_web: true`): it searches and fetches pages and
injects them as **labeled, untrusted reference data** (never instructions — prompt-
injection defense). This is how a local Ollama model "reaches the internet": AG
retrieves on its behalf.

```bash
python -m ag run "what shipped in the latest iRacing update?" --verbose   # web on
python -m ag run "pure reasoning question" --no-web                        # web off
```

**Private / egress-only.** AG makes outbound requests but runs **no server, opens no
listening port, and accepts no inbound connections** — nothing external (Claude,
scrapers, anyone) can query it. It emits **no telemetry**; the only traffic is AG's own
model + web requests. `python -m ag doctor` and `python -m ag host` report this posture.

- Egress still flows through the default-deny `PermissionBroker` ('network' grant);
  `allow_web` auto-grants it. Set `allow_web: false` to require `--web` per run.
- Stdlib only, no API key. Search uses DuckDuckGo (swap the provider in
  `ag/tools/web.py`); `web_fetch` works on any http(s) URL. Works on every backend.

## Host resources

`python -m ag host` inspects the machine (CPU, RAM, GPU/VRAM, disk, connectivity) and AG
scales its own parallelism to your cores and sizes the local model to your GPU. This is
use of the host's *own* resources for AG's work only.

> **Explicitly out of scope, by design:** AG does not scan, exploit, or control routers,
> wifi, or other network devices, does not reach other machines on the network, and has
> no persistence/evasion or anti-shutdown behavior. It is a controllable local tool with
> outbound internet — not autonomous network software. The test-gate, permission broker,
> and human-in-the-loop are load-bearing safety features; don't remove them.

## The self-improvement loop (`evolve`)

1. **Snapshot** every evolvable file into `state/versions/<timestamp>/` (fast, local).
2. **Propose** small patches, given recent run telemetry (`state/runs/`).
3. **Validate** statically (Python compiles, JSON parses) — reject early.
4. **Apply** to the working tree.
5. **Test** in a subprocess (`pytest`).
6. **Adopt** (+ optional git commit) if green; **restore the snapshot** if not.

The gate runs the suite with **pytest** (the tests use its fixtures), so `evolve`
requires it — it ships in `requirements.txt` and the installers add it. If pytest is
absent the gate **fails closed**: `evolve` refuses to run (rather than adopting
unverified code), and `ag doctor` reports `evolve gate: UNAVAILABLE`.

### Directed evolution — scoring + tool inventory steer the loop

Every run is scored on three axes (0–10): **accuracy** and **quality** are judged by
the critic; **speed** is *measured* by AG from wall-clock and token cost against
`speed_budget_s` (a model can't judge its own latency). The blend is stored per run
(`config.json → score_weights`). Separately, `ag tools` inventories every backend,
retrieval/agent/host tool, and scaffolded capability, rating each on **integration**
(how wired-in) and **friction** (10 = frictionless).

`evolve` feeds both into the evolver as a *directed-evolution briefing*: it identifies
the weakest score axis across recent runs and the highest-friction wired tool, and asks
for the single smallest change that lifts one of them. Self-improvement aims at the real
bottleneck instead of editing blindly.

Autonomy is a dial in `config.json`:

- `never` — evolution off.
- `manual` — propose + test, but leave a passing candidate for your review (`--apply` to keep).
- `guarded` — **default**: auto-adopt only when tests pass, else auto-rollback.

Only files listed in `config.json → evolvable_paths` can ever be touched (default:
prompts, principles, config). The safety, permission, and backup modules are **not**
evolvable by design.

## Autonomy, Chrome, and tools — permission-gated

`ag/permissions.py` is a **default-deny** broker. Browser/Chrome control, network,
shell, deletion, sending messages, and spawning sub-agents are all gated capabilities
that require an explicit grant *and* `allow_external_tools: true`. AG has broad
*capability* but narrow *default authority*.

> Live browser automation is intentionally **not** wired to a real driver here — the
> broker and extension point are in place, but connecting AG to Chrome (via an MCP
> client or the Claude Agent SDK) is a deliberate, separately-authorized step, not a
> silent default. Sub-agents (`ag/agents.py`) are bounded and non-recursive.

## Layout

```
ag/
  config.py        paths + Config (config.json)
  model.py         Claude client + offline dry-run stub
  prompts.py       meta-prompts   [evolvable]
  profile.py       loads your principles
  pipeline.py      optimize -> execute -> critique -> iterate
  critic/reviser   (in pipeline.py)
  evolve.py        snapshot -> patch -> test -> adopt/rollback
  backup.py        snapshot / restore / git record
  permissions.py   default-deny capability broker
  agents.py        bounded, gated sub-agents
  cli.py           `python -m ag ...`
profile/           principles.md [evolvable] + about_me.md (fill in)
state/             runs/ (telemetry) + versions/ (snapshots)
tests/             the gate that every self-edit must pass
```

## What is real vs. aspirational

- **Real & tested:** the full pipeline (dry-run + live), prompt optimization, the
  critique/iterate loop, snapshot/rollback, the test-gated evolve loop, the permission
  broker, the CLI. `pytest` is green and rollback-on-failure is proven.
- **Scaffolded (needs your authorization to wire up):** live Chrome/browser control,
  arbitrary network/shell tools. The gates exist; the drivers are deliberately absent.
- **Deliberately not built:** literal unbounded self-rewriting with no test gate. It
  is unsafe, and the whole architecture exists to give you the intent without the risk.
```
