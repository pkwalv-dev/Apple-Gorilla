# Apple-Gorilla (AG)

A self-improving prompt executor built on the Claude API. You give it a raw prompt;
it engineers that prompt for the model, executes it, reviews and iterates on the
output (fact-check / vibe-check / formatting), and — separately and safely — evolves
its own tunable source over time.

> **Design stance.** The brief asked AG to "rewrite its source code after every
> output." Unbounded, unsupervised self-modification is how a system bricks itself,
> so AG implements the *disciplined* version of that intent. Every self-edit clears
> **two automatic gates or is rolled back in milliseconds**: a **safety gate** (AG's
> own test suite must stay green) and a **fitness gate** (the edit must *measurably
> not regress* AG's objective benchmark — `ag/bench.py`). AG never runs on unverified
> code, and never keeps a change that made it worse. That "keep-if-better" selection
> is the shared core of the strongest published self-improving systems — STOP
> ([arXiv:2310.02304](https://arxiv.org/abs/2310.02304)), the Darwin Gödel Machine
> ([arXiv:2505.22954](https://arxiv.org/abs/2505.22954)), and AlphaEvolve
> ([arXiv:2506.13131](https://arxiv.org/abs/2506.13131)) — and it is the whole point;
> don't remove it.

## What it does

| Stage | Module | Job |
|-------|--------|-----|
| **Optimize** | `ag/pipeline.py` → `optimize` | Rewrites your raw prompt into an efficient, unambiguous prompt for the model **without changing intent**. |
| **Execute** | `ag/model.py` | Calls Claude (adaptive thinking + effort) or an offline stub. |
| **Critique** | `pipeline.py` → `critique` | Scores the answer on correctness, fidelity, "vibe", and formatting, judged against *your* principles. |
| **Iterate** | `pipeline.py` → `revise` | Applies the reviewer's fixes and re-checks, up to `max_iterations`. |
| **Benchmark** | `ag/bench.py` | Scores AG on a held-out suite of objectively-checkable tasks — its **fitness function**. No model grades itself; every check is programmatic. |
| **Evolve** | `ag/evolve.py` | Proposes small patches to AG's evolvable files, snapshots, **tests + re-benchmarks**, and adopts a change *only if it doesn't regress fitness* — else rolls back. Records the measured delta to an evolution archive. |

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

The web app now includes a **Self-improvement panel** — Benchmark, Evolve, History,
and Status buttons (plus a proposer selector for split-backend evolution) — so AG's
core self-improvement runs from the GUI, not just the CLI.

**One-click launch (no terminal).** Double-click **`AG.bat`** (Windows) or
**`AG.command`** (macOS) in the repo to start the web app and open your browser. You
can copy that launcher to your Desktop — on first run it asks once where the AG folder
is (or drag the folder onto the window) and remembers it, picks the right Python, and
installs dependencies. Right-click → *Send to → Desktop (create shortcut)* for an icon.

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
python -m ag run "<prompt>" [--verbose] [--show-prompt] [--tools] [--model ...]
python -m ag memory add "<fact>" | recall "<query>" | list | clear   # persistent memory
python -m ag ingest <export>   # distill a claude.ai data export into your profile
python -m ag bench [-v] [--json]  # score AG on its objective fitness benchmark
python -m ag evolve            # attempt a test- + fitness-gated self-improvement
python -m ag evolve --history  # the measured fitness lineage of past self-edits
python -m ag tools [-v|--json] # inventory tools/apps + integration & friction ratings
python -m ag update [--apply]  # check/pull a newer Ollama model build (on demand)
python -m ag versions          # list source snapshots
python -m ag rollback <id>     # restore a snapshot instantly
python -m ag profile           # show the loaded intelligence principles
python -m ag doctor            # environment / readiness check
```

## Beyond search — memory, tools, and a reasoning loop

Plain `run` optimizes → executes → critiques → iterates a *text* answer. Two switches
let AG actually **compute and act**, so it is useful even with the internet off:

- **Persistent memory** (`use_memory`, on by default). AG recalls durable facts into
  context each run, so sessions aren't cold-started. Manage it with `ag memory add/recall/
  list/clear`; the reasoning loop can also `remember`/`recall` mid-task. Stored under
  `state/memory/` (git-ignored, capped at `max_memories`).
- **Local tools + reasoning loop** (`--tools`, or `allow_local_tools`). AG runs a bounded
  reason → act → observe loop (`max_tool_steps`) and can call: `calc` (exact arithmetic,
  fixing the small-model math weakness), `read_file`/`list_dir` (gated `filesystem_read`),
  `python_exec` (gated `code_exec`, sandboxed subprocess with a timeout), and memory. It
  degrades to a single answer if the model calls nothing.

```bash
python -m ag run "Compute 3847 * 2913 exactly." --tools --no-web -v
#   [reason] tool: calc({"expr": "3847*2913"})  ->  observation: 11206311
```

> **Safety:** file access and `python_exec` are powerful and **off by default** — they
> require `--tools`/`allow_local_tools` (which grants `filesystem_read` + `code_exec`
> through the default-deny broker) and `allow_external_tools`. `python_exec` runs real
> Python on this machine; only enable tools for prompts you trust.

**What the `evolve` loop can and can't do (honest scope).** `evolve` only rewrites files
in `evolvable_paths` (prompts, the web tool, the GUI theme, principles, config), and now
adopts a change only if it passes the test suite **and does not regress AG's benchmark
fitness**. That upgrade is the substantive one: evolution is no longer *blind* (adopt
anything that compiles and keeps tests green) — it is **empirically selected** against an
objective score, the same keep-if-better principle behind STOP, the Darwin Gödel Machine,
and AlphaEvolve. It remains a bounded prompt/parameter/design *optimiser*: it does not yet
rewrite core subsystems like the pipeline or the permission broker — those are built as
real, tested code, and only their tunable surfaces are exposed to `evolve`. The ceiling it
*can* climb is set by the benchmark: **as the task suite grows, so does the range of
capability AG can measurably improve** — which is why the suite is data you can extend
(`state/bench/tasks.jsonl`), and why AG is instructed never to edit the benchmark to cheat
its own score.

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

1. **Measure** the incumbent's fitness on the benchmark (cached by a content-hash of
   the evolvable files, so an unchanged incumbent is scored at most once).
2. **Snapshot** every evolvable file into `state/versions/<timestamp>/` (fast, local).
3. **Propose** small patches, given recent telemetry *and which benchmark tasks
   currently fail* — evolution aims at a concrete, verifiable gap.
4. **Validate** statically (Python compiles, JSON parses) — reject early.
5. **Apply** to the working tree.
6. **Gate 1 — safety:** run the suite in a subprocess (`pytest`). Fail → rollback.
7. **Gate 2 — fitness:** re-benchmark the candidate `bench_samples` times and compare
   means by **standard error**. A *statistically significant* **regression** → rollback;
   noise-level differences are treated as neutral. AG never keeps a change that made it
   measurably worse.
8. **Adopt** (+ optional git commit to `ag/evolve`, never `main`) and **record the
   fitness delta** to the evolution archive (`state/archive/`, inspect with
   `ag evolve --history`).

The gate runs the suite with **pytest** (the tests use its fixtures), so `evolve`
requires it — it ships in `requirements.txt` and the installers add it. If pytest is
absent the gate **fails closed**: `evolve` refuses to run (rather than adopting
unverified code), and `ag doctor` reports `evolve gate: UNAVAILABLE`.

### The fitness function (`ag/bench.py`) — why this raises the ceiling

Self-improvement is only real if "better" is *measured*, not hoped. `ag/bench.py` is a
held-out suite of tasks with a single, **programmatically checkable** answer (arithmetic,
logic, factual recall, format compliance, instruction-following). Fitness is the weighted
pass rate on a 0–10 scale. Two properties make the signal trustworthy:

- **No model grades itself.** Every verifier is code (exact match / numeric / regex /
  JSON) — so evolution can't reward-hack by learning to flatter a critic.
- **It's the utility function `evolve` optimises.** A self-edit is kept only if this
  number holds or rises. That is the entire mechanism behind measured self-improvement.

**Fitness is a random variable, so the gate is statistical.** A model at temperature
> 0 makes each benchmark run a *noisy draw*, not a fixed number — so comparing two
single runs would adopt changes that only *look* better by luck. AG instead samples the
benchmark `bench_samples` times (default 3), takes the mean, and estimates its **standard
error** (`sem = stdev / √n`). A change is called *improved* or *regressed* only if the
means differ by more than `fitness_k` combined standard errors (the decision margin is
`max(fitness_tol, fitness_k · √(sem_incumbent² + sem_candidate²))`) — otherwise it is a
*neutral* lateral move. AG reacts to signal, not noise. Set `bench_samples: 1` to fall
back to the cheap single-shot gate.

```bash
python -m ag bench -v            # score AG now; -v shows every task's pass/fail
python -m ag bench -n 5          # mean fitness ± standard error over 5 runs
python -m ag evolve              # try one improvement, gated on statistical significance
python -m ag evolve --history    # the measured fitness trajectory over time
```

**Grow the benchmark → grow what AG can improve.** The suite is data: drop a
`state/bench/tasks.jsonl` (one task per line: `{"id","prompt","check","expect"}`) to add
capabilities you want AG to get measurably better at. This is also how AG **compares
itself to other AI**: `ag bench` is a common yardstick — run it here, run it against any
other model you point a backend at, and the fitness scores are directly comparable.

### Directed evolution — scoring + tool inventory steer the loop

Every run is scored on three axes (0–10): **accuracy** and **quality** are judged by
the critic; **speed** is *measured* by AG from wall-clock and token cost against
`speed_budget_s` (a model can't judge its own latency). The blend is stored per run
(`config.json → score_weights`). Separately, `ag tools` inventories every backend,
retrieval/agent/host tool, and scaffolded capability, rating each on **integration**
(how wired-in) and **friction** (10 = frictionless).

`evolve` feeds both into the evolver as a *directed-evolution briefing* — the weakest
score axis, the highest-friction wired tool, and **the specific benchmark tasks that
currently fail** (the sharpest target of all: a concrete capability gap with a verifiable
pass/fail). It asks for the single smallest change that closes one, then *measures whether
it did*. Self-improvement aims at the real bottleneck and is kept only if the number moves
the right way.

### Split-backend evolution — a strong proposer, a local measurer

Proposing a good self-edit takes real reasoning; *measuring* whether it helped is
~170 cheap benchmark calls. Those costs are asymmetric, so AG lets you split them:

```bash
# on the GPU desktop: Claude proposes the edit, Ollama measures + runs it
python -m ag evolve --evolver-backend anthropic     # deploy backend stays ollama
# or make it permanent:  config.json -> "evolver_backend": "anthropic"
```

The **proposer** (`evolver_backend`) writes the patch; the **deploy backend**
(`backend`) always runs the fitness benchmark. That ordering is the whole point of
the design: **an adopted change is kept only if it measurably helps the model you
actually run.** A Claude-proposed prompt tweak that helps Claude but not your local
7B model is *rejected*, because the gate scores it on Ollama. This is also the answer
to "do Claude's improvements persist offline?" — the evolved files are plain text
committed to `ag/evolve`, and because they were *selected against the offline model*,
they keep helping it with no network and no Claude present. One Claude call per
improvement; the ~170 measurement calls are free and local.

> **What "self-improvement" means here (honest framing).** Evolve tunes AG's
> *scaffolding* — prompts, principles, config, tool code — not the model's weights.
> It makes AG measurably better at *using* a fixed model (better prompting extracts
> real latent capability), verified and compounding, but asymptotic toward that
> model's ceiling. It is not making the underlying brain smarter; it is making AG a
> better operator of it.

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
  bench.py         objective benchmark = AG's fitness function [evolvable tasks]
  archive.py       evolution lineage + measured fitness deltas + fitness cache
  evolve.py        measure -> patch -> test-gate -> fitness-gate -> adopt/rollback
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
  critique/iterate loop, snapshot/rollback, the **objective benchmark / fitness function**,
  the **test- *and* fitness-gated evolve loop** with keep-if-better selection and its
  measured-delta archive, the permission broker, the CLI. `pytest` is green (115 tests);
  rollback-on-failure *and* rollback-on-regression are both proven.
- **Scaffolded (needs your authorization to wire up):** live Chrome/browser control,
  arbitrary network/shell tools. The gates exist; the drivers are deliberately absent.
- **Deliberately not built:** literal unbounded self-rewriting with no test gate. It
  is unsafe, and the whole architecture exists to give you the intent without the risk.
```
