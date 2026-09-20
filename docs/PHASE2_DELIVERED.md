# Phase 2 — delivered

Branch `ag/omni-upgrade`, on top of Phase 1 (`fe2e9ec`).
**542 tests passing** (was 482; +60). Portability audit clean. Docs: `MODELS.md`.

This phase does two jobs: **harden the dual-model deployment** (Qwen3-8B primary +
abliterated Qwen2.5-Coder specialist, incl. autonomous agents on the specialist for
authorised security work) and **close the verifier-integrity holes** in the evolve
loop. The framing: the specialist's *refusal* is routed around; its
*instruction-following* is measured and compensated for; and the gates that judge
self-edits are now outside the reach of the thing they judge.

---

## Model switching, preserved and extended

- `ag run --model X` **actually switches now** — see bug #1 below.
- `ag bench --model X --record` measures any local model on the honest benchmark
  and folds the scores into the routing capability doc (`--record`).
- `ag models` — a control panel: both configured brains, install state, measured
  per-category scores, learned-vs-seed routing preferences.
- `delegate` (the spawn-agent tool) accepts `"model": "specialist"`, so an
  autonomous agent tree can run code/security-heavy leaves on the abliterated
  coder while the primary coordinates. New `security` task tag (pentest, recon,
  CTF, vuln analysis…) seeds to the specialist — the instruct primary's fine-tune
  refuse-baits exactly these phrasings.
- `ag doctor` reports the specialist's install state and flags the classic
  silent-degradation case: configured but not pulled.

## Working around the abliterated model's instruction-following

Five compensations, all verified by tests (design detail in `docs/MODELS.md`):
an `instruction` benchgen category (exact-echo / JSON-only / ALL-CAPS /
one-word — winnable by construction, so a low score is unambiguous evidence);
guidance that cites the *measured* instruction rate and stays silent when
unmeasured; malformed tool calls repaired in-loop (capped, so a hopeless model
still terminates); degenerate output (empty / repetition loops of any period)
detected, retried under a tighter contract, else truncated and **labelled**;
structured steps (`complete_json`, incl. the evolve proposer) verify-and-repair
instead of assuming parseable JSON.

**What does not change with the model:** the PermissionBroker is the boundary.
Sub-agents on the specialist get a child broker derived from the parent's grants;
`code_exec`/`network`/`shell` stay default-deny regardless of model disposition.

## Verifier integrity (the load-bearing fix)

config.json is evolvable and `evolvable_paths` lives in it. Without guards, a
candidate could (a) declare the gate files evolvable and edit them next cycle,
(b) set `fitness_gate: false` directly, or (c) change the instrument the bench
subprocess re-reads (fewer tasks, different model) — re-grading itself on a
friendlier ruler. Now:

- `PROTECTED_FILES` (evolve/bench/benchgen/scoring/permissions/archive/backup/
  conftest) can never be a patch target, whatever evolvable_paths says.
- `SAFETY_CONFIG_KEYS` (the gate knobs, the bench instrument, the model
  selection, the authority boundaries) are frozen to evolution; safe tuning
  (speed budget, run budgets, prompts) stays evolvable.
- Gate files are **hash-pinned** across every candidate cycle; a byte change
  rolls the candidate back and records an `integrity` regression.
- A **regression corpus** (`state/evolve/regressions.jsonl`) records every
  gate rejection and is fed back into the proposer's briefing — failures become
  lessons, not repeated attempts.

## Run budgets (Phase 3 item)

`ag/budget.py` + `budget_model_calls` / `budget_tool_calls` / `budget_tokens` /
`budget_wall_s` in config. Enforced by the loop, not the model; every ceiling
stops the run with the stop *declared in the answer* rather than a silently
truncated result. All default 0 = unlimited (existing behaviour unchanged).

## Bugs found by writing the tests

1. **`ag run --model X` did nothing on the default backend.** It set
   `cfg.model` (the Anthropic field); the local backend reads
   `cfg.ollama_model`. The one switch whose purpose is switching silently
   didn't. Now routed per-backend.
2. **`looks_degenerate` missed arbitrary-period repetition.** Fixed window
   sizes (120/60/30) let a 37-char loop sail through — caught because a test
   used period 37. The period is now discovered from the text.
3. **The safety-config attack chain above** — found by reviewing what the
   static validator *didn't* check, before any candidate could try it.
4. **The evolve proposer's JSON was assumed.** `extract_json(res.text) or {}`
   turned a weak backend's prose into a silent "no patches proposed". Now
   verify-repair-honest-None.
5. (test-side) The broker is two-key: GATED capabilities need
   `allow_external_tools` **and** the grant. Good — the negative test now
   proves a `model=` argument can't reach `delegate` without `spawn_agent`.

## Tally of test additions

`test_model_robust` (10), `test_reason_recovery` (13), `test_routing_models`
(18), `test_evolve_integrity` (11), `test_budget` (8) = 60.

## Still roadmap

- Split `server.py` into app/api/ui so the GUI becomes evolvable (Ceiling 9).
- Parallel `spawn_many`, reuse-first skill acquisition, context packer (Phase 3).
- LoRA-from-verified-traces pipeline discipline, speculative draft+verify (P5).
- Co-evolving open-ended task proposal, federation (P6).
