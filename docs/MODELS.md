# Models: Qwen3-8B primary + abliterated Qwen2.5-Coder specialist

AG runs **two local models as one system**. Neither is trusted with everything.

| role | model (default) | what it's for |
|---|---|---|
| **primary** | `qwen3:8b` | every turn: reasoning, tool orchestration, conversation, planning, math, research synthesis. The orchestration brain. |
| **specialist** | `ag-coder-abliterated:latest` | consulted subtasks: long code generation, strict format output, **authorised security work** (pentest/CTF/recon — phrasings an instruct model's fine-tune treats as refuse-bait). Also the automatic fallback when the primary errors, empties, or refuses. |

The primary is always in charge. The specialist is a *tool it can reach for* —
never a competing controller — plus a deterministic net the *runtime* (not the
model) falls back to when the primary hard-fails.

## How a task gets to the specialist

Four routes, all ending at the same model:

1. **Routing doc preference.** `ag/routing.py` tags every prompt
   (`code`, `security`, `creative`, `math`, `research`), and the capability doc
   (`state/routing/profiles.json`, seeded then *learned from honest signals*)
   says which role each tag favours. `security` and `code` seed to the
   specialist. The primary sees this as prompt guidance with a
   `consult_specialist` tool.
2. **Explicit consult.** The primary calls `consult_specialist` with a focused
   subtask when a step suits it better.
3. **Failure fallback.** Primary hard-fails → runtime retries once on the
   specialist and records the outcome (`fallback_win` / `failure`), so routing
   sharpens from real evidence. A model cannot orchestrate its way out of its
   own crash — only the runtime can do that for it.
4. **Autonomous sub-agents.** `delegate` (the spawn-agent tool) accepts
   `"model": "specialist"`, so an autonomous agent tree — e.g. a recon agent
   spawning focused workers on an authorised pentest — can run its code-heavy
   leaves on the abliterated coder while the primary coordinates.
   `ag run --model <name>` forces a whole run onto one model.

## Working around the abliterated model's limitations

Abliteration removes refusal behaviour but **degrades instruction-following**:
prose-wrapped JSON, dropped output contracts, repetition loops, empty replies.
The harness assumes none of its discipline and compensates at five points:

1. **Measurement first.** `ag/benchgen.py` has an `instruction` category —
   tasks that are trivially winnable *if* you comply (exact echo, JSON-only
   output, ALL-CAPS rewrites, one-word answers). A low score there cannot be a
   knowledge gap. Measure any model with:
   `ag bench --model ag-coder-abliterated:latest --generated --record`
2. **The doc cites measurements.** `--record` folds scores into the routing doc.
   When the specialist's measured strict-instruction rate is low (< 60%),
   the primary's guidance says so explicitly: *keep tightly-formatted steps on
   yourself; give it open-ended or code-heavy subtasks.* Unmeasured means the
   doc says nothing — it never asserts a weakness it has no evidence for.
3. **Malformed tool calls are repaired, not answered with.** If the model emits
   something that *looks* like a tool call but doesn't parse, the reason loop
   names the problem in an observation and asks again from the same transcript
   (capped at 2 consecutive repairs, so a model that simply cannot comply still
   terminates — with its prose answer, not an infinite loop).
4. **Degenerate output is detected and retried.** `looks_degenerate()` catches
   empty replies and repetition loops of *any* period. The specialist consult
   path retries once under a tighter contract; a persistent loop is truncated
   and **labelled** `[truncated … use with care]` — a degraded answer is never
   passed off as a clean one. `complete_json()` does the same for structured
   steps: verify the JSON, repair the prompt, retry, return `None` honestly if
   the model can't comply.
5. **The learning loop is insulated.** Fitness gating (Gates 1–3) runs on the
   primary deploy backend regardless of which model proposed or answered; the
   instruction category above is part of that fitness function, so a patch
   that degrades instruction-following loses fitness and is rolled back.

## What does NOT change with the model

This is the load-bearing rule: **model choice is who thinks, never what is
permitted.** The PermissionBroker is the security boundary, not the model's
disposition:

- Sub-agents on the specialist get a child broker derived from the parent's
  grants. `delegate` with a `model` argument still requires the `spawn_agent`
  grant (and the `allow_external_tools` master switch) just to exist.
- `code_exec`, `network`, `shell`, `filesystem_write_outside_repo` stay
  default-deny for every agent and model. On a pentest this means AG drafts
  scan/exploit reasoning and runs what *you've* granted — point it only at
  systems you're authorised to test.
- Evolve candidates may not touch the gate files (`ag/evolve.py`,
  `ag/bench.py`, `ag/benchgen.py`, `ag/permissions.py`, …) or the safety keys
  of config.json (`fitness_gate`, `evolvable_paths`, `ollama_model`, the bench
  instrument — see `SAFETY_CONFIG_KEYS` in `ag/evolve.py`), however the patch
  arrives. Gate files are hash-pinned across each candidate cycle.
- Run budgets (`budget_model_calls`, `budget_tool_calls`, `budget_tokens`,
  `budget_wall_s` in config.json) cap what any autonomous run can spend; an
  exhausted budget stops the loop with the stop *declared in the answer*.

## Operator controls

```bash
ag models                      # the panel: install state, measured strengths,
                               # learned + seed routing preferences
ag bench --model <name> --generated --record   # measure a model honestly
ag run --model qwen3:8b ...    # force one model for a run
ag doctor                      # reports both halves; flags a configured-but-
                               # missing specialist (the silent-degradation case)
```

Config keys: `ollama_model` (primary), `specialist_model` (abliterated coder),
`model_routing` (on/off). Sensible pairings: primary ≈ 7–9B instruct
(qwen3:8b, llama3.1:8b), specialist ≈ small coder abliterated
(qwen2.5-coder abliterated). The routing doc is keyed by *role*, so renaming
either model keeps everything that was learned.
