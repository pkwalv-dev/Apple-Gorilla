# Phase 4 — delivered

Branch `ag/omni-upgrade`, on top of the mainline merge (`f1e1a50`).
**607 tests passing** (584 after the merge + 23 new). Portability audit clean.

This phase closes the roadmap: the remaining plan items, plus a full merge with
upstream's household-cluster feature.

---

## Merge with `origin/main` (`1741fbf`)

Upstream landed a household cluster (LAN discovery, remote sub-agents, distributed
compute) that touched `reason.py`, `server.py`, `fleet.py`, `config.py` — the four
files this branch changed most. Four conflicts, all composed rather than chosen:

- **reason.py** — upstream's `stop_check`/`on_step` loop hooks AND this branch's
  budget enforcement + malformed-tool-call repair, in the same `solve` loop.
- **server.py** — upstream's page changes (the new cluster tab) were ported into
  `ag/server_ui.py`, keeping this branch's UI split: app/API in server.py (now
  with both the auth gate and the cluster listener), page-as-data in server_ui.py.
- **fleet.py** — upstream's per-agent `node/location/pid/heartbeat` fields inside
  this branch's mutation lock.
- **config.py** — both feature blocks kept.

All 584 tests pass after the merge, including upstream's 22 cluster tests.

## Context packer (final Phase 3 item: C11's prompt-side half)

`ag/contextpack.py` + `context_budget_chars` in config. The pipeline's context
blocks (machine briefing, settled decisions, conversation, user profile, memory,
web sources) were appended with an implicit it-will-fit assumption; on an 8–16k
token local model, overflow truncates at the FRONT of the prompt — where the
executor instructions live. Now:

- each block carries a priority and a truncatable flag;
- over budget, least-important blocks are clipped to a floor, or dropped — each
  **with a marker inside the text**, because a model answering from clipped
  evidence must know it was clipped;
- the budget accounting includes the markers themselves (a packer that ignores
  the size of its own markers would overshoot the very thing it guards — found
  by a test, fixed);
- budget 0 = unlimited, byte-identical to before.

## Benchmark task proposals (Phase 6 seed: co-evolving measurement)

The benchmark is the ceiling of everything the loop can become, so it should
grow — but a model-proposed task carries an *unverified* expected answer, and a
wrong one poisons the fitness function in the direction that matters most
(confidently wrong grading). The deal:

- the model may **propose**: `ag bench --propose 5 [--focus instruction]` —
  shape-validated (known checker, well-typed expect, regex compiles, non-trivial
  non-duplicate prompt) and saved to `state/bench/proposals.jsonl`;
- only a human may **accept**: `ag bench --proposals`, `ag bench --accept <id>`
  moves a task into the curated suite (which outranks everything by design);
  `--reject` discards;
- rows carry `proposed: true` provenance.

The wrong-expect test pins the structural defence: a `2+2 = 5` proposal passes
shape validation honestly (5 IS a number) — and still never reaches the suite
without a human accept. AG may suggest new measurement; it may never install it.

## Test tally

`test_contextpack_proposals.py` — 23 tests: verbatim/fitting passthrough, clip
priority, in-text markers, drop-down-to-marker vs keep-at-floor (with the correct
semantics pinned after the accounting fix), end-to-end pipeline under a tight
budget, the seven invalid-proposal shapes, pending/accept/reject round trips,
idempotent accept, and the self-installation defence.

## Roadmap status — what's left is hardware, not code

- Speculative draft+verify generation and federation need live model endpoints
  (two real models to compare, two real machines to federate). Building them
  blind would produce unverifiable scaffolding, which this project is against.
- Everything else in IMPROVEMENT_PLAN.md is now in the tree and green.
