# Phase 3 — delivered

Branch `ag/omni-upgrade`, on top of Phase 2 (`d13b31f`).
**562 tests passing** (was 542; +20). Portability audit clean.

Ceilings closed this round: **C9** (server monolith), **C12** (serial sub-agents),
**C10** (no reuse-first acquisition); plus the honest half of the Phase 5
training-data item, and the generated capability statement.

---

## C9 — the GUI is now evolvable

`ag/server.py` was 2419 lines: route handlers, auth, media, NDJSON — and, glued in,
~1180 lines of HTML/CSS/JS page. The UI is the one surface a human actually looks
at, so it belongs in the keep-if-better loop. The split was done byte-exactly:

- `ag/server_ui.py` — the page as data (template + `render_page()`), newly added
  to `evolvable_paths` in config.json.
- `ag/server.py` — 1243 lines of pure app/API, importing `render_page`.

Its guard is deliberately NOT the fitness function (the bench never renders a
page): it is `tests/test_server.py`'s pins on the page's JS hooks — `id="log"`,
`getReader`, `doEvolve()`, `id="directive"`, fetch paths. A candidate that breaks
the page's *contract with the API* fails Gate 1 and rolls back; a candidate that
changes appearance clears it. All 40 server tests pass unchanged.

## C12 — parallel sub-agents

`agents.spawn_many(tasks, workers=…)` now runs children concurrently
(`ThreadPoolExecutor`, default `min(4, n)`). The three guarantees, each tested:

- **order-stable**: `results[i]` always answers `tasks[i]` even when completion
  order is the reverse of input order (staggered-sleep test);
- **crash-isolated**: one dead child becomes its entry's output string
  `(sub-agent failed: …)`, the batch returns the rest;
- **shared state is locked**: fleet's `agents.jsonl` was load-modify-write — safe
  under serial spawns *by accident only*. Mutations now hold a lock, and a
  16-thread spawn storm is asserted to lose zero records.

Each task dict may carry `"model": "specialist"` — an autonomous workflow can
spread code/security-heavy leaves onto the abliterated coder in parallel while
the primary coordinates.

## C10 — reuse-first acquisition

`acquire.author_skill` now checks `find_reusable(registry, spec)` *before* the
model call: if an existing enabled skill's vocabulary covers ≥60% of the
request's distinctive tokens, it is returned as the answer and **no model call
happens** (test uses a client that raises if called). Counterweight tested: a
non-covering spec still authors. This kills the registry-NIH failure mode —
three near-identical CSV converters, each billed a model call.

## Verified LoRA rows (Phase 5, honest half)

A training row trains behaviour: a refusal teaches refusing, a loop teaches
looping. `_pairs_from_teacher` now passes every teacher answer through the same
two detectors the runtime uses — `pipeline._is_failure` and
`model.looks_degenerate` — so the adapter is never taught output the harness
itself would reject. Rows written *before* this gate (or by a since-degraded
teacher) are re-verified on reuse: age confers no immunity. Dropped rows are
counted and emitted, never silently kept.

## `ag capability` (Phase 5)

Writes `docs/CAPABILITY.md` from *evidence only*: routing-doc measured scores
(or "none yet" when unmeasured), the tool inventory, format family count, the
host probe, gate status, and the default-deny authority summary. Where there is
no evidence the file says there is no evidence — pinned by test, including a
control assertion that it does not praise itself.

## Test tally

`test_fleet_acquire_lora.py` — 20 tests: order stability, real parallelism
(timing-bounded), crash isolation, per-task model overrides, grant enforcement,
spawn-storm record integrity, reuse coverage both directions, reuse-without-a-
model-call, verified-teacher-row filtering (fresh and on-reuse), capability
honesty.

## Still roadmap

- Context packer with an explicit token budget (pipeline assembly).
- Bench task *proposal* flow (model suggests → human approves via the guidance
  queue → lands in curated tasks), the Phase 6 seed.
- Speculative draft+verify generation; federation. Both need live hardware to
  verify against; noted, not faked.
