# Phase 1 — delivered

Branch `ag/omni-upgrade`, commit `666d321`, on top of `208bce2`.
**482 tests passing** (was 349). Portability audit clean. No new hard dependencies.

This implements the Phase 1 ceilings from `IMPROVEMENT_PLAN.md`. The framing
throughout: AG cannot be AGI on 8B local weights, so the work goes into the
*harness* — how much it can read, how well it fits the machine it is on, how
honestly it measures itself, and how gracefully it asks for help.

---

## What is new

| Module | Ceiling lifted | What it does |
|---|---|---|
| `ag/formats.py` | C3 — `read_file` was utf-8 only | ~30 formats: json/jsonl/csv/xml/html/ini/toml/yaml/zip/tar/gzip/docx/xlsx/pptx/epub/pdf/sqlite/images/media/executables/base64. Every reading carries an **earned** confidence. Unknown bytes get ranked hypotheses instead of a shrug. Never raises. |
| `ag/osadapt.py` | C5 — `host.py` was read-only | Probes shell, 19 package managers, path conventions, case sensitivity, WSL/container/admin. Unknown OS → POSIX fallback with uncertainty **declared**. |
| `ag/guidance.py` | C4 — no ask-the-human channel | Non-blocking escalation queue. Two triggers: low confidence, or irreversibility. |
| `ag/mcp_server.py` | C7 — no MCP server | Stdlib JSON-RPC 2.0 over stdio. 8 tools, 6 `ag://` resources. |
| `ag/benchgen.py` | C1 — fitness was 14 fixed tasks | 28 generators, unbounded tasks, answers computed in Python, held-out split. |
| `ag/cache.py` | C6 — re-parsing and re-embedding | LRU / File / TTL caches wired into memory, embeddings, and model probes. |

## Changed behaviour you will notice

```bash
ag read report.pdf              # or .docx .xlsx .sqlite .zip — anything
ag inspect mystery.bin          # what IS this? structure, no content dump
ag osinfo                       # how AG adapts to THIS machine
ag guidance                     # questions AG raised; `guidance answer <id> "..."`
ag mcp                          # serve AG to other agents over MCP
ag bench --generated --tier 3   # unbounded suite instead of the 14 fixed tasks
ag bench --split validation     # score on tasks evolution never optimises against
ag serve --host 0.0.0.0         # now prints an access token; loopback unchanged
```

The executor prompt now includes a ~5-line briefing about the actual machine
(`cfg.os_context`, default on) and any decisions you have already settled via
`guidance`. That removes a whole class of confidently-wrong answers — `apt install`
suggested on a Mac, backslashes on Linux — for a few dozen tokens.

## The gate that matters most

**Gate 3 (generalisation).** Gates 1 (tests) and 2 (fitness) cannot detect
overfitting, because Gate 2 *is* the thing being gamed. A candidate that improves
the scored tasks while regressing held-out ones is now rolled back with verdict
`overfit`. Only a *statistically significant* drop rejects — noise must not veto
a real improvement, or evolution stalls and a gate that blocks everything is as
useless as one that blocks nothing.

Enable with `bench_generated: true` + `bench_validate: true` in `config.json`.
Curated `state/bench/tasks.jsonl` still outranks generated tasks when present:
human input beats machine defaults.

## Security

`ag serve` on a non-loopback address publishes an endpoint that runs the model,
reads files through the tool layer, and can trigger evolve. It now requires a
token, **generated automatically** so the safe path is also the default path —
the token is not an option you have to remember. Accepted via `?token=`,
`Authorization: Bearer`, or the cookie set on first page load; compared with
`hmac.compare_digest`. Loopback stays frictionless and unauthenticated.

MCP callers go through the same `PermissionBroker` as everything else. A remote
agent **cannot** widen AG's permissions by asking: `code_exec` stays denied unless
you set `allow_code_exec` locally.

---

## Bugs found by writing the tests

Each of these produced a *silently wrong answer* rather than an error, which is
the failure mode worth the most to catch.

1. **`benchgen` splits were not actually disjoint.** The salt separated RNG
   *seeds*, not the *tasks* they produce. Generators with small parameter spaces
   emitted identical prompts into both splits — which would have quietly turned
   the new overfitting detector into a rubber stamp. Membership is now decided by
   hashing the prompt itself (blake2b, not `hash()`, which is randomised per
   process and would reshuffle splits between runs). Verified: 0 overlap at
   n=300, both splits still covering all six categories.

2. **`.csv` forced a comma delimiter.** Semicolon-separated European exports —
   extremely common — fused into a single column `name;age` that *looked* like a
   successful parse. Evidence from the data now outranks the extension.

3. **Base64 detection matched ordinary prose.** The base64 alphabet is a superset
   of lowercase letters, so plain text was "decoded" into noise. Detection now
   requires the decode to yield something recognisable.

4. **UTF-16 text was reported as "unrecognised binary".** Printable-ratio is an
   8-bit-encoding test; wide-encoded text fails it by construction because every
   other byte is a NUL. A large share of the world's text files on Windows were
   unreadable.

5. **UTF-16 was also guessed for any even-length input**, because "it decoded
   without error" was treated as evidence — it is not; almost anything decodes.
   `café naïve` became CJK mojibake. Now requires the NUL-interleaving pattern.

6. **Two-column tables were structurally unreachable** — the delimiter score
   threshold could never be met with one delimiter per line.

7. **`guidance` ignored an explicit `reversible=False`** unless the text also
   matched a keyword, so "apply the migration" never escalated. The caller's own
   declaration is the strongest evidence available and now wins outright.

8. **`ollama_has_model` cached a positive forever.** A model you deleted stayed
   "present" for the life of the process. Now TTL-cached per *host*: one HTTP
   probe answers every model question, deletions are noticed, and failures are
   never cached (a momentarily-down Ollama must not disable routing for the
   session).

## One deliberate non-fix

The bundle portability auditor **correctly** failed on the new optional `import
yaml`. The fix was to teach it the real distinction — guarded-with-fallback vs
hard import, executable string literal vs docstring — and verify it is now
*strictly more precise*, not weaker. Grepping source text instead would have
pushed authors toward `importlib.import_module` and hidden the dependency from
the auditor entirely. Sharpen a failing guard; never evade it.

---

## Next (Phase 2+, not yet built)

- Tiered `evolvable_paths` (`tune` / `extend` / frozen `core`) — C2.
- Hash-pinned verifier integrity test, so evolution cannot weaken its own gates.
- `Budget(wall_s, model_calls, tokens, tool_calls)` — C11.
- Split `server.py` (2328 lines) into app/api/ui so the GUI becomes evolvable — C9.
- Parallel sub-agents — C12; reuse-first skill acquisition — C10.
