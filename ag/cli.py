"""Command-line interface for Apple-Gorilla.

  ag run "<prompt>"      execute with context, tools, and memory
  ag evolve              propose + test-gate a self-improvement (auto-rollback)
  ag versions            list source snapshots
  ag rollback <id>       restore a snapshot
  ag doctor              environment / readiness check
  ag profile             show the loaded intelligence principles
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import backup, evolve as evolve_mod
from .config import Config, ensure_dirs
from .model import make_client
from .pipeline import run as run_pipeline
from .profile import load_principles


def _client(args, cfg):
    return make_client(cfg, backend=getattr(args, "backend", None),
                       dry_run=getattr(args, "dry_run", False))


def _make_broker(cfg, *, web: bool, tools: bool):
    """Build a permission broker granting exactly what this run is allowed to use."""
    if not (web or tools):
        return None
    from .permissions import PermissionBroker
    broker = PermissionBroker(allow_external_tools=True)
    if web:
        broker.grant("network")
    if tools:
        # calc/recall/remember need no grant; these enable file + code tools.
        broker.grant("filesystem_read")
        broker.grant("code_exec")
        # Directed acquisition: authoring/fetch are safe by default because the
        # acquisition_autonomy gate stops before install/test in "ask" mode; the
        # install itself is still separately gated and logged.
        if getattr(cfg, "allow_acquire", True):
            broker.grant("write_skill")
            broker.grant("github_fetch")
            broker.grant("install_package")
            if getattr(cfg, "acquisition_autonomy", "ask") == "auto":
                broker.grant("acquire_auto")
    return broker


def cmd_run(args) -> int:
    cfg = Config.load()
    if args.model:
        # Route --model to the field the chosen backend actually reads: the local
        # brain is cfg.ollama_model, the anthropic one cfg.model. Before this check,
        # --model on the default local backend silently asked for nothing.
        backend = getattr(args, "backend", None) or cfg.backend
        if backend in ("ollama", "dry"):
            cfg.ollama_model = args.model
        else:
            cfg.model = args.model
    client = _client(args, cfg)
    # Permanent internet is on via cfg.allow_web; --web / --no-web override per run.
    # With neither flag, ambient web fires only for research-type prompts that need
    # external/current info — not chat or self-contained tasks (keeps them fast).
    if args.web is None:
        from . import routing
        web_effective = cfg.allow_web and ("research" in routing.tags_for(args.prompt))
    else:
        web_effective = args.web
    tools_effective = cfg.allow_local_tools or getattr(args, "tools", False)
    cfg.allow_local_tools = tools_effective
    broker = _make_broker(cfg, web=web_effective, tools=tools_effective)
    trace = None
    if args.verbose:
        def trace(ev):  # surface tool/memory/web activity live on stderr
            if ev.get("stage") in ("reason", "memory", "web", "conversation") or ev.get("level") == "error":
                print(f"[{ev['stage']}] {ev['msg']}", file=sys.stderr)
    # Resolve which conversation this CLI run belongs to: consecutive `ag run`s chain
    # into one working session, and a long silence starts a fresh one.
    session_id = ""
    if getattr(cfg, "working_memory", True):
        try:
            from .memory import working
            session_id = working.resolve_session(
                getattr(args, "session", "") or None,
                idle_reset_min=getattr(cfg, "working_idle_reset_min", 45))
        except Exception:
            session_id = ""
    rec = run_pipeline(client, cfg, args.prompt, verbose=args.verbose,
                       web=web_effective, broker=broker, emit=trace,
                       session_id=session_id)
    # Fill long-term memory from normal CLI use too (best-effort, gated by auto_memory),
    # and advance the working buffer so the next `ag run` remembers this turn.
    try:
        from .pipeline import capture_memory
        saved = capture_memory(client, cfg, args.prompt, rec.answer, emit=trace,
                               session_id=session_id, web_sources=rec.web_sources)
        if saved and args.verbose:
            print(f"[memory] saved {len(saved)} durable fact(s)", file=sys.stderr)
    except Exception:
        pass
    if args.show_prompt:
        print("=== ENGINEERED PROMPT ===")
        print(rec.engineered_prompt)
        print("=== ANSWER ===")
    print(rec.answer)
    if args.verbose:
        sc = rec.scorecard or {}
        print(f"\n[meta] elapsed={rec.elapsed_s}s dry_run={rec.dry_run}",
              file=sys.stderr)
        print(f"[score] speed={sc.get('speed','?')}", file=sys.stderr)
    return 0


def cmd_evolve(args) -> int:
    cfg = Config.load()
    if getattr(args, "history", False):
        return _print_evolve_history()
    client = _client(args, cfg)
    if getattr(args, "propose", False):
        # List candidate changes for the human to review — apply nothing (mirrors the
        # web app's propose→select→apply flow; the GUI is where you pick and apply).
        res = evolve_mod.propose(client, cfg,
                                 directive=getattr(args, "note", "") or "")
        if not res.attempted:
            print(f"not attempted: {res.reason}")
            return 0
        if res.rationale:
            print(f"rationale: {res.rationale}")
        if not res.patches:
            print("no changes proposed")
            return 0
        print(f"{res.reason}:")
        for p in res.patches:
            flag = "OK " if p.valid else "SKIP"
            print(f"  [{flag}] {p.id}  {p.path}  ({p.n_bytes} B)"
                  + ("" if p.valid else f"  -- {p.error}"))
        print("\nApply selected changes from the web app (python -m ag serve), where "
              "you can pick which to keep; safety tests still gate whatever you apply.")
        return 0
    # Split-backend evolution: a stronger model proposes the edit, the deploy backend
    # (client) still measures fitness — so an adopted change is verified on the model
    # you actually run. --evolver-backend overrides config.evolver_backend.
    evolver_client = None
    evolver_backend = getattr(args, "evolver_backend", None) or cfg.evolver_backend
    deploy_backend = getattr(args, "backend", None) or cfg.backend
    if evolver_backend and not args.dry_run and evolver_backend != deploy_backend:
        try:
            evolver_client = make_client(cfg, backend=evolver_backend)
            print(f"proposer: {evolver_backend}  |  fitness measured on: "
                  f"{deploy_backend} (the model you run)")
        except Exception as e:
            print(f"warning: evolver backend '{evolver_backend}' unavailable ({e}) — "
                  f"proposing with the deploy backend instead")
    result = evolve_mod.evolve(client, cfg, apply=args.apply, dry_run=args.dry_run,
                               evolver_client=evolver_client,
                               directive=getattr(args, "note", "") or "")
    print(f"attempted={result.attempted} adopted={result.adopted} "
          f"rolled_back={result.rolled_back}")
    print(f"reason: {result.reason}")
    if result.incumbent_fitness is not None:
        cand = result.candidate_fitness
        line = f"fitness: incumbent={result.incumbent_fitness}/10"
        if cand is not None:
            spread = f"±{result.candidate_stdev}" if result.candidate_stdev else ""
            line += (f" -> candidate={cand}{spread}/10  (Δ{result.fitness_delta:+}, "
                     f"{result.verdict}, margin={result.margin}, n={result.samples})")
        print(line)
    if result.rationale:
        print(f"rationale: {result.rationale}")
    if result.changed:
        print(f"files: {', '.join(result.changed)}")
    if result.snapshot_id:
        print(f"snapshot: {result.snapshot_id} (rollback with: ag rollback "
              f"{result.snapshot_id})")
    if result.commit:
        print(f"commit: {result.commit}")
    return 0


def _print_evolve_history() -> int:
    from . import archive
    rows = archive.history(limit=20)
    if not rows:
        print("(no evolution history yet — run `ag evolve`)")
        return 0
    print("evolution lineage (newest first):")
    for r in rows:
        mark = "OK adopted" if r.get("adopted") else "· " + (r.get("verdict") or "n/a")
        inc, cand = r.get("incumbent_fitness"), r.get("candidate_fitness")
        fit = ""
        if inc is not None and cand is not None:
            fit = f"  {inc}->{cand}/10 (Δ{r.get('delta'):+})"
        elif inc is not None:
            fit = f"  incumbent={inc}/10"
        print(f"  {r.get('ts', '?')}  {mark}{fit}")
        if r.get("rationale"):
            print(f"      {r['rationale'][:100]}")
    return 0


def cmd_bench(args) -> int:
    from . import bench
    cfg = Config.load()
    # --model benches a specific LOCAL model (e.g. the specialist) without touching
    # config.json — the operator may want the abliterated coder's real numbers, and
    # `--record` folds them into the routing doc as measured evidence.
    bench_model = getattr(args, "model", None)
    if bench_model:
        import dataclasses
        cfg = dataclasses.replace(cfg, ollama_model=bench_model, backend="ollama")
    client = _client(args, cfg)
    mode = getattr(args, "mode", None) or cfg.bench_mode
    n = max(1, getattr(args, "samples", 1) or 1)
    workers = getattr(args, "workers", None)

    # An explicitly generated suite is built here and passed through, so --split
    # validation scores AG on tasks the evolve loop has never optimised against.
    tasks = None
    if getattr(args, "generated", False) or getattr(args, "split", "train") != "train":
        tasks = bench.load_generated(
            cfg, split=getattr(args, "split", "train") or "train",
            n=(getattr(args, "tasks", 0) or None),
            seed=getattr(args, "seed", None), tier=getattr(args, "tier", None))
        print(f"generated suite: {len(tasks)} tasks "
              f"(split={getattr(args, 'split', 'train')}, "
              f"seed={getattr(args, 'seed', None) or cfg.bench_seed})",
              file=sys.stderr)
    # Multiple samples surface the nondeterminism the evolve gate reasons about:
    # report the mean fitness and its standard error, not a single noisy draw.
    if n > 1:
        runs = [bench.run_benchmark(client, cfg, mode=mode, tasks=tasks,
                                    workers=workers) for _ in range(n)]
        stat = bench.summarize([r.fitness for r in runs])
        res = runs[-1]
        if getattr(args, "json", False):
            print(json.dumps({**res.as_dict(), "stat": stat.as_dict()}))
            return 0
        print(f"fitness: {stat.mean}±{stat.sem}/10  (mean of n={stat.n}, "
              f"stdev={stat.stdev})   mode={mode}")
        print("  per-run: " + ", ".join(str(r.fitness) for r in runs))
        return 0
    res = bench.run_benchmark(client, cfg, mode=mode, tasks=tasks,
                              workers=workers)
    # --json emits ONLY the JSON, so machine callers (the evolve fitness gate shells
    # out to this) can parse stdout cleanly.
    if getattr(args, "json", False):
        print(json.dumps(res.as_dict()))
        return 0
    if getattr(args, "record", False):
        from . import routing
        role = routing.record_bench(cfg, bench_model or cfg.ollama_model,
                                    fitness=res.fitness, by_category=res.by_category)
        print(f"recorded in the routing capability doc as: {role}")
    print(f"fitness: {res.fitness}/10   ({res.passed}/{res.n} passed, "
          f"pass_rate={res.pass_rate})   mode={res.mode}   {res.elapsed_s}s")
    if res.by_category:
        # Where AG is weak is more actionable than the average it rolls up into.
        parts = [f"{c} {d['passed']}/{d['n']}"
                 for c, d in sorted(res.by_category.items())]
        print("  by category: " + "  ".join(parts))
        weak = res.weakest_category
        if weak and res.by_category[weak]["pass_rate"] < 1.0:
            print(f"  weakest: {weak} "
                  f"({res.by_category[weak]['pass_rate']:.0%} pass)")
    if getattr(args, "verbose", False):
        for t in res.per_task:
            flag = "PASS" if t["passed"] else "FAIL"
            print(f"  [{flag}] {t['id']:22} ({t['category']})  -> {t['answer'][:60]}")
    else:
        failed = res.failed_ids
        if failed:
            print("  failing: " + ", ".join(failed))
    return 0


def cmd_versions(args) -> int:
    ids = backup.list_snapshots()
    if not ids:
        print("(no snapshots yet)")
        return 0
    for sid in ids:
        print(sid)
    return 0


def cmd_rollback(args) -> int:
    restored = backup.restore_by_id(args.snapshot_id)
    print(f"restored {len(restored)} file(s) from {args.snapshot_id}:")
    for r in restored:
        print(f"  {r}")
    return 0


def cmd_doctor(args) -> int:
    ensure_dirs()
    cfg = Config.load()
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY")
                   or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    print(f"backend:          {cfg.backend}")
    print(f"model:            {cfg.model} (anthropic) / {cfg.ollama_model} (ollama)")
    print(f"autonomy_level:   {cfg.autonomy_level}")
    print(f"external tools:   {'allowed' if cfg.allow_external_tools else 'DENIED (default)'}")
    from .model import has_oauth_profile, oauth_token_status
    oauth = has_oauth_profile()
    print(f"API credentials:  {'present' if has_key else 'missing'}")
    if oauth and not has_key:
        tok = oauth_token_status()
        hint = {"expired": " — RE-LOGIN NEEDED (ant auth login / sign in via Claude Code)",
                "valid": " — token valid", "unknown": "", "none": ""}.get(tok, "")
        print(f"OAuth profile:    present, token {tok}{hint}")
    else:
        print(f"OAuth profile:    {'present' if oauth else 'none'}")
    print(f"internet (web):   {'ON' if cfg.allow_web else 'off'}"
          f"  | web app: run `ag serve`")
    try:
        import anthropic  # noqa: F401
        print("anthropic SDK:    installed")
    except Exception:
        print("anthropic SDK:    NOT installed (pip install -r requirements.txt)")
    # Probe Ollama without hard-failing.
    ollama_status = "not reachable"
    installed = []
    try:
        import urllib.request
        with urllib.request.urlopen(cfg.ollama_host.rstrip("/") + "/api/tags",
                                    timeout=1.5) as r:
            tags = json.loads(r.read().decode("utf-8")).get("models", [])
            installed = [m.get("name", "?") for m in tags]
            names = ", ".join(installed) or "(no models pulled)"
            ollama_status = f"reachable — {names}"
    except Exception:
        pass
    print(f"ollama:           {ollama_status}")
    # The dual-model pair is the deployment's load-bearing configuration: report
    # both halves, with measured instruction-following when a bench --record run
    # has produced it. An abliterated specialist that is configured but absent is
    # the classic silent-degradation case, so absence is said plainly.
    if cfg.specialist_model:
        if not installed:
            spec_status = "unknown (ollama not reachable)"
        elif cfg.specialist_model in installed or any(
                n.startswith(cfg.specialist_model.split(":")[0]) for n in installed):
            spec_status = "installed"
        else:
            spec_status = "NOT INSTALLED (routing/fallback degrade silently — " \
                          "pull it or clear specialist_model)"
        extra = ""
        from . import routing
        rate = routing.instruction_pass_rate(routing.load(cfg), "specialist")
        if rate is not None:
            extra = f", measured strict-instruction {rate:.0%}"
        print(f"specialist:       {cfg.specialist_model} — {spec_status}{extra}")
    print(f"snapshots:        {len(backup.list_snapshots())}")
    from .evolve import gate_available
    gate = "ready" if gate_available() else "UNAVAILABLE (pip install pytest)"
    print(f"evolve gate:      {gate}")
    from . import bench, archive
    n_tasks = len(bench.load_tasks(cfg=cfg))
    fgate = "ON (keep-if-better)" if cfg.fitness_gate else "off"
    print(f"fitness gate:     {fgate} — {n_tasks} benchmark tasks ({cfg.bench_mode})")
    if cfg.evolver_backend:
        print(f"evolver backend:  {cfg.evolver_backend} proposes; "
              f"fitness measured on deploy backend")
    adopted = archive.adopted_history(limit=1000)
    if adopted:
        last = adopted[0]
        print(f"last improvement: {last.get('ts')} "
              f"Δ{last.get('delta')} -> {last.get('candidate_fitness')}/10")

    # Effective backend for a plain `run` — mirror make_client: auto uses Anthropic
    # when creds exist, else the configured offline backend (default local Ollama).
    if cfg.backend == "auto":
        if has_key or oauth:
            eff = "anthropic"
        elif (getattr(cfg, "offline_backend", "") or "dry").lower() == "ollama":
            eff = f"ollama ({cfg.ollama_model})"
        else:
            eff = "dry-run (stub answers)"
    else:
        eff = cfg.backend
    print(f"status:           ready — effective backend: {eff}")
    return 0


def cmd_profile(args) -> int:
    print(load_principles())
    return 0


def cmd_capability(args) -> int:
    """Write docs/CAPABILITY.md: what this deployment measurably IS, right now.

    A capability statement someone hand-writes drifts into marketing. This one is
    assembled only from things the system can point at: the tool inventory, the
    routing doc's measured scores, the benchmark's weakest category, the host
    probe, and the permission defaults. Where there is no evidence, the file SAYS
    there is no evidence — an honest gap is worth more than a confident adjective.
    """
    import datetime
    from . import bench as bench_mod, bundle, inventory, osadapt, routing
    cfg = Config.load()
    out_path = getattr(args, "out", "") or "docs/CAPABILITY.md"
    lines = []
    w = lines.append
    w("# AG capability statement")
    w("")
    w(f"_generated by `ag capability` on "
      f"{datetime.date.today().isoformat()} — do not hand-edit; regenerate._")
    w("")
    # -- models ---------------------------------------------------------------
    doc = routing.summary(cfg)
    w("## Models")
    w("")
    w(f"- primary (reasons, orchestrates, answers): `{cfg.ollama_model}`")
    w(f"- specialist (code/security, refusal fallback): "
      f"`{cfg.specialist_model or '(none)'}`")
    w(f"- routing {'on' if cfg.model_routing else 'OFF'}; measured scores:")
    measured = False
    for role in ("primary", "specialist"):
        b = (doc.get("bench") or {}).get(role)
        if b:
            measured = True
            w(f"  - {role}: fitness {b['fitness']}/10 (bench {b['when']})")
    if not measured:
        w("  - none yet — run `ag bench --model <name> --generated --record` "
          "for each configured model")
    w("")
    # -- tools ----------------------------------------------------------------
    inv = inventory.summary(cfg)
    counts = inv["counts"]
    w("## Tools")
    w("")
    w(f"- {counts.get('total', '?')} inventoried: "
      f"{counts.get('available', '?')} available, "
      f"{counts.get('degraded', '?')} degraded, "
      f"{counts.get('unavailable', '?')} unavailable here; "
      f"avg friction {inv.get('avg_friction', '?')}/10")
    weakest = inv.get("highest_friction_wired")
    if weakest:
        w(f"- highest-friction wired tool: {weakest}")
    w("")
    # -- formats + OS ---------------------------------------------------------
    from ag import formats
    w("## Input + platform")
    w("")
    kinds = sorted({f["kind"] for f in formats.supported_formats()})
    w(f"- reads {len(kinds)} format families: {', '.join(kinds[:18])}"
      + (", …" if len(kinds) > 18 else ""))
    plat = osadapt.detect()
    w(f"- this host: {plat.family} ({plat.shell_flavor}), "
      f"adapted via ag/osadapt.py")
    w("")
    # -- self-improvement ------------------------------------------------------
    w("## Self-improvement loop")
    w("")
    from .evolve import gate_available
    w(f"- test gate: {'ready' if gate_available() else 'UNAVAILABLE'}")
    w(f"- fitness gate: {'on' if getattr(cfg, 'fitness_gate', True) else 'off'},"
      f" held-out gate: {'on' if getattr(cfg, 'bench_validate', False) else 'off'}")
    w(f"- autonomy: {cfg.autonomy_level}; evolvable surface: "
      f"{len(cfg.evolvable_paths)} files (+ gates hash-pinned)")
    w("")
    # -- authority -------------------------------------------------------------
    w("## Authority (defaults — the operator can widen them)")
    w("")
    w("- default-deny broker: code_exec, network, shell, writes outside the repo,")
    w("  deletion, sub-agent spawning, and package installs all start DENIED")
    w("- model choice never widens a grant; MCP and HTTP surfaces reuse the broker")
    w("")
    p = __import__("pathlib").Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"capability statement written: {p}")
    return 0


def cmd_models(args) -> int:
    """The dual-model control panel: which brain does what, what's actually
    installed, and what the evidence says each is good at."""
    from . import routing
    from .model import ollama_has_model
    cfg = Config.load()
    doc = routing.summary(cfg)
    data = {"backend": cfg.backend, "primary": cfg.ollama_model,
            "specialist": cfg.specialist_model, "routing": cfg.model_routing,
            "doc": doc}
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2))
        return 0
    print("AG dual-model setup  (primary reasons + orchestrates; the specialist is "
          "consulted for code/security and is the refusal fallback)\n")
    for role, name in (("primary", cfg.ollama_model),
                       ("specialist", cfg.specialist_model)):
        if not name:
            print(f"  {role:<11} (not configured)")
            continue
        avail = "?" if cfg.backend != "ollama" else (
            "installed" if ollama_has_model(cfg, name) else "NOT INSTALLED")
        print(f"  {role:<11} {name}  [{avail}]")
        strengths = doc["roles"].get(role, {}).get("strengths", [])
        if strengths:
            print(f"  {'':<11} strengths: {', '.join(strengths)}")
        b = (doc.get("bench") or {}).get(role)
        if b:
            print(f"  {'':<11} measured {b['when']}: fitness {b['fitness']}/10")
            cats = b.get("by_category") or {}
            if cats:
                parts = [f"{c} {d['pass_rate']:.0%}" for c, d in sorted(cats.items())]
                print(f"  {'':<11}   by category: " + "  ".join(parts))
        rate = routing.instruction_pass_rate(doc, role)
        if rate is not None:
            print(f"  {'':<11} strict-instruction pass rate: {rate:.0%}")
    # What the doc currently favours, seed + learning + measurement.
    prefs = {t: (doc.get("stats", {}).get(t) or {}) for t in routing.TAGS
             if t != "general"}
    learned = {t: routing._learned_preference(doc, t) for t in prefs}
    lines = []
    for t in prefs:
        pref = learned[t] or doc["seed_preference"].get(t, "primary")
        src = "learned" if learned[t] else "seed"
        lines.append(f"{t}->{pref} ({src})")
    print(f"\n  routing: {'on' if cfg.model_routing else 'OFF'}; preferences: "
          + ", ".join(lines))
    print("  measure a model:  ag bench --model <name> --record")
    print("  force a model:    ag run --model <name> ...   (config: ollama_model / "
          "specialist_model)")
    return 0


def cmd_tools(args) -> int:
    from . import inventory
    cfg = Config.load()
    data = inventory.summary(cfg)
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2))
        return 0
    c = data["counts"]
    print("AG tool & app inventory (integration = how wired-in; "
          "friction = 10 is frictionless):\n")
    print(f"  {'TOOL':<32} {'CATEGORY':<11} {'STATUS':<12} INTEG  FRICTION")
    print(f"  {'-'*32} {'-'*11} {'-'*12} -----  --------")
    for t in data["tools"]:
        print(f"  {t['name']:<32} {t['category']:<11} {t['status']:<12} "
              f"{t['integration']:>4}   {t['friction']:>6}")
    print(f"\n  {c['available']} available · {c['degraded']} degraded · "
          f"{c['unavailable']} unavailable  |  "
          f"avg integration {data['avg_integration']}, "
          f"avg friction {data['avg_friction']}")
    if data["highest_friction_wired"]:
        print(f"  highest-friction wired tool (best evolve target): "
              f"{data['highest_friction_wired']}")
    if args.verbose:
        print("\nnotes:")
        for t in data["tools"]:
            print(f"  - {t['name']}: {t['notes']}")
    return 0


def cmd_serve(args) -> int:
    from . import server
    server.serve(host=args.host, port=args.port, open_browser=args.open,
                 token=getattr(args, "token", "") or "",
                 no_auth=bool(getattr(args, "no_auth", False)))
    return 0


def cmd_read(args) -> int:
    """Read any file, in any format, with AG's interpretation of it."""
    from . import formats
    from .osadapt import detect
    interp = formats.read_path(detect().normalize_path(args.path))
    if getattr(args, "json", False):
        print(json.dumps(interp.as_dict(), indent=2, default=str))
        return 0
    print(f"{interp.label}  [{interp.kind}]  confidence {interp.confidence:.2f}  "
          f"{interp.n_bytes} bytes  (handler: {interp.handler})")
    for n in interp.notes:
        print(f"  note: {n}")
    print("-" * 70)
    print(interp.text)
    return 0 if interp.kind != "error" else 1


def cmd_inspect(args) -> int:
    """Identify a file's format and structure without dumping its contents."""
    from . import formats
    from .osadapt import detect
    interp = formats.read_path(detect().normalize_path(args.path))
    out = {"kind": interp.kind, "label": interp.label,
           "confidence": interp.confidence, "bytes": interp.n_bytes,
           "handler": interp.handler, "truncated": interp.truncated,
           "notes": interp.notes, "structure": interp.structured}
    if getattr(args, "json", False):
        print(json.dumps(out, indent=2, default=str))
        return 0
    print(f"path:       {args.path}")
    print(f"format:     {interp.label}  [{interp.kind}]")
    print(f"confidence: {interp.confidence:.2f}   handler: {interp.handler}")
    print(f"size:       {interp.n_bytes} bytes"
          + ("  (truncated)" if interp.truncated else ""))
    for n in interp.notes:
        print(f"note:       {n}")
    if interp.structured:
        print("structure:")
        print("  " + json.dumps(interp.structured, indent=2,
                                default=str)[:2000].replace("\n", "\n  "))
    return 0


def cmd_osinfo(args) -> int:
    from . import osadapt
    if getattr(args, "json", False):
        print(json.dumps(osadapt.detect().as_dict(), indent=2, default=str))
        return 0
    print(osadapt.report())
    return 0


def cmd_guidance(args) -> int:
    """The escalation queue: what AG asked, and answering it."""
    from . import guidance
    action = getattr(args, "action", "list")
    if action == "list":
        items = guidance.pending()
        if not items:
            print("(no pending guidance requests)")
            return 0
        print(f"{len(items)} pending request(s):\n")
        for i, r in enumerate(items, 1):
            print(r.render(index=i))
            print()
        return 0
    if action == "answer":
        if not args.target or not args.text:
            print("usage: ag guidance answer <id> \"your answer\"", file=sys.stderr)
            return 2
        r = guidance.answer(args.target, args.text)
        if r is None:
            print(f"no pending request matching {args.target!r}", file=sys.stderr)
            return 1
        print(f"answered {r.id[:8]}: {r.answer}")
        # An answer is durable knowledge: store it so the same question, asked by a
        # later run, is already settled rather than escalated a second time.
        try:
            from . import memory
            memory.remember(f"Guidance: {r.question} -> {r.answer}",
                            origin=memory.Origin.USER)
            print("(stored as a durable memory)")
        except Exception:
            pass
        return 0
    if action == "answered":
        for r in guidance.answered(limit=args.k or 20):
            print(r.render())
            print()
        return 0
    if action == "clear":
        print(f"cleared {guidance.clear()} pending request(s)")
        return 0
    if action == "stats":
        print(json.dumps(guidance.stats(), indent=2))
        return 0
    return 2


def cmd_mcp(args) -> int:
    """Serve AG over MCP (stdio JSON-RPC) so other agents can drive it."""
    from . import mcp_server
    cfg = Config.load()
    return mcp_server.serve(cfg)


def cmd_host(args) -> int:
    from . import host
    info = host.inspect()
    print("host resources (AG adapts its own work to these):")
    print(f"  system:     {info.system} / {info.machine}")
    print(f"  cpu cores:  {info.cpu_count}  (AG parallelism -> {host.pick_concurrency()})")
    print(f"  ram:        {info.ram_gb if info.ram_gb else '?'} GB")
    print(f"  gpu:        {info.gpu}")
    print(f"  disk free:  {info.disk_free_gb if info.disk_free_gb else '?'} GB")
    print(f"  internet:   {'reachable' if info.internet else 'not reachable'}")
    print("network: AG makes outbound requests; `ag serve` adds a local web UI.")
    print("no telemetry: AG only makes requests you or the pipeline initiate.")
    return 0


def cmd_setup_ollama(args) -> int:
    cfg = Config.load()
    cfg.backend = "ollama"
    if args.model:
        cfg.ollama_model = args.model
    if args.host:
        cfg.ollama_host = args.host
    cfg.save()
    print(f"config saved: backend=ollama, model={cfg.ollama_model}, host={cfg.ollama_host}")
    print("next steps:")
    print("  1) install Ollama from https://ollama.com (Windows/macOS/Linux)")
    print(f"  2) ollama pull {cfg.ollama_model}")
    print("  3) ollama serve         (Windows: the installer runs it for you)")
    print("  4) python -m ag doctor  (should show ollama: reachable)")
    print('  5) python -m ag run "your prompt"')
    return 0


def cmd_login(args) -> int:
    """Sign in to Claude so the `auto` backend uses it instead of the local model."""
    from . import model
    if getattr(args, "clear", False):
        print("Removed the saved API key." if model.clear_api_key()
              else "No saved API key to remove.")
        return 0
    if getattr(args, "key", None):
        try:
            model.save_api_key(args.key)
        except ValueError as e:
            print(f"error: {e}")
            return 1
        print(f"Signed in. `auto` will now use Claude ({Config.load().model}).")
        return 0
    st = model.signin_status()
    if st["signed_in"]:
        print(f"Already signed in to Claude (method: {st['method']}).")
        return 0
    print("Not signed in to Claude. Two ways to sign in:")
    print("  1) API key — create one at https://console.anthropic.com/settings/keys, then:")
    print("       python -m ag login --key sk-ant-...")
    print("     (or paste it into the web app's Sign in box: python -m ag serve)")
    print("  2) Claude subscription — install Claude Code and run its login, or")
    print("     `ant auth login`; AG reads that OAuth profile automatically.")
    return 0


def _standing(mgr, m) -> str:
    """How well-founded a memory is, in one trailing tag — so the list never reads as a
    flat wall of equally-true statements."""
    b = mgr.belief(m)
    via = f"{m.origin}:{m.asserter}" if m.asserter else m.origin
    about = " about:self" if m.subject == "self" else ""
    if m.disputed:
        return f"  (DISPUTED, {b:.2f}, via {via}{about})"
    if b < mgr.trust_threshold:
        stale = " stale," if m.volatile and m.confidence > b + 0.05 else ""
        return f"  (unconfirmed,{stale} {b:.2f}, via {via}{about})"
    return f"  ({b:.2f}, via {via}{about})"


def cmd_memory(args) -> int:
    from . import memory
    if args.action == "add":
        m = memory.remember(args.text or "", max_memories=Config.load().max_memories)
        print(f"remembered: {m.text}" if m else "nothing to remember")
    elif args.action == "recall":
        mgr = memory.get_manager("root")
        hits = memory.recall(args.text or "", k=args.k)
        if not hits:
            print("(no relevant memories)")
        for m in hits:
            print(f"- {m.text}{_standing(mgr, m)}")
    elif args.action == "list":
        mgr = memory.get_manager("root")
        mems = memory.all_memories()
        print(f"{len(mems)} memory item(s):")
        for m in mems:
            print(f"  [{m.kind[:4]}] [{m.id}] {m.text}{_standing(mgr, m)}")
    elif args.action == "disputed":
        # AG does not quietly pick a winner between two stable claims that cannot both
        # be true; it holds both in doubt and shows them here to be settled.
        items = memory.disputed()
        print(f"{len(items)} disputed memor(y/ies):" if items
              else "(nothing disputed)")
        for m in items:
            other = m.meta.get("contradicts", "?")
            print(f"  [{m.id}] {m.text}\n      contradicts [{other}]")
    elif args.action == "stale":
        items = memory.needs_verification(k=args.k)
        print(f"{len(items)} belief(s) due a re-check:" if items
              else "(nothing stale)")
        mgr = memory.get_manager("root")
        for m in items:
            print(f"  [{m.id}] {m.text}{_standing(mgr, m)}")
    elif args.action == "verify":
        m = memory.verify(args.text or "", confirmed=not args.reject)
        if m is None:
            print("no such memory")
        else:
            print(f"{'rejected' if args.reject else 'confirmed'}: {m.text} "
                  f"(confidence {m.confidence:.2f})")
    elif args.action == "clear":
        print(f"cleared {memory.clear()} memory item(s)")
    elif args.action == "stats":
        import json as _json
        print(_json.dumps(memory.get_manager("root").stats(), indent=2))
    elif args.action == "reflect":
        cfg = Config.load()
        from .model import make_client
        client = make_client(cfg)
        out = memory.reflect(client, cfg)
        print(f"reflected: {out}")
    return 0


def cmd_skills(args) -> int:
    from . import skills
    reg = skills.get_registry("root")
    if args.action == "list":
        items = reg.list(include_disabled=True)
        print(f"{len(items)} skill(s):")
        for s in items:
            flag = "" if s.enabled else " (disabled)"
            caps = f" [caps: {', '.join(s.capabilities)}]" if s.capabilities else ""
            print(f"  {s.name}{flag} - {s.description}{caps}")
    elif args.action in ("disable", "enable"):
        ok = reg.set_enabled(args.name or "", args.action == "enable")
        print("done" if ok else f"skill not found: {args.name}")
    elif args.action == "remove":
        print("removed" if reg.remove(args.name or "") else f"skill not found: {args.name}")
    return 0


def cmd_acquire(args) -> int:
    """Author + gate + register a skill on demand. Running this command is the user's
    explicit approval for the install/test the acquisition performs."""
    cfg = Config.load()
    from .model import make_client
    from .permissions import PermissionBroker
    from . import acquire
    client = make_client(cfg, backend=getattr(args, "backend", None))
    broker = PermissionBroker(allow_external_tools=True)
    broker.grant("write_skill")
    broker.grant("install_package")
    broker.grant("github_fetch")
    broker.grant("code_exec")

    def trace(ev):
        if ev.get("stage") == "acquire" or ev.get("level") == "error":
            print(f"[{ev.get('stage')}] {ev.get('msg', '')}", file=sys.stderr)

    res = acquire.author_skill(client, cfg, args.spec or "", broker=broker,
                               approve=True, emit=trace)
    print(res.reason)
    if res.test_output and not res.acquired:
        print(res.test_output, file=sys.stderr)
    return 0 if res.acquired else 1


def cmd_fleet(args) -> int:
    from . import fleet
    if args.action == "list":
        agents = fleet.list_agents()
        killed = " (KILL SWITCH ENGAGED)" if fleet.kill_active() else ""
        print(f"{len(agents)} agent(s){killed}:")
        for a in agents:
            print(f"  {a.agent} [{a.status}] role={a.role} parent={a.parent} "
                  f"depth={a.depth} skills={a.skills_acquired}")
    elif args.action in ("disable", "enable"):
        ok = fleet.set_status(args.name or "", "disabled" if args.action == "disable" else "active")
        print("done" if ok else f"agent not found: {args.name}")
    elif args.action == "kill":
        fleet.engage_kill()
        print("kill switch ENGAGED — all spawning halted, running loops will stop")
    elif args.action == "revive":
        fleet.clear_kill()
        print("kill switch cleared — spawning allowed again")
    elif args.action == "clear":
        print(f"cleared {fleet.clear()} agent record(s)")
    return 0


def cmd_lora(args) -> int:
    cfg = Config.load()
    from . import lora
    if args.action == "status":
        f = lora.feasibility(cfg)
        print(f"GPU: {f.gpu} · VRAM: {f.vram_gb} GB · CUDA: {f.cuda}")
        print(f"base model: {cfg.lora_base_model} (4-bit: {cfg.lora_4bit})")
        print(f"dataset: {lora.dataset_size()} example(s)")
        if f.missing_deps:
            print("missing deps: " + ", ".join(f.missing_deps))
        for n in f.notes:
            print("  - " + n)
        print("trainable now: " + ("YES" if f.ok else "no (see notes above)"))
    elif args.action == "build-data":
        def trace(ev):
            if ev.get("stage") == "lora":
                print(f"[lora] {ev.get('msg','')}", file=sys.stderr)
        st = lora.build_dataset(cfg, emit=trace,
                                refresh_teacher=bool(getattr(args, "refresh_teacher", False)))
        print(f"dataset: {st.total} example(s) "
              f"({st.from_memory} from memory, {st.from_teacher} from teacher) -> {st.path}")
    elif args.action == "train":
        def trace(ev):
            if ev.get("stage") == "lora" or ev.get("level") == "error":
                print(f"[lora] {ev.get('msg','')}", file=sys.stderr)
        res = lora.train(cfg, emit=trace)
        print(res.reason)
        if res.ok:
            print(f"adapter: {res.adapter_path}")
        return 0 if res.ok else 1
    elif args.action == "list":
        ads = lora.list_adapters()
        print(f"{len(ads)} adapter(s):")
        for a in ads:
            print(f"  {a['id']} - base={a.get('base','?')} examples={a.get('examples','?')}")
    elif args.action == "options":
        for b in lora.available_bases():
            print(f"  [{b['fit']:>9}] {b['id']} ({b['params_b']}B) - {b['note']}")
        rec = lora.recommended_config()
        print(f"recommended for this GPU: base={rec['base']} seq={rec['max_seq']}")
    elif args.action == "merge":
        if not args.target:
            print("usage: ag lora merge <adapter_id>  (see `ag lora list`)"); return 1
        def trace(ev):
            if ev.get("stage") == "lora" or ev.get("level") == "error":
                print(f"[lora] {ev.get('msg','')}", file=sys.stderr)
        res = lora.merge_to_gguf(cfg, args.target, emit=trace)
        print(res.reason)
        return 0 if res.ok else 1
    return 0


def cmd_media(args) -> int:
    """Local image/video generation: what's installed, does the workflow load, make one."""
    from . import comfy, images, video
    cfg = Config.load()
    if args.action == "status":
        info = comfy.status(cfg)
        print(f"image backend : {images.backend_for(cfg)}")
        print(f"comfyui       : {info['host']} "
              f"({'reachable' if info['reachable'] else 'not running'}"
              f"{', installed' if info['installed'] else ', not installed'})")
        print(f"a1111         : {cfg.sd_host} "
              f"({'reachable' if images.sd_reachable(cfg) else 'not running'})")
        for kind in ("image", "video"):
            print(f"{kind:<14}: {info.get(kind + '_workflow', '(none)')}")
            for m in info.get(f"{kind}_models", []):
                print(f"  needs       : {m}")
        return 0

    if args.action == "check":
        # Validate the workflows against the server that will actually run them.
        if not comfy.reachable(cfg):
            print(f"ComfyUI is not running at {cfg.comfy_host} — start it first "
                  "(nothing can be checked against a server that isn't up)")
            return 1
        bad = 0
        for kind, name in (("image", cfg.comfy_image_workflow),
                           ("video", cfg.comfy_video_workflow)):
            try:
                problems = comfy.validate(comfy.load_workflow(name), cfg)
            except Exception as e:
                print(f"{kind}: {name}: {e}")
                bad += 1
                continue
            if problems:
                bad += 1
                print(f"{kind}: {name}: {len(problems)} problem(s)")
                for pr in problems:
                    print(f"  - {pr}")
            else:
                print(f"{kind}: {name}: ok")
        return 1 if bad else 0

    prompt = (args.prompt or "").strip()
    if not prompt:
        print("give a prompt")
        return 2
    try:
        if args.action == "image":
            r = images.generate(prompt, cfg)
            print(f"saved {r.path} ({r.width}x{r.height}, {r.steps} steps)")
        else:
            r = video.generate(prompt, cfg, seconds=args.seconds or None)
            print(f"saved {r.path} ({r.seconds}s, {r.width}x{r.height}, "
                  f"{r.frames} frames @ {r.fps}fps)")
    except Exception as e:
        print(f"failed: {e}")
        return 1
    return 0


def cmd_bundle(args) -> int:
    from . import bundle
    if args.check:
        checks = bundle.check()
        allok = all(c.ok for c in checks)
        for c in checks:
            print(f"  [{'OK ' if c.ok else 'XX '}] {c.name}" + (f" - {c.detail}" if c.detail else ""))
        print("portable" if allok else "NOT fully portable — see failures above")
        return 0 if allok else 1
    dest = bundle.export(args.out or None)
    print(f"bundle written: {dest}")
    return 0


def cmd_update(args) -> int:
    from . import update
    cfg = Config.load()
    model = args.model or cfg.ollama_model
    status = update.check_model_update(cfg, model)
    print(f"model:  {model}")
    print(f"status: {status.state} — {status.reason}")
    if args.apply:
        if status.state == "up-to-date":
            print("already up to date; nothing to pull.")
            return 0
        print(f"pulling {model} ...")
        ok, final = update.pull_model(cfg, model,
                                      emit=lambda ev: print("  " + ev["msg"]))
        print(("updated: " if ok else "failed: ") + final)
        return 0 if ok else 1
    if status.available:
        print("a newer build is available — run `ag update --apply` to update now.")
    return 0


def cmd_ingest(args) -> int:
    from . import ingest
    cfg = Config.load()
    try:
        msgs = ingest.extract_human_messages(args.path)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"extracted {len(msgs)} of your messages from {args.path}")
    if not msgs:
        print("no human messages found — check the export path/format",
              file=sys.stderr)
        return 1
    client = _client(args, cfg)
    md = ingest.distill(client, cfg, msgs)
    if args.write:
        dest = ingest.write_profile(md)
        print(f"wrote {dest}")
        print("review it, then run:  python -m ag run \"<prompt>\" --verbose")
    else:
        print("\n--- proposed profile (preview) — re-run with --write to save ---\n")
        print(md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ag", description="Apple-Gorilla executor")
    p.add_argument("--dry-run", action="store_true",
                   help="use the offline stub instead of the live API")
    p.add_argument("--backend", choices=["auto", "anthropic", "ollama", "dry"],
                   default=None,
                   help="model backend (default: config.json 'backend', usually auto)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the full pipeline on a prompt")
    r.add_argument("prompt")
    r.add_argument("--model", default=None)
    r.add_argument("--verbose", "-v", action="store_true")
    r.add_argument("--show-prompt", action="store_true")
    r.add_argument("--web", dest="web", action="store_true", default=None,
                   help="force internet access on for this run")
    r.add_argument("--no-web", dest="web", action="store_false",
                   help="force internet access off for this run")
    r.add_argument("--tools", action="store_true",
                   help="enable local tools (calc/file-read/python-exec/memory) + "
                        "the reasoning loop for this run")
    r.add_argument("--session", default="",
                   help="pin this run to a named working-memory session (default: "
                        "continue the recent one, or start fresh after a long gap)")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("evolve",
                       help="attempt a test- + fitness-gated self-improvement")
    e.add_argument("--apply", action="store_true",
                   help="in manual mode, commit a passing candidate")
    e.add_argument("--history", action="store_true",
                   help="show the measured fitness lineage instead of evolving")
    e.add_argument("--propose", action="store_true",
                   help="list candidate changes for review and apply NOTHING "
                        "(pick + apply them in the web app)")
    e.add_argument("--note", "--directive", dest="note", default="",
                   help="free-text instruction steering WHAT to improve this cycle "
                        "(safety + fitness gates still apply)")
    e.add_argument("--evolver-backend", dest="evolver_backend",
                   choices=["auto", "anthropic", "ollama", "dry"], default=None,
                   help="model that PROPOSES edits (e.g. anthropic); fitness is still "
                        "measured on the deploy backend. Default: config.evolver_backend")
    e.set_defaults(func=cmd_evolve)

    bn = sub.add_parser("bench",
                        help="score AG on its objective benchmark (its fitness fn)")
    bn.add_argument("--mode", choices=["execute", "optimize_execute"], default=None,
                    help="execute = base model only; optimize_execute = full prompt path")
    bn.add_argument("--samples", "-n", type=int, default=1,
                    help="repeat N times and report mean fitness ± standard error")
    bn.add_argument("--verbose", "-v", action="store_true",
                    help="show every task's pass/fail and answer")
    bn.add_argument("--json", action="store_true", help="emit the raw result JSON")
    bn.add_argument("--generated", action="store_true",
                    help="use the generated, unbounded suite (ag/benchgen.py) "
                         "instead of the fixed seed tasks")
    bn.add_argument("--split", choices=["train", "validation"], default="train",
                    help="generated suite split (validation = held out from evolve)")
    bn.add_argument("--tasks", type=int, default=0,
                    help="how many generated tasks (default: config bench_generated_n)")
    bn.add_argument("--tier", type=int, default=None,
                    help="generated difficulty: 1 easy, 2 normal, 3 hard, 0 mixed")
    bn.add_argument("--seed", type=int, default=None,
                    help="generated suite seed (same seed = same tasks)")
    bn.add_argument("--workers", type=int, default=None,
                    help="run tasks concurrently (default: config bench_workers)")
    bn.add_argument("--model", default=None,
                    help="bench a specific local model (e.g. the abliterated "
                         "specialist) — does not change config.json")
    bn.add_argument("--record", action="store_true",
                    help="fold this run's scores into the routing capability doc as "
                         "measured per-role evidence")
    bn.set_defaults(func=cmd_bench)

    sub.add_parser("versions", help="list source snapshots").set_defaults(
        func=cmd_versions)

    rb = sub.add_parser("rollback", help="restore a snapshot by id")
    rb.add_argument("snapshot_id")
    rb.set_defaults(func=cmd_rollback)

    sub.add_parser("doctor", help="environment / readiness check").set_defaults(
        func=cmd_doctor)
    sub.add_parser("profile", help="show loaded intelligence principles").set_defaults(
        func=cmd_profile)

    mo = sub.add_parser("models", help="dual-model panel: primary + abliterated "
                                       "specialist, availability, measured strengths")
    mo.add_argument("--json", action="store_true")
    mo.set_defaults(func=cmd_models)

    cap = sub.add_parser("capability", help="generate docs/CAPABILITY.md from "
                                            "measured evidence (not adjectives)")
    cap.add_argument("--out", default="", help="output path (default docs/CAPABILITY.md)")
    cap.set_defaults(func=cmd_capability)

    tl = sub.add_parser("tools",
                        help="inventory tools/apps AG can use + integration/friction")
    tl.add_argument("--json", action="store_true", help="emit the raw inventory JSON")
    tl.add_argument("--verbose", "-v", action="store_true", help="include notes")
    tl.set_defaults(func=cmd_tools)
    sub.add_parser("host", help="inspect host resources + network posture").set_defaults(
        func=cmd_host)

    rd = sub.add_parser("read", help="read ANY file format (pdf/docx/xlsx/sqlite/"
                                     "zip/binary) with AG's interpretation")
    rd.add_argument("path")
    rd.add_argument("--json", action="store_true", help="emit the raw interpretation")
    rd.set_defaults(func=cmd_read)

    ins = sub.add_parser("inspect", help="identify a file's format + structure "
                                         "without dumping its contents")
    ins.add_argument("path")
    ins.add_argument("--json", action="store_true")
    ins.set_defaults(func=cmd_inspect)

    osi = sub.add_parser("osinfo", help="how AG adapts to THIS machine "
                                        "(shell, package manager, paths)")
    osi.add_argument("--json", action="store_true")
    osi.set_defaults(func=cmd_osinfo)

    gd = sub.add_parser("guidance", help="questions AG raised for you "
                                         "(list/answer/answered/clear/stats)")
    gd.add_argument("action", nargs="?", default="list",
                    choices=["list", "answer", "answered", "clear", "stats"])
    gd.add_argument("target", nargs="?", default="", help="request id (for answer)")
    gd.add_argument("text", nargs="?", default="", help="your answer")
    gd.add_argument("-k", type=int, default=20, help="how many to show")
    gd.set_defaults(func=cmd_guidance)

    sub.add_parser("mcp", help="serve AG over MCP (stdio) so other agents can use it"
                   ).set_defaults(func=cmd_mcp)

    sv = sub.add_parser("serve", help="run the browser web app (any OS / phone)")
    sv.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 (local only) or 0.0.0.0 (reachable from phone/LAN)")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--open", action="store_true", help="open a browser on start")
    sv.add_argument("--token", default="",
                    help="access token required on non-loopback binds "
                         "(auto-generated when omitted)")
    sv.add_argument("--no-auth", dest="no_auth", action="store_true",
                    help="disable the token on a network bind (NOT recommended: "
                         "anyone who can reach the port can run this instance)")
    sv.set_defaults(func=cmd_serve)

    lg = sub.add_parser("login",
                        help="sign in to Claude so `auto` uses it (saves an API key)")
    lg.add_argument("--key", default=None,
                    help="Anthropic API key to save (sk-ant-...)")
    lg.add_argument("--clear", action="store_true", help="remove the saved API key")
    lg.set_defaults(func=cmd_login)

    so = sub.add_parser("setup-ollama",
                        help="switch config to the local Ollama backend")
    so.add_argument("--model", default=None,
                    help="ollama model tag (e.g. qwen2.5:14b, llama3.1:8b)")
    so.add_argument("--host", default=None,
                    help="ollama host URL (default http://127.0.0.1:11434)")
    so.set_defaults(func=cmd_setup_ollama)

    mem = sub.add_parser("memory", help="AG's layered memory (add/recall/list/clear/"
                                        "stats/reflect/disputed/stale/verify)")
    mem.add_argument("action", choices=["add", "recall", "list", "clear", "stats",
                                        "reflect", "disputed", "stale", "verify"])
    mem.add_argument("text", nargs="?", default="",
                     help="fact to add, recall query, or memory id to verify")
    mem.add_argument("-k", type=int, default=5, help="recall/stale: max items")
    mem.add_argument("--reject", action="store_true",
                     help="verify: mark the claim false instead of confirming it")
    mem.set_defaults(func=cmd_memory)

    sk = sub.add_parser("skills", help="acquired skills (list/enable/disable/remove)")
    sk.add_argument("action", choices=["list", "enable", "disable", "remove"])
    sk.add_argument("name", nargs="?", default="", help="skill name (for enable/disable/remove)")
    sk.set_defaults(func=cmd_skills)

    acq = sub.add_parser("acquire", help="author + test + register a new skill on demand")
    acq.add_argument("spec", help="the capability to acquire, in plain language")
    acq.set_defaults(func=cmd_acquire)

    md = sub.add_parser("media", help="local image/video generation "
                                      "(status/check/image/video)")
    md.add_argument("action", choices=["status", "check", "image", "video"])
    md.add_argument("prompt", nargs="?", default="", help="what to generate")
    md.add_argument("--seconds", type=float, default=0.0,
                    help="clip length for 'video' (default: config video_frames)")
    md.set_defaults(func=cmd_media)

    fl = sub.add_parser("fleet", help="agent swarm control (list/enable/disable/kill/revive/clear)")
    fl.add_argument("action", choices=["list", "enable", "disable", "kill", "revive", "clear"])
    fl.add_argument("name", nargs="?", default="", help="agent name (for enable/disable)")
    fl.set_defaults(func=cmd_fleet)

    lo = sub.add_parser("lora", help="LoRA fine-tune a local model "
                        "(status/options/build-data/train/list/merge)")
    lo.add_argument("action", choices=["status", "options", "build-data", "train",
                                       "list", "merge"])
    lo.add_argument("target", nargs="?", default="", help="adapter id (for merge)")
    lo.add_argument("--refresh-teacher", action="store_true",
                    help="build-data: regenerate the teacher set instead of reusing it "
                         "(one model call per task — costs API tokens on a cloud backend)")
    lo.set_defaults(func=cmd_lora)

    bn = sub.add_parser("bundle", help="export a portable bundle, or --check portability")
    bn.add_argument("--check", action="store_true", help="audit portability constraints only")
    bn.add_argument("--out", default="", help="output .zip path (default: alongside the repo)")
    bn.set_defaults(func=cmd_bundle)

    up = sub.add_parser("update",
                        help="check/apply an Ollama model update (on demand, no polling)")
    up.add_argument("--apply", action="store_true", help="pull the newer build now")
    up.add_argument("--model", default=None,
                    help="model tag to check (default: config ollama_model)")
    up.set_defaults(func=cmd_update)

    ing = sub.add_parser("ingest",
                         help="distill a claude.ai data export into your profile")
    ing.add_argument("path", help="path to the export .zip, conversations.json, or dir")
    ing.add_argument("--write", action="store_true",
                     help="save to profile/about_me.md (otherwise preview only)")
    ing.set_defaults(func=cmd_ingest)
    return p


def main(argv=None) -> int:
    # AG prints Unicode (Δ, ·, —, →) in its output; the legacy Windows console defaults
    # to cp1252, which raises UnicodeEncodeError on those. Force UTF-8 so the CLI works
    # everywhere (a no-op on platforms that are already UTF-8).
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    # dry_run may be absent on some subparsers; normalize.
    if not hasattr(args, "dry_run"):
        args.dry_run = False
    try:
        return args.func(args)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
