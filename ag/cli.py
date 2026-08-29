"""Command-line interface for Apple-Gorilla.

  ag run "<prompt>"      optimize -> execute -> critique -> iterate
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
    return broker


def cmd_run(args) -> int:
    cfg = Config.load()
    if args.model:
        cfg.model = args.model
    client = _client(args, cfg)
    # Permanent internet is on via cfg.allow_web; --web / --no-web override per run.
    web_effective = cfg.allow_web if args.web is None else args.web
    tools_effective = cfg.allow_local_tools or getattr(args, "tools", False)
    cfg.allow_local_tools = tools_effective
    broker = _make_broker(cfg, web=web_effective, tools=tools_effective)
    trace = None
    if args.verbose:
        def trace(ev):  # surface tool/memory/web activity live on stderr
            if ev.get("stage") in ("reason", "memory", "web") or ev.get("level") == "error":
                print(f"[{ev['stage']}] {ev['msg']}", file=sys.stderr)
    rec = run_pipeline(client, cfg, args.prompt, verbose=args.verbose,
                       web=web_effective, broker=broker, emit=trace)
    if args.show_prompt:
        print("=== ENGINEERED PROMPT ===")
        print(rec.engineered_prompt)
        print("=== ANSWER ===")
    print(rec.answer)
    if args.verbose:
        sc = rec.scorecard or {}
        print(f"\n[meta] iterations={rec.iterations} elapsed={rec.elapsed_s}s "
              f"dry_run={rec.dry_run}", file=sys.stderr)
        print(f"[score] overall={sc.get('overall','?')} "
              f"accuracy={sc.get('accuracy','?')} quality={sc.get('quality','?')} "
              f"speed={sc.get('speed','?')}", file=sys.stderr)
    return 0


def cmd_evolve(args) -> int:
    cfg = Config.load()
    if getattr(args, "history", False):
        return _print_evolve_history()
    client = _client(args, cfg)
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
                               evolver_client=evolver_client)
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
        mark = "✓ adopted" if r.get("adopted") else "· " + (r.get("verdict") or "n/a")
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
    client = _client(args, cfg)
    mode = getattr(args, "mode", None) or cfg.bench_mode
    n = max(1, getattr(args, "samples", 1) or 1)
    # Multiple samples surface the nondeterminism the evolve gate reasons about:
    # report the mean fitness and its standard error, not a single noisy draw.
    if n > 1:
        runs = [bench.run_benchmark(client, cfg, mode=mode) for _ in range(n)]
        stat = bench.summarize([r.fitness for r in runs])
        res = runs[-1]
        if getattr(args, "json", False):
            print(json.dumps({**res.as_dict(), "stat": stat.as_dict()}))
            return 0
        print(f"fitness: {stat.mean}±{stat.sem}/10  (mean of n={stat.n}, "
              f"stdev={stat.stdev})   mode={mode}")
        print("  per-run: " + ", ".join(str(r.fitness) for r in runs))
        return 0
    res = bench.run_benchmark(client, cfg, mode=mode)
    # --json emits ONLY the JSON, so machine callers (the evolve fitness gate shells
    # out to this) can parse stdout cleanly.
    if getattr(args, "json", False):
        print(json.dumps(res.as_dict()))
        return 0
    print(f"fitness: {res.fitness}/10   ({res.passed}/{res.n} passed, "
          f"pass_rate={res.pass_rate})   mode={res.mode}   {res.elapsed_s}s")
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
    try:
        import urllib.request
        with urllib.request.urlopen(cfg.ollama_host.rstrip("/") + "/api/tags",
                                    timeout=1.5) as r:
            tags = json.loads(r.read().decode("utf-8")).get("models", [])
            names = ", ".join(m.get("name", "?") for m in tags) or "(no models pulled)"
            ollama_status = f"reachable — {names}"
    except Exception:
        pass
    print(f"ollama:           {ollama_status}")
    print(f"snapshots:        {len(backup.list_snapshots())}")
    from .evolve import gate_available
    gate = "ready" if gate_available() else "UNAVAILABLE (pip install pytest)"
    print(f"evolve gate:      {gate}")
    from . import bench, archive
    n_tasks = len(bench.load_tasks())
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

    # Effective backend for a plain `run`.
    if cfg.backend == "auto":
        eff = "anthropic" if (has_key or oauth) else "dry-run (stub answers)"
    else:
        eff = cfg.backend
    print(f"status:           ready — effective backend: {eff}")
    return 0


def cmd_profile(args) -> int:
    print(load_principles())
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
    server.serve(host=args.host, port=args.port, open_browser=args.open)
    return 0


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


def cmd_memory(args) -> int:
    from . import memory
    if args.action == "add":
        m = memory.remember(args.text or "", max_memories=Config.load().max_memories)
        print(f"remembered: {m.text}" if m else "nothing to remember")
    elif args.action == "recall":
        hits = memory.recall(args.text or "", k=args.k)
        if not hits:
            print("(no relevant memories)")
        for m in hits:
            print(f"- {m.text}")
    elif args.action == "list":
        mems = memory.all_memories()
        print(f"{len(mems)} memory item(s):")
        for m in mems:
            print(f"  [{m.id}] {m.text}")
    elif args.action == "clear":
        print(f"cleared {memory.clear()} memory item(s)")
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
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("evolve",
                       help="attempt a test- + fitness-gated self-improvement")
    e.add_argument("--apply", action="store_true",
                   help="in manual mode, commit a passing candidate")
    e.add_argument("--history", action="store_true",
                   help="show the measured fitness lineage instead of evolving")
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

    tl = sub.add_parser("tools",
                        help="inventory tools/apps AG can use + integration/friction")
    tl.add_argument("--json", action="store_true", help="emit the raw inventory JSON")
    tl.add_argument("--verbose", "-v", action="store_true", help="include notes")
    tl.set_defaults(func=cmd_tools)
    sub.add_parser("host", help="inspect host resources + network posture").set_defaults(
        func=cmd_host)

    sv = sub.add_parser("serve", help="run the browser web app (any OS / phone)")
    sv.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 (local only) or 0.0.0.0 (reachable from phone/LAN)")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--open", action="store_true", help="open a browser on start")
    sv.set_defaults(func=cmd_serve)

    so = sub.add_parser("setup-ollama",
                        help="switch config to the local Ollama backend")
    so.add_argument("--model", default=None,
                    help="ollama model tag (e.g. qwen2.5:14b, llama3.1:8b)")
    so.add_argument("--host", default=None,
                    help="ollama host URL (default http://127.0.0.1:11434)")
    so.set_defaults(func=cmd_setup_ollama)

    mem = sub.add_parser("memory", help="AG's persistent memory (add/recall/list/clear)")
    mem.add_argument("action", choices=["add", "recall", "list", "clear"])
    mem.add_argument("text", nargs="?", default="", help="fact to add, or recall query")
    mem.add_argument("-k", type=int, default=5, help="recall: max items")
    mem.set_defaults(func=cmd_memory)

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
