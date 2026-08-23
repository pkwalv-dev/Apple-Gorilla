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


def cmd_run(args) -> int:
    cfg = Config.load()
    if args.model:
        cfg.model = args.model
    client = _client(args, cfg)
    # Permanent internet is on via cfg.allow_web; --web / --no-web override per run.
    web_effective = cfg.allow_web if args.web is None else args.web
    broker = None
    if web_effective:
        from .permissions import PermissionBroker
        broker = PermissionBroker(allow_external_tools=True)
        broker.grant("network")
    rec = run_pipeline(client, cfg, args.prompt, verbose=args.verbose,
                       web=web_effective, broker=broker)
    if args.show_prompt:
        print("=== ENGINEERED PROMPT ===")
        print(rec.engineered_prompt)
        print("=== ANSWER ===")
    print(rec.answer)
    if args.verbose:
        last = rec.critiques[-1] if rec.critiques else {}
        print(f"\n[meta] iterations={rec.iterations} "
              f"score={last.get('score','?')} elapsed={rec.elapsed_s}s "
              f"dry_run={rec.dry_run}", file=sys.stderr)
    return 0


def cmd_evolve(args) -> int:
    cfg = Config.load()
    client = _client(args, cfg)
    result = evolve_mod.evolve(client, cfg, apply=args.apply, dry_run=args.dry_run)
    print(f"attempted={result.attempted} adopted={result.adopted} "
          f"rolled_back={result.rolled_back}")
    print(f"reason: {result.reason}")
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
    from .model import has_oauth_profile
    oauth = has_oauth_profile()
    print(f"API credentials:  {'present' if has_key else 'missing'}")
    print(f"OAuth profile:    {'present (ant auth login)' if oauth else 'none'}")
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
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("evolve", help="attempt a test-gated self-improvement")
    e.add_argument("--apply", action="store_true",
                   help="in manual mode, commit a passing candidate")
    e.set_defaults(func=cmd_evolve)

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
                    help="ollama host URL (default http://localhost:11434)")
    so.set_defaults(func=cmd_setup_ollama)

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
