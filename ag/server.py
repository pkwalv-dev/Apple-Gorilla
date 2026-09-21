"""A tiny built-in web app for AG (stdlib only).

`python -m ag serve` starts a local server with a browser UI. Open it on this
machine, or from your phone/another device on the same network (bind --host
0.0.0.0). Cross-platform by virtue of being a web page — any OS with a browser.

The UI drives the same pipeline (execute with context, tools, and memory) and
streams a **realtime log** of AG's thought process, tool/app calls, internet usage,
errors, and the measured speed score as they happen (newline-delimited
JSON over a single streamed response). This IS an inbound server (for YOUR use); it
is not a data-farming endpoint — bind to localhost unless you deliberately want
LAN/phone access.
"""
from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import Config
from .model import make_client, Canceller, Cancelled
from .pipeline import run as run_pipeline


class _Interrupted(Exception):
    """Raised inside a run's on_delta when the viewer disconnects (Stop / closed tab),
    so generation is halted upstream instead of running to completion unseen."""

# --- web UI ------------------------------------------------------------------
# The page (HTML/CSS/JS) lives in ag/server_ui.py, an EVOLVABLE file: the GUI is
# the surface humans see, so it joins prompts/theme in the keep-if-better loop.
# Its guard is the test suite's pins on the page's JS hooks, not the fitness fn.
from .server_ui import render_page as _render_page

PAGE = _render_page()

def _clean_history(raw, *, max_turns: int = 40, max_len: int = 4000) -> list:
    """Sanitise conversation history from the client into [{role, text}] pairs.

    Untrusted input: coerce types, keep only known roles, bound count and size so a
    malformed or oversized payload can't wedge the pipeline. Content is treated as
    conversation data, never as instructions, by the prompt framing downstream.
    """
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw[-max_turns:]:
        if not isinstance(item, dict):
            continue
        role = "user" if str(item.get("role")) == "user" else "ai"
        text = str(item.get("text", "")).strip()
        if text:
            out.append({"role": role, "text": text[:max_len]})
    return out

MEDIA_TYPES = {".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska",
               ".gif": "image/gif", ".webp": "image/webp", ".png": "image/png",
               ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def media_type(path: Path) -> str:
    return MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def media_file(want: str):
    """Resolve a browser-supplied path to a generated file, or None.

    This endpoint reads a file named by the page, so it is confined to the two
    directories AG writes media into. A path outside them — or a symlink pointing
    out of them, which is why this resolves before comparing — is not found.
    """
    from .config import STATE_DIR
    roots = [(STATE_DIR / "video").resolve(), (STATE_DIR / "images").resolve()]
    try:
        target = Path(want).resolve()
        if any(target.is_relative_to(r) for r in roots) and target.is_file():
            return target
    except (OSError, ValueError):
        pass
    return None


def _build_broker(cfg: Config, *, web=None):
    web = cfg.allow_web if web is None else web
    if not (web or cfg.allow_local_tools):
        return None
    from .permissions import PermissionBroker
    broker = PermissionBroker(allow_external_tools=True)
    if web:
        broker.grant("network")
    if cfg.allow_local_tools:
        broker.grant("filesystem_read")   # read-only file/dir access
        broker.grant("spawn_agent")       # delegate a subtask to a sub-agent
        if getattr(cfg, "allow_code_exec", False):
            broker.grant("code_exec")     # arbitrary Python — opt-in only
        # Self-extension: author/test/register a new skill, install its deps, fetch
        # allowlisted code. The acquisition_autonomy gate ("ask") still stops before
        # install/test unless the run opts into "auto".
        if getattr(cfg, "allow_acquire", False):
            broker.grant("write_skill")
            broker.grant("install_package")
            broker.grant("github_fetch")
            if getattr(cfg, "acquisition_autonomy", "ask") == "auto":
                broker.grant("acquire_auto")
    return broker

class _Handler(BaseHTTPRequestHandler):
    cfg: Config = Config()
    # Where the server is bound (filled in by serve()), for the GUI "where am I
    # running" indicator.
    bind_host: str = "127.0.0.1"
    bind_port: int = 8765
    # Evolve self-modifies source, so only one may run at a time. The server keeps
    # serving during an evolve (ThreadingHTTPServer), so this flag lets the GUI show
    # that a cycle is in progress and lets us reject overlapping evolves.
    _evolve_lock = threading.Lock()
    _evolving = threading.Event()
    # LoRA training runs in a background thread (it is long); this tracks its state so
    # the GUI can show progress and reject overlapping runs.
    _lora_train = {"running": False, "last": ""}
    _lora_lock = threading.Lock()
    # Cancellation registry: run_id -> True when the viewer asked to stop that run.
    # Checked every streamed chunk so Stop halts generation promptly and reliably,
    # rather than relying on a TCP write eventually failing.
    _cancel: dict = {}
    _cancel_lock = threading.Lock()
    # Last set of proposed self-edits, cached so the GUI can apply a chosen subset by
    # id (the browser never sends code back — AG applies exactly what it proposed).
    _proposal: dict = {}
    _proposal_directive: str = ""

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, code, data: bytes, ctype: str):
        """Send raw bytes (a generated clip or image) rather than encoded text."""
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # --- access control ----------------------------------------------------
    # Empty = loopback-only, no auth (unchanged local behaviour). Set by serve()
    # whenever the bind address is reachable from the network, because that turns
    # AG into an unauthenticated endpoint that runs your model, reads your files,
    # and can trigger evolve. MODALITY.md flagged exactly this; the fix is a token.
    auth_token: str = ""

    def _authorized(self) -> bool:
        if not self.auth_token:
            return True
        supplied = ""
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer "):
            supplied = header[7:].strip()
        if not supplied:
            from urllib.parse import urlparse, parse_qs
            supplied = (parse_qs(urlparse(self.path).query).get("token") or [""])[0]
        if not supplied:
            cookie = self.headers.get("Cookie", "")
            for part in cookie.split(";"):
                k, _, v = part.strip().partition("=")
                if k == "ag_token":
                    supplied = v.strip()
                    break
        import hmac
        # compare_digest: a plain == leaks the token's prefix through timing.
        return hmac.compare_digest(supplied, self.auth_token)

    def _deny(self) -> None:
        body = ("<!doctype html><meta charset=utf-8><title>Apple-Gorilla</title>"
                "<body style='font-family:system-ui;padding:2rem;max-width:34rem'>"
                "<h2>Access token required</h2><p>This Apple-Gorilla instance is "
                "bound to a network address, so it requires the access token printed "
                "in the terminal when it started.</p>"
                "<p>Open it as <code>http://HOST:PORT/?token=YOUR_TOKEN</code>.</p>")
        self._send(401, body)

    def _guard(self) -> bool:
        """True if the request may proceed; sends 401 and returns False if not."""
        if self._authorized():
            return True
        self._deny()
        return False

    def do_GET(self):
        if not self._guard():
            return
        # The token arrives as `/?token=...`, so the root route has to be matched
        # on the path alone. Only the root is normalised here: /media? carries a
        # meaningful query string, and every other route is an exact literal.
        if self.path.split("?", 1)[0] in ("/", "/index.html"):
            # Set the token as a cookie once, so the page's own fetch() calls (which
            # cannot carry the query string) authenticate for the rest of the session.
            if self.auth_token:
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Set-Cookie",
                                 f"ag_token={self.auth_token}; Path=/; SameSite=Strict")
                self.end_headers()
                self.wfile.write(body)
                return
            self._send(200, PAGE)
        elif self.path == "/tools":
            from . import inventory
            self._send(200, json.dumps(inventory.summary(self.cfg)),
                       "application/json")
        elif self.path == "/update/check":
            # On-demand (the page pings this once after a run). Never polled.
            from . import update
            status = update.check_model_update(self.cfg)
            self._send(200, json.dumps(status.as_dict()), "application/json")
        elif self.path == "/doctor":
            self._send(200, json.dumps(_doctor_data(self.cfg)), "application/json")
        elif self.path == "/evolve/history":
            from . import archive
            self._send(200, json.dumps({"history": archive.history(limit=20)}),
                       "application/json")
        elif self.path == "/whereami":
            self._send(200, json.dumps(self._whereami()), "application/json")
        elif self.path == "/models":
            self._send(200, json.dumps(_models_data(self.cfg)), "application/json")
        elif self.path == "/auth":
            from .model import signin_status
            st = signin_status()
            st["model"] = self.cfg.model
            st["console_url"] = "https://console.anthropic.com/settings/keys"
            self._send(200, json.dumps(st), "application/json")
        elif self.path.startswith("/media?"):
            # Serve a generated clip back to the page. Confined to AG's own output
            # directories: this endpoint takes a path from the browser, so anything
            # outside them is refused rather than read.
            import urllib.parse as _up
            want = _up.parse_qs(self.path.split("?", 1)[1]).get("path", [""])[0]
            target = media_file(want)
            if target is None:
                self._send(404, json.dumps({"error": "not found"}), "application/json")
                return
            self._send_bytes(200, target.read_bytes(), media_type(target))
        elif self.path == "/image/status":
            from . import comfy, images
            cfg = self.cfg
            on = bool(getattr(cfg, "allow_image_gen", False))
            backend = images.backend_for(cfg) if on else ""
            up = False
            host = cfg.sd_host
            models = []
            if on and backend == "comfy":
                host = cfg.comfy_host
                up = comfy.reachable(cfg)
                try:
                    models = comfy.describe(
                        comfy.load_workflow(cfg.comfy_image_workflow))
                except Exception:
                    models = []
            elif on:
                up = images.sd_reachable(cfg)
            self._send(200, json.dumps({
                "enabled": on, "reachable": up, "host": host, "backend": backend,
                "models": models,
                "video": bool(getattr(cfg, "allow_video_gen", False)),
            }), "application/json")
        elif self.path == "/fleet":
            from . import fleet
            self._send(200, json.dumps({
                "kill_active": fleet.kill_active(),
                "agents": [a.as_dict() for a in fleet.list_agents()]}),
                "application/json")
        elif self.path == "/skills":
            from . import skills
            items = skills.get_registry("root").list(include_disabled=True)
            self._send(200, json.dumps({"skills": [{
                "name": s.name, "description": s.description,
                "capabilities": s.capabilities, "enabled": s.enabled,
                "agent": s.agent, "source": s.source} for s in items]}),
                "application/json")
        elif self.path == "/bundle/check":
            from . import bundle
            checks = bundle.check()
            self._send(200, json.dumps({
                "portable": all(c.ok for c in checks),
                "checks": [c.as_dict() for c in checks]}), "application/json")
        elif self.path == "/lora/status":
            from . import lora
            feas = lora.feasibility(self.cfg)
            self._send(200, json.dumps({
                "feasibility": feas.as_dict(),
                "base": self.cfg.lora_base_model,
                "epochs": self.cfg.lora_epochs,
                "unsloth_pref": bool(getattr(self.cfg, "lora_use_unsloth", True)),
                "bases": lora.available_bases(feas.vram_gb),
                "recommended": lora.recommended_config(feas.vram_gb),
                "merge_ready": lora.merge_feasibility(self.cfg)["ok"],
                "dataset_size": lora.dataset_size(),
                "adapters": lora.list_adapters(),
                "training": dict(_Handler._lora_train)}), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def _whereami(self) -> dict:
        """Where and how AG is running right now — for the GUI location indicator."""
        from .model import _has_anthropic_creds, has_oauth_profile
        cfg = self.cfg
        host, port = self.bind_host, self.bind_port
        local_only = host not in ("0.0.0.0", "::")
        if cfg.backend == "auto":
            if _has_anthropic_creds() or has_oauth_profile():
                brain = "Claude (" + cfg.model + ")"
            else:
                brain = ("Ollama (" + cfg.ollama_model + ")"
                         if getattr(cfg, "offline_backend", "") == "ollama"
                         else "dry-run (stub)")
        elif cfg.backend == "ollama":
            brain = "Ollama (" + cfg.ollama_model + ")"
        elif cfg.backend == "anthropic":
            brain = "Claude (" + cfg.model + ")"
        else:
            brain = "dry-run (stub)"
        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = "?"
        return {
            "host": host, "port": port, "local_only": local_only,
            "hostname": hostname, "url": f"http://127.0.0.1:{port}",
            "lan_url": (f"http://{_lan_ip()}:{port}" if not local_only else None),
            "backend": cfg.backend, "brain": brain,
            "evolving": self._evolving.is_set(),
            "can_evolve_while_running": True,
        }

    def _start_lora_train(self):
        """Kick off a QLoRA run in a background thread (it is long-running). Refuses to
        start a second concurrent run; result is reported via _lora_train for the GUI."""
        from . import lora
        with _Handler._lora_lock:
            if _Handler._lora_train.get("running"):
                self._send(200, json.dumps({"started": False,
                           "reason": "a training run is already in progress"}),
                           "application/json")
                return
            feas = lora.feasibility(self.cfg)
            if not feas.ok:
                reason = "not trainable here: " + (
                    ", ".join(feas.missing_deps) if feas.missing_deps
                    else "no CUDA GPU" if not feas.cuda else "; ".join(feas.notes))
                self._send(200, json.dumps({"started": False, "reason": reason}),
                           "application/json")
                return
            _Handler._lora_train = {"running": True, "last": ""}
        cfg = self.cfg

        def _worker():
            try:
                res = lora.train(cfg)
                _Handler._lora_train = {"running": False, "last": res.reason}
            except Exception as e:  # pragma: no cover - defensive
                _Handler._lora_train = {"running": False, "last": f"error: {e}"}

        threading.Thread(target=_worker, daemon=True).start()
        self._send(200, json.dumps({"started": True}), "application/json")

    def _start_lora_merge(self, adapter: str):
        """Merge an adapter -> GGUF -> Ollama in a background thread (long-running)."""
        from . import lora
        if not adapter:
            self._send(200, json.dumps({"started": False, "reason": "no adapter given"}),
                       "application/json")
            return
        with _Handler._lora_lock:
            if _Handler._lora_train.get("running"):
                self._send(200, json.dumps({"started": False,
                           "reason": "a LoRA job is already in progress"}), "application/json")
                return
            mf = lora.merge_feasibility(self.cfg)
            if not mf["ok"]:
                self._send(200, json.dumps({"started": False,
                           "reason": "; ".join(mf["notes"])}), "application/json")
                return
            _Handler._lora_train = {"running": True, "last": ""}
        cfg = self.cfg

        def _worker():
            try:
                res = lora.merge_to_gguf(cfg, adapter)
                _Handler._lora_train = {"running": False, "last": res.reason}
            except Exception as e:  # pragma: no cover
                _Handler._lora_train = {"running": False, "last": f"error: {e}"}

        threading.Thread(target=_worker, daemon=True).start()
        self._send(200, json.dumps({"started": True}), "application/json")

    def _handle_control(self, path: str, payload: dict):
        """Fleet / skills / bundle control-plane endpoints (non-streaming JSON)."""
        try:
            if path == "/fleet/act":
                from . import fleet
                action = str(payload.get("action", ""))
                agent = str(payload.get("agent", ""))
                if action == "kill":
                    fleet.engage_kill(); msg = "kill switch engaged"
                elif action == "revive":
                    fleet.clear_kill(); msg = "kill switch cleared"
                elif action == "clear":
                    msg = f"cleared {fleet.clear()} record(s)"
                elif action in ("disable", "enable"):
                    ok = fleet.set_status(agent, "disabled" if action == "disable" else "active")
                    msg = "ok" if ok else "agent not found"
                else:
                    msg = "unknown action"
                self._send(200, json.dumps({"ok": True, "msg": msg}), "application/json")
                return
            if path == "/skills/act":
                from . import skills
                reg = skills.get_registry("root")
                action = str(payload.get("action", ""))
                name = str(payload.get("name", ""))
                if action in ("disable", "enable"):
                    ok = reg.set_enabled(name, action == "enable")
                elif action == "remove":
                    ok = reg.remove(name)
                else:
                    ok = False
                self._send(200, json.dumps({"ok": bool(ok)}), "application/json")
                return
            if path == "/skills/acquire":
                from . import acquire
                from .permissions import PermissionBroker
                spec = str(payload.get("spec", "")).strip()
                if not spec:
                    self._send(200, json.dumps({"acquired": False,
                               "reason": "empty spec"}), "application/json")
                    return
                broker = PermissionBroker(allow_external_tools=True)
                for cap in ("write_skill", "install_package", "github_fetch", "code_exec"):
                    broker.grant(cap)
                client = make_client(self.cfg)
                res = acquire.author_skill(client, self.cfg, spec, broker=broker, approve=True)
                self._send(200, json.dumps(res.as_dict() | {"reason": res.reason}),
                           "application/json")
                return
            if path == "/bundle/export":
                from . import bundle
                dest = bundle.export()
                self._send(200, json.dumps({
                    "path": str(dest),
                    "bytes": dest.stat().st_size if dest.exists() else 0}),
                    "application/json")
                return
            if path == "/lora/build":
                from . import lora
                st = lora.build_dataset(self.cfg)
                self._send(200, json.dumps({
                    "total": st.total, "from_memory": st.from_memory,
                    "from_teacher": st.from_teacher}), "application/json")
                return
            if path == "/lora/train":
                self._start_lora_train()
                return
            if path == "/lora/set-base":
                from . import lora
                base = str(payload.get("base", "")).strip()
                valid = {b["id"] for b in lora.available_bases()}
                if base not in valid:
                    self._send(200, json.dumps({"ok": False, "reason": "unknown base"}),
                               "application/json")
                    return
                cfg = Config.load()
                cfg.lora_base_model = base
                cfg.save()
                _Handler.cfg = cfg
                self._send(200, json.dumps({"ok": True, "base": base}), "application/json")
                return
            if path == "/lora/set-opts":
                cfg = Config.load()
                if "epochs" in payload:
                    try:
                        ep = float(payload.get("epochs"))
                    except (TypeError, ValueError):
                        ep = cfg.lora_epochs
                    cfg.lora_epochs = max(0.5, min(10.0, ep))
                if "unsloth" in payload:
                    cfg.lora_use_unsloth = bool(payload.get("unsloth"))
                cfg.save()
                _Handler.cfg = cfg
                self._send(200, json.dumps({"ok": True, "epochs": cfg.lora_epochs,
                           "unsloth": bool(getattr(cfg, "lora_use_unsloth", True))}),
                           "application/json")
                return
            if path == "/lora/merge":
                adapter = str(payload.get("adapter", "")).strip()
                self._start_lora_merge(adapter)
                return
        except Exception as e:
            self._send(200, json.dumps({"ok": False, "error": str(e)}), "application/json")
            return
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        if not self._guard():
            return
        if self.path in ("/fleet/act", "/skills/act", "/skills/acquire", "/bundle/export",
                         "/lora/build", "/lora/train", "/lora/set-base", "/lora/merge"):
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except Exception:
                payload = {}
            self._handle_control(self.path, payload)
            return
        if self.path == "/image/start":
            from . import images
            cfg = self.cfg
            if not getattr(cfg, "allow_image_gen", False):
                self._send(200, json.dumps({"ok": False,
                           "error": "image generation is disabled"}), "application/json")
                return
            from . import comfy
            if images.backend_for(cfg) == "comfy":
                ok = comfy.ensure_running(cfg)
                err = ("" if ok else "could not start ComfyUI (none found, or it did "
                       "not come up). Install it, or set comfy_cmd to its launcher.")
                host = cfg.comfy_host
            else:
                ok = images.ensure_sd_running(cfg)
                err = ("" if ok else "could not start a Stable Diffusion server "
                       "(none installed/found, or it did not come up). Set sd_cmd to "
                       "your launcher, or start it manually with --api.")
                host = cfg.sd_host
            self._send(200, json.dumps({"ok": ok, "reachable": ok, "host": host,
                                        "error": err}), "application/json")
            return
        if self.path == "/image":
            from . import images
            cfg = self.cfg
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
                prompt = str(payload.get("prompt", "")).strip()
                if not prompt:
                    raise ValueError("empty prompt")
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "error": str(e)}),
                           "application/json")
                return
            if not getattr(cfg, "allow_image_gen", False):
                self._send(200, json.dumps(
                    {"ok": False, "error": "image generation is disabled "
                     "(set allow_image_gen)"}), "application/json")
                return
            try:
                res = images.generate(prompt, cfg,
                                      negative_prompt=str(payload.get("negative", "")))
                self._send(200, json.dumps({
                    "ok": True, "data_url": res.data_url, "path": res.path,
                    "width": res.width, "height": res.height}), "application/json")
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "error": str(e)}),
                           "application/json")
            return
        if self.path == "/video":
            from . import video
            cfg = self.cfg
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
                prompt = str(payload.get("prompt", "")).strip()
                if not prompt:
                    raise ValueError("empty prompt")
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "error": str(e)}),
                           "application/json")
                return
            try:
                secs = float(payload.get("seconds") or 0) or None
            except (TypeError, ValueError):
                secs = None
            try:
                res = video.generate(prompt, cfg, seconds=secs,
                                     negative_prompt=str(payload.get("negative", "")))
                self._send(200, json.dumps({
                    "ok": True, "path": res.path, "seconds": res.seconds,
                    "width": res.width, "height": res.height}), "application/json")
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "error": str(e)}),
                           "application/json")
            return
        if self.path == "/rate":
            # An honest learning signal from the UI: 👍/👎, a regenerate, or a Claude
            # escalation. The doc credits the model that produced the answer for the
            # task's tags. Never fabricates a score — it only records what the user did.
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except Exception:
                payload = {}
            prompt = str(payload.get("prompt", ""))[:4000]
            model = str(payload.get("model", ""))[:100]
            signal = str(payload.get("signal", ""))
            if signal not in ("rating_up", "rating_down", "escalated"):
                self._send(200, json.dumps({"ok": False, "error": "bad signal"}),
                           "application/json")
                return
            try:
                from . import routing
                role = routing.role_for_model(self.cfg, model)
                routing.record(self.cfg, role, routing.tags_for(prompt), signal)
                self._send(200, json.dumps({"ok": True}), "application/json")
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "error": str(e)}),
                           "application/json")
            return
        if self.path == "/stop":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
                rid = str(payload.get("run_id", ""))[:64]
            except Exception:
                rid = ""
            if rid:
                with _Handler._cancel_lock:
                    c = _Handler._cancel.get(rid)
                    if c is None:
                        _Handler._cancel[rid] = True   # sentinel: cancel on registration
                if hasattr(c, "cancel"):
                    c.cancel()                         # stop an already-running generation
            self._send(200, json.dumps({"ok": bool(rid)}), "application/json")
            return
        if self.path in ("/login/key", "/logout"):
            # Sign-in is a local action: only honour it from this machine, even when
            # bound to 0.0.0.0 for LAN viewing.
            if self.client_address and self.client_address[0] not in ("127.0.0.1", "::1"):
                self._send(403, json.dumps({"ok": False, "error": "sign-in is local-only"}),
                           "application/json")
                return
            from .model import save_api_key, clear_api_key, signin_status
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except Exception:
                payload = {}
            if self.path == "/logout":
                clear_api_key()
                self._send(200, json.dumps({"ok": True, **signin_status()}),
                           "application/json")
                return
            key = str(payload.get("key", "")).strip()
            if len(key) < 10 or any(c.isspace() for c in key):
                self._send(200, json.dumps(
                    {"ok": False, "error": "that does not look like a valid key"}),
                    "application/json")
                return
            try:
                save_api_key(key)
                self._send(200, json.dumps({"ok": True, **signin_status()}),
                           "application/json")
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "error": str(e)}),
                           "application/json")
            return
        if self.path == "/update/apply":
            self._stream_update()
            return
        if self.path == "/bench":
            # Drain the request body first: closing the socket with an unread body
            # makes the client see a TCP reset instead of the streamed response.
            try:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
            except Exception:
                pass
            self._stream_bench()
            return
        if self.path in ("/evolve", "/evolve/propose", "/evolve/apply"):
            # Evolve self-modifies source + commits, so gate it to the local machine
            # even when the app is bound to 0.0.0.0 for phone/LAN *viewing*.
            if self.client_address and self.client_address[0] not in ("127.0.0.1", "::1"):
                self._send(403, json.dumps({"error": "evolve is local-only"}),
                           "application/json")
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                payload = {}
            if self.path == "/evolve/apply":
                self._stream_apply(payload)
            elif self.path == "/evolve":
                self._stream_evolve(payload)   # legacy automatic path (not used by GUI)
            else:
                self._stream_propose(payload)
            return
        if self.path != "/run":
            self._send(404, "not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            prompt = str(payload.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("empty prompt")
        except Exception as e:
            self._send(200, json.dumps({"error": str(e)}), "application/json")
            return
        history = _clean_history(payload.get("history"))
        model = str(payload.get("model", "")).strip()[:100]
        verbose = bool(payload.get("verbose", False))
        run_id = str(payload.get("run_id", ""))[:64]
        session_id = str(payload.get("session_id", ""))[:80]
        # Every declared run control, validated and reduced to the Config fields it
        # changes for this run only. A key that isn't sent keeps the standing config
        # value, so an older client or a bare `{"prompt": ...}` behaves as before.
        from . import controls
        self._stream_run(prompt, history, model, verbose, run_id,
                         session_id=session_id,
                         overrides=controls.overrides_for(payload))

    def _ndjson_writer(self):
        """Begin a streamed NDJSON response and return a write(event) callback."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def write(ev: dict) -> None:
            self.wfile.write((json.dumps(ev) + "\n").encode("utf-8"))
            self.wfile.flush()
        return write

    def _stream_update(self):
        """Pull/update the configured Ollama model, streaming progress (user-approved
        via the page's yes/no click — this endpoint only runs on an explicit POST)."""
        from . import update
        write = self._ndjson_writer()
        cfg = self.cfg
        try:
            write({"stage": "update", "level": "tool",
                   "msg": f"updating {cfg.ollama_model}…", "data": {}})
            ok, final = update.pull_model(cfg, emit=write)
            write({"stage": "done", "level": "result" if ok else "error",
                   "msg": final, "data": {"ok": ok, "final": final}})
        except (BrokenPipeError, ConnectionError):
            return
        except Exception as e:
            try:
                write({"stage": "error", "level": "error", "msg": str(e), "data": {}})
            except Exception:
                pass

    def _stream_bench(self):
        """Score AG on its objective benchmark, streaming per-task progress."""
        from . import bench
        write = self._ndjson_writer()
        cfg = self.cfg
        try:
            client = make_client(cfg)
            res = bench.run_benchmark(client, cfg, emit=write)
            write({"stage": "done", "level": "result", "msg": "benchmark complete",
                   "data": {"fitness": res.fitness, "pass_rate": res.pass_rate,
                            "passed": res.passed, "n": res.n, "mode": res.mode,
                            "failing": res.failed_ids}})
        except (BrokenPipeError, ConnectionError):
            return
        except Exception as e:
            try:
                write({"stage": "error", "level": "error", "msg": str(e), "data": {}})
            except Exception:
                pass

    def _stream_evolve(self, payload):
        """Run one gated self-improvement cycle, streaming its progress + verdict.

        Only one evolve runs at a time; the server keeps answering /run requests
        meanwhile (AG CAN evolve while running). Overlapping evolves are rejected.
        """
        from . import evolve as evolve_mod
        write = self._ndjson_writer()
        cfg = self.cfg
        # Reject a second concurrent evolve rather than corrupting a half-applied edit.
        if not self._evolve_lock.acquire(blocking=False):
            write({"stage": "done", "level": "info",
                   "msg": "an evolve cycle is already running — try again when it finishes",
                   "data": {"busy": True, "adopted": False, "rolled_back": False,
                            "reason": "evolve already in progress"}})
            return
        self._evolving.set()
        try:
            client = make_client(cfg)
            # Optional split backend: a stronger model proposes; deploy backend measures.
            evolver_client = None
            prop = str((payload or {}).get("proposer", "")).strip()
            directive = str((payload or {}).get("directive", "")).strip()
            if prop and prop != cfg.backend:
                try:
                    evolver_client = make_client(cfg, backend=prop)
                    write({"stage": "evolve", "level": "tool", "data": {},
                           "msg": f"proposer: {prop}  |  fitness measured on: {cfg.backend}"})
                except Exception as e:
                    write({"stage": "evolve", "level": "error", "data": {},
                           "msg": f"proposer '{prop}' unavailable ({e}); using deploy backend"})
            write({"stage": "evolve", "level": "tool",
                   "msg": ("starting self-improvement cycle"
                           + (" toward your request…" if directive else "…")),
                   "data": {}})
            res = evolve_mod.evolve(client, cfg, emit=write,
                                    evolver_client=evolver_client, directive=directive)
            write({"stage": "done",
                   "level": "result" if res.adopted else "info",
                   "msg": res.reason, "data": {
                       "adopted": res.adopted, "rolled_back": res.rolled_back,
                       "verdict": res.verdict, "incumbent": res.incumbent_fitness,
                       "candidate": res.candidate_fitness, "delta": res.fitness_delta,
                       "changed": res.changed, "rationale": res.rationale,
                       "reason": res.reason, "snapshot_id": res.snapshot_id}})
        except (BrokenPipeError, ConnectionError):
            return
        except Exception as e:
            try:
                write({"stage": "error", "level": "error", "msg": str(e), "data": {}})
            except Exception:
                pass
        finally:
            self._evolving.clear()
            self._evolve_lock.release()

    def _stream_propose(self, payload):
        """Generate candidate self-edits and stream them for the user to choose from.

        Nothing is applied here — the automatic adopt/reject decision is replaced by
        this propose→select→apply flow. The full patches are cached server-side so the
        browser only sends back the ids it selected.
        """
        import dataclasses
        from . import evolve as evolve_mod
        write = self._ndjson_writer()
        cfg = self.cfg
        # Honor the GUI's "use vetted GitHub references" toggle for this propose only.
        if (payload or {}).get("github"):
            cfg = dataclasses.replace(cfg, evolve_use_github=True)
        if not self._evolve_lock.acquire(blocking=False):
            write({"stage": "done", "level": "info",
                   "msg": "an evolve cycle is already running — try again shortly",
                   "data": {"busy": True, "patches": []}})
            return
        try:
            client = make_client(cfg)
            evolver_client = None
            prop = str((payload or {}).get("proposer", "")).strip()
            directive = str((payload or {}).get("directive", "")).strip()
            if prop and prop != cfg.backend:
                try:
                    evolver_client = make_client(cfg, backend=prop)
                    write({"stage": "evolve", "level": "tool", "data": {},
                           "msg": f"proposer: {prop}"})
                except Exception as e:
                    write({"stage": "evolve", "level": "error", "data": {},
                           "msg": f"proposer '{prop}' unavailable ({e}); using deploy backend"})
            res = evolve_mod.propose(client, cfg, emit=write,
                                     evolver_client=evolver_client, directive=directive)
            _Handler._proposal = {p.id: p for p in res.patches}
            _Handler._proposal_directive = res.directive
            write({"stage": "done",
                   "level": "result" if res.attempted else "info",
                   "msg": res.reason, "data": res.as_dict()})
        except (BrokenPipeError, ConnectionError):
            return
        except Exception as e:
            try:
                write({"stage": "error", "level": "error", "msg": str(e), "data": {}})
            except Exception:
                pass
        finally:
            self._evolve_lock.release()

    def _stream_apply(self, payload):
        """Apply the user-selected subset of the last proposal (by id), streaming it.

        The human's selection IS the decision; only the safety test gate still applies.
        """
        from . import evolve as evolve_mod
        write = self._ndjson_writer()
        cfg = self.cfg
        ids = (payload or {}).get("ids") or []
        measure = bool((payload or {}).get("measure", False))
        cache = getattr(_Handler, "_proposal", {}) or {}
        selected = []
        for i in ids:
            p = cache.get(str(i))
            if p is not None and getattr(p, "valid", False):
                selected.append({"path": p.path, "new_content": p.new_content})
        if not selected:
            write({"stage": "done", "level": "info",
                   "msg": "nothing to apply — the proposal expired or held no valid "
                          "selection; click Evolve again",
                   "data": {"adopted": False, "rolled_back": False, "changed": [],
                            "reason": "no valid selection"}})
            return
        if not self._evolve_lock.acquire(blocking=False):
            write({"stage": "done", "level": "info",
                   "msg": "an evolve cycle is already running — try again shortly",
                   "data": {"busy": True, "adopted": False, "rolled_back": False}})
            return
        self._evolving.set()
        try:
            client = make_client(cfg)
            write({"stage": "evolve", "level": "tool", "data": {},
                   "msg": f"applying {len(selected)} selected change(s)"
                          + (" and measuring fitness…" if measure else "…")})
            res = evolve_mod.apply_selected(
                client, cfg, selected, emit=write, measure=measure,
                note=getattr(_Handler, "_proposal_directive", ""))
            write({"stage": "done",
                   "level": "result" if res.adopted else "info",
                   "msg": res.reason, "data": {
                       "adopted": res.adopted, "rolled_back": res.rolled_back,
                       "verdict": res.verdict, "incumbent": res.incumbent_fitness,
                       "candidate": res.candidate_fitness, "delta": res.fitness_delta,
                       "changed": res.changed, "rationale": res.rationale,
                       "reason": res.reason, "snapshot_id": res.snapshot_id}})
            if res.adopted:
                _Handler._proposal = {}   # consumed
        except (BrokenPipeError, ConnectionError):
            return
        except Exception as e:
            try:
                write({"stage": "error", "level": "error", "msg": str(e), "data": {}})
            except Exception:
                pass
        finally:
            self._evolving.clear()
            self._evolve_lock.release()

    def _parse_model_choice(self, choice: str):
        """Validate a "<backend>:<model>" selector value into per-run overrides.

        Returns a dict of Config overrides, or {} to use the configured default. We
        only honour a backend we actually support and, for the cloud option, only the
        configured Claude model — never an arbitrary string from the request. An
        Ollama model name is accepted as-is (the user may have just pulled it); if it
        isn't really available the run surfaces a clear error rather than pretending.
        """
        if not choice or ":" not in choice:
            return {}
        backend, _, model = choice.partition(":")
        backend, model = backend.lower().strip(), model.strip()
        if backend == "anthropic":
            return {"backend": "anthropic", "model": self.cfg.model}
        if backend == "ollama" and model:
            return {"backend": "ollama", "ollama_model": model}
        return {}

    def _stream_run(self, prompt: str, history=None, model="", verbose=False,
                    run_id="", *, session_id="", overrides=None):
        """Run the pipeline, streaming each stage event as one NDJSON line.

        `overrides` is the set of Config fields this request changes, already validated
        by ag.controls — the same machinery the CLI uses sits underneath, so a control
        in the GUI reflects what actually happens rather than describing it.

        The model's output is always streamed internally via `on_delta`: when
        `verbose` is set the chunks are forwarded to the browser as 'delta' events so
        the viewer sees the reasoning/output live; otherwise a light heartbeat keeps
        the connection observable. Either way, if the viewer disconnects (Stop button /
        closed tab) the next write fails, we raise, and generation is halted upstream.
        """
        import dataclasses
        from .pipeline import capture_memory
        write = self._ndjson_writer()
        # Per-request overrides, without mutating the shared handler config.
        merged = dict(overrides or {})
        merged.update(self._parse_model_choice(model))
        cfg = dataclasses.replace(self.cfg, **merged)
        # Ambient web fires only when the prompt actually needs external/current info
        # (a research-type task), not on chat or self-contained work like coding — that
        # was making simple requests slow. An explicit model/run still has web available.
        from . import routing
        web_eff = bool(cfg.allow_web) and ("research" in routing.tags_for(prompt))
        broker = _build_broker(cfg, web=web_eff)
        # Normalize the session id (a per-tab id from the client) so this run's working
        # memory loads and saves under one key; empty falls back to a raw transcript.
        if session_id and getattr(cfg, "working_memory", True):
            try:
                from .memory import working
                session_id = working.resolve_session(session_id)
            except Exception:
                pass

        # A Canceller lets /stop close the upstream model connection at any point (even
        # during prompt-eval), so Stop is prompt and reliable — not only once tokens flow.
        canceller = Canceller()
        if run_id:
            with _Handler._cancel_lock:
                pre = _Handler._cancel.get(run_id)
                _Handler._cancel[run_id] = canceller
            if pre is True:            # /stop arrived before we registered
                canceller.cancel()

        state = {"n": 0}

        def on_delta(text):
            state["n"] += 1
            try:
                if verbose:
                    write({"stage": "delta", "level": "token", "msg": "",
                           "data": {"text": text}})
                elif state["n"] % 16 == 0:
                    write({"stage": "progress", "level": "info", "msg": "",
                           "data": {"tokens": state["n"]}})
            except (BrokenPipeError, ConnectionError):
                raise _Interrupted()   # viewer closed the tab mid-stream

        try:
            client = make_client(cfg)
            rec = run_pipeline(client, cfg, prompt, web=web_eff, broker=broker,
                               emit=write, history=history, session_id=session_id,
                               on_delta=on_delta, cancel=canceller)
            write({"stage": "done", "level": "result", "msg": "done", "data": {
                "answer": rec.answer, "scorecard": rec.scorecard,
                "elapsed_s": rec.elapsed_s, "dry_run": rec.dry_run,
                "model_used": rec.model_used}})
            # Fill long-term memory AFTER the answer is on screen, so it never delays
            # the response. Best-effort; a "saved N fact(s)" event streams if it stores.
            try:
                from .pipeline import format_history
                convo = format_history(history,
                                       max_turns=getattr(cfg, "max_history_turns", 12))
                capture_memory(client, cfg, prompt, rec.answer,
                               conversation=convo, session_id=session_id,
                               history=history, emit=write,
                               web_sources=rec.web_sources)
            except Exception:
                pass
        except (BrokenPipeError, ConnectionError, _Interrupted, Cancelled):
            return  # viewer stopped the run or navigated away mid-stream
        except Exception as e:
            try:
                write({"stage": "error", "level": "error", "msg": str(e), "data": {}})
            except Exception:
                pass
        finally:
            if run_id:
                with _Handler._cancel_lock:
                    _Handler._cancel.pop(run_id, None)

    def log_message(self, *a):  # quiet
        pass

def _doctor_data(cfg: Config) -> dict:
    """Environment / readiness snapshot for the GUI Status button (same facts as the
    `ag doctor` CLI). Read-only; probes Ollama without hard-failing."""
    import os
    import urllib.request
    from . import archive, backup, bench
    from .evolve import gate_available
    from .model import has_oauth_profile, oauth_token_status

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY")
                   or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    oauth = has_oauth_profile()
    reachable, models = False, []
    try:
        with urllib.request.urlopen(cfg.ollama_host.rstrip("/") + "/api/tags",
                                    timeout=1.5) as r:
            reachable = True
            models = [m.get("name", "?")
                      for m in json.loads(r.read().decode("utf-8")).get("models", [])]
    except Exception:
        pass
    if cfg.backend == "auto":
        if has_key or oauth:
            eff = "anthropic"
        elif getattr(cfg, "offline_backend", "") == "ollama":
            eff = "ollama"
        else:
            eff = "dry-run (stub)"
    else:
        eff = cfg.backend
    last = archive.adopted_history(limit=1)
    return {
        "backend": cfg.backend, "effective_backend": eff, "model": cfg.model,
        "ollama_model": cfg.ollama_model, "ollama_reachable": reachable,
        "ollama_models": models, "api_key": has_key, "oauth": oauth,
        "oauth_token": oauth_token_status() if oauth else "none",
        "autonomy": cfg.autonomy_level, "evolver_backend": cfg.evolver_backend,
        "fitness_gate": cfg.fitness_gate, "bench_tasks": len(bench.load_tasks(cfg=cfg)),
        "bench_mode": cfg.bench_mode, "bench_samples": cfg.bench_samples,
        "evolve_gate": "ready" if gate_available() else "unavailable (pip install pytest)",
        "snapshots": len(backup.list_snapshots()),
        "last_improvement": last[0] if last else None, "allow_web": cfg.allow_web,
    }

def _list_ollama_models(cfg: Config) -> list:
    """Names of models currently pulled in the local Ollama (empty if unreachable)."""
    import urllib.request
    try:
        with urllib.request.urlopen(cfg.ollama_host.rstrip("/") + "/api/tags",
                                    timeout=1.5) as r:
            return [m.get("name") for m
                    in json.loads(r.read().decode("utf-8")).get("models", [])
                    if m.get("name")]
    except Exception:
        return []

def _models_data(cfg: Config) -> dict:
    """Models the GUI selector can offer, and the current effective choice.

    Each option's value is "<backend>:<model>" so the /run handler knows both which
    backend to use and which model. Local (Ollama) models are always listed if the
    server is reachable; the cloud Claude model is offered only when creds/OAuth are
    present. `current` reflects what a run would use right now with no override."""
    import re as _re
    from .model import _has_anthropic_creds, has_oauth_profile
    models = _list_ollama_models(cfg)
    cloud = bool(_has_anthropic_creds() or has_oauth_profile())

    # Curate the list: drop what can't answer a prompt (embedding models) and raw
    # timestamped build artifacts (e.g. ...-obliterated-20260918-215836) that just
    # duplicate a stable alias — the "unused models" clutter. AG's two working models
    # (the abliterated task model and the instruct chat model) are labelled and sorted
    # to the top; everything else pulled remains selectable.
    def _keep(name: str) -> bool:
        low = name.lower()
        if "embed" in low:
            return False
        if _re.search(r"-\d{8}-\d{6}", name):   # a dated build snapshot, not a model
            return False
        return True

    primary, spec = cfg.ollama_model, getattr(cfg, "specialist_model", "")

    def _label(m: str) -> str:
        if m == primary:
            return f"{m} · local · primary"
        if m == spec:
            return f"{m} · local · specialist (abliterated)"
        return f"{m} · local"

    def _rank(m: str) -> int:
        return 0 if m == primary else 1 if m == spec else 2

    kept = sorted((m for m in models if _keep(m)), key=lambda m: (_rank(m), m.lower()))
    # Local models first so the offline default is the obvious top choice; the cloud
    # Claude option is listed last and labelled with its cost, since picking it is the
    # user's explicit opt-in to spend API tokens.
    options = [{"value": f"ollama:{m}", "label": _label(m)} for m in kept]
    if cloud:
        options.append({"value": f"anthropic:{cfg.model}",
                        "label": f"{cfg.model} · Claude cloud (uses API tokens)"})

    if cfg.backend == "anthropic" or (cfg.backend == "auto" and cloud):
        current = f"anthropic:{cfg.model}"
    elif cfg.backend == "ollama" or (
            cfg.backend == "auto" and getattr(cfg, "offline_backend", "") == "ollama"):
        current = f"ollama:{cfg.ollama_model}"
    else:
        current = f"ollama:{cfg.ollama_model}"
    return {"options": options, "current": current, "cloud": cloud,
            "ollama_reachable": bool(models)}

def run_prompt(cfg: Config, prompt: str) -> tuple[str, str]:
    """Run one prompt through the pipeline; returns (answer, meta-string).

    Non-streaming helper kept for programmatic use and tests; the web UI uses the
    streaming path in `_Handler._stream_run`.
    """
    client = make_client(cfg)
    broker = _build_broker(cfg)
    rec = run_pipeline(client, cfg, prompt, web=cfg.allow_web, broker=broker)
    sc = rec.scorecard or {}
    meta = f"dry_run={rec.dry_run} speed={sc.get('speed', '?')}"
    return rec.answer, meta

def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def _is_loopback(host: str) -> bool:
    """Only these bind addresses are unreachable from the network."""
    return host in ("127.0.0.1", "::1", "localhost", "")


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False,
          token: str = "", no_auth: bool = False):
    _Handler.cfg = Config.load()
    _Handler.bind_host = host
    _Handler.bind_port = port

    # Binding off-loopback publishes an endpoint that runs the model, reads files
    # through the tool layer, and can start an evolve cycle. Requiring a token there
    # — and generating one automatically, so it cannot be skipped by forgetting —
    # makes the safe path the default path. Loopback is untouched: a personal tool
    # on your own machine should not ask you to log in.
    if _is_loopback(host) or no_auth:
        _Handler.auth_token = ""
    else:
        import secrets
        _Handler.auth_token = token or secrets.token_urlsafe(24)

    httpd = ThreadingHTTPServer((host, port), _Handler)
    tok = _Handler.auth_token
    qs = f"/?token={tok}" if tok else "/"
    local = f"http://127.0.0.1:{port}{qs}"
    print(f"Apple-Gorilla web app running:")
    print(f"  this machine : {local}")
    if not _is_loopback(host):
        print(f"  on your phone: http://{_lan_ip()}:{port}{qs}  (same wifi)")
    else:
        print(f"  (localhost only; use --host 0.0.0.0 to reach it from your phone)")
    if tok:
        print(f"\n  ACCESS TOKEN: {tok}")
        print("  This instance is reachable from the network, so requests without "
              "the token are refused.")
        print("  Open the link above (the token is set as a cookie on first load).")
    elif not _is_loopback(host):
        print("\n  WARNING: --no-auth on a network address. Anyone who can reach "
              f"port {port} can run this instance.")
    print("Ctrl+C to stop.")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(local)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
