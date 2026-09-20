"""Worker node — run `ag node` on any device you own to lend its compute to AG.

The node advertises itself with a UDP beacon (its CPU/GPU/RAM from host.inspect) and
serves a small HTTP API. It runs a dispatched sub-agent on THIS machine's model +
tools, so a task placed here uses this device's power. Stdlib only.

Trust: the node accepts a run only from a coordinator the operator has APPROVED in the
dashboard (that approval calls /pair here). An un-paired coordinator's run is refused
with a message telling the user to approve this node. Set net_auto_approve_nodes on the
node to skip pairing (trusts anyone on the LAN — convenient, less safe).
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from ..config import Config
from . import CLUSTER_DIR, identity, lan_ip
from .discovery import Beacon

TRUSTED_FILE = CLUSTER_DIR / "trusted.json"


def _node_caps() -> dict:
    from .. import host
    try:
        caps = host.inspect(probe_internet=False).as_dict()
    except Exception:
        caps = {}
    caps["elevated"] = _is_elevated()
    return caps


def _is_elevated() -> bool:
    """Report (never acquire) whether this process already has admin/root — decided by
    how the operator launched it. AG does not self-escalate."""
    try:
        if os.name == "nt":
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        return os.geteuid() == 0
    except Exception:
        return False


def _load_trusted() -> dict:
    try:
        return json.loads(TRUSTED_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_trusted(d: dict) -> None:
    try:
        CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
        TRUSTED_FILE.write_text(json.dumps(d), encoding="utf-8")
    except OSError:
        pass


def _coordinator_ping(url: str, path: str, *, body: Optional[dict] = None,
                      timeout: float = 4.0) -> Optional[dict]:
    if not url:
        return None
    try:
        u = url.rstrip("/") + path
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            u, data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="POST" if data is not None else "GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


class _NodeHandler(BaseHTTPRequestHandler):
    cfg: Config = Config()
    node_id: str = ""
    node_name: str = ""
    running: dict = {}          # agent -> {"started": ts, "coordinator": id}
    _lock = threading.Lock()

    def log_message(self, *a):  # keep the console quiet
        pass

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _read(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(n) or b"{}") if n else {}
        except Exception:
            return {}

    def do_GET(self):
        if self.path == "/hello":
            self._send(200, {"node_id": self.node_id, "name": self.node_name,
                             "caps": _node_caps(), "port": self.cfg.net_node_port})
        elif self.path == "/health":
            with self._lock:
                running = list(self.running.keys())
            self._send(200, {"ok": True, "node_id": self.node_id,
                             "running": running})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        payload = self._read()
        if self.path == "/pair":
            cid = str(payload.get("coordinator_id", ""))
            if not cid:
                self._send(400, {"ok": False, "error": "no coordinator_id"})
                return
            trusted = _load_trusted()
            trusted[cid] = {"name": payload.get("coordinator_name", ""),
                            "url": payload.get("coordinator_url", ""), "ts": time.time()}
            _save_trusted(trusted)
            self._send(200, {"ok": True, "node_id": self.node_id})
            return
        if self.path == "/run":
            self._handle_run(payload)
            return
        if self.path == "/compute":
            self._handle_compute(payload)
            return
        self._send(404, {"ok": False, "error": "not found"})

    def _handle_compute(self, payload: dict):
        """Run one distributed-compute shard on this device. Same trust gate as /run: a
        coordinator must be approved before it can spend this machine's cycles."""
        cid = str(payload.get("coordinator_id", ""))
        if not self._trusts(cid):
            self._send(403, {"ok": False,
                "error": "this node is not approved for that coordinator — approve it "
                         "in the AG dashboard (Cluster tab) and try again"})
            return
        from .compute import run_shard
        try:
            out = run_shard(int(payload.get("iterations", 0)),
                            int(payload.get("seed", 0)))
            out["ok"] = True
            out["node_id"] = self.node_id
            out["name"] = self.node_name
            self._send(200, out)
        except Exception as e:
            self._send(200, {"ok": False, "error": str(e)})

    def _trusts(self, coordinator_id: str) -> bool:
        if getattr(self.cfg, "net_auto_approve_nodes", False):
            return True
        return coordinator_id in _load_trusted()

    def _handle_run(self, payload: dict):
        cid = str(payload.get("coordinator_id", ""))
        curl = str(payload.get("coordinator_url", ""))
        if not self._trusts(cid):
            self._send(403, {"ok": False,
                "error": "this node is not approved for that coordinator — approve it "
                         "in the AG dashboard (Cluster tab) and try again"})
            return
        agent = str(payload.get("agent") or f"remote.{int(time.time()*1000)%100000}")
        role = str(payload.get("role", "worker"))
        task = str(payload.get("task", ""))
        parent = str(payload.get("parent_agent", "root"))
        depth = int(payload.get("depth", 1) or 1)
        location = f"{self.node_name} @ {lan_ip()}"
        with self._lock:
            self.running[agent] = {"started": time.time(), "coordinator": cid}
        try:
            output = _execute(self.cfg, agent=agent, role=role, task=task,
                              parent=parent, depth=depth, coordinator_url=curl,
                              node_id=self.node_id, location=location)
            self._send(200, {"ok": True, "output": output, "agent": agent,
                             "node_id": self.node_id, "location": location})
        except Exception as e:
            self._send(200, {"ok": False, "error": str(e), "agent": agent})
        finally:
            with self._lock:
                self.running.pop(agent, None)


def _full_broker(cfg: Config):
    """A remote sub-agent is NOT capability-limited: it gets this node's full local
    toolset, including code execution and self-extension. The gate is that the run only
    happens for a coordinator the operator approved — see _trusts()."""
    from ..permissions import PermissionBroker
    broker = PermissionBroker(allow_external_tools=True)
    for cap in ("network", "filesystem_read", "filesystem_write_outside_repo",
                "spawn_agent", "code_exec", "write_skill", "install_package",
                "github_fetch", "acquire_auto"):
        broker.grant(cap)
    return broker


def _execute(cfg: Config, *, agent: str, role: str, task: str, parent: str, depth: int,
             coordinator_url: str, node_id: str, location: str) -> str:
    """Run one sub-agent's reason loop on this machine, heartbeating to the coordinator
    and honouring a stop it signals (the swarm UI's per-agent kill)."""
    from ..model import make_client
    from .. import reason

    client = make_client(cfg)
    broker = _full_broker(cfg)
    pid = os.getpid()

    def on_step():
        _coordinator_ping(coordinator_url, "/cluster/heartbeat",
                          body={"agent": agent, "node_id": node_id,
                                "location": location, "pid": pid})

    def stop_check():
        r = _coordinator_ping(coordinator_url, "/cluster/stopped",
                              body={"agent": agent})
        return bool(r and r.get("stop"))

    system = (f"You are a focused sub-agent with the single role: {role}. "
              "Do only this task. Be concise and return just the result.")
    on_step()  # first heartbeat registers where it is running
    res = reason.solve(client, cfg, system=system, user=task, broker=broker,
                       agent=agent, parents=[parent], depth=depth,
                       stop_check=stop_check, on_step=on_step)
    return res.answer


def serve_node(host: str = "0.0.0.0", port: int = 0, name: str = "") -> None:
    """Start this device as a worker node: beacon + HTTP server. Blocks until Ctrl+C."""
    cfg = Config.load()
    ident = identity(name or cfg.net_node_name)
    port = port or int(getattr(cfg, "net_node_port", 8767) or 8767)
    _NodeHandler.cfg = cfg
    _NodeHandler.node_id = ident["node_id"]
    _NodeHandler.node_name = ident["name"]

    beacon = Beacon(node_id=ident["node_id"], name=ident["name"], host=lan_ip(),
                    port=port, beacon_port=int(getattr(cfg, "net_beacon_port", 8766)),
                    caps_fn=_node_caps,
                    interval=float(getattr(cfg, "net_beacon_interval", 5.0)))
    beacon.start()
    httpd = ThreadingHTTPServer((host, port), _NodeHandler)
    print(f"AG node '{ident['name']}' ({ident['node_id']}) online:")
    print(f"  serving : http://{lan_ip()}:{port}")
    print(f"  beacon  : UDP {cfg.net_beacon_port} (broadcasting caps every "
          f"{cfg.net_beacon_interval:g}s)")
    caps = _node_caps()
    print(f"  offering: {caps.get('cpu_count','?')} CPU · {caps.get('ram_gb','?')} GB RAM"
          f" · GPU {caps.get('gpu','?')}")
    if getattr(cfg, "net_auto_approve_nodes", False):
        print("  trust   : AUTO — runs from any LAN coordinator are accepted")
    else:
        print("  trust   : waiting to be approved in your AG dashboard (Cluster tab)")
    if _is_elevated():
        print("  privs   : ELEVATED (admin/root) — this node's agents run with full "
              "system rights on THIS device")
    else:
        print("  privs   : standard user — agents are limited to your account. To grant "
              "admin on this box, YOU relaunch elevated (Run as administrator / sudo).")
    print("Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        beacon.stop()
        httpd.shutdown()
