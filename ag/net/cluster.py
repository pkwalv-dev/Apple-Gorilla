"""Coordinator side of the household cluster.

Keeps the registry of discovered worker nodes, tracks which the operator has APPROVED,
scores each node's capability, picks where a sub-agent should run, and dispatches the
run over HTTP. State lives under state/cluster/ (git-ignored). Stdlib only; every
network op is best-effort and degrades to "node offline" rather than raising.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

from ..config import Config
from . import CLUSTER_DIR, identity, lan_ip

NODES_FILE = CLUSTER_DIR / "nodes.jsonl"

_LOCK = threading.RLock()


@dataclass
class NodeInfo:
    node_id: str
    name: str = ""
    host: str = ""
    port: int = 0
    caps: dict = field(default_factory=dict)   # host.inspect() dict of the node
    last_seen: float = 0.0                     # epoch of the most recent beacon
    approved: bool = False                     # operator cleared it in the UI
    self_node: bool = False                    # this is the coordinator's own machine

    def as_dict(self, stale_s: float = 20.0) -> dict:
        d = asdict(self)
        d["online"] = self.online(stale_s)
        d["status"] = self.status(stale_s)
        d["score"] = self.score()
        return d

    def online(self, stale_s: float = 20.0) -> bool:
        return bool(self.last_seen) and (time.time() - self.last_seen) <= stale_s

    def status(self, stale_s: float = 20.0) -> str:
        if not self.approved:
            return "pending"
        return "online" if self.online(stale_s) else "offline"

    def score(self) -> float:
        """A rough 'how much can this node do' number for placement — cores weighted by
        RAM, with a big bonus for a real (non-integrated) GPU."""
        c = self.caps or {}
        cpu = float(c.get("cpu_count") or 1)
        ram = float(c.get("ram_gb") or 1)
        gpu = str(c.get("gpu") or "")
        gpu_bonus = 8.0 if gpu and "unknown" not in gpu.lower() and "integrated" not in gpu.lower() else 0.0
        # A measured bench (MIPS) refines the static estimate once a node has run one.
        mips_bonus = min(float(c.get("benchMips") or 0) / 20.0, 12.0)
        return round(cpu * 1.0 + ram * 0.3 + gpu_bonus + mips_bonus, 2)


# --- registry io -----------------------------------------------------------
def _load() -> Dict[str, NodeInfo]:
    out: Dict[str, NodeInfo] = {}
    if not NODES_FILE.exists():
        return out
    try:
        for line in NODES_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out[d["node_id"]] = NodeInfo(**{k: d.get(k)
                                    for k in NodeInfo.__dataclass_fields__ if k in d})
            except Exception:
                continue
    except OSError:
        pass
    return out


def _save(nodes: Dict[str, NodeInfo]) -> None:
    try:
        CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
        NODES_FILE.write_text(
            "".join(json.dumps(asdict(n)) + "\n" for n in nodes.values()),
            encoding="utf-8")
    except OSError:
        pass


# --- discovery ingest ------------------------------------------------------
def ingest_beacon(msg: dict) -> None:
    """Record/refresh a node from a received beacon. Keeps the operator's `approved`
    decision across beacons; a re-appearing node is not un-approved."""
    nid = msg.get("node_id")
    if not nid:
        return
    me = identity().get("node_id")
    with _LOCK:
        nodes = _load()
        n = nodes.get(nid) or NodeInfo(node_id=nid)
        n.name = msg.get("name") or n.name
        n.host = msg.get("host") or n.host
        n.port = int(msg.get("port") or n.port or 0)
        if msg.get("caps"):
            n.caps = msg["caps"]
        n.last_seen = float(msg.get("ts") or time.time())
        if nid == me:
            n.self_node = True
            n.approved = True   # this machine is always trusted to run its own work
        elif getattr(Config.load(), "net_auto_approve_nodes", False):
            n.approved = True
        nodes[nid] = n
        _save(nodes)


def ensure_self(cfg: Optional[Config] = None) -> None:
    """Make sure THIS machine appears in the registry (always approved, always 'online'),
    so the dashboard shows the coordinator alongside the workers even before any beacon."""
    cfg = cfg or Config.load()
    from .. import host
    me = identity(getattr(cfg, "net_node_name", ""))
    try:
        caps = host.inspect(probe_internet=False).as_dict()
    except Exception:
        caps = {}
    with _LOCK:
        nodes = _load()
        n = nodes.get(me["node_id"]) or NodeInfo(node_id=me["node_id"])
        n.name = me["name"]
        n.host = lan_ip()
        n.port = int(getattr(cfg, "net_node_port", 8767) or 8767)
        n.caps = caps or n.caps
        n.last_seen = time.time()
        n.self_node = True
        n.approved = True
        nodes[me["node_id"]] = n
        _save(nodes)


# --- queries ---------------------------------------------------------------
def list_nodes(cfg: Optional[Config] = None) -> List[NodeInfo]:
    cfg = cfg or Config.load()
    with _LOCK:
        return sorted(_load().values(), key=lambda n: (not n.self_node, n.name))


def get(node_id: str) -> Optional[NodeInfo]:
    with _LOCK:
        return _load().get(node_id)


def approve(node_id: str, approved: bool = True) -> bool:
    with _LOCK:
        nodes = _load()
        if node_id not in nodes:
            return False
        nodes[node_id].approved = approved
        _save(nodes)
        node = nodes[node_id]
    if approved and not node.self_node:
        _send_pairing(node)     # tell the node this coordinator is now trusted
    return True


def forget(node_id: str) -> bool:
    with _LOCK:
        nodes = _load()
        if node_id not in nodes:
            return False
        del nodes[node_id]
        _save(nodes)
    return True


def add_static(host: str, port: int) -> Optional[NodeInfo]:
    """Add a node by address for networks where broadcast is blocked. We probe /hello to
    learn its identity + caps; returns the NodeInfo, or None if unreachable."""
    info = _http_json(f"http://{host}:{port}/hello", timeout=3.0)
    if not info or not info.get("node_id"):
        return None
    info["host"] = host
    info["port"] = port
    info["ts"] = time.time()
    ingest_beacon(info)
    return get(info["node_id"])


# --- placement -------------------------------------------------------------
def usable_nodes(cfg: Optional[Config] = None, *, want_gpu: bool = False) -> List[NodeInfo]:
    cfg = cfg or Config.load()
    stale = float(getattr(cfg, "net_node_stale_s", 20.0) or 20.0)
    out = []
    for n in list_nodes(cfg):
        if n.self_node or not n.approved or not n.online(stale):
            continue
        if want_gpu:
            gpu = str((n.caps or {}).get("gpu") or "")
            if not gpu or "unknown" in gpu.lower() or "integrated" in gpu.lower():
                continue
        out.append(n)
    return out


def pick_node(cfg: Optional[Config] = None, *, want_gpu: bool = False) -> Optional[NodeInfo]:
    """The best remote node for a task, or None to run locally. GPU tasks only ever go to
    a GPU node; otherwise the highest-capability approved+online node wins."""
    cfg = cfg or Config.load()
    if getattr(cfg, "net_placement", "auto") != "auto":
        return None
    cands = usable_nodes(cfg, want_gpu=want_gpu)
    if not cands:
        return None
    return max(cands, key=lambda n: n.score())


# --- dispatch --------------------------------------------------------------
def dispatch_run(node: NodeInfo, *, agent: str, role: str, task: str,
                 parent_agent: str, depth: int, cfg: Optional[Config] = None,
                 timeout: float = 600.0) -> Optional[str]:
    """Ask an approved node to run one sub-agent and return its output text. The
    coordinator identifies itself so the node can confirm it is the trusted one and can
    call back for heartbeats / stop checks. Returns None on any failure (caller then
    falls back to local)."""
    cfg = cfg or Config.load()
    me = identity()
    port = int(getattr(cfg, "net_node_port", 8767) or 8767)  # our own node/callback port
    body = {
        "coordinator_id": me.get("node_id"),
        "coordinator_name": me.get("name"),
        "coordinator_url": f"http://{lan_ip()}:{_coordinator_web_port(cfg)}",
        "agent": agent, "role": role, "task": task,
        "parent_agent": parent_agent, "depth": depth,
    }
    res = _http_json(f"http://{node.host}:{node.port}/run", body=body, timeout=timeout)
    if not res or not res.get("ok"):
        return None
    return res.get("output", "")


def dispatch_compute(node: NodeInfo, *, iterations: int, seed: int,
                     timeout: float = 300.0) -> Optional[dict]:
    """Ask an approved node to run one compute shard. Returns its result dict, or None."""
    me = identity()
    body = {"coordinator_id": me.get("node_id"), "iterations": iterations, "seed": seed}
    return _http_json(f"http://{node.host}:{node.port}/compute", body=body, timeout=timeout)


def fanout_bench(iterations: int, cfg: Optional[Config] = None, *,
                 emit=None) -> dict:
    """Split one benchmark across every approved, online device — INCLUDING this one —
    weighted by capability, run the shards in parallel, verify each, and aggregate. This
    is the 'use every device's resources for mutual processing' primitive."""
    import threading
    from . import compute, tasks

    cfg = cfg or Config.load()
    ensure_self(cfg)
    stale = float(getattr(cfg, "net_node_stale_s", 20.0) or 20.0)
    me = identity().get("node_id")

    # Eligible = this machine (runs its shard in-process) + approved, online remotes.
    eligible = []
    for n in list_nodes(cfg):
        if n.self_node or (n.approved and n.online(stale)):
            eligible.append(n)
    if not eligible:
        return {"ok": False, "reason": "no devices available — approve a node, or it is offline"}

    weighted = [(n.node_id, n.name, n.score()) for n in eligible]
    plan = compute.plan_shards(weighted, iterations)
    by_id = {n.node_id: n for n in eligible}

    parent = tasks.create(title=f"distributed bench · {iterations/1e6:.1f}M iterations",
                          kind="fanout", status="running",
                          payload={"iterations": iterations, "shards": len(plan)})
    if emit:
        emit(f"fanout: {len(plan)} shard(s) across "
             + ", ".join(f"{s['name']} ({s['iterations']/1e6:.1f}M)" for s in plan))

    results: dict = {}
    threads = []

    def _work(shard):
        nid = shard["node_id"]
        node = by_id[nid]
        child = tasks.create(title=f"shard → {shard['name']}", kind="bench",
                             status="running", node_id=nid, parent_id=parent["id"],
                             shard_index=plan.index(shard),
                             payload={"iterations": shard["iterations"], "seed": shard["seed"]})
        if node.self_node or nid == me:
            try:
                r = compute.run_shard(shard["iterations"], shard["seed"]); r["ok"] = True
            except Exception as e:
                r = {"ok": False, "error": str(e)}
            r["name"] = node.name
        else:
            r = dispatch_compute(node, iterations=shard["iterations"], seed=shard["seed"])
            if not r:
                r = {"ok": False, "error": "node unreachable", "name": node.name}
            r.setdefault("name", node.name)
        r["iterations"] = shard["iterations"]
        results[nid] = r
        tasks.update(child["id"], status="done" if r.get("ok") else "failed",
                     result={k: r.get(k) for k in ("mips", "checksum", "canary", "ms")},
                     error=r.get("error", ""))
        # feed measured throughput back into the node's caps for future scheduling
        if r.get("ok") and r.get("mips"):
            _record_mips(nid, float(r["mips"]))

    for shard in plan:
        t = threading.Thread(target=_work, args=(shard,), daemon=True)
        threads.append(t); t.start()
    for t in threads:
        t.join(timeout=320.0)

    ordered = [results.get(s["node_id"], {"ok": False, "name": s["name"]}) for s in plan]
    summary = compute.aggregate(ordered)
    summary["ok"] = True
    summary["per_node"] = [{"name": r.get("name", "?"), "mips": r.get("mips", 0),
                            "iterations": r.get("iterations", 0), "ok": bool(r.get("ok")),
                            "verified": r.get("canary") == compute.canary_checksum()}
                           for r in ordered]
    tasks.update(parent["id"], status="done", result=summary)
    summary["task_id"] = parent["id"]
    return summary


def _record_mips(node_id: str, mips: float) -> None:
    with _LOCK:
        nodes = _load()
        if node_id in nodes:
            caps = dict(nodes[node_id].caps or {})
            caps["benchMips"] = mips
            nodes[node_id].caps = caps
            _save(nodes)


def _send_pairing(node: NodeInfo) -> bool:
    me = identity()
    body = {"coordinator_id": me.get("node_id"), "coordinator_name": me.get("name"),
            "coordinator_url": f"http://{lan_ip()}:{_coordinator_web_port()}"}
    res = _http_json(f"http://{node.host}:{node.port}/pair", body=body, timeout=4.0)
    return bool(res and res.get("ok"))


def _coordinator_web_port(cfg: Optional[Config] = None) -> int:
    # The coordinator's web UI port — nodes call back here for heartbeat/stop. Overridable
    # via the running server; defaults to the documented 8770.
    return int(_WEB_PORT[0])


_WEB_PORT = [8770]


def set_web_port(port: int) -> None:
    """The web server records its port here so dispatched nodes can call back to it."""
    _WEB_PORT[0] = int(port)


# --- http helper -----------------------------------------------------------
def _http_json(url: str, *, body: Optional[dict] = None, timeout: float = 5.0) -> Optional[dict]:
    try:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="POST" if data is not None else "GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
