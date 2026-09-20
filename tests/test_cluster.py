"""Tests for the household cluster: fleet swarm control, node registry, discovery
payloads, capability placement, and remote-run trust. All hermetic — no real network."""
import json

import pytest

from ag.config import Config


# --- fleet: per-agent kill, staleness, heartbeat, location ------------------
def _patch_fleet(monkeypatch, tmp_path):
    from ag import fleet
    monkeypatch.setattr(fleet, "FLEET_DIR", tmp_path)
    monkeypatch.setattr(fleet, "AGENTS_FILE", tmp_path / "agents.jsonl")
    monkeypatch.setattr(fleet, "STOP_FILE", tmp_path / "STOP")
    monkeypatch.setattr(fleet, "KILL_DIR", tmp_path / "kill")


def test_record_spawn_captures_location(monkeypatch, tmp_path):
    from ag import fleet
    _patch_fleet(monkeypatch, tmp_path)
    fleet.record_spawn("root.w-1", role="w", parent="root", depth=1,
                       node="node9", location="desktop @ 192.168.1.5", pid=4242)
    rec = fleet.get("root.w-1")
    assert rec.node == "node9" and rec.pid == 4242
    assert rec.as_dict()["location"] == "desktop @ 192.168.1.5"
    assert rec.heartbeat > 0


def test_kill_one_agent_and_descendants(monkeypatch, tmp_path):
    from ag import fleet
    _patch_fleet(monkeypatch, tmp_path)
    fleet.record_spawn("root.a-1", role="a", parent="root", depth=1)
    fleet.record_spawn("root.a-1.b-2", role="b", parent="root.a-1", depth=2)
    assert fleet.kill_agent("root.a-1") is True
    assert fleet.is_killed("root.a-1") is True
    assert fleet.is_killed("root.a-1.b-2") is True    # inherits the ancestor's kill
    assert fleet.should_stop("root.a-1.b-2") is True
    assert fleet.get("root.a-1").status == "killed"
    assert fleet.kill_agent("nope") is False


def test_reap_stale_agents(monkeypatch, tmp_path):
    from ag import fleet
    _patch_fleet(monkeypatch, tmp_path)
    fleet.record_spawn("root.fresh-1", role="w", parent="root", depth=1)
    fleet.record_spawn("root.old-2", role="w", parent="root", depth=1)
    # Age one agent's heartbeat past the stale window by rewriting the file.
    recs = fleet._load()
    recs["root.old-2"].heartbeat = 1.0    # epoch 1970 — very stale
    fleet._save(recs)
    reaped = fleet.reap_stale(stale_s=60.0)
    assert reaped == ["root.old-2"]
    assert fleet.get("root.old-2").status == "killed"
    assert fleet.get("root.fresh-1").status == "active"


def test_heartbeat_refreshes(monkeypatch, tmp_path):
    from ag import fleet
    _patch_fleet(monkeypatch, tmp_path)
    fleet.record_spawn("root.w-1", role="w", parent="root", depth=1)
    recs = fleet._load(); recs["root.w-1"].heartbeat = 1.0; fleet._save(recs)
    fleet.heartbeat("root.w-1", node="n2", location="laptop", pid=7)
    rec = fleet.get("root.w-1")
    assert rec.heartbeat > 1.0 and rec.node == "n2" and rec.pid == 7


# --- cluster registry: ingest, approve, placement ---------------------------
def _patch_cluster(monkeypatch, tmp_path, node_id="me-000"):
    from ag.net import cluster
    monkeypatch.setattr(cluster, "NODES_FILE", tmp_path / "nodes.jsonl")
    monkeypatch.setattr(cluster, "identity", lambda name="": {"node_id": node_id, "name": "me"})
    monkeypatch.setattr(cluster, "_send_pairing", lambda node: True)  # no network
    return cluster


def _beacon(node_id, name, host, cpu=8, ram=16.0, gpu="NVIDIA RTX 4060"):
    return {"magic": "ag-cluster/1", "node_id": node_id, "name": name, "host": host,
            "port": 8767, "ts": 9e9,  # far-future ts => always "online"
            "caps": {"cpu_count": cpu, "ram_gb": ram, "gpu": gpu}}


def test_ingest_and_approve_node(monkeypatch, tmp_path):
    cluster = _patch_cluster(monkeypatch, tmp_path)
    cluster.ingest_beacon(_beacon("n1", "desktop", "192.168.1.5"))
    n = cluster.get("n1")
    assert n and n.name == "desktop" and n.approved is False      # pending until approved
    assert n.status(stale_s=20.0) == "pending"
    assert cluster.approve("n1") is True
    assert cluster.get("n1").approved is True
    assert cluster.get("n1").status(stale_s=20.0) == "online"


def test_ingest_preserves_approval(monkeypatch, tmp_path):
    cluster = _patch_cluster(monkeypatch, tmp_path)
    cluster.ingest_beacon(_beacon("n1", "desktop", "192.168.1.5"))
    cluster.approve("n1")
    cluster.ingest_beacon(_beacon("n1", "desktop", "192.168.1.5"))  # re-appears
    assert cluster.get("n1").approved is True                      # not un-approved


def test_pick_node_prefers_gpu_and_capability(monkeypatch, tmp_path):
    cluster = _patch_cluster(monkeypatch, tmp_path)
    cluster.ingest_beacon(_beacon("cpu", "laptop", "192.168.1.6", cpu=4, ram=8.0,
                                  gpu="integrated"))
    cluster.ingest_beacon(_beacon("gpu", "desktop", "192.168.1.5", cpu=16, ram=32.0,
                                  gpu="NVIDIA RTX 4060"))
    for nid in ("cpu", "gpu"):
        cluster.approve(nid)
    cfg = Config(net_cluster=True, net_placement="auto")
    # A GPU task must land on the GPU node only.
    assert cluster.pick_node(cfg, want_gpu=True).node_id == "gpu"
    # A general task picks the highest-capability node (also the desktop here).
    assert cluster.pick_node(cfg, want_gpu=False).node_id == "gpu"


def test_placement_local_disables_remote(monkeypatch, tmp_path):
    cluster = _patch_cluster(monkeypatch, tmp_path)
    cluster.ingest_beacon(_beacon("gpu", "desktop", "192.168.1.5"))
    cluster.approve("gpu")
    cfg = Config(net_cluster=True, net_placement="local")
    assert cluster.pick_node(cfg) is None                          # local-only: never remote


def test_unapproved_node_is_not_usable(monkeypatch, tmp_path):
    cluster = _patch_cluster(monkeypatch, tmp_path)
    cluster.ingest_beacon(_beacon("n1", "desktop", "192.168.1.5"))  # never approved
    cfg = Config(net_cluster=True, net_placement="auto")
    assert cluster.pick_node(cfg) is None


# --- discovery: beacon payload is valid + parseable -------------------------
def test_beacon_payload_roundtrips():
    from ag.net.discovery import Beacon
    b = Beacon(node_id="x1", name="dev", host="10.0.0.2", port=8767, beacon_port=8766,
               caps_fn=lambda: {"cpu_count": 8})
    msg = json.loads(b._payload().decode("utf-8"))
    assert msg["magic"] == "ag-cluster/1" and msg["node_id"] == "x1"
    assert msg["caps"]["cpu_count"] == 8 and msg["port"] == 8767


# --- node: trust gate + full capability broker ------------------------------
def test_node_trust_requires_pairing(monkeypatch, tmp_path):
    from ag.net import node
    monkeypatch.setattr(node, "TRUSTED_FILE", tmp_path / "trusted.json")
    node._NodeHandler.cfg = Config(net_auto_approve_nodes=False)
    h = node._NodeHandler.__new__(node._NodeHandler)      # avoid BaseHTTPRequestHandler init
    assert h._trusts("coord-A") is False                 # unpaired => refused
    node._save_trusted({"coord-A": {"name": "main", "url": "", "ts": 1}})
    assert h._trusts("coord-A") is True                  # paired via UI approval


def test_node_auto_approve_trusts_anyone(monkeypatch, tmp_path):
    from ag.net import node
    monkeypatch.setattr(node, "TRUSTED_FILE", tmp_path / "trusted.json")
    node._NodeHandler.cfg = Config(net_auto_approve_nodes=True)
    h = node._NodeHandler.__new__(node._NodeHandler)
    assert h._trusts("anyone") is True


def test_remote_subagent_broker_is_unlimited():
    from ag.net.node import _full_broker
    broker = _full_broker(Config())
    for cap in ("code_exec", "filesystem_read", "spawn_agent", "network", "write_skill"):
        assert broker.check(cap) is True                 # sub-agents are NOT capped on a node


# --- agents: placement heuristic + graceful fallback ------------------------
def test_wants_gpu_heuristic():
    from ag import agents
    assert agents._wants_gpu("render", "make a video clip") is True
    assert agents._wants_gpu("writer", "summarize this text") is False


# --- distributed compute: deterministic bench, sharding, aggregate ----------
def test_bench_is_deterministic():
    from ag.net import compute
    a = compute.bench(50_000, seed=123)
    b = compute.bench(50_000, seed=123)
    assert a["checksum"] == b["checksum"]                 # same input => same checksum
    assert compute.bench(50_000, seed=124)["checksum"] != a["checksum"]
    assert a["iterations"] == 50_000


def test_kernels_agree_and_bench_reports_kernel():
    from ag.net import compute
    # Whatever path bench takes, it must match the scalar reference for the same input,
    # or remote verification (which compares checksums) would break.
    ref = f"{compute._sum_scalar(0, 300_000, 999) & compute.MASK:08x}"
    b = compute.bench(300_000, 999)
    assert b["checksum"] == ref
    assert b["kernel"] in ("numpy", "scalar")
    np = compute._numpy()
    if np is not None:
        assert f"{compute._sum_numpy(np, 0, 300_000, 999) & compute.MASK:08x}" == ref


def test_run_shard_carries_verifiable_canary():
    from ag.net import compute
    r = compute.run_shard(30_000, seed=7)
    assert r["canary"] == compute.canary_checksum()       # coordinator can verify it


def test_plan_shards_weights_by_capability():
    from ag.net import compute
    plan = compute.plan_shards([("weak", "phone", 5.0), ("strong", "desktop", 45.0)],
                               10_000_000)
    by = {p["node_id"]: p for p in plan}
    assert by["strong"]["iterations"] > by["weak"]["iterations"]   # muscle => more work
    assert all(p["iterations"] >= compute.MIN_SHARD for p in plan)


def test_aggregate_verifies_and_counts():
    from ag.net import compute
    good = compute.canary_checksum()
    shards = [{"ok": True, "mips": 10, "iterations": 1000, "canary": good},
              {"ok": True, "mips": 20, "iterations": 2000, "canary": good},
              {"ok": False, "iterations": 0}]
    agg = compute.aggregate(shards)
    assert agg["shards_done"] == 2 and agg["shards_failed"] == 1
    assert agg["aggregate_mips"] == 30 and agg["verified"] is True
    bad = [{"ok": True, "mips": 5, "iterations": 500, "canary": "deadbeef"}]
    assert compute.aggregate(bad)["verified"] is False    # a lying node fails verification


# --- task log ---------------------------------------------------------------
def test_task_log_lifecycle(monkeypatch, tmp_path):
    from ag.net import tasks
    monkeypatch.setattr(tasks, "TASKS_FILE", tmp_path / "tasks.jsonl")
    p = tasks.create(title="parent", kind="fanout", status="running")
    c = tasks.create(title="shard", kind="bench", parent_id=p["id"], shard_index=0)
    tasks.update(c["id"], status="done", result={"mips": 9})
    assert tasks.get(c["id"])["status"] == "done"
    assert tasks.get(c["id"])["finished"] > 0
    assert [t["id"] for t in tasks.children(p["id"])] == [c["id"]]
    assert len(tasks.recent()) == 2


# --- fanout end to end on the local device (no network) ---------------------
def test_fanout_runs_on_local_device(monkeypatch, tmp_path):
    from ag.net import cluster, tasks
    monkeypatch.setattr(cluster, "NODES_FILE", tmp_path / "nodes.jsonl")
    monkeypatch.setattr(tasks, "TASKS_FILE", tmp_path / "tasks.jsonl")
    monkeypatch.setattr(cluster, "identity", lambda name="": {"node_id": "self", "name": "here"})
    cfg = Config(net_cluster=True, net_placement="auto")
    res = cluster.fanout_bench(300_000, cfg)              # only this machine is eligible
    assert res["ok"] is True and res["shards_done"] == 1
    assert res["verified"] is True                        # local shard always verifies
    assert res["per_node"][0]["name"] == "here"
    assert tasks.get(res["task_id"])["status"] == "done"


def test_try_remote_returns_none_without_nodes(monkeypatch, tmp_path):
    from ag import agents
    from ag.net import cluster
    monkeypatch.setattr(cluster, "NODES_FILE", tmp_path / "nodes.jsonl")
    monkeypatch.setattr(cluster, "identity", lambda name="": {"node_id": "me", "name": "me"})
    cfg = Config(net_cluster=True, net_placement="auto")
    out = agents._try_remote(cfg, "root.w-1", role="w", task="hi",
                             parent_agent="root", depth=1, emit=None)
    assert out is None                                   # no nodes => fall back to local
