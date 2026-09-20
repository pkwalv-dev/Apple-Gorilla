"""Household cluster: discover your other devices on the LAN and share compute.

AG on this machine is the *coordinator*; each other device you own runs an `ag node`
*worker*. Workers announce themselves with a stdlib UDP beacon (`discovery.py`); the
coordinator keeps a registry of them, matches sub-agents to each node's CPU/GPU/RAM,
and dispatches work over plain HTTP (`cluster.py`, `node.py`).

Safety model (chosen by the operator): discovery is OPEN on the local network — any
device can be *seen* — but a node refuses to *run* anything until the controlling AG
user APPROVES it in the dashboard. That human approval, minted through the UI, is the
credential; there is no pre-shared secret to type. Everything here is stdlib-only and
best-effort: a missing/unreachable node degrades to "offline", never an exception into
a run.
"""
from __future__ import annotations

import json
import socket
import uuid
from pathlib import Path

from ..config import STATE_DIR

CLUSTER_DIR = STATE_DIR / "cluster"
IDENTITY_FILE = CLUSTER_DIR / "identity.json"

BEACON_MAGIC = "ag-cluster/1"


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "device"


def lan_ip() -> str:
    """This machine's LAN address (best-effort; falls back to loopback)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def identity(name: str = "") -> dict:
    """This machine's stable cluster identity {node_id, name}, created once and reused by
    both the coordinator and the worker so a device is the same node in every view."""
    try:
        CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
        if IDENTITY_FILE.exists():
            data = json.loads(IDENTITY_FILE.read_text(encoding="utf-8"))
        else:
            data = {}
    except Exception:
        data = {}
    if not data.get("node_id"):
        data["node_id"] = uuid.uuid4().hex[:12]
    if name:
        data["name"] = name
    if not data.get("name"):
        data["name"] = _hostname()
    try:
        IDENTITY_FILE.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass
    return data
