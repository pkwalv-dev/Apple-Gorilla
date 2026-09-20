"""UDP beacon discovery — stdlib only, zero dependencies.

A worker node broadcasts a small JSON beacon on the LAN every few seconds:

    {"magic":"ag-cluster/1","node_id":"…","name":"desktop","host":"192.168.1.5",
     "port":8767,"caps":{…host.inspect…},"ts":<epoch>}

The coordinator runs a listener that collects beacons into the node registry
(`cluster.py`). Broadcast reaches every device on the same subnet without any central
server; a device that cannot broadcast (some locked-down networks) can still be added
by static address in the UI. Nothing here authenticates — being *seen* is open; being
*run on* is gated by UI approval in `node.py`.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from typing import Callable, Optional

from . import BEACON_MAGIC


def _broadcast_addrs() -> list:
    # 255.255.255.255 reaches the local subnet on every common home network; we also
    # send to the /24 directed broadcast of our own IP as a fallback for stacks that
    # drop the global one.
    from . import lan_ip
    addrs = ["255.255.255.255"]
    ip = lan_ip()
    if ip and ip.count(".") == 3 and not ip.startswith("127."):
        addrs.append(ip.rsplit(".", 1)[0] + ".255")
    return addrs


class Beacon:
    """Periodically broadcasts this node's presence. Runs in a daemon thread."""

    def __init__(self, *, node_id: str, name: str, host: str, port: int,
                 beacon_port: int, caps_fn: Callable[[], dict], interval: float = 5.0):
        self.node_id = node_id
        self.name = name
        self.host = host
        self.port = port
        self.beacon_port = beacon_port
        self.caps_fn = caps_fn
        self.interval = max(1.0, float(interval))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _payload(self) -> bytes:
        try:
            caps = self.caps_fn() or {}
        except Exception:
            caps = {}
        msg = {"magic": BEACON_MAGIC, "node_id": self.node_id, "name": self.name,
               "host": self.host, "port": self.port, "caps": caps, "ts": time.time()}
        return json.dumps(msg).encode("utf-8")

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except OSError:
            pass
        while not self._stop.is_set():
            data = self._payload()
            for addr in _broadcast_addrs():
                try:
                    sock.sendto(data, (addr, self.beacon_port))
                except OSError:
                    continue
            self._stop.wait(self.interval)
        sock.close()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()


class Listener:
    """Listens for node beacons and calls `on_beacon(dict)` for each valid one. Runs in
    a daemon thread; safe to start once alongside the coordinator's web server."""

    def __init__(self, beacon_port: int, on_beacon: Callable[[dict], None]):
        self.beacon_port = beacon_port
        self.on_beacon = on_beacon
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except OSError:
            pass
        try:
            sock.bind(("", self.beacon_port))
        except OSError:
            return  # port busy (e.g. a node on this box already binds it) — give up quietly
        sock.settimeout(1.0)
        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            if not isinstance(msg, dict) or msg.get("magic") != BEACON_MAGIC:
                continue
            # Trust the sender's on-wire address over any self-reported host.
            if not msg.get("host"):
                msg["host"] = addr[0]
            try:
                self.on_beacon(msg)
            except Exception:
                pass
        sock.close()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()


def probe_once(beacon_port: int, timeout: float = 3.0) -> list:
    """One-shot listen for beacons — used by `ag cluster scan` on the CLI. Returns the
    deduplicated node dicts heard within `timeout` seconds."""
    seen: dict = {}
    lis = Listener(beacon_port, lambda m: seen.__setitem__(m.get("node_id"), m))
    lis.start()
    time.sleep(max(0.5, timeout))
    lis.stop()
    return [v for k, v in seen.items() if k]
