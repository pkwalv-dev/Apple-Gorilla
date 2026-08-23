"""Host introspection and resource use (read-only, self-directed).

AG inspects the machine it runs on and adapts to it: scale parallelism to the CPU,
size the local model to the GPU, confirm it has outbound connectivity. This is
*use* of the host's own resources for AG's legitimate work — it does not touch other
devices, network infrastructure, or anything AG doesn't own on this box.

Network posture: AG is EGRESS-ONLY. It opens no listening sockets and runs no server,
so nothing can connect *to* it. The connectivity check below is a single outbound
probe AG makes about itself; it accepts no inbound traffic and emits no telemetry.
"""
from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class HostInfo:
    system: str
    machine: str
    cpu_count: int
    ram_gb: Optional[float]
    gpu: str
    disk_free_gb: Optional[float]
    internet: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _ram_gb() -> Optional[float]:
    # Best-effort, stdlib only, cross-platform.
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
                         / 1e9, 1)
    except (ValueError, OSError):
        pass
    try:  # macOS / BSD
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                             text=True, timeout=3)
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return round(int(out.stdout.strip()) / 1e9, 1)
    except Exception:
        pass
    return None


def _gpu() -> str:
    # NVIDIA (Windows/Linux, incl. the user's RTX rig).
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass
    # Apple / others.
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                                 capture_output=True, text=True, timeout=8)
            for line in out.stdout.splitlines():
                if "Chipset Model:" in line:
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return "unknown / integrated"


def _disk_free_gb() -> Optional[float]:
    try:
        return round(shutil.disk_usage(os.getcwd()).free / 1e9, 1)
    except Exception:
        return None


def has_internet(timeout: float = 2.0) -> bool:
    """Single outbound connectivity probe (AG checking its own reach)."""
    for host, port in (("1.1.1.1", 443), ("8.8.8.8", 53)):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def inspect(probe_internet: bool = True) -> HostInfo:
    return HostInfo(
        system=platform.system(),
        machine=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=_ram_gb(),
        gpu=_gpu(),
        disk_free_gb=_disk_free_gb(),
        internet=has_internet() if probe_internet else False,
    )


def pick_concurrency(cap: int = 8) -> int:
    """How many parallel workers AG should use for its own fan-out (web fetches,
    sub-agents). Uses the host's cores, bounded so AG stays a good citizen."""
    return max(1, min(cap, (os.cpu_count() or 2)))
