"""Inventory of the tools and applications AG can interact with, each rated on
how well-integrated and how frictionless it is.

This is AG taking stock of its own reach. For every backend, tool, and gated
capability it can (or could) use, we report:

  - status:       "available" | "degraded" | "unavailable"
  - integration:  0-10, how wired-in it is (fully functional vs. scaffolded vs. absent)
  - friction:     0-10, how frictionless to actually use (10 = zero setup; a key,
                  a running service, or a manual permission grant lowers it)
  - notes:        what it is, and what would raise its scores

The scores are deliberately probe-driven (they reflect *this* machine right now),
so the evolve loop can steer at the highest-friction integration — the place where
AG's usefulness is most bottlenecked. Read-only: nothing here performs a tool action.
"""
from __future__ import annotations

import json
import os
import shutil
import urllib.request
from dataclasses import dataclass, asdict
from typing import List, Optional

from .config import Config
from .permissions import GATED


@dataclass
class ToolReport:
    name: str
    category: str            # backend | retrieval | agent | host | capability | gate
    status: str              # available | degraded | unavailable
    integration: int         # 0..10 (how wired-in)
    friction: int            # 0..10 (10 = frictionless)
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _has_anthropic_creds() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _anthropic_sdk() -> bool:
    import importlib.util
    return importlib.util.find_spec("anthropic") is not None


def _ollama_models(cfg: Config) -> Optional[List[str]]:
    """Return pulled model names if Ollama is reachable, else None."""
    try:
        url = cfg.ollama_host.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(url, timeout=1.5) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m.get("name", "?") for m in data.get("models", [])]
    except Exception:
        return None


def _report_anthropic(cfg: Config) -> ToolReport:
    from .model import has_oauth_profile
    sdk = _anthropic_sdk()
    creds = _has_anthropic_creds() or has_oauth_profile()
    if sdk and creds:
        return ToolReport("anthropic (Claude API)", "backend", "available", 10, 7,
                          "Highest-quality backend. Friction: needs an API key or "
                          "OAuth login; pip-installed SDK.")
    if not sdk:
        return ToolReport("anthropic (Claude API)", "backend", "unavailable", 10, 4,
                          "SDK not installed. `pip install -r requirements.txt` and "
                          "set ANTHROPIC_API_KEY (or `ant auth login`) to enable.")
    return ToolReport("anthropic (Claude API)", "backend", "degraded", 10, 5,
                      "SDK installed but no credentials. Set ANTHROPIC_API_KEY or "
                      "run `ant auth login` to activate.")


def _report_ollama(cfg: Config) -> ToolReport:
    models = _ollama_models(cfg)
    if models is None:
        return ToolReport("ollama (local models)", "backend", "unavailable", 9, 6,
                          "Not reachable at " + cfg.ollama_host + ". Install from "
                          "ollama.com and run `ollama serve`.")
    if not models:
        return ToolReport("ollama (local models)", "backend", "degraded", 9, 7,
                          "Reachable but no models pulled. `ollama pull "
                          f"{cfg.ollama_model}` to enable keyless local answers.")
    ready = any(cfg.ollama_model in m for m in models)
    note = "Keyless, offline, local. " + (
        f"Configured model '{cfg.ollama_model}' is pulled." if ready
        else f"Configured model '{cfg.ollama_model}' NOT pulled (have: "
             f"{', '.join(models)}).")
    return ToolReport("ollama (local models)", "backend",
                      "available" if ready else "degraded", 10,
                      9 if ready else 7, note)


def _report_dry(cfg: Config) -> ToolReport:
    return ToolReport("dry-run stub", "backend", "available", 10, 10,
                      "Offline deterministic stub for exercising the machinery. "
                      "Zero setup; no real reasoning.")


def _report_web(cfg: Config) -> ToolReport:
    # Retrieval is stdlib-only; the only friction is the 'network' grant, which
    # allow_web auto-grants. Live reachability depends on the network at run time.
    if cfg.allow_web:
        return ToolReport("web search + fetch", "retrieval", "available", 8, 9,
                          "DuckDuckGo search + page fetch, injected as UNTRUSTED "
                          "reference data. allow_web auto-grants 'network'. Friction: "
                          "a single scrape-parser (no API); brittle to layout changes.")
    return ToolReport("web search + fetch", "retrieval", "degraded", 8, 6,
                      "Wired but allow_web=false: each run needs an explicit --web "
                      "grant. Set allow_web=true for standing access.")


def _report_subagents(cfg: Config) -> ToolReport:
    if cfg.allow_external_tools:
        return ToolReport("sub-agents", "agent", "available", 6, 6,
                          "Bounded, non-recursive role-scoped calls. Gated by "
                          "'spawn_agent'; runs sequentially (no parallel fan-out yet).")
    return ToolReport("sub-agents", "agent", "degraded", 6, 5,
                      "Implemented but gated off: needs allow_external_tools + a "
                      "'spawn_agent' grant. No parallelism yet.")


def _report_host(cfg: Config) -> ToolReport:
    return ToolReport("host introspection", "host", "available", 9, 10,
                      "Read-only CPU/RAM/GPU/disk/connectivity probe used to size "
                      "AG's own work. Zero setup, no external service.")


def _report_evolve_gate(cfg: Config) -> ToolReport:
    from .evolve import gate_available
    if gate_available():
        return ToolReport("self-improvement gate (pytest)", "gate", "available", 10, 8,
                          "The test suite that gates every self-edit. Present, so "
                          "`ag evolve` can verify candidates.")
    return ToolReport("self-improvement gate (pytest)", "gate", "unavailable", 10, 4,
                      "pytest missing -> `ag evolve` fails closed. `pip install "
                      "pytest` to enable directed self-improvement.")


# Capabilities that exist as *gates* but have no wired driver yet (README: the
# broker + extension points are in place; the drivers are deliberately absent).
_SCAFFOLDED = {
    "browser": "Chrome / browser automation. Broker gate exists; no driver wired "
               "(would connect via an MCP client or the Agent SDK).",
    "shell": "Arbitrary shell execution. Gated; no executor wired by design.",
    "delete": "Destructive deletion. Gated; intentionally not wired.",
    "send_message": "Email / chat / outbound comms. Gated; no transport wired.",
    "filesystem_write_outside_repo": "Writing outside the repo. Gated; not wired.",
}


def _report_scaffolded(cfg: Config) -> List[ToolReport]:
    out = []
    for cap in sorted(GATED):
        if cap == "network":
            continue  # covered by the web retrieval report
        note = _SCAFFOLDED.get(cap, "Gated capability; no driver wired.")
        out.append(ToolReport(cap, "capability", "unavailable", 2, 3,
                              "Scaffolded (default-deny gate present, driver absent, "
                              "by design). " + note))
    return out


def inventory(cfg: Optional[Config] = None) -> List[ToolReport]:
    """Probe and rate every tool/app AG can interact with, right now."""
    cfg = cfg or Config.load()
    reports = [
        _report_anthropic(cfg),
        _report_ollama(cfg),
        _report_dry(cfg),
        _report_web(cfg),
        _report_subagents(cfg),
        _report_host(cfg),
        _report_evolve_gate(cfg),
    ]
    reports.extend(_report_scaffolded(cfg))
    return reports


def summary(cfg: Optional[Config] = None) -> dict:
    """Structured inventory + aggregate friction/integration, for telemetry/GUI."""
    reports = inventory(cfg)
    dicts = [r.as_dict() for r in reports]
    avail = [r for r in reports if r.status == "available"]
    # The most impactful place to improve: the wired-but-highest-friction tool.
    improvable = [r for r in reports if r.status != "unavailable"]
    worst = min(improvable, key=lambda r: r.friction, default=None)
    return {
        "tools": dicts,
        "counts": {
            "total": len(reports),
            "available": len(avail),
            "degraded": sum(r.status == "degraded" for r in reports),
            "unavailable": sum(r.status == "unavailable" for r in reports),
        },
        "avg_integration": round(sum(r.integration for r in reports) / len(reports), 1),
        "avg_friction": round(sum(r.friction for r in reports) / len(reports), 1),
        "highest_friction_wired": worst.name if worst else None,
    }
