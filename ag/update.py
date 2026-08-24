"""On-demand Ollama model updates — check now, pull only on approval.

This is deliberately NOT a background poller. A network check happens only when an
update is *instituted* by the user: the web app pings the check once after a run
completes (and shows a one-click yes/no), or you run `ag update`. Nothing is ever
pulled without an explicit approval.

How the check works without downloading anything: Ollama stores each model as an
OCI-style manifest whose sha256 IS the registry's `Docker-Content-Digest`. We hash
the local manifest file and compare it to the digest the registry reports for the
same tag. Different digest -> a newer build exists. Everything is best-effort and
fails *safe*: if we cannot determine the state, we report "unknown", never a false
"up to date". Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .config import Config

_REGISTRY = "https://registry.ollama.ai"
_MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"


@dataclass
class ModelUpdateStatus:
    model: str
    state: str               # "update-available" | "up-to-date" | "unknown"
    local_digest: Optional[str] = None
    remote_digest: Optional[str] = None
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.state == "update-available"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["available"] = self.available
        return d


def _split_model(name: str) -> tuple[str, str, str]:
    """'qwen2.5:7b' -> ('library', 'qwen2.5', '7b'); 'ns/repo:tag' respects ns."""
    repo, _, tag = name.partition(":")
    tag = tag or "latest"
    if "/" in repo:
        ns, _, r = repo.partition("/")
        return ns, r, tag
    return "library", repo, tag


def _models_dir() -> Path:
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return Path(env)
    return Path.home() / ".ollama" / "models"


def local_manifest_digest(model: str) -> Optional[str]:
    """sha256 of the local manifest file for `model`, or None if not installed."""
    ns, repo, tag = _split_model(model)
    path = _models_dir() / "manifests" / "registry.ollama.ai" / ns / repo / tag
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return "sha256:" + hashlib.sha256(data).hexdigest()


def remote_manifest_digest(model: str, *, timeout: float = 4.0) -> Optional[str]:
    """The registry's Docker-Content-Digest for `model`, or None if unreachable."""
    ns, repo, tag = _split_model(model)
    url = f"{_REGISTRY}/v2/{ns}/{repo}/manifests/{tag}"
    req = urllib.request.Request(url, headers={"Accept": _MANIFEST_ACCEPT},
                                 method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            digest = r.headers.get("Docker-Content-Digest")
            if digest:
                return digest.strip()
            # Fallback: hash the body (matches how Ollama derives the digest).
            body = r.read()
            return "sha256:" + hashlib.sha256(body).hexdigest()
    except Exception:
        return None


def check_model_update(cfg: Optional[Config] = None,
                       model: Optional[str] = None) -> ModelUpdateStatus:
    """Compare the local model manifest to the registry. Fails safe to 'unknown'."""
    cfg = cfg or Config.load()
    model = model or cfg.ollama_model
    local = local_manifest_digest(model)
    if local is None:
        return ModelUpdateStatus(model, "unknown", None, None,
                                 "model not installed locally (or custom models dir)")
    remote = remote_manifest_digest(model)
    if remote is None:
        return ModelUpdateStatus(model, "unknown", local, None,
                                 "could not reach the model registry")
    if local == remote:
        return ModelUpdateStatus(model, "up-to-date", local, remote,
                                 "local build matches the registry")
    return ModelUpdateStatus(model, "update-available", local, remote,
                             "a newer build is published for this tag")


def pull_model(cfg: Optional[Config] = None, model: Optional[str] = None,
               *, emit=None) -> tuple[bool, str]:
    """Pull/update the model via Ollama (the 'apply' step). Streams progress to
    `emit` (a dict callback) if given. Returns (success, final_status)."""
    cfg = cfg or Config.load()
    model = model or cfg.ollama_model
    host = cfg.ollama_host.rstrip("/")
    payload = json.dumps({"name": model, "stream": True}).encode("utf-8")
    req = urllib.request.Request(f"{host}/api/pull", data=payload,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    last = ""
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("error"):
                    return False, str(obj["error"])
                status = obj.get("status", "")
                if status and status != last:
                    last = status
                    _emit(emit, model, status, obj)
            return True, last or "pull complete"
    except Exception as e:  # network / ollama down
        return False, f"pull failed: {e}"


def _emit(emit, model: str, status: str, obj: dict) -> None:
    if emit is None:
        return
    try:
        pct = ""
        total, done = obj.get("total"), obj.get("completed")
        if total and done:
            pct = f" {round(100 * done / total)}%"
        emit({"stage": "update", "level": "tool",
              "msg": f"pulling {model}: {status}{pct}", "data": {}})
    except Exception:
        pass
