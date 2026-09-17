"""Vetted-GitHub fetch — pull proven code from an allowlist, not the open internet.

`fetch()` reads a single raw file from a repository whose "owner/repo" is on the
user-controlled `github_allowlist` (config.json). It is gated by the `github_fetch`
capability, reads only (never writes, clones, or executes), and returns the file text
as UNTRUSTED reference data — the caller decides whether to use it. This is how AG
draws on battle-tested implementations instead of authoring everything from scratch,
while the allowlist keeps "draw on GitHub" from meaning "run anything on GitHub".
"""
from __future__ import annotations

import urllib.request
from typing import List, Optional

from ..permissions import PermissionBroker

RAW = "https://raw.githubusercontent.com"
MAX_CHARS = 20000


def allowed(repo: str, allowlist: List[str]) -> bool:
    repo = (repo or "").strip().lower()
    return any(repo == a.strip().lower() for a in (allowlist or []))


def fetch(repo: str, path: str, *, broker: PermissionBroker,
          allowlist: List[str], ref: str = "main", timeout: float = 15.0) -> str:
    """Return the text of `path` in `owner/repo` at `ref`, if the repo is allowlisted.

    Gated by 'github_fetch'. Reject anything off the allowlist with a clear message.
    """
    broker.require("github_fetch")
    repo = (repo or "").strip()
    if repo.count("/") != 1:
        return "github_fetch error: repo must be 'owner/repo'"
    if not allowed(repo, allowlist):
        return (f"github_fetch refused: {repo} is not on the allowlist. "
                f"Add it to config.json -> github_allowlist to permit it.")
    path = (path or "").lstrip("/")
    url = f"{RAW}/{repo}/{ref}/{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"github_fetch error: {e}"
    if len(data) > MAX_CHARS:
        return data[:MAX_CHARS] + f"\n…[truncated at {MAX_CHARS} chars]"
    return data
