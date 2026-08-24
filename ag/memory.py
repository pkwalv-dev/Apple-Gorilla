"""Persistent memory — the thing AG was missing.

A run used to start cold every time. This gives AG a durable store it can write
facts into and recall relevant ones from on later runs, so context carries across
sessions. Stdlib only; stored as JSONL under state/memory/ (git-ignored, bounded).

Recall is deliberately simple and offline: a keyword-overlap score between the
query and each memory, so it works with no model, no network, and no embeddings.
It is not semantic search — it is a dependable, inspectable baseline that the
reasoning loop and pipeline can lean on. Retrieval never raises; a corrupt line is
skipped, not fatal.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from .config import STATE_DIR, ensure_dirs

MEMORY_DIR = STATE_DIR / "memory"
MEMORY_FILE = MEMORY_DIR / "memories.jsonl"

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an and or but of to in on at for with is are was were be been being this "
    "that it its as by from into be i you he she they we my your our their".split()
)


@dataclass
class Memory:
    id: str
    text: str
    tags: List[str]
    created: str

    def as_dict(self) -> dict:
        return asdict(self)


def _tokens(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


def _ensure() -> None:
    ensure_dirs()
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("")


def remember(text: str, tags: Optional[List[str]] = None, *,
             max_memories: int = 200) -> Optional[Memory]:
    """Append a fact to memory. Returns the stored Memory (None for empty text).

    De-dupes exact-text repeats so recall isn't drowned by the same note.
    """
    text = (text or "").strip()
    if not text:
        return None
    _ensure()
    for m in all_memories():
        if m.text.strip() == text:
            return m  # already known — no duplicate
    mem = Memory(
        id=time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}",
        text=text, tags=[t.strip() for t in (tags or []) if t.strip()],
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(mem.as_dict()) + "\n")
    _prune(max_memories)
    return mem


def all_memories() -> List[Memory]:
    _ensure()
    out: List[Memory] = []
    for line in MEMORY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out.append(Memory(id=d["id"], text=d["text"],
                              tags=list(d.get("tags", [])), created=d.get("created", "")))
        except Exception:
            continue  # skip a corrupt line rather than failing recall
    return out


def recall(query: str, *, k: int = 5) -> List[Memory]:
    """Return up to k memories most relevant to `query` by keyword overlap."""
    q = _tokens(query)
    if not q:
        return []
    scored = []
    for m in all_memories():
        mt = _tokens(m.text + " " + " ".join(m.tags))
        if not mt:
            continue
        overlap = len(q & mt)
        if overlap:
            # Normalise by query size so short, on-topic memories rank fairly.
            scored.append((overlap / len(q), m))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in scored[:k]]


def memory_context(query: str, *, k: int = 5) -> str:
    """A labeled block of recalled memories for injecting into a prompt (or '')."""
    hits = recall(query, k=k)
    if not hits:
        return ""
    lines = [f"- {m.text}" for m in hits]
    return "\n".join(lines)


def forget(mem_id: str) -> bool:
    """Delete one memory by id. Returns True if something was removed."""
    mems = all_memories()
    kept = [m for m in mems if m.id != mem_id]
    if len(kept) == len(mems):
        return False
    _write_all(kept)
    return True


def clear() -> int:
    """Erase all memories. Returns how many were removed."""
    n = len(all_memories())
    _write_all([])
    return n


def _prune(max_memories: int) -> int:
    if max_memories <= 0:
        return 0
    mems = all_memories()
    if len(mems) <= max_memories:
        return 0
    kept = mems[-max_memories:]  # newest survive (file is append-ordered)
    _write_all(kept)
    return len(mems) - len(kept)


def _write_all(mems: List[Memory]) -> None:
    _ensure()
    MEMORY_FILE.write_text(
        "".join(json.dumps(m.as_dict()) + "\n" for m in mems), encoding="utf-8")
