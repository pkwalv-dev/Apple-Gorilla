"""Storage behind an interface — the bet that gives the highest ceiling.

AG talks to memory only through `MemoryStore` (add / get / update / delete / all /
write_all). Nothing in AG touches a file or a schema directly, so the storage engine
can grow from flat JSONL today to SQLite, a vector database, or a graph store later
with *zero changes to AG*. The structure never needs rebuilding — only the engine
behind it.

`JsonlStore` is the default engine: plain-text, append-friendly, inspectable, and
portable (it's just files under `state/memory/` that travel with the agent). Memory
is **namespaced per agent** and split by layer:

    state/memory/<agent>/episodic.jsonl
    state/memory/<agent>/semantic.jsonl
    state/memory/<agent>/procedural.jsonl

Per-agent namespacing is what lets a spawned AG carry its own memory while a parent
can still read across the lineage; per-layer files let high-volume episodic memory be
pruned independently of durable semantic/procedural knowledge.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .types import Memory, MemoryKind


class MemoryStore:
    """Interface for a namespaced, layered memory store."""

    def add(self, mem: Memory) -> Memory:  # pragma: no cover - interface
        raise NotImplementedError

    def get(self, agent: str, mem_id: str) -> Optional[Memory]:  # pragma: no cover
        raise NotImplementedError

    def update(self, mem: Memory) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def delete(self, agent: str, mem_id: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    def all(self, agent: str, kinds: Optional[Iterable[str]] = None) -> List[Memory]:  # pragma: no cover
        raise NotImplementedError

    def write_all(self, agent: str, kind: str, mems: List[Memory]) -> None:  # pragma: no cover
        raise NotImplementedError

    def agents(self) -> List[str]:  # pragma: no cover - interface
        raise NotImplementedError


class JsonlStore(MemoryStore):
    """Default engine: one JSONL file per (agent, layer). Never raises on read — a
    corrupt line is skipped, not fatal — so recall is always available."""

    def __init__(self, base_dir: Path):
        self.base = Path(base_dir)

    # --- paths -------------------------------------------------------------
    def _agent_dir(self, agent: str) -> Path:
        d = self.base / _safe(agent)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _file(self, agent: str, kind: str) -> Path:
        return self._agent_dir(agent) / f"{MemoryKind.valid(kind)}.jsonl"

    # --- reads -------------------------------------------------------------
    def _load_file(self, path: Path) -> List[Memory]:
        if not path.exists():
            return []
        out: List[Memory] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Memory.from_dict(json.loads(line)))
            except Exception:
                continue  # tolerate a corrupt line rather than losing the whole store
        return out

    def all(self, agent: str, kinds: Optional[Iterable[str]] = None) -> List[Memory]:
        kinds = tuple(kinds) if kinds else MemoryKind.ALL
        out: List[Memory] = []
        for kind in kinds:
            out.extend(self._load_file(self._file(agent, kind)))
        return out

    def get(self, agent: str, mem_id: str) -> Optional[Memory]:
        for m in self.all(agent):
            if m.id == mem_id:
                return m
        return None

    def agents(self) -> List[str]:
        if not self.base.exists():
            return []
        return sorted(p.name for p in self.base.iterdir() if p.is_dir())

    # --- writes ------------------------------------------------------------
    def add(self, mem: Memory) -> Memory:
        path = self._file(mem.agent, mem.kind)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(mem.to_dict(), ensure_ascii=False) + "\n")
        return mem

    def write_all(self, agent: str, kind: str, mems: List[Memory]) -> None:
        path = self._file(agent, kind)
        path.write_text(
            "".join(json.dumps(m.to_dict(), ensure_ascii=False) + "\n" for m in mems),
            encoding="utf-8",
        )

    def update(self, mem: Memory) -> None:
        mems = self._load_file(self._file(mem.agent, mem.kind))
        replaced = False
        for i, m in enumerate(mems):
            if m.id == mem.id:
                mems[i] = mem
                replaced = True
                break
        if not replaced:
            mems.append(mem)
        self.write_all(mem.agent, mem.kind, mems)

    def delete(self, agent: str, mem_id: str) -> bool:
        for kind in MemoryKind.ALL:
            mems = self._load_file(self._file(agent, kind))
            kept = [m for m in mems if m.id != mem_id]
            if len(kept) != len(mems):
                self.write_all(agent, kind, kept)
                return True
        return False


def _safe(name: str) -> str:
    """Namespace names become directory names; keep them filesystem-safe."""
    keep = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in (name or "root"))
    return keep or "root"
