"""Memory records and the layered taxonomy.

AG's memory is not a flat list of notes. It is organized into the layers a durable
learner needs, so that raw experience can be distilled into knowledge and knowledge
into reusable skill:

- EPISODIC   — what happened: one record per run/exchange (prompt, answer, score).
               High volume, decays fastest. The raw material reflection learns from.
- SEMANTIC   — distilled truth: generalized facts about the user/world extracted from
               many episodes ("uses metric units", "has an RTX 4060").
- PROCEDURAL — learned know-how: reusable strategies that measurably worked ("for
               arithmetic, call calc first"). This is where remembering becomes
               *capability*, and it is the store AG's skills draw on.

Every record is model-agnostic plain data: it carries an optional cached embedding
(so recall is by meaning, not keywords) and `links` to other memories (so the store
is a graph that can be reasoned over, not just a bag of strings). Nothing here is
tied to a specific LLM or storage engine.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class MemoryKind:
    """The memory layers. Plain strings so they serialize transparently to JSONL."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

    ALL = (EPISODIC, SEMANTIC, PROCEDURAL)

    @classmethod
    def valid(cls, kind: str) -> str:
        return kind if kind in cls.ALL else cls.SEMANTIC


def new_id() -> str:
    """A sortable, unique id: timestamp + millisecond tiebreaker."""
    return time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class Memory:
    """A single unit of memory in any layer.

    The record is deliberately a superset that fits all three layers; unused fields
    stay at their defaults. `embedding` is a cached semantic vector (None until
    embedded, or when no embedder is available). `links` are ids of related memories
    — the edges that make the store a graph.
    """

    id: str
    text: str
    kind: str = MemoryKind.SEMANTIC
    agent: str = "root"                     # namespace: which AG owns this memory
    tags: List[str] = field(default_factory=list)
    source: str = "manual"                  # manual | auto | reflect | ingest | inherit
    importance: float = 0.5                 # 0..1 — drives ranking and retention
    use_count: int = 0                      # bumped each time recall surfaces it
    created: str = field(default_factory=now_iso)
    last_used: str = ""                     # iso timestamp of last recall
    embedding: Optional[List[float]] = None  # cached vector; absent => keyword-only
    links: List[str] = field(default_factory=list)  # ids of related memories (graph edges)
    meta: Dict[str, Any] = field(default_factory=dict)  # layer-specific extras

    def to_dict(self) -> dict:
        return {
            "id": self.id, "text": self.text, "kind": self.kind, "agent": self.agent,
            "tags": self.tags, "source": self.source, "importance": self.importance,
            "use_count": self.use_count, "created": self.created,
            "last_used": self.last_used, "embedding": self.embedding,
            "links": self.links, "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        """Tolerant load: unknown/missing fields fall back to sane defaults so the
        record format can evolve without invalidating stored memory."""
        return cls(
            id=str(d.get("id") or new_id()),
            text=str(d.get("text", "")),
            kind=MemoryKind.valid(str(d.get("kind", MemoryKind.SEMANTIC))),
            agent=str(d.get("agent", "root")),
            tags=[str(t) for t in (d.get("tags") or [])],
            source=str(d.get("source", "manual")),
            importance=_clamp01(d.get("importance", 0.5)),
            use_count=int(d.get("use_count", 0) or 0),
            created=str(d.get("created") or now_iso()),
            last_used=str(d.get("last_used", "") or ""),
            embedding=_as_vec(d.get("embedding")),
            links=[str(x) for x in (d.get("links") or [])],
            meta=dict(d.get("meta") or {}),
        )


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.5
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def _as_vec(x: Any) -> Optional[List[float]]:
    if not x:
        return None
    try:
        return [float(v) for v in x]
    except (TypeError, ValueError):
        return None
