"""AG's memory system — a layered, model-agnostic, graph-linked learner.

Public surface:

- Architecture:  Memory, MemoryKind, MemoryManager, MemoryStore, JsonlStore, Embedder
- Per-agent:     get_manager(agent, parents=...)   -> a namespaced MemoryManager
- Learning:      reflect(client, cfg), consolidate()
- Back-compat:   remember / recall / memory_context / all_memories / forget / clear
                 operate on the default "root" namespace so existing callers
                 (pipeline, reason loop, CLI) keep working unchanged.

The whole system is decoupled from any specific LLM (embeddings + reflection run on
whatever backend AG is attached to, defaulting to local Ollama) and from any specific
storage engine (everything goes through MemoryStore). It is portable by construction:
memory is just data under state/memory/ that travels with the agent.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from ..config import STATE_DIR
from .embed import Embedder, HashingEmbedder, OllamaEmbedder, cosine, get_embedder
from .manager import SHARED_AGENT, MemoryManager
from .reflect import consolidate as _consolidate
from .reflect import reflect as _reflect
from .store import JsonlStore, MemoryStore
from .types import Memory, MemoryKind, new_id, now_iso

MEMORY_DIR = STATE_DIR / "memory"

__all__ = [
    "Memory", "MemoryKind", "MemoryManager", "MemoryStore", "JsonlStore",
    "Embedder", "HashingEmbedder", "OllamaEmbedder", "cosine", "get_embedder",
    "get_manager", "reflect", "consolidate", "remember", "recall",
    "memory_context", "all_memories", "forget", "clear", "MEMORY_DIR",
]

# --- namespaced managers (cached per agent) --------------------------------
_MANAGERS: dict = {}


def get_manager(agent: str = "root", *, parents: Sequence[str] = (),
                cfg=None) -> MemoryManager:
    """Return the MemoryManager for an agent namespace, cached. Pass `parents` to set
    the inheritance lineage (a spawned AG inherits its parent's + shared memory)."""
    key = (agent, tuple(parents))
    mgr = _MANAGERS.get(key)
    if mgr is None:
        if cfg is None:
            try:
                from ..config import Config
                cfg = Config.load()
            except Exception:
                cfg = None
        mgr = MemoryManager(cfg, agent=agent, parents=parents)
        _MANAGERS[key] = mgr
    return mgr


def _root() -> MemoryManager:
    return get_manager("root")


def reset() -> None:
    """Drop cached managers (used by tests that redirect the store)."""
    _MANAGERS.clear()


# --- learning ---------------------------------------------------------------
def reflect(client, cfg, *, agent: str = "root", emit=None) -> dict:
    return _reflect(get_manager(agent, cfg=cfg), client, cfg, emit=emit)


def consolidate(*, agent: str = "root") -> dict:
    return _consolidate(get_manager(agent))


# --- back-compat surface (default "root" namespace) ------------------------
def remember(text: str, tags: Optional[List[str]] = None, *,
             max_memories: int = 200, kind: str = MemoryKind.SEMANTIC,
             importance: float = 0.5, source: str = "manual") -> Optional[Memory]:
    """Store a durable fact in root memory. Signature is a superset of the old one so
    existing callers (`memory.remember(text, max_memories=...)`) keep working."""
    return _root().remember(text, tags=tags, kind=kind, importance=importance,
                            source=source)


_SELF_REF_MARKERS = (
    # identity questions
    "your name", "who are you", "what are you", "about yourself",
    "what is apple-gorilla", "who is apple-gorilla", "know about yourself",
    "gotten smarter", "gotten any smarter", "abliterated model", "describe yourself",
    "your capabilities", "do you have persistent memory", "are you sentient",
    "are you conscious",
    # (false) self-descriptions AG must not learn or recall about itself
    "protocol layer", "not an autonomous agent",
    "no persistent memory", "do not have persistent memory",
    "does not have persistent memory", "without persistent memory",
    "no built-in file access", "do not have built-in file access",
    "no direct file access", "without direct file access",
    "self-iteration", "limited to text-based", "cannot fulfill this request",
)


def is_self_reference(text: str) -> bool:
    """True if the text is about AG's own identity/nature/capabilities. Such content is
    the system prompt's domain (prompts.AG_IDENTITY), not memory's: storing or recalling
    it lets a stale/wrong self-description override the authoritative identity. So memory
    neither captures nor recalls it."""
    t = (text or "").lower()
    return any(mk in t for mk in _SELF_REF_MARKERS)


def recall(query: str, *, k: int = 5) -> List[Memory]:
    return _root().recall(query, k=k)


def memory_context(query: str, *, k: int = 5) -> str:
    return _root().memory_context(query, k=k)


def all_memories(*, agent: str = "root") -> List[Memory]:
    return get_manager(agent).store.all(agent)


def forget(mem_id: str, *, agent: str = "root") -> bool:
    return get_manager(agent).store.delete(agent, mem_id)


def clear(*, agent: str = "root") -> int:
    mgr = get_manager(agent)
    n = len(mgr.store.all(agent))
    for kind in MemoryKind.ALL:
        mgr.store.write_all(agent, kind, [])
    return n
