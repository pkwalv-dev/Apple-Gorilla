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
from .types import (CONFIDENCE_PRIORS, Memory, MemoryKind, Origin, Subject,
                    combine_confidence, effective_confidence, evidence_key,
                    guess_subject, is_identity_claim, new_id, now_iso, prior_for)

MEMORY_DIR = STATE_DIR / "memory"

__all__ = [
    "Memory", "MemoryKind", "MemoryManager", "MemoryStore", "JsonlStore",
    "Embedder", "HashingEmbedder", "OllamaEmbedder", "cosine", "get_embedder",
    "get_manager", "reflect", "consolidate", "remember", "recall",
    "memory_context", "context", "all_memories", "forget", "clear", "MEMORY_DIR",
    "Origin", "Subject", "CONFIDENCE_PRIORS", "prior_for", "combine_confidence",
    "effective_confidence", "evidence_key", "guess_subject", "is_identity_claim",
    "is_self_reference", "verify", "disputed", "needs_verification",
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
             importance: float = 0.5, source: str = "manual",
             origin: Optional[str] = None, confidence: Optional[float] = None,
             volatile: Optional[bool] = None, asserter: str = "",
             subject: Optional[str] = None) -> Optional[Memory]:
    """Store a durable fact in root memory. Signature is a superset of the old one so
    existing callers (`memory.remember(text, max_memories=...)`) keep working.

    `origin` says what kind of evidence the claim rests on and sets its starting
    confidence; it defaults from `source`, so callers that do not care about veracity
    still get a sensible prior rather than blind trust."""
    return _root().remember(text, tags=tags, kind=kind, importance=importance,
                            source=source, origin=origin, confidence=confidence,
                            volatile=volatile, asserter=asserter, subject=subject)


def is_self_reference(text: str) -> bool:
    """True if the text is about AG rather than about the world or the user.

    This is now a *classification*, not a veto: the write gate in MemoryManager decides
    what may be stored about AG and by whom, and recall drops only identity claims. The
    function is kept because one caller still needs the blunt version — LoRA dataset
    building (`ag.lora`), where ANY self-description in a training pair would be baked
    into weights and then argue with the system prompt forever.
    """
    return guess_subject(text) == Subject.SELF


def recall(query: str, *, k: int = 5,
           min_confidence: Optional[float] = None) -> List[Memory]:
    return _root().recall(query, k=k, min_confidence=min_confidence)


def memory_context(query: str, *, k: int = 5) -> str:
    return _root().memory_context(query, k=k)


def context(query: str, *, k: int = 5, agent: str = "root") -> dict:
    """Recall split into what is established and what is merely reported, so a caller
    can inject the two under different headings instead of asserting both as fact."""
    return get_manager(agent).context(query, k=k)


def verify(mem_id: str, *, by: str = Origin.USER, confirmed: bool = True,
           agent: str = "root") -> Optional[Memory]:
    """Confirm or reject a belief — resets staleness, settles a dispute, or collapses
    a claim that turned out to be wrong."""
    return get_manager(agent).verify(mem_id, by=by, confirmed=confirmed)


def disputed(*, agent: str = "root") -> List[Memory]:
    """Memories in open contradiction with another — AG deliberately does not pick a
    winner on its own for facts that cannot change."""
    return get_manager(agent).disputed()


def needs_verification(*, k: int = 10, agent: str = "root") -> List[Memory]:
    """Beliefs that have gone stale and still matter: the re-check queue."""
    return get_manager(agent).needs_verification(k=k)


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
