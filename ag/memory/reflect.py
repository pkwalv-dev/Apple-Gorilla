"""Reflection — the step that turns remembering into learning.

Storing raw episodes is not learning; learning is when experience *changes future
behavior*. Reflection is the process that promotes knowledge up the layers:

    episodic (what happened)  --distill-->  semantic (what's true)
                              --mine------>  procedural (what works)

Two passes, both best-effort and non-raising:

- consolidate() — no model required. Merges near-duplicate semantic memories and
  ages out low-value episodic memory. Cheap housekeeping that keeps recall sharp and
  runs anywhere.
- reflect(client, cfg) — uses the current model to read recent episodes and emit
  generalized facts and reusable procedures, which are stored as SEMANTIC and
  PROCEDURAL memory and linked back to the episodes they came from. This is the
  engine the evolutionary loop later feeds on.

Reflection is model-agnostic: whichever backend AG is attached to does the distilling,
and the resulting knowledge is plain data any future model can read.
"""
from __future__ import annotations

from typing import List, Optional

from .manager import MemoryManager
from .types import Memory, MemoryKind

_REFLECT_FLAG = "reflected"


def consolidate(mgr: MemoryManager) -> dict:
    """Model-free maintenance: merge duplicate semantic facts, enforce layer caps.
    Returns a small summary dict. Never raises."""
    summary = {"merged": 0}
    try:
        sem = mgr.store.all(mgr.agent, [MemoryKind.SEMANTIC])
        kept: List[Memory] = []
        for m in sem:
            dup = None
            for k in kept:
                from .embed import cosine
                if m.embedding and k.embedding and cosine(m.embedding, k.embedding) >= mgr.merge_threshold:
                    dup = k
                    break
                if m.text.strip().lower() == k.text.strip().lower():
                    dup = k
                    break
            if dup is None:
                kept.append(m)
            else:
                dup.importance = max(dup.importance, m.importance)
                dup.use_count += m.use_count
                summary["merged"] += 1
        if summary["merged"]:
            mgr.store.write_all(mgr.agent, MemoryKind.SEMANTIC, kept)
        for kind in MemoryKind.ALL:
            mgr._enforce_cap(mgr.agent, kind)
    except Exception as e:  # pragma: no cover - defensive
        summary["error"] = str(e)
    return summary


def reflect(mgr: MemoryManager, client, cfg, *, max_episodes: int = 20,
            emit=None) -> dict:
    """Read recent un-reflected episodes and distill semantic facts + procedures.

    Requires a real model client; on the dry-run stub or any failure it degrades to
    consolidate() and returns quietly. Stored knowledge links back to its source
    episodes, building the graph reflection reasons over next time.
    """
    result = {"facts": 0, "procedures": 0, "episodes": 0}
    try:
        from ..model import DryRunClient
        if client is None or isinstance(client, DryRunClient):
            result["skipped"] = "no reasoning backend"
            result.update(consolidate(mgr))
            return result

        episodes = [m for m in mgr.store.all(mgr.agent, [MemoryKind.EPISODIC])
                    if not m.meta.get(_REFLECT_FLAG)]
        episodes = episodes[-max_episodes:]
        if not episodes:
            result.update(consolidate(mgr))
            return result
        result["episodes"] = len(episodes)

        from .. import prompts
        joined = "\n\n".join(f"[{i}] {e.text}" for i, e in enumerate(episodes))
        res = client.complete(
            system=prompts.MEMORY_REFLECTOR_SYSTEM,
            user=f"# Recent episodes (raw experience)\n{joined}\n",
            cfg=cfg, max_tokens=800,
        )
        from ..pipeline import extract_json
        data = extract_json(res.text) or {}

        ep_ids = [e.id for e in episodes]
        for fact in (data.get("facts") or [])[:8]:
            fact = str(fact).strip()
            if fact:
                m = mgr.remember(fact, kind=MemoryKind.SEMANTIC, source="reflect",
                                importance=0.6, links=ep_ids[-3:])
                if m:
                    result["facts"] += 1
        for proc in (data.get("procedures") or [])[:5]:
            text = _fmt_procedure(proc)
            if text:
                m = mgr.remember(text, kind=MemoryKind.PROCEDURAL, source="reflect",
                                importance=0.7, links=ep_ids[-3:])
                if m:
                    result["procedures"] += 1

        # Mark episodes reflected so we don't re-distill them next time.
        for e in episodes:
            e.meta[_REFLECT_FLAG] = True
            try:
                mgr.store.update(e)
            except Exception:
                pass
        consolidate(mgr)
        if emit:
            emit("memory", f"reflected {result['episodes']} episodes -> "
                 f"{result['facts']} facts, {result['procedures']} procedures",
                 level="tool")
    except Exception as e:
        result["error"] = str(e)
    return result


def _fmt_procedure(proc) -> str:
    if isinstance(proc, str):
        return proc.strip()
    if isinstance(proc, dict):
        name = str(proc.get("name", "")).strip()
        when = str(proc.get("when", "")).strip()
        steps = proc.get("steps") or []
        if isinstance(steps, list):
            steps = "; ".join(str(s).strip() for s in steps if str(s).strip())
        parts = [p for p in (name, (f"when {when}" if when else ""), str(steps)) if p]
        return " — ".join(parts)
    return ""
