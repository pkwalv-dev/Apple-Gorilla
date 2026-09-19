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
  PROCEDURAL memory and linked back to the episodes they came from.

What reflection produces is *inference*, not testimony, so it enters memory at the
INFERENCE prior — below the trust line. A reflected fact is a hypothesis AG carries
until the user (or a second independent origin) confirms it, which is exactly what
keeps a plausible-sounding generalization from hardening into a believed fact about
someone's life.

Provenance is grounded, not guessed: each item is linked to the episodes it was
actually drawn from — the model is asked to cite them, and when it does not, the link
falls back to the episodes with the strongest lexical overlap. A wrong provenance edge
is worse than none, because it is what a later audit would trust.
"""
from __future__ import annotations

import re
from typing import List, Optional

from .manager import MemoryManager
from .types import Memory, MemoryKind, Origin, combine_confidence

_REFLECT_FLAG = "reflected"

_FACT_CONFIDENCE = None       # None => use the INFERENCE prior from types
_PROC_CONFIDENCE = 0.45       # procedures are judged by whether they worked, not by fiat


def consolidate(mgr: MemoryManager) -> dict:
    """Model-free maintenance: fold duplicate semantic facts together, enforce layer
    caps. Folding is *corroboration-aware* — two records of the same claim raise belief
    only when they came from different origins, so a duplicate that is merely the same
    source recorded twice is dropped without inflating confidence. Never raises."""
    summary = {"merged": 0, "corroborated": 0}
    try:
        from .embed import cosine
        sem = mgr.store.all(mgr.agent, [MemoryKind.SEMANTIC])
        kept: List[Memory] = []
        for m in sem:
            dup = None
            for k in kept:
                if m.text.strip().lower() == k.text.strip().lower():
                    dup = k
                    break
                if (m.embedding and k.embedding
                        and cosine(m.embedding, k.embedding) >= mgr.merge_threshold):
                    dup = k
                    break
            if dup is None:
                kept.append(m)
                continue
            summary["merged"] += 1
            dup.importance = max(dup.importance, m.importance)
            dup.use_count += m.use_count
            for e in m.evidence:
                if dup.attest(str(e.get("origin", Origin.UNKNOWN))):
                    dup.confidence = combine_confidence(dup.confidence, m.confidence)
                    summary["corroborated"] += 1
            for lid in m.links:
                if lid not in dup.links:
                    dup.links.append(lid)
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
    consolidate() and returns quietly. Everything learned here is stored as inference —
    low-confidence by construction — and linked back to the episodes that support it.
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

        for item in (data.get("facts") or [])[:8]:
            text, cites, vol = _unpack(item, "fact")
            text = str(text or "").strip()
            if not text:
                continue
            m = mgr.remember(text, kind=MemoryKind.SEMANTIC, source="reflect",
                             origin=Origin.INFERENCE, confidence=_FACT_CONFIDENCE,
                             importance=0.6, volatile=vol,
                             links=_provenance(text, cites, episodes))
            if m:
                result["facts"] += 1
        for item in (data.get("procedures") or [])[:5]:
            proc, cites, _vol = _unpack(item, "procedure")
            text = _fmt_procedure(proc)
            if not text:
                continue
            # A procedure is advice about method, not a claim about the world, so it
            # never goes stale on a clock — only evidence retires it.
            m = mgr.remember(text, kind=MemoryKind.PROCEDURAL, source="reflect",
                             origin=Origin.INFERENCE, confidence=_PROC_CONFIDENCE,
                             importance=0.7, volatile=False,
                             links=_provenance(text, cites, episodes))
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
                 f"{result['facts']} facts, {result['procedures']} procedures "
                 f"(unconfirmed until corroborated)",
                 level="tool")
    except Exception as e:
        result["error"] = str(e)
    return result


def _unpack(item, field: str):
    """Accept either a bare value or {"<field>": ..., "from": [...], "volatile": ...}.

    Tolerant on purpose: a model that cites its sources gets grounded provenance, and
    one that answers in the older bare-string form still works rather than losing the
    learning entirely. Returns (payload, cited episode indices, volatile-or-None)."""
    if isinstance(item, dict) and ("from" in item or "episodes" in item
                                   or "volatile" in item or field in item):
        payload = item.get(field, item.get("text", item))
        cites = item.get("from", item.get("episodes")) or []
        vol = item.get("volatile")
        if isinstance(payload, dict) and field == "procedure":
            payload = {k: v for k, v in payload.items()
                       if k not in ("from", "episodes", "volatile")}
        return (payload, [c for c in cites if isinstance(c, int)],
                None if vol is None else bool(vol))
    return item, [], None


def _provenance(text: str, cites: List[int], episodes: List[Memory]) -> List[str]:
    """The episodes this knowledge actually rests on.

    Cited indices win. Otherwise fall back to the best lexical matches — an approximate
    edge to the right episodes beats a precise edge to the wrong ones. If nothing
    matches, record no provenance at all rather than a fabricated trail."""
    ids = [episodes[i].id for i in cites if 0 <= i < len(episodes)]
    if ids:
        return ids[:3]
    q = _words(text)
    if not q:
        return []
    scored = [(len(q & _words(e.text)) / len(q), e.id) for e in episodes]
    return [eid for s, eid in sorted(scored, reverse=True) if s >= 0.2][:2]


_W = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set:
    return {w for w in _W.findall((text or "").lower()) if len(w) > 2}


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
