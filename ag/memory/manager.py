"""MemoryManager — the object AG actually uses.

It composes the pieces into one coherent memory system for a single agent namespace:

    store (where)  +  embedder (meaning)  +  recall blend  +  graph links  +  lineage

Recall is a weighted blend, not a single signal — semantic similarity, lexical
overlap, recency, and importance — then expanded one hop along memory links, so a
strong hit pulls in what it is connected to. Every recall bumps `use_count`/`last_used`,
which feeds retention: memory that proves useful survives; noise ages out.

Lineage: an agent recalls from its own namespace, then its parents', then the shared
tier. This is heredity for a fleet — a spawned AG is born knowing what its lineage
knows, keeps its own private experience, and can `promote` a genuine discovery upward
for siblings to inherit.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .embed import Embedder, cosine, get_embedder
from .store import JsonlStore, MemoryStore
from .types import Memory, MemoryKind, new_id, now_iso

SHARED_AGENT = "shared"  # the tier every agent inherits from (fleet-wide knowledge)

_DEFAULT_WEIGHTS = {"semantic": 0.55, "keyword": 0.2, "recency": 0.15, "importance": 0.1}
_DEFAULT_CAPS = {"episodic": 2000, "semantic": 1000, "procedural": 500}

_WORD_MIN = 2


class MemoryManager:
    def __init__(self, cfg=None, *, agent: str = "root",
                 parents: Sequence[str] = (), store: Optional[MemoryStore] = None,
                 embedder: Optional[Embedder] = None, base_dir=None):
        from ..config import STATE_DIR
        self.cfg = cfg
        self.agent = agent or "root"
        # Lineage: own -> parents -> shared. De-duped, order-preserving.
        chain = [self.agent, *[p for p in parents if p], SHARED_AGENT]
        seen: set = set()
        self.lineage = [a for a in chain if not (a in seen or seen.add(a))]
        self.store = store or JsonlStore(base_dir or (STATE_DIR / "memory"))
        self._embedder = embedder
        self.weights = dict(_DEFAULT_WEIGHTS)
        self.weights.update(getattr(cfg, "memory_weights", None) or {})
        self.caps = dict(_DEFAULT_CAPS)
        self.caps.update(getattr(cfg, "memory_caps", None) or {})
        self.halflife_days = float(getattr(cfg, "memory_recency_halflife_days", 30.0) or 30.0)
        self.merge_threshold = float(getattr(cfg, "memory_merge_threshold", 0.92) or 0.92)

    # --- embedder (lazy so construction never needs the network) -----------
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self.cfg)
        return self._embedder

    def _embed(self, text: str) -> Optional[List[float]]:
        try:
            v = self.embedder.embed_one(text)
            return v or None
        except Exception:
            return None

    # --- write -------------------------------------------------------------
    def remember(self, text: str, *, kind: str = MemoryKind.SEMANTIC,
                 tags: Optional[List[str]] = None, importance: float = 0.5,
                 source: str = "manual", links: Optional[List[str]] = None,
                 meta: Optional[dict] = None, agent: Optional[str] = None) -> Optional[Memory]:
        """Store a memory in this agent's namespace. Near-duplicates within the same
        layer are merged (importance/recency refreshed, use_count summed) instead of
        stored twice, so recall isn't drowned by repetition."""
        text = (text or "").strip()
        if not text:
            return None
        agent = agent or self.agent
        kind = MemoryKind.valid(kind)
        emb = self._embed(text)

        existing = self.store.all(agent, [kind])
        dup = self._find_duplicate(text, emb, existing)
        if dup is not None:
            dup.importance = max(dup.importance, _clamp01(importance))
            dup.use_count += 1
            dup.last_used = now_iso()
            for l in (links or []):
                if l not in dup.links:
                    dup.links.append(l)
            if emb and not dup.embedding:
                dup.embedding = emb
            self.store.update(dup)
            return dup

        mem = Memory(
            id=new_id(), text=text, kind=kind, agent=agent,
            tags=[t.strip() for t in (tags or []) if t.strip()],
            source=source, importance=_clamp01(importance),
            embedding=emb, links=list(links or []), meta=dict(meta or {}),
        )
        self.store.add(mem)
        self._enforce_cap(agent, kind)
        return mem

    def record_episode(self, prompt: str, answer: str, *, score: Optional[float] = None,
                       tags: Optional[List[str]] = None, meta: Optional[dict] = None) -> Optional[Memory]:
        """Log one exchange as episodic memory — the raw experience reflection learns
        from. Importance is seeded from the run's score when available."""
        text = f"Q: {(prompt or '').strip()[:400]}\nA: {(answer or '').strip()[:400]}"
        m = dict(meta or {})
        if score is not None:
            m["score"] = score
        imp = 0.4 if score is None else _clamp01(0.2 + 0.06 * float(score))
        return self.remember(text, kind=MemoryKind.EPISODIC, tags=tags,
                             importance=imp, source="auto", meta=m)

    def link(self, a_id: str, b_id: str) -> None:
        """Add an undirected edge between two memories (a graph, not a bag)."""
        for src, dst in ((a_id, b_id), (b_id, a_id)):
            for agent in self.lineage:
                m = self.store.get(agent, src)
                if m is not None:
                    if dst not in m.links:
                        m.links.append(dst)
                        self.store.update(m)
                    break

    def promote(self, mem_id: str, *, to: str = SHARED_AGENT) -> Optional[Memory]:
        """Copy a memory up to a shared/parent namespace so the lineage inherits it."""
        for agent in self.lineage:
            m = self.store.get(agent, mem_id)
            if m is not None:
                return self.remember(m.text, kind=m.kind, tags=m.tags,
                                     importance=max(m.importance, 0.6),
                                     source="inherit", meta=m.meta, agent=to)
        return None

    # --- recall ------------------------------------------------------------
    def recall(self, query: str, *, k: int = 5, kinds: Optional[Iterable[str]] = None,
               span_lineage: bool = True, expand: bool = True) -> List[Memory]:
        """Return up to k memories most relevant to `query`, blended across signals
        and expanded one hop along links. Empty query -> nothing."""
        query = (query or "").strip()
        if not query:
            return []
        qvec = self._embed(query)
        qtok = _tokens(query)
        agents = self.lineage if span_lineage else [self.agent]

        pool: Dict[str, Tuple[float, Memory]] = {}
        for agent in agents:
            for m in self.store.all(agent, kinds):
                if m.id in pool:
                    continue
                pool[m.id] = (self._score(m, qvec, qtok), m)

        ranked = sorted(pool.values(), key=lambda t: t[0], reverse=True)
        top = [m for s, m in ranked if s > 0][:k]

        if expand and top:
            self._expand_links(top, pool, k)

        self._touch(top)
        return top

    def _expand_links(self, top: List[Memory], pool: Dict[str, Tuple[float, Memory]],
                      k: int) -> None:
        have = {m.id for m in top}
        for m in list(top):
            for lid in m.links:
                if lid in have:
                    continue
                linked = pool.get(lid)
                found = linked[1] if linked else self._get_any(lid)
                if found is not None and found.id not in have:
                    top.append(found)
                    have.add(found.id)
                if len(top) >= k + max(2, k // 2):  # bounded expansion
                    return

    def _get_any(self, mem_id: str) -> Optional[Memory]:
        for agent in self.lineage:
            m = self.store.get(agent, mem_id)
            if m is not None:
                return m
        return None

    def _score(self, m: Memory, qvec, qtok: set) -> float:
        w = self.weights
        sem = cosine(qvec, m.embedding) if (qvec and m.embedding) else 0.0
        # Redistribute the semantic weight onto keyword when no vectors are in play,
        # so recall degrades gracefully to lexical rather than scoring everything 0.
        kw = _overlap(qtok, _tokens(m.text + " " + " ".join(m.tags)))
        if sem == 0.0 and (not qvec or m.embedding is None):
            kw_w = w["keyword"] + w["semantic"]
            sem_w = 0.0
        else:
            kw_w, sem_w = w["keyword"], w["semantic"]
        rec = self._recency(m)
        imp = m.importance
        score = sem_w * sem + kw_w * kw + w["recency"] * rec + w["importance"] * imp
        return score

    def _recency(self, m: Memory) -> float:
        ts = m.last_used or m.created
        try:
            age_days = (time.time() - time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%S"))) / 86400.0
        except Exception:
            return 0.5
        age_days = max(0.0, age_days)
        return math.exp(-age_days / max(1e-6, self.halflife_days))

    def _touch(self, mems: List[Memory]) -> None:
        for m in mems:
            m.use_count += 1
            m.last_used = now_iso()
            try:
                self.store.update(m)
            except Exception:
                pass

    # --- context block (back-compat surface used by the pipeline) ----------
    def memory_context(self, query: str, *, k: int = 5,
                       kinds: Optional[Iterable[str]] = None) -> str:
        from . import is_self_reference
        hits = self.recall(query, k=k, kinds=kinds)
        # Never inject AG self-descriptions into the executor context — identity is set
        # authoritatively by the system prompt; a recalled self-description would fight it.
        hits = [m for m in hits if not is_self_reference(m.text)]
        return "\n".join(f"- {m.text}" for m in hits)

    # --- maintenance -------------------------------------------------------
    def _find_duplicate(self, text: str, emb, existing: List[Memory]) -> Optional[Memory]:
        low = text.lower()
        for m in existing:
            if m.text.strip().lower() == low:
                return m
        if emb:
            for m in existing:
                if m.embedding and cosine(emb, m.embedding) >= self.merge_threshold:
                    return m
        return None

    def _enforce_cap(self, agent: str, kind: str) -> None:
        cap = int(self.caps.get(kind, 0) or 0)
        if cap <= 0:
            return
        mems = self.store.all(agent, [kind])
        if len(mems) <= cap:
            return
        # Retention = importance + usage + recency. Lowest-value memories age out.
        scored = sorted(mems, key=lambda m: self._retention(m))
        keep = set(m.id for m in scored[len(mems) - cap:])
        self.store.write_all(agent, kind, [m for m in mems if m.id in keep])

    def _retention(self, m: Memory) -> float:
        return m.importance + 0.1 * math.log1p(m.use_count) + 0.5 * self._recency(m)

    def stats(self) -> dict:
        out = {"agent": self.agent, "lineage": self.lineage, "layers": {}}
        for kind in MemoryKind.ALL:
            out["layers"][kind] = len(self.store.all(self.agent, [kind]))
        out["embedder"] = self.embedder.name
        return out


def _clamp01(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.5
    return 0.0 if v < 0 else 1.0 if v > 1 else v


import re as _re

_WORD = _re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an and or but of to in on at for with is are was were be been being this "
    "that it its as by from into be i you he she they we my your our their do does".split()
)


def _tokens(text: str) -> set:
    return {w for w in _WORD.findall((text or "").lower())
            if w not in _STOP and len(w) >= _WORD_MIN}


def _overlap(q: set, mt: set) -> float:
    if not q or not mt:
        return 0.0
    return len(q & mt) / len(q)  # normalized by query size, as before
