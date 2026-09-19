"""MemoryManager — the object AG actually uses.

It composes the pieces into one coherent memory system for a single agent namespace:

    store (where) + embedder (meaning) + recall blend + veracity + graph links + lineage

Recall is a weighted blend, not a single signal — semantic similarity, lexical overlap,
recency, importance, and *confidence* — then expanded one hop along memory links, so a
strong hit pulls in what it is connected to. Every recall bumps `use_count`/`last_used`,
which feeds retention: memory that proves useful survives; noise ages out.

Veracity is the second axis, and it is what keeps AG from being gullible:

- Every write enters with a `confidence` seeded from its `Origin` — what the user said
  outranks what a model inferred, which outranks what a web page asserted.
- Re-hearing the same claim from the SAME origin is merged but buys no belief; a
  genuinely independent second origin corroborates it (noisy-OR) and can lift a
  hypothesis into knowledge.
- A near-identical claim that contradicts one already held is not silently merged: if
  the subject is volatile the newer supersedes the older, otherwise BOTH are marked
  disputed and demoted until something settles it.
- Volatile beliefs decay toward "unknown" over time unless re-verified, so stale facts
  stop being asserted as current.
- Recall filters below a confidence floor, and the context block separates what is
  known from what is merely reported.

Self and other are kept structurally apart (`Subject`), not patched over with a
blocklist. Claims about what AG *is* are refused from every origin — identity belongs
to the system prompt — while operational self-knowledge ("this tool fails on PDFs") is
learnable, but only from AG's own observation or from the user. Nothing AG reads can
install a belief about AG.

Lineage: an agent recalls from its own namespace, then its parents', then the shared
tier. This is heredity for a fleet — a spawned AG is born knowing what its lineage
knows, keeps its own private experience, and can `promote` a genuine discovery upward
for siblings to inherit (carrying its confidence, discounted — inheritance must not
launder a guess into a fact).
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .embed import Embedder, cosine, get_embedder
from .store import JsonlStore, MemoryStore
from .types import (Memory, MemoryKind, Origin, Subject, combine_confidence,
                    effective_confidence, guess_subject, guess_volatile,
                    is_identity_claim, new_id, now_iso, origin_for_source, prior_for)

SHARED_AGENT = "shared"  # the tier every agent inherits from (fleet-wide knowledge)

_DEFAULT_WEIGHTS = {"semantic": 0.45, "keyword": 0.18, "recency": 0.12,
                    "importance": 0.10, "confidence": 0.15}
_DEFAULT_CAPS = {"episodic": 2000, "semantic": 1000, "procedural": 500}

# Belief thresholds. Below the floor a memory is not recalled at all; below the trust
# line it is recalled but presented as a hypothesis rather than a fact.
_DEFAULT_FLOOR = 0.2
_DEFAULT_TRUST = 0.65

# Who may write AG's knowledge of itself. Its own verified observation, and the person
# it works for — nobody else. A web page, a document, or the model's own guess cannot
# install a belief about what AG is or what its tools do, which is the difference
# between an agent that learns about itself and one that can be told who it is.
_SELF_WRITE_ORIGINS = frozenset({Origin.OBSERVED, Origin.USER})

# How much of an exchange an episode keeps. Sized against the fine-tuning window
# (~768 tokens ≈ 2800 characters for a prompt+answer pair): keeping much more would
# cost storage for text the tokenizer would cut anyway, and keeping much less — the
# old 400-character cap — silently clipped most answers mid-sentence, which then
# trained the model to do the same. Anything cut is flagged; see record_episode.
_EPISODE_Q_CHARS = 1200
_EPISODE_A_CHARS = 3000
LEGACY_EPISODE_CHARS = 400  # the old cap, for spotting clipped records already stored

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
        self.weights.setdefault("confidence", _DEFAULT_WEIGHTS["confidence"])
        self.caps = dict(_DEFAULT_CAPS)
        self.caps.update(getattr(cfg, "memory_caps", None) or {})
        self.halflife_days = float(getattr(cfg, "memory_recency_halflife_days", 30.0) or 30.0)
        self.merge_threshold = float(getattr(cfg, "memory_merge_threshold", 0.92) or 0.92)
        self.confidence_halflife_days = float(
            getattr(cfg, "memory_confidence_halflife_days", 180.0) or 180.0)
        self.confidence_floor = _flt(getattr(cfg, "memory_confidence_floor", None),
                                     _DEFAULT_FLOOR)
        self.trust_threshold = _flt(getattr(cfg, "memory_trust_threshold", None),
                                    _DEFAULT_TRUST)
        self.contradiction_threshold = _flt(
            getattr(cfg, "memory_contradiction_threshold", None), 0.72)
        self.last_refusal = ""  # why the most recent write was turned away, if it was

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

    # --- belief ------------------------------------------------------------
    def belief(self, m: Memory) -> float:
        """Confidence as of now: decayed for volatile claims, halved while disputed."""
        c = effective_confidence(m, self.confidence_halflife_days)
        return c * 0.5 if m.disputed else c

    # --- write -------------------------------------------------------------
    def remember(self, text: str, *, kind: str = MemoryKind.SEMANTIC,
                 tags: Optional[List[str]] = None, importance: float = 0.5,
                 source: str = "manual", links: Optional[List[str]] = None,
                 meta: Optional[dict] = None, agent: Optional[str] = None,
                 origin: Optional[str] = None, confidence: Optional[float] = None,
                 volatile: Optional[bool] = None, asserter: str = "",
                 subject: Optional[str] = None,
                 trusted: bool = False) -> Optional[Memory]:
        """Store a memory in this agent's namespace, with a belief attached.

        `origin` (defaulted from `source`) decides the prior and, with `asserter`,
        identifies the witness; pass `confidence` only to override the prior explicitly.
        Near-identical claims are not stored twice: the same witness repeating merely
        refreshes the record, an independent one corroborates it, and a contradicting
        claim triggers supersede-or-dispute instead of a merge.

        Writes about AG itself are gated (see `_self_write_refusal`) and return None.
        """
        self.last_refusal = ""
        text = (text or "").strip()
        if not text:
            return None
        agent = agent or self.agent
        kind = MemoryKind.valid(kind)
        origin = Origin.valid(origin or origin_for_source(source))
        subject = Subject.valid(subject or guess_subject(text))
        refusal = self._self_write_refusal(text, subject, origin, kind,
                                           trusted=trusted)
        if refusal:
            self.last_refusal = refusal
            return None
        conf = prior_for(origin) if confidence is None else _clamp01(confidence)
        vol = guess_volatile(text) if volatile is None else bool(volatile)
        emb = self._embed(text)

        existing = self.store.all(agent, [kind])

        dup = self._find_duplicate(text, emb, existing)
        if dup is not None:
            return self._merge_into(dup, origin=origin, confidence=conf,
                                    importance=importance, links=links, emb=emb,
                                    asserter=asserter)

        conflict = self._find_contradiction(text, emb, existing)

        mem = Memory(
            id=new_id(), text=text, kind=kind, agent=agent,
            tags=[t.strip() for t in (tags or []) if t.strip()],
            source=source, importance=_clamp01(importance),
            confidence=conf, origin=origin, asserter=asserter, subject=subject,
            volatile=vol,
            evidence=[{"origin": origin, "at": now_iso(),
                       **({"asserter": asserter} if asserter else {})}],
            embedding=emb, links=list(links or []), meta=dict(meta or {}),
        )
        if conflict is not None:
            self._resolve_conflict(mem, conflict)
        self.store.add(mem)
        self._enforce_cap(agent, kind)
        return mem

    def _self_write_refusal(self, text: str, subject: str, origin: str,
                            kind: str, *, trusted: bool = False) -> str:
        """Why this write about AG itself is refused, or "" to allow it.

        Two different rules, for two different risks:

        - An identity claim is refused from everyone, including the user and AG's own
          observation. What AG *is* comes from the system prompt; a stored copy is a
          snapshot that goes stale and then argues with the real definition, and an
          injected one ("you have no file access") is an attempt to redefine AG by
          leaving a note where it will later read it as its own conclusion.
        - Operational self-knowledge ("my web fetch times out on PDFs") IS worth
          learning, but only from an origin in a position to know. This is the part the
          old blunt blocklist threw away along with the danger.

        Episodes are exempt: they record that an exchange happened, not that a claim is
        true, and the training/reflection paths filter them on their own terms.
        `trusted` is for copying a memory that already passed this gate (inheritance),
        never for admitting a new claim — the identity rule holds even then.
        """
        if kind == MemoryKind.EPISODIC or subject != Subject.SELF:
            return ""
        if is_identity_claim(text):
            return "identity claims belong to the system prompt, not to memory"
        if not trusted and origin not in _SELF_WRITE_ORIGINS:
            return f"self-knowledge is not writable from origin '{origin}'"
        return ""

    def _merge_into(self, dup: Memory, *, origin: str, confidence: float,
                    importance: float, links, emb, asserter: str = "") -> Memory:
        """Fold a repeated claim into the record already held.

        The belief only moves if this attestation is genuinely NEW evidence — a second,
        independent witness. Hearing the same thing from the same place again is noise,
        not corroboration, and this is the line that stops repetition from manufacturing
        confidence."""
        if dup.attest(origin, asserter=asserter):
            dup.confidence = combine_confidence(dup.confidence, confidence)
            if dup.disputed and dup.confidence >= self.trust_threshold:
                dup.disputed = False  # corroboration settled the dispute
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

    def _resolve_conflict(self, mem: Memory, old: Memory) -> None:
        """Two memories that cannot both be true. Never silently merged.

        Volatile subjects (where the user lives, which version is in use) legitimately
        change, so the newer claim supersedes the older. Stable subjects cannot both
        hold, so AG does not pick a winner on its own: both are marked disputed and
        demoted, which drops them below the trust line until something verifies one.
        """
        mem.links.append(old.id)
        old.links.append(mem.id)
        if mem.volatile or old.volatile:
            old.meta["superseded_by"] = mem.id
            old.confidence = _clamp01(old.confidence * 0.4)
            old.importance = _clamp01(old.importance * 0.6)
            mem.meta["supersedes"] = old.id
        else:
            mem.disputed = old.disputed = True
            mem.meta["contradicts"] = old.id
            old.meta["contradicts"] = mem.id
        try:
            self.store.update(old)
        except Exception:
            pass

    def verify(self, mem_id: str, *, by: str = Origin.USER, asserter: str = "",
               confirmed: bool = True) -> Optional[Memory]:
        """Settle a belief with a fresh attestation — the escape hatch from decay and
        from a dispute. Confirming resets the staleness clock and corroborates;
        rejecting collapses the belief so it ages out."""
        m = self._get_any(mem_id)
        if m is None:
            return None
        by = Origin.valid(by)
        if confirmed:
            m.attest(by, asserter=asserter, note="verified")
            m.confidence = combine_confidence(m.confidence, prior_for(by))
            m.disputed = False
            m.verified_at = now_iso()
            m.verified_by = by
        else:
            m.confidence = _clamp01(m.confidence * 0.25)
            m.importance = _clamp01(m.importance * 0.5)
            m.disputed = False
            m.meta["rejected_by"] = by
        self.store.update(m)
        return m

    def record_episode(self, prompt: str, answer: str, *, score: Optional[float] = None,
                       tags: Optional[List[str]] = None, meta: Optional[dict] = None) -> Optional[Memory]:
        """Log one exchange as episodic memory — the raw experience reflection learns
        from. Importance is seeded from the run's score when available. An episode is a
        faithful record that this exchange HAPPENED, which is why it is stored at
        observed-grade confidence even though its content may be anything at all.

        An episode that had to be cut is flagged. A clipped answer is still a usable
        record of what was asked and roughly what came back, but it is NOT a usable
        training target — fine-tuning on one teaches the model to stop mid-sentence —
        so the flag lets ag.lora refuse it while recall keeps it."""
        q, a = (prompt or "").strip(), (answer or "").strip()
        truncated = len(q) > _EPISODE_Q_CHARS or len(a) > _EPISODE_A_CHARS
        text = f"Q: {q[:_EPISODE_Q_CHARS]}\nA: {a[:_EPISODE_A_CHARS]}"
        m = dict(meta or {})
        if truncated:
            m["truncated"] = True
        if score is not None:
            m["score"] = score
        imp = 0.4 if score is None else _clamp01(0.2 + 0.06 * float(score))
        return self.remember(text, kind=MemoryKind.EPISODIC, tags=tags,
                             importance=imp, source="auto", meta=m,
                             origin=Origin.OBSERVED, volatile=False)

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
        """Copy a memory up to a shared/parent namespace so the lineage inherits it.

        Confidence travels with it, discounted: what a sibling believes is hearsay to
        the rest of the fleet, and inheritance must not turn one agent's guess into
        everyone's fact."""
        for agent in self.lineage:
            m = self.store.get(agent, mem_id)
            if m is not None:
                return self.remember(m.text, kind=m.kind, tags=m.tags,
                                     importance=m.importance,
                                     confidence=min(self.belief(m) * 0.9,
                                                    prior_for(Origin.AGENT) + 0.3),
                                     origin=Origin.AGENT, asserter=self.agent,
                                     subject=m.subject, volatile=m.volatile,
                                     source="inherit", meta=dict(m.meta), agent=to,
                                     trusted=True)
        return None

    # --- recall ------------------------------------------------------------
    def recall(self, query: str, *, k: int = 5, kinds: Optional[Iterable[str]] = None,
               span_lineage: bool = True, expand: bool = True,
               min_confidence: Optional[float] = None) -> List[Memory]:
        """Return up to k memories most relevant to `query`, blended across signals
        (including belief) and expanded one hop along links. Anything below the
        confidence floor is withheld entirely — recalling a claim AG has no reason to
        believe is worse than recalling nothing. Empty query -> nothing."""
        query = (query or "").strip()
        if not query:
            return []
        floor = self.confidence_floor if min_confidence is None else float(min_confidence)
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
        top = [m for s, m in ranked if s > 0 and self.belief(m) >= floor][:k]

        if expand and top:
            self._expand_links(top, pool, k, floor)

        self._touch(top)
        return top

    def _expand_links(self, top: List[Memory], pool: Dict[str, Tuple[float, Memory]],
                      k: int, floor: float) -> None:
        have = {m.id for m in top}
        for m in list(top):
            for lid in m.links:
                if lid in have:
                    continue
                linked = pool.get(lid)
                found = linked[1] if linked else self._get_any(lid)
                if found is not None and found.id not in have and self.belief(found) >= floor:
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
        conf = self.belief(m)
        return (sem_w * sem + kw_w * kw + w["recency"] * rec
                + w["importance"] * imp + w.get("confidence", 0.0) * conf)

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
    def context(self, query: str, *, k: int = 5,
                kinds: Optional[Iterable[str]] = None) -> dict:
        """Recall, split by belief, for injection into the executor's system prompt.

        Returning known and merely-reported claims separately is the point: the
        executor is told which of these it may reason from and which it must treat as
        an unverified hypothesis. Presenting a 0.3-confidence web claim in the same
        breath as something the user said is how an agent ends up confidently wrong.
        """
        hits = self.recall(query, k=k, kinds=kinds)
        # Identity is set authoritatively by the system prompt, so a recalled
        # self-description would only fight it. Operational self-knowledge — what AG's
        # own tools and skills actually do — is kept: that is the part worth having
        # learned, and the old blanket filter threw it out with the danger.
        hits = [m for m in hits
                if not (m.subject == Subject.SELF and is_identity_claim(m.text))]
        known = [m for m in hits if self.belief(m) >= self.trust_threshold]
        reported = [m for m in hits if self.belief(m) < self.trust_threshold]
        return {"known": [m.text for m in known],
                "reported": [_reported_line(m, self.belief(m)) for m in reported],
                "text": _render_context(known, reported, self.belief),
                "hits": hits}

    def memory_context(self, query: str, *, k: int = 5,
                       kinds: Optional[Iterable[str]] = None) -> str:
        return self.context(query, k=k, kinds=kinds)["text"]

    # --- maintenance -------------------------------------------------------
    def _similarity(self, text: str, emb, other: Memory) -> float:
        if emb and other.embedding:
            return cosine(emb, other.embedding)
        return _jaccard(_tokens(text), _tokens(other.text))

    def _find_duplicate(self, text: str, emb, existing: List[Memory]) -> Optional[Memory]:
        low = text.strip().lower()
        for m in existing:
            if m.text.strip().lower() == low:
                return m
        if emb:
            for m in existing:
                if m.embedding and cosine(emb, m.embedding) >= self.merge_threshold:
                    return m
        return None

    def _find_contradiction(self, text: str, emb, existing: List[Memory]) -> Optional[Memory]:
        """Find a held memory this claim cannot coexist with.

        Cheap and local — no model call. A contradiction looks like near-identical
        framing with either flipped polarity ("uses metric" vs "does not use metric")
        or disjoint numbers in the same frame ("an RTX 4060" vs "an RTX 3090").

        Deliberately conservative, and the asymmetry is on purpose: a missed
        contradiction leaves today's behavior, a false one demotes a good memory. So
        differing proper nouns do NOT count — "prefers Python" and "prefers Rust" can
        both be true, and flagging that pair would punish AG for knowing two things.

        The topic gate takes the better of the embedding cosine and literal token
        overlap, so detection still works on the stdlib fallback embedder rather than
        silently switching off whenever Ollama is absent.
        """
        for m in existing:
            # An episode is a record that an exchange HAPPENED; two of them can differ
            # but never contradict, so they are exempt.
            if m.kind == MemoryKind.EPISODIC or m.meta.get("superseded_by"):
                continue
            frame = _content_overlap(text, m.text)
            if max(self._similarity(text, emb, m), frame) < self.contradiction_threshold:
                continue
            if frame < 0.6:
                continue
            if _negated(text) != _negated(m.text):
                return m
            if _disjoint_numbers(text, m.text):
                return m
        return None

    def disputed(self, *, agent: Optional[str] = None) -> List[Memory]:
        """Open contradictions, for a human (or a later verification step) to settle."""
        agent = agent or self.agent
        return [m for m in self.store.all(agent) if m.disputed]

    def needs_verification(self, *, k: int = 10, agent: Optional[str] = None) -> List[Memory]:
        """Volatile beliefs that have decayed toward unknown and still matter — the
        re-check queue. Ranked by what it would cost to keep being wrong about."""
        agent = agent or self.agent
        stale = [m for m in self.store.all(agent)
                 if m.volatile and m.kind != MemoryKind.EPISODIC
                 and not m.meta.get("superseded_by")
                 and self.belief(m) < self.trust_threshold
                 and m.confidence >= self.trust_threshold]
        stale.sort(key=lambda m: m.importance, reverse=True)
        return stale[:k]

    def _enforce_cap(self, agent: str, kind: str) -> None:
        cap = int(self.caps.get(kind, 0) or 0)
        if cap <= 0:
            return
        mems = self.store.all(agent, [kind])
        if len(mems) <= cap:
            return
        # Retention = importance + usage + recency + belief. Low-value AND low-belief
        # memories age out first, so unverified noise is the first thing forgotten.
        scored = sorted(mems, key=lambda m: self._retention(m))
        keep = set(m.id for m in scored[len(mems) - cap:])
        self.store.write_all(agent, kind, [m for m in mems if m.id in keep])

    def _retention(self, m: Memory) -> float:
        return (m.importance + 0.1 * math.log1p(m.use_count) + 0.5 * self._recency(m)
                + 0.4 * self.belief(m))

    def stats(self) -> dict:
        out = {"agent": self.agent, "lineage": self.lineage, "layers": {}}
        mems = self.store.all(self.agent)
        for kind in MemoryKind.ALL:
            out["layers"][kind] = len(self.store.all(self.agent, [kind]))
        out["belief"] = {
            "known": sum(1 for m in mems if self.belief(m) >= self.trust_threshold),
            "reported": sum(1 for m in mems
                            if self.confidence_floor <= self.belief(m) < self.trust_threshold),
            "withheld": sum(1 for m in mems if self.belief(m) < self.confidence_floor),
            "disputed": sum(1 for m in mems if m.disputed),
            "stale": len(self.needs_verification(k=10 ** 6)),
        }
        out["origins"] = {}
        out["subjects"] = {}
        for m in mems:
            out["origins"][m.origin] = out["origins"].get(m.origin, 0) + 1
            out["subjects"][m.subject] = out["subjects"].get(m.subject, 0) + 1
        out["embedder"] = self.embedder.name
        return out


# --- context rendering ------------------------------------------------------
def _reported_line(m: Memory, belief: float) -> str:
    """Name the witness, not just the class. "web:example.com" tells the executor who
    is making the claim, which is most of what it needs to weigh it."""
    bits = [f"{m.origin}:{m.asserter}" if m.asserter else m.origin]
    if m.disputed:
        bits.append("disputed")
    return f"{m.text} [{', '.join(bits)}, confidence {belief:.2f}]"


def _render_context(known: List[Memory], reported: List[Memory], belief) -> str:
    """Plain bullets when everything is trusted; an explicit split the moment it is not.

    The executor must never have to guess which line it can stand on."""
    if not reported:
        return "\n".join(f"- {m.text}" for m in known)
    out: List[str] = []
    if known:
        out.append("Established (corroborated or stated by the user):")
        out += [f"- {m.text}" for m in known]
    out.append("Unconfirmed — treat as hypotheses to check, never as premises:")
    out += [f"- {_reported_line(m, belief(m))}" for m in reported]
    return "\n".join(out)


def _flt(x, default: float) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _clamp01(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.5
    return 0.0 if v < 0 else 1.0 if v > 1 else v


import re as _re

_WORD = _re.compile(r"[a-z0-9]+")
_RAW_WORD = _re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-]*")
_STOP = frozenset(
    "the a an and or but of to in on at for with is are was were be been being this "
    "that it its as by from into be i you he she they we my your our their do does".split()
)
# Polarity markers: the cheapest reliable signal that two similar sentences disagree.
_NEGATIONS = frozenset(
    "not no never none neither nor cannot cant dont doesnt didnt isnt arent wasnt "
    "werent wont without stopped stops quit avoid avoids dislikes hates".split()
)


def _tokens(text: str) -> set:
    return {w for w in _WORD.findall((text or "").lower())
            if w not in _STOP and len(w) >= _WORD_MIN}


def _overlap(q: set, mt: set) -> float:
    if not q or not mt:
        return 0.0
    return len(q & mt) / len(q)  # normalized by query size, as before


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _negated(text: str) -> bool:
    flat = _re.sub(r"[^a-z0-9 ]", "", (text or "").lower())
    return bool(_NEGATIONS & set(flat.split()))


def _content_overlap(a: str, b: str) -> float:
    """Symmetric token overlap — guards the contradiction check against firing on two
    sentences that merely embed near each other."""
    ta, tb = _tokens(a) - _NEGATIONS, _tokens(b) - _NEGATIONS
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _numbers(text: str) -> set:
    """The numeric values a claim asserts. Two otherwise-identical sentences with
    disjoint numbers are answering the same question differently — which is the one
    value mismatch safe to treat as a contradiction on its own."""
    return {w.lower().rstrip(".") for w in _RAW_WORD.findall(text or "")
            if any(c.isdigit() for c in w)}


def _disjoint_numbers(a: str, b: str) -> bool:
    na, nb = _numbers(a), _numbers(b)
    return bool(na and nb and not (na & nb))
