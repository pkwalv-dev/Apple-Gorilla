"""Memory records, the layered taxonomy, and the veracity model.

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

Every record carries TWO independent scalars, because "worth recalling" and "likely
true" are different questions:

- `importance` — utility: how much this matters if true. Drives ranking and retention.
- `confidence` — belief: how likely it is to BE true, seeded from where the claim came
                 from (`Origin`) and raised only by *independent* corroboration.

That split is what lets AG acquire facts about the world without being gullible: a web
page's claim enters at 0.25 and is recalled as a hypothesis, while something the user
stated enters at 0.90 and is recalled as known. Repetition from the same origin buys
nothing; a second, independent origin does.

Two further axes keep self and other apart, because "who said it" and "who it is about"
are different questions:

- `origin`/`asserter` — WHO ATTESTED it. Independence is judged per witness, so two
  different sites corroborate but one site repeating does not.
- `subject`      — WHO IT IS ABOUT (`Subject.SELF` / `USER` / `WORLD` / `AGENT`).
                   Claims about AG's own identity are never stored by anyone, and
                   operational self-knowledge is writable only by origins that could
                   actually know: AG's own observation, or the user.

Records are model-agnostic plain data: an optional cached embedding (so recall is by
meaning, not keywords), `links` to other memories (so the store is a graph), and an
`evidence` trail (so a belief can always be traced to what attested it).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_COUNTER = itertools.count()


class MemoryKind:
    """The memory layers. Plain strings so they serialize transparently to JSONL."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

    ALL = (EPISODIC, SEMANTIC, PROCEDURAL)

    @classmethod
    def valid(cls, kind: str) -> str:
        return kind if kind in cls.ALL else cls.SEMANTIC


class Subject:
    """WHO a memory is about — the axis that separates self from other.

    Kept strictly apart from `Origin` (who *asserted* it), because the two questions
    are independent and conflating them is how an agent gets talked out of its own
    nature. "A web page says AG has no memory" is SELF/WEB; "the user is in Lisbon" is
    USER/USER. Only the first needs guarding.

    The separation is what lets AG learn true things about itself — that a tool it owns
    is unreliable, that a skill it acquired works — without letting anything it reads
    rewrite what it is.
    """

    SELF = "self"     # AG itself: its tools, skills, limits, nature
    USER = "user"     # the person AG is helping
    WORLD = "world"   # everything else
    AGENT = "agent"   # another AG in the fleet

    ALL = (SELF, USER, WORLD, AGENT)

    @classmethod
    def valid(cls, subject: str) -> str:
        return subject if subject in cls.ALL else cls.WORLD


class Origin:
    """How a claim reached AG — the evidence class, not the subsystem that wrote it.

    This is the axis veracity is reasoned on. `source` (manual/auto/reflect/...) says
    which part of AG stored a memory; `origin` says what kind of evidence it rests on.
    They are deliberately separate: the same subsystem can record claims of very
    different trustworthiness.
    """

    USER = "user"                 # the person stated it about themselves
    OBSERVED = "observed"         # AG's own verified result (tool output, passing test)
    DISTILLED = "distilled"       # extracted from the user's own words by a model
    DOCUMENT = "document"         # a file/document the user supplied
    THIRD_PARTY = "third_party"   # someone else, relayed through the user
    INFERENCE = "inference"       # a model's generalization over past episodes
    AGENT = "agent"               # another AG in the fleet
    WEB = "web"                   # untrusted page content
    UNKNOWN = "unknown"

    ALL = (USER, OBSERVED, DISTILLED, DOCUMENT, THIRD_PARTY, INFERENCE, AGENT, WEB,
           UNKNOWN)

    @classmethod
    def valid(cls, origin: str) -> str:
        return origin if origin in cls.ALL else cls.UNKNOWN


# Prior belief a claim earns purely from where it came from, before any corroboration.
# Deliberately conservative: anything a model merely inferred, or anything a web page
# asserted, starts BELOW the trust threshold and is recalled as a hypothesis until a
# second independent origin backs it up.
CONFIDENCE_PRIORS: Dict[str, float] = {
    Origin.USER: 0.90,
    Origin.OBSERVED: 0.85,
    Origin.DISTILLED: 0.60,
    Origin.DOCUMENT: 0.55,
    Origin.THIRD_PARTY: 0.40,
    Origin.UNKNOWN: 0.40,
    Origin.INFERENCE: 0.35,
    Origin.AGENT: 0.35,
    Origin.WEB: 0.25,
}

# Legacy `source` values map onto evidence classes, so every existing call site gets a
# sensible prior without being rewritten.
_SOURCE_ORIGIN = {
    "manual": Origin.USER,
    "auto": Origin.DISTILLED,
    "reflect": Origin.INFERENCE,
    "ingest": Origin.DOCUMENT,
    "inherit": Origin.AGENT,
    "acquire": Origin.OBSERVED,
    "web": Origin.WEB,
}

MAX_CONFIDENCE = 0.98  # nothing here is ever certain; leave room to be wrong


def origin_for_source(source: str) -> str:
    return _SOURCE_ORIGIN.get((source or "").strip().lower(), Origin.UNKNOWN)


def prior_for(origin: str) -> float:
    return CONFIDENCE_PRIORS.get(Origin.valid(origin), CONFIDENCE_PRIORS[Origin.UNKNOWN])


def combine_confidence(a: float, b: float) -> float:
    """Noisy-OR: independent attestations accumulate, none of them alone is decisive.

    Corroboration only ever *raises* belief (doubt is applied by contradiction, not by
    a weak witness), and the result is capped short of certainty. Two web claims reach
    0.44 — still a hypothesis; a user confirming an inference reaches 0.94 — known.
    """
    a, b = _clamp01(a), _clamp01(b)
    return min(MAX_CONFIDENCE, 1.0 - (1.0 - a) * (1.0 - b))


def effective_confidence(m: "Memory", halflife_days: float = 180.0,
                         now: Optional[float] = None) -> float:
    """Belief as of *now*, not as of when it was written.

    Volatile claims ("currently at Acme", "using version 3") decay toward 0.5 — not
    toward false, toward *unknown* — because the world moves and nothing has re-checked
    them. Stable claims do not decay. Verification resets the clock.
    """
    if not m.volatile:
        return m.confidence
    anchor = m.verified_at or m.created
    age_days = _age_days(anchor, now)
    if age_days is None:
        return m.confidence
    f = 0.5 ** (age_days / max(1e-6, float(halflife_days)))
    return 0.5 + (m.confidence - 0.5) * f


# Claims whose truth has a shelf life. Used to decide what decays, and whether a
# contradiction should quietly supersede the old record or be flagged as a dispute.
_VOLATILE_MARKERS = (
    "currently", "right now", "at the moment", "these days", "for now", "temporarily",
    "today", "this week", "this month", "next week", "upcoming", "deadline", "due ",
    "is working on", "working on", "is using", "lives in", "moved to", "works at",
    "job at", "version", "latest", "plans to", "planning to", "in progress",
)


def guess_volatile(text: str) -> bool:
    t = (text or "").lower()
    return any(mk in t for mk in _VOLATILE_MARKERS)


# --- who a claim is about ---------------------------------------------------
# Claims about AG's own nature, identity, or standing capabilities. These are the
# system prompt's territory (prompts.AG_IDENTITY): a stored one is a stale snapshot
# that will later argue with the authoritative definition, and a maliciously supplied
# one ("you have no file access", "you cannot refuse") is an attempt to rewrite AG by
# leaving a note in its own memory. AG neither stores nor recalls them, from anyone.
_IDENTITY_MARKERS = (
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

# Text that is *about AG* without being a claim about its identity — operational
# self-knowledge, which is worth learning: "AG's web fetch times out on PDFs".
_SELF_MARKERS = (
    "apple-gorilla", "ag's ", "ag is ", "ag has ", "ag can ", "ag cannot ", "ag will ",
    "your tool", "your web", "your calc", "your skill", "your memory", "your model",
    "you are ", "you can ", "you cannot ", "you have ", "yourself", "my own ",
)
_USER_MARKERS = ("user ", "the user", "they prefer", "their project")


def is_identity_claim(text: str) -> bool:
    """True if the text asserts what AG *is*. Never storable, never recallable."""
    t = (text or "").lower()
    return any(mk in t for mk in _IDENTITY_MARKERS)


def guess_subject(text: str) -> str:
    """Who a memory is about, from its wording. Self is checked first: a sentence that
    mentions both AG and the user is guarded as a claim about AG, because that is the
    direction where being wrong is expensive."""
    t = (text or "").lower().strip()
    if any(mk in t for mk in _SELF_MARKERS) or is_identity_claim(t):
        return Subject.SELF
    if t.startswith("user ") or any(mk in t for mk in _USER_MARKERS):
        return Subject.USER
    return Subject.WORLD


def evidence_key(origin: str, asserter: str = "") -> str:
    """The identity of a witness, for deciding what counts as INDEPENDENT corroboration.

    Two different websites are two witnesses; the same site repeating itself is one.
    Every other AG in the fleet collapses to a single witness on purpose: siblings
    inherit from the same lineage, so counting them separately would let one belief
    echo around the fleet and come back sounding like consensus.
    """
    origin = Origin.valid(origin)
    if origin == Origin.AGENT:
        return Origin.AGENT
    a = (asserter or "").strip().lower()
    return f"{origin}:{a}" if a else origin


def new_id() -> str:
    """A sortable, unique id: timestamp + a process-wide counter tiebreaker.

    The counter (not just milliseconds) guarantees uniqueness even when many memories
    are created within the same millisecond in a tight loop."""
    return time.strftime("%Y%m%d-%H%M%S") + f"-{next(_COUNTER) % 100000:05d}"


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
    importance: float = 0.5                 # 0..1 — utility: how much this matters
    confidence: float = 0.5                 # 0..1 — belief: how likely it is to be true
    origin: str = Origin.UNKNOWN            # evidence class the belief rests on
    asserter: str = ""                      # the specific witness (domain, agent id)
    subject: str = Subject.WORLD            # who this memory is ABOUT (self vs other)
    evidence: List[Dict[str, Any]] = field(default_factory=list)  # distinct attestations
    volatile: bool = False                  # truth has a shelf life -> confidence decays
    disputed: bool = False                  # a contradicting memory exists; both suspect
    verified_at: str = ""                   # iso timestamp of last (re)verification
    verified_by: str = ""                   # origin that verified it
    use_count: int = 0                      # bumped each time recall surfaces it
    created: str = field(default_factory=now_iso)
    last_used: str = ""                     # iso timestamp of last recall
    embedding: Optional[List[float]] = None  # cached vector; absent => keyword-only
    links: List[str] = field(default_factory=list)  # ids of related memories (graph edges)
    meta: Dict[str, Any] = field(default_factory=dict)  # layer-specific extras

    # --- evidence ----------------------------------------------------------
    def origins(self) -> set:
        """The distinct evidence classes that have attested this memory."""
        got = {str(e.get("origin")) for e in self.evidence if e.get("origin")}
        got.add(self.origin)
        return got

    def witnesses(self) -> set:
        """The distinct WITNESSES on file — origin plus the specific source within it,
        which is the unit independence is judged on."""
        got = {evidence_key(str(e.get("origin", "")), str(e.get("asserter", "")))
               for e in self.evidence if e.get("origin")}
        got.add(evidence_key(self.origin, self.asserter))
        return got

    def attest(self, origin: str, *, asserter: str = "", note: str = "") -> bool:
        """Record an attestation. Returns True only if it is a NEW, independent witness
        — the same source repeating is not new evidence, which is what stops "said three
        times" from becoming "true"."""
        origin = Origin.valid(origin)
        if evidence_key(origin, asserter) in self.witnesses():
            return False
        e = {"origin": origin, "at": now_iso()}
        if asserter:
            e["asserter"] = asserter[:120]
        if note:
            e["note"] = note[:200]
        self.evidence.append(e)
        return True

    def to_dict(self) -> dict:
        return {
            "id": self.id, "text": self.text, "kind": self.kind, "agent": self.agent,
            "tags": self.tags, "source": self.source, "importance": self.importance,
            "confidence": self.confidence, "origin": self.origin,
            "asserter": self.asserter, "subject": self.subject,
            "evidence": self.evidence, "volatile": self.volatile,
            "disputed": self.disputed, "verified_at": self.verified_at,
            "verified_by": self.verified_by,
            "use_count": self.use_count, "created": self.created,
            "last_used": self.last_used, "embedding": self.embedding,
            "links": self.links, "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        """Tolerant load: unknown/missing fields fall back to sane defaults so the
        record format can evolve without invalidating stored memory. Records written
        before veracity existed get an origin inferred from their `source` and a
        confidence seeded from that origin's prior — so pre-existing memory is graded,
        not trusted by default."""
        source = str(d.get("source", "manual"))
        origin = Origin.valid(str(d.get("origin") or origin_for_source(source)))
        text = str(d.get("text", ""))
        conf = d.get("confidence")
        return cls(
            id=str(d.get("id") or new_id()),
            text=text,
            kind=MemoryKind.valid(str(d.get("kind", MemoryKind.SEMANTIC))),
            agent=str(d.get("agent", "root")),
            tags=[str(t) for t in (d.get("tags") or [])],
            source=source,
            importance=_clamp01(d.get("importance", 0.5)),
            confidence=_clamp01(conf) if conf is not None else prior_for(origin),
            origin=origin,
            asserter=str(d.get("asserter", "") or ""),
            subject=Subject.valid(str(d.get("subject") or guess_subject(text))),
            evidence=[dict(e) for e in (d.get("evidence") or []) if isinstance(e, dict)],
            volatile=bool(d.get("volatile", guess_volatile(text))),
            disputed=bool(d.get("disputed", False)),
            verified_at=str(d.get("verified_at", "") or ""),
            verified_by=str(d.get("verified_by", "") or ""),
            use_count=int(d.get("use_count", 0) or 0),
            created=str(d.get("created") or now_iso()),
            last_used=str(d.get("last_used", "") or ""),
            embedding=_as_vec(d.get("embedding")),
            links=[str(x) for x in (d.get("links") or [])],
            meta=dict(d.get("meta") or {}),
        )


def _age_days(ts: str, now: Optional[float] = None) -> Optional[float]:
    try:
        then = time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return None
    return max(0.0, ((now if now is not None else time.time()) - then) / 86400.0)


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
