"""Per-session working memory — the buffer that makes AG conversationally coherent.

Long-term memory (semantic/procedural) answers "what do I know across all time?".
Working memory answers a different, narrower question: "what is going on in THIS
conversation, right now?" — and it is deliberately kept apart from the durable store so
the two can never contaminate each other.

A conversation is not a flat transcript. Held as one, it either overflows a small local
context window (and the oldest turns are silently dropped, so AG "forgets what we
decided") or it is truncated mid-thought. So a session is held as three parts, each
answering a different need:

- recent   — the last few exchanges, VERBATIM. This is precision: AG can quote exactly
             what was just said and resolve "it"/"that" to the right thing.
- summary  — a rolling digest of everything that has scrolled off the tail. This is
             coverage: turn 40 still knows the gist of turn 3 instead of hitting a
             cliff. It is extractive by construction and never leaves the session.
- ledger   — pinned facts and decisions ("we settled on session ids", "the file is
             reason.py"). This is the anti-drop guarantee: a decision made 30 turns ago
             survives no matter how far it scrolls past the verbatim tail.

Isolation is structural, not policed: recall only ever loads THIS session's file, so a
past conversation's buffer cannot bleed into a new one. That is the property that makes
episodic cross-conversation recall unnecessary in the first place — working memory owns
in-session recall; the durable layers own everything that generalizes across sessions.

The buffer is ephemeral. Nothing here is promoted to long-term except through the
existing gated distiller/reflection path (see pipeline.capture_memory), so a weak local
summarizer that hallucinates can only ever muddle one conversation that then ends — it
cannot write a false belief into the durable store.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..config import STATE_DIR

SESSIONS_DIR = STATE_DIR / "sessions"
_CURRENT_POINTER = SESSIONS_DIR / "current.json"

# Budgets. Sized for an 8GB local 7B: the whole rendered block stays ~3.5k chars
# regardless of how long the conversation runs, spent on a digest + pinned facts +
# a verbatim tail rather than on a blindly-truncated transcript.
DEFAULT_RECENT_TURNS = 6          # verbatim EXCHANGES kept (a user+ai pair is one)
DEFAULT_SUMMARY_CHARS = 700
DEFAULT_TURN_CHARS = 1200         # clamp any single very long verbatim turn
DEFAULT_LEDGER_MAX = 12
DEFAULT_IDLE_RESET_MIN = 45       # CLI: a silence longer than this starts a new session

# Cheap, reliable cues that a turn recorded a decision worth pinning. Heuristic on
# purpose — a model pass can refine the ledger later, but this needs no extra call and
# never invents an entry, which is the failure mode that matters.
_DECISION_CUES = (
    "let's go with", "lets go with", "we'll use", "well use", "we will use",
    "decided", "decision", "the plan is", "go with", "let's do", "lets do",
    "final answer", "the answer is", "conclusion", "we agreed", "settled on",
    "from now on", "going forward", "make sure to", "remember to", "the goal is",
)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class WorkingMemory:
    """One conversation's working state. Plain data, serialized to a single JSON file."""

    session_id: str
    summary: str = ""
    recent: List[Dict[str, str]] = field(default_factory=list)   # {"role","text"}
    ledger: List[Dict[str, str]] = field(default_factory=list)   # {"text","turn"}
    turn_count: int = 0
    created: str = field(default_factory=now_iso)
    updated: str = field(default_factory=now_iso)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkingMemory":
        return cls(
            session_id=str(d.get("session_id") or new_session_id()),
            summary=str(d.get("summary", "") or ""),
            recent=[{"role": ("user" if str(t.get("role")) == "user" else "ai"),
                     "text": str(t.get("text", ""))}
                    for t in (d.get("recent") or []) if isinstance(t, dict)],
            ledger=[{"text": str(e.get("text", "")), "turn": str(e.get("turn", ""))}
                    for e in (d.get("ledger") or []) if isinstance(e, dict)
                    and str(e.get("text", "")).strip()],
            turn_count=int(d.get("turn_count", 0) or 0),
            created=str(d.get("created") or now_iso()),
            updated=str(d.get("updated") or now_iso()),
        )

    def is_empty(self) -> bool:
        return not (self.summary or self.recent or self.ledger)


# --- session identity -------------------------------------------------------
_COUNTER = itertools.count()


def new_session_id() -> str:
    """A sortable, unique session id. The process-wide counter is the tiebreaker: two
    ids minted in the same millisecond (a fast test, or a rapid rotate) must still
    differ, or a rotated session collides with the one it replaced."""
    return (time.strftime("%Y%m%d-%H%M%S")
            + f"-{int(time.time() * 1000) % 1000:03d}-{next(_COUNTER) % 1000:03d}")


def _safe(session_id: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in (session_id or "")]
    s = "".join(keep).strip("-")
    return s[:80] or new_session_id()


def _path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{_safe(session_id)}.json"


def resolve_session(session_id: Optional[str] = None, *,
                    idle_reset_min: int = DEFAULT_IDLE_RESET_MIN) -> str:
    """Decide which session this run belongs to.

    An explicit id (the web client sends one per chat tab) is used as-is. With no id —
    the CLI case — the most recent session is continued if the gap since the last run is
    short, and a fresh one is started otherwise, so `ag run` chains within a working
    session and resets when you walk away. The pointer is refreshed on every resolve, so
    idle is measured from the last run, not the first.
    """
    if session_id and session_id.strip():
        sid = session_id.strip()[:80]
        _write_pointer(sid)
        return sid
    sid = new_session_id()
    try:
        if _CURRENT_POINTER.exists():
            ptr = json.loads(_CURRENT_POINTER.read_text(encoding="utf-8"))
            last_id = str(ptr.get("id") or "")
            last_at = float(ptr.get("at") or 0.0)
            if last_id and (time.time() - last_at) <= idle_reset_min * 60:
                sid = last_id
    except Exception:
        pass
    _write_pointer(sid)
    return sid


def _write_pointer(session_id: str) -> None:
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        _CURRENT_POINTER.write_text(
            json.dumps({"id": session_id, "at": time.time()}), encoding="utf-8")
    except Exception:
        pass


# --- load / save ------------------------------------------------------------
def load(session_id: str) -> WorkingMemory:
    """Load this session's buffer, or a fresh empty one. Never raises — a corrupt file
    yields a new buffer rather than breaking the run."""
    p = _path(session_id)
    if p.exists():
        try:
            return WorkingMemory.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return WorkingMemory(session_id=session_id)


def save(wm: WorkingMemory) -> None:
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        wm.updated = now_iso()
        _path(wm.session_id).write_text(
            json.dumps(asdict(wm), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def seed_from_history(wm: WorkingMemory, history) -> WorkingMemory:
    """Prime a brand-new buffer from client-supplied history (e.g. a page reload lands
    mid-conversation). Only ever runs on an empty buffer, so it cannot duplicate turns
    the update path has already folded in."""
    if wm.turn_count or wm.recent or not history:
        return wm
    for h in history:
        if not isinstance(h, dict):
            continue
        text = str(h.get("text", "")).strip()
        if not text:
            continue
        role = "user" if str(h.get("role")) == "user" else "ai"
        wm.recent.append({"role": role, "text": text[:DEFAULT_TURN_CHARS]})
    # Keep only the verbatim tail; anything older is coverage the summary would own, but
    # we have no model call here, so drop it rather than fabricate a summary.
    wm.recent = wm.recent[-2 * DEFAULT_RECENT_TURNS:]
    return wm


# --- rendering (what gets injected) -----------------------------------------
def render(wm: WorkingMemory, *, summary_chars: int = DEFAULT_SUMMARY_CHARS) -> str:
    """Assemble the working-memory block for the executor system prompt. Returns "" when
    there is nothing to show, so the caller can fall back to a raw transcript."""
    if wm.is_empty():
        return ""
    out: List[str] = []
    if wm.summary:
        s = wm.summary.strip()
        if len(s) > summary_chars:
            s = s[:summary_chars].rstrip() + " …"
        out.append("Summary of earlier turns:\n" + s)
    if wm.ledger:
        lines = "\n".join(f"- {e['text']}" for e in wm.ledger if e.get("text"))
        if lines:
            out.append("Key facts & decisions established this session:\n" + lines)
    if wm.recent:
        turns = []
        for t in wm.recent:
            who = "User" if t.get("role") == "user" else "Apple-Gorilla"
            txt = str(t.get("text", "")).strip().replace("\r", "")
            if len(txt) > DEFAULT_TURN_CHARS:
                txt = txt[:DEFAULT_TURN_CHARS] + " …"
            turns.append(f"{who}: {txt}")
        out.append("Recent turns (verbatim):\n" + "\n".join(turns))
    return "\n\n".join(out)


# --- update (after each exchange) -------------------------------------------
def update(wm: WorkingMemory, user_text: str, ai_text: str, *,
           client=None, cfg=None, recent_turns: int = DEFAULT_RECENT_TURNS,
           summary_chars: int = DEFAULT_SUMMARY_CHARS, emit=None) -> WorkingMemory:
    """Fold one finished exchange into the buffer: append it verbatim, pin any decision
    it recorded, and — when the verbatim tail overflows — digest the oldest turns into
    the rolling summary and evict them. Best-effort and non-raising."""
    user_text = (user_text or "").strip()
    ai_text = (ai_text or "").strip()
    if not user_text and not ai_text:
        return wm
    wm.turn_count += 1
    if user_text:
        wm.recent.append({"role": "user", "text": user_text[:DEFAULT_TURN_CHARS]})
    if ai_text:
        wm.recent.append({"role": "ai", "text": ai_text[:DEFAULT_TURN_CHARS]})

    _update_ledger(wm, user_text, ai_text)

    # Evict everything past the verbatim window into the summary.
    keep = 2 * max(1, recent_turns)
    if len(wm.recent) > keep:
        overflow = wm.recent[:-keep]
        wm.recent = wm.recent[-keep:]
        wm.summary = _resummarize(wm.summary, overflow, client=client, cfg=cfg,
                                  summary_chars=summary_chars, emit=emit)
    return wm


def _update_ledger(wm: WorkingMemory, user_text: str, ai_text: str) -> None:
    turn = str(wm.turn_count)
    for text in (user_text, ai_text):
        low = text.lower()
        if not any(cue in low for cue in _DECISION_CUES):
            continue
        entry = _first_relevant_sentence(text) or text
        entry = entry.strip()[:200]
        if entry and not any(e["text"] == entry for e in wm.ledger):
            wm.ledger.append({"text": entry, "turn": turn})
    if len(wm.ledger) > DEFAULT_LEDGER_MAX:
        wm.ledger = wm.ledger[-DEFAULT_LEDGER_MAX:]


def _first_relevant_sentence(text: str) -> str:
    """The sentence in `text` that carries a decision cue — a tighter pin than the whole
    turn, so the ledger stays a list of points rather than paragraphs."""
    for raw in text.replace("\n", ". ").split(". "):
        low = raw.lower()
        if any(cue in low for cue in _DECISION_CUES):
            return raw.strip()
    return ""


def _resummarize(summary: str, evicted: List[Dict[str, str]], *, client=None,
                 cfg=None, summary_chars: int = DEFAULT_SUMMARY_CHARS,
                 emit=None) -> str:
    """Fold evicted turns into the running summary. Uses the model when one is available;
    otherwise falls back to an extractive concatenation so coverage degrades gracefully
    rather than being lost. The result is always session-local and never promoted."""
    evicted_text = "\n".join(
        f"{'User' if t.get('role') == 'user' else 'Apple-Gorilla'}: "
        f"{str(t.get('text',''))[:DEFAULT_TURN_CHARS]}" for t in evicted)
    if not evicted_text.strip():
        return summary
    model_summary = _model_summary(summary, evicted_text, client=client, cfg=cfg)
    if model_summary:
        if emit:
            try:
                emit("memory", "condensed older turns into the session summary",
                     level="tool")
            except Exception:
                pass
        return model_summary.strip()[: summary_chars * 2]
    # Fallback: keep the prior summary plus a clipped extract of what was evicted.
    merged = (summary + "\n" + evicted_text).strip()
    if len(merged) > summary_chars * 2:
        merged = "…\n" + merged[-summary_chars * 2:]
    return merged


def _model_summary(summary: str, evicted_text: str, *, client=None, cfg=None) -> str:
    if client is None:
        return ""
    try:
        from ..model import DryRunClient
        if isinstance(client, DryRunClient):
            return ""
        from .. import prompts
        prior = f"# Running summary so far\n{summary}\n\n" if summary else ""
        user = (prior + "# Turns now being archived (fold these in)\n" + evicted_text)
        res = client.complete(system=prompts.SUMMARIZER_SYSTEM, user=user, cfg=cfg,
                              max_tokens=350)
        return (getattr(res, "text", "") or "").strip()
    except Exception:
        return ""
