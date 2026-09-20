"""The guidance protocol — how AG asks for help instead of guessing.

A local 8B model will meet things it genuinely cannot resolve: an ambiguous
instruction, a destructive action that needs sign-off, a format it inferred but
could not parse, a task above its weight class. Today it has exactly two options,
and both are bad — guess confidently, or refuse and stop. This module adds the
third: **escalate, with structure.**

A guidance request is not a plea. It is a record of work already done:

    question       what AG needs decided, in one sentence
    blocked_on     what it cannot proceed past
    tried          what it already attempted, so nobody repeats it
    options        the candidate resolutions, each with its tradeoff
    recommendation which one AG would pick, and why
    urgency        blocking | soon | whenever
    confidence     how sure AG is that it even framed the question right

That shape is the difference between "I need help" and a decision the human can make
in five seconds. It is also machine-readable, so the CLI, the web app, and the MCP
server all render the same queue, and an answer flows back into the run that was
waiting on it.

Requests persist under `state/guidance/` so a question raised by a background job is
still there when the operator looks. Nothing here calls a model or blocks a thread:
the reason loop records a request and continues with its best-effort fallback, which
keeps AG useful while still surfacing that it was unsure.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import STATE_DIR

GUIDANCE_DIR = STATE_DIR / "guidance"
PENDING_FILE = GUIDANCE_DIR / "pending.jsonl"
ANSWERED_FILE = GUIDANCE_DIR / "answered.jsonl"

URGENCIES = ("blocking", "soon", "whenever")
MAX_PENDING = 200


@dataclass
class Option:
    """One way forward, and what it costs."""

    label: str
    detail: str = ""
    tradeoff: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class GuidanceRequest:
    """A structured request for human judgement."""

    id: str
    question: str
    blocked_on: str = ""
    tried: List[str] = field(default_factory=list)
    options: List[dict] = field(default_factory=list)
    recommendation: str = ""
    urgency: str = "soon"
    confidence: float = 0.5           # AG's confidence that it framed this right
    context: str = ""                 # the task this arose in
    agent: str = "root"
    session: str = ""
    created: str = ""
    answered: str = ""
    answer: str = ""
    status: str = "pending"           # pending | answered | withdrawn

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GuidanceRequest":
        return cls(
            id=str(d.get("id") or _new_id()),
            question=str(d.get("question", "")),
            blocked_on=str(d.get("blocked_on", "")),
            tried=[str(t) for t in (d.get("tried") or [])],
            options=[o if isinstance(o, dict) else {"label": str(o)}
                     for o in (d.get("options") or [])],
            recommendation=str(d.get("recommendation", "")),
            urgency=_valid_urgency(d.get("urgency")),
            confidence=_clamp(d.get("confidence", 0.5)),
            context=str(d.get("context", "")),
            agent=str(d.get("agent", "root")),
            session=str(d.get("session", "")),
            created=str(d.get("created", "")),
            answered=str(d.get("answered", "")),
            answer=str(d.get("answer", "")),
            status=str(d.get("status", "pending")),
        )

    def render(self, *, index: Optional[int] = None) -> str:
        """The human-facing view — the whole point of the structure."""
        head = f"[{self.id[:8]}]" if index is None else f"{index}. [{self.id[:8]}]"
        lines = [f"{head} {self.question}",
                 f"    urgency: {self.urgency}   confidence: {self.confidence:.2f}"]
        if self.context:
            lines.append(f"    task: {_trim(self.context, 160)}")
        if self.blocked_on:
            lines.append(f"    blocked on: {self.blocked_on}")
        if self.tried:
            lines.append("    already tried:")
            lines += [f"      - {t}" for t in self.tried]
        if self.options:
            lines.append("    options:")
            for i, o in enumerate(self.options, 1):
                label = o.get("label", "")
                detail = o.get("detail", "")
                trade = o.get("tradeoff", "")
                lines.append(f"      {i}) {label}" + (f" — {detail}" if detail else ""))
                if trade:
                    lines.append(f"         tradeoff: {trade}")
        if self.recommendation:
            lines.append(f"    AG would: {self.recommendation}")
        if self.status == "answered":
            lines.append(f"    ANSWERED {self.answered}: {self.answer}")
        return "\n".join(lines)

    def fallback_note(self) -> str:
        """What the model is told after filing a request, so the run can continue.

        Escalating must not mean stalling. AG proceeds with its recommendation and
        *says* that it did — an answer marked provisional is far more useful than
        silence, and far more honest than an unmarked guess.
        """
        rec = self.recommendation or (self.options[0].get("label")
                                      if self.options else "your best judgement")
        return (f"Guidance request {self.id[:8]} recorded for the user. "
                f"Do not wait for an answer — continue now using: {rec}. "
                f"In your final answer, state plainly that you made this assumption "
                f"and that you have asked the user to confirm it.")


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _clamp(x, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return 0.5


def _valid_urgency(u) -> str:
    s = str(u or "soon").lower()
    return s if s in URGENCIES else "soon"


def _trim(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + "…"


def _ensure() -> None:
    GUIDANCE_DIR.mkdir(parents=True, exist_ok=True)


def _read(path: Path) -> List[GuidanceRequest]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(GuidanceRequest.from_dict(json.loads(line)))
        except Exception:
            continue      # a corrupt line must never hide the rest of the queue
    return out


def _write(path: Path, items: List[GuidanceRequest]) -> None:
    _ensure()
    path.write_text(
        "".join(json.dumps(i.as_dict(), ensure_ascii=False) + "\n" for i in items),
        encoding="utf-8")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

def ask(question: str, *, blocked_on: str = "", tried=(), options=(),
        recommendation: str = "", urgency: str = "soon", confidence: float = 0.5,
        context: str = "", agent: str = "root",
        session: str = "") -> GuidanceRequest:
    """File a guidance request and return it. Never raises into a run."""
    req = GuidanceRequest(
        id=_new_id(), question=str(question).strip(),
        blocked_on=str(blocked_on).strip(),
        tried=[str(t) for t in tried if str(t).strip()],
        options=[o.as_dict() if isinstance(o, Option) else _coerce_option(o)
                 for o in options],
        recommendation=str(recommendation).strip(),
        urgency=_valid_urgency(urgency), confidence=_clamp(confidence),
        context=_trim(context, 600), agent=agent or "root", session=session,
        created=time.strftime("%Y-%m-%dT%H:%M:%S"), status="pending")
    if not req.question:
        req.question = "(no question text provided)"
        req.confidence = 0.1
    try:
        items = _read(PENDING_FILE)
        # De-duplicate: an agent in a loop must not file the same question fifty
        # times. Same question + same context = the same request.
        for existing in items:
            if (existing.question == req.question
                    and existing.context == req.context
                    and existing.status == "pending"):
                return existing
        items.append(req)
        _write(PENDING_FILE, items[-MAX_PENDING:])
    except OSError:
        pass          # an unwritable state dir must not break the run
    return req


def _coerce_option(o) -> dict:
    if isinstance(o, dict):
        return {"label": str(o.get("label", "")),
                "detail": str(o.get("detail", "")),
                "tradeoff": str(o.get("tradeoff", ""))}
    return {"label": str(o), "detail": "", "tradeoff": ""}


def pending(*, urgency: str = "", limit: int = 50) -> List[GuidanceRequest]:
    items = [r for r in _read(PENDING_FILE) if r.status == "pending"]
    if urgency:
        items = [r for r in items if r.urgency == urgency]
    rank = {u: i for i, u in enumerate(URGENCIES)}
    items.sort(key=lambda r: (rank.get(r.urgency, 9), r.created))
    return items[:limit]


def get(request_id: str) -> Optional[GuidanceRequest]:
    rid = str(request_id or "")
    for r in _read(PENDING_FILE) + _read(ANSWERED_FILE):
        if r.id == rid or r.id.startswith(rid):
            return r
    return None


def answer(request_id: str, text: str) -> Optional[GuidanceRequest]:
    """Answer a request: move it to the answered log and return it.

    An answer is also durable *knowledge* — the same question should not need
    asking twice — so the caller (CLI/server) stores it as a semantic memory.
    """
    items = _read(PENDING_FILE)
    hit = None
    keep = []
    for r in items:
        if hit is None and (r.id == request_id or r.id.startswith(str(request_id))):
            r.status = "answered"
            r.answer = str(text)
            r.answered = time.strftime("%Y-%m-%dT%H:%M:%S")
            hit = r
        else:
            keep.append(r)
    if hit is None:
        return None
    _write(PENDING_FILE, keep)
    done = _read(ANSWERED_FILE)
    done.append(hit)
    _write(ANSWERED_FILE, done[-MAX_PENDING:])
    return hit


def withdraw(request_id: str) -> bool:
    items = _read(PENDING_FILE)
    keep = [r for r in items
            if not (r.id == request_id or r.id.startswith(str(request_id)))]
    if len(keep) == len(items):
        return False
    _write(PENDING_FILE, keep)
    return True


def clear() -> int:
    n = len(_read(PENDING_FILE))
    _write(PENDING_FILE, [])
    return n


def answered(limit: int = 50) -> List[GuidanceRequest]:
    return list(reversed(_read(ANSWERED_FILE)))[:limit]


def stats() -> dict:
    p = pending(limit=10_000)
    return {
        "pending": len(p),
        "blocking": sum(1 for r in p if r.urgency == "blocking"),
        "soon": sum(1 for r in p if r.urgency == "soon"),
        "whenever": sum(1 for r in p if r.urgency == "whenever"),
        "answered": len(_read(ANSWERED_FILE)),
    }


def context_for_prompt(*, limit: int = 3) -> str:
    """Recently answered guidance, injected so AG doesn't re-ask a settled question."""
    done = answered(limit=limit)
    if not done:
        return ""
    lines = ["# Guidance the user already gave (treat as decided)"]
    for r in done:
        lines.append(f"- Q: {_trim(r.question, 140)}\n  A: {_trim(r.answer, 200)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Heuristics: when *should* AG escalate?
# --------------------------------------------------------------------------- #

# Phrases that indicate an irreversible or expensive action. A request touching one
# of these is a sign-off case even when the model feels confident, because the cost
# of being wrong is asymmetric.
_IRREVERSIBLE = (
    "delete", "rm -rf", "drop table", "truncate", "format", "overwrite",
    "force push", "git reset --hard", "revoke", "uninstall", "wipe",
    "send email", "post to", "publish", "deploy", "pay", "purchase",
    "shutdown", "reboot", "kill -9", "chmod 777", "sudo",
)


def should_escalate(*, confidence: float, action: str = "",
                    reversible: bool = True, threshold: float = 0.45) -> tuple:
    """Decide whether to ask. Returns (escalate: bool, reason: str).

    Two independent triggers, because they catch different failures:
      - low confidence catches "I don't know what they mean";
      - irreversibility catches "I know exactly what they mean, and if I'm wrong
        about it nothing can be undone".
    """
    act = (action or "").lower()
    hit = next((k for k in _IRREVERSIBLE if k in act), "")
    # An explicit `reversible=False` from the caller is the strongest evidence there
    # is — it beats keyword matching, which can only recognise damage it has a word
    # for. Requiring BOTH signals meant a caller who correctly declared "this cannot
    # be undone" was ignored whenever the phrasing was unusual ("apply the
    # migration"), which is exactly the case the flag exists to cover.
    if not reversible:
        detail = f"the action involves {hit!r}, which " if hit else "this action "
        return True, (f"{detail}cannot be undone — confirm before proceeding")
    if confidence < threshold:
        return True, (f"confidence {confidence:.2f} is below the {threshold:.2f} "
                      "threshold for acting without confirmation")
    if hit and confidence < 0.8:
        return True, (f"the action involves {hit!r} and confidence is only "
                      f"{confidence:.2f}")
    return False, ""
