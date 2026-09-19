"""Model routing — one brain, a specialist it can call, and a doc it learns from.

AG runs on one PRIMARY model (a strong local instruct model, qwen3:8b) that reasons,
uses tools, and answers every turn. A SPECIALIST model (the abliterated Qwen2.5-Coder)
is not a competing controller — it is a tool the primary can reach for, and a net the
runtime falls back to. Two things decide when the specialist is used:

- a capability doc (this module's `profiles.json`) that distils each model's strengths
  and weaknesses and is injected into the primary's prompt as guidance, so the primary
  can choose to delegate a subtask to the specialist; and
- a deterministic failure-fallback the runtime enforces — a model cannot orchestrate
  its way out of its own crash, refusal, or empty answer, so when the primary hard-fails
  the runtime retries on the specialist.

The doc LEARNS, but only from honest signals — AG does not fabricate a quality score.
It learns from: hard failures, fallback wins (primary failed, specialist succeeded),
Claude escalations (the user judged it too hard for the local models), and an explicit
user rating. Per (task-tag, model) it keeps uses/wins/losses; guidance is re-rendered
from those tallies, so routing sharpens as evidence accumulates.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from .config import STATE_DIR

ROUTING_DIR = STATE_DIR / "routing"
PROFILES_FILE = ROUTING_DIR / "profiles.json"

# Task tags a prompt can carry. Deliberately small and about the KIND of work, so the
# doc's per-tag tallies stay legible and learnable rather than sparse.
TAGS = ("code", "creative", "math", "research", "general")

_TAG_PATTERNS = {
    "code": re.compile(
        r"\b(code|coding|program|programming|function|debug|refactor|implement|"
        r"compile|script|regex|api|json|sql|bug|stack ?trace|traceback|class|method|"
        r"\.py|\.js|\.ts|\.rs|\.go|\.cpp|\.java|npm|pip|git)\b", re.I),
    "creative": re.compile(
        r"\b(write|story|poem|haiku|lyrics|fiction|character|dialogue|screenplay|"
        r"novel|brainstorm|imagine|creative)\b", re.I),
    "math": re.compile(
        r"\b(calculate|compute|solve|equation|integral|derivative|probability|"
        r"algebra|matrix|proof|arithmetic|sum of|factor(?:ial)?)\b", re.I),
    "research": re.compile(
        r"\b(search|look up|latest|news|current|who is|what happened|cite|source|"
        r"according to|research|find out)\b", re.I),
}

# The seeded profile: what each ROLE is good and bad at, before any evidence. Keyed by
# role ("primary"/"specialist") so it survives a change of the underlying model name.
_SEED = {
    "roles": {
        "primary": {
            "strengths": ["general reasoning", "conversation", "planning",
                          "tool orchestration", "math and logic", "research synthesis"],
            "weaknesses": ["may refuse blunt/uncensored requests",
                           "less specialized at long code generation"],
        },
        "specialist": {
            "strengths": ["code generation and debugging", "strict output formats",
                          "will not refuse a task", "uncensored/unfiltered content"],
            "weaknesses": ["weaker open-ended conversation",
                           "narrates tool steps / report scaffolding if over-prompted"],
        },
    },
    # Which role the seed prefers for each tag (before learning shifts it).
    "seed_preference": {"code": "specialist", "creative": "primary", "math": "primary",
                        "research": "primary", "general": "primary"},
    "stats": {},          # tag -> role -> {"uses","wins","losses"}
    "updated": "",
}

# Outcome signals the doc learns from. Each maps to a (wins, losses) delta.
_SIGNALS = {
    "success": (1, 0),
    "rating_up": (2, 0),
    "failure": (0, 1),
    "rating_down": (0, 2),
    "fallback_win": (2, 0),     # this role succeeded where the other had just failed
    "escalated": (0, 1),        # user sent it to Claude — the local role fell short
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# --- persistence ------------------------------------------------------------
def load(cfg=None) -> dict:
    """Load the capability doc, seeding a fresh one on first use. Never raises."""
    try:
        if PROFILES_FILE.exists():
            d = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
            if isinstance(d, dict) and "roles" in d:
                d.setdefault("stats", {})
                d.setdefault("seed_preference", dict(_SEED["seed_preference"]))
                return d
    except Exception:
        pass
    doc = json.loads(json.dumps(_SEED))   # deep copy
    doc["updated"] = now_iso()
    save(doc)
    return doc


def save(doc: dict) -> None:
    try:
        ROUTING_DIR.mkdir(parents=True, exist_ok=True)
        doc["updated"] = now_iso()
        PROFILES_FILE.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    except Exception:
        pass


# --- task classification ----------------------------------------------------
def tags_for(prompt: str) -> Set[str]:
    """The task tags a prompt carries; always includes 'general' as a floor."""
    found = {tag for tag, pat in _TAG_PATTERNS.items() if pat.search(prompt or "")}
    found.add("general")
    return found


# --- guidance (injected into the primary's prompt) --------------------------
def specialist_preferred(doc: dict, tags: Set[str]) -> bool:
    """Whether the doc, seed + learning combined, currently favours the specialist for
    any of these tags. Learning overrides the seed once there is real evidence."""
    for tag in tags:
        if tag == "general":
            continue
        learned = _learned_preference(doc, tag)
        pref = learned or doc.get("seed_preference", {}).get(tag, "primary")
        if pref == "specialist":
            return True
    return False


def _learned_preference(doc: dict, tag: str) -> Optional[str]:
    """The role with the better win-rate for a tag, once each has enough evidence to
    matter; None while the tag is still unproven (so the seed stands)."""
    st = (doc.get("stats") or {}).get(tag) or {}
    best, best_rate = None, -1.0
    for role in ("primary", "specialist"):
        r = st.get(role) or {}
        n = int(r.get("uses", 0))
        if n < 3:                     # too little evidence to overrule the seed
            continue
        rate = int(r.get("wins", 0)) / max(1, n)
        if rate > best_rate:
            best, best_rate = role, rate
    return best


def guidance(cfg, specialist_name: str) -> str:
    """A short block for the primary's system prompt: what the specialist is for and how
    to call it. Reflects the current doc (seed shifted by whatever has been learned)."""
    doc = load(cfg)
    spec = doc["roles"]["specialist"]
    prefer = [t for t in TAGS if t != "general"
              and (_learned_preference(doc, t)
                   or doc["seed_preference"].get(t)) == "specialist"]
    prefer_line = (", ".join(prefer) if prefer else "tasks it is clearly better at")
    return (
        f"# A specialist model is available to you\n"
        f"You are AG's primary model and answer directly. A specialist model "
        f"('{specialist_name}') can be called with the consult_specialist tool when a "
        f"subtask suits it better. It is strong at: {', '.join(spec['strengths'])}. "
        f"Prefer it for: {prefer_line}. Also use it if you would otherwise refuse a "
        f"legitimate request. Do this only when it genuinely helps — answer directly "
        f"otherwise, and never mention this routing to the user."
    )


# --- learning ---------------------------------------------------------------
def record(cfg, role: str, tags, signal: str) -> None:
    """Fold one honest outcome signal into the doc's per-tag tallies. `role` is
    'primary' or 'specialist'; `tags` the task tags; `signal` one of _SIGNALS."""
    if role not in ("primary", "specialist") or signal not in _SIGNALS:
        return
    dw, dl = _SIGNALS[signal]
    doc = load(cfg)
    stats = doc.setdefault("stats", {})
    for tag in set(tags) | {"general"}:
        cell = stats.setdefault(tag, {}).setdefault(
            role, {"uses": 0, "wins": 0, "losses": 0})
        cell["uses"] += 1
        cell["wins"] += dw
        cell["losses"] += dl
    save(doc)


def role_for_model(cfg, model: str) -> str:
    """Map a model name to its role, so a rating on an answer credits the right side."""
    return "specialist" if model and model == getattr(cfg, "specialist_model", "") \
        else "primary"


def summary(cfg) -> dict:
    """The doc as data, for a status view or test."""
    return load(cfg)
