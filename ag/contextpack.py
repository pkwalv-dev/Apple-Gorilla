"""Context packing with an explicit budget and a declared outcome.

The pipeline layers context blocks onto the system prompt (machine briefing,
settled guidance decisions, conversation, user profile, memory, web sources).
Each append assumes it fits. On a local model with a 8-16k token context that
assumption fails SILENTLY, and the truncation happens at the front of the prompt
— which is where the executor's own instructions live. The least valuable block
survives and the most valuable one is quietly amputated.

This module replaces the assumption with a decision:

- every section declares a priority and whether it may be clipped;
- sections that can't fit whole are clipped to a floor, with a marker INSIDE the
  text saying so (the model should know its evidence was cut, and a trace should
  show it);
- sections that still can't fit are dropped, with the same marker;
- the base instructions are never touched by the packer — the caller only hands
  over context blocks.

Budget 0 means unlimited and the packer returns sections verbatim, so existing
behaviour is byte-identical unless the operator sets a budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

# A clipped/dropped section must SAY so where the model reads it — a silent cut is
# how a model ends up answering confidently from evidence it no longer has.
_CLIP_MARK = " [… '{name}' clipped from {orig} to {kept} chars to fit the context budget …]"
_DROP_MARK = "['{name}' omitted entirely — context budget exhausted]"
KEEP_FLOOR = 600            # below this many chars a clipped section stops being useful


@dataclass
class Section:
    """One context block. Lower priority number = more important = cut LAST."""
    name: str
    text: str
    priority: int = 50
    truncatable: bool = True


@dataclass
class PackNote:
    name: str
    action: str               # "kept" | "clipped" | "dropped"
    detail: str = ""


def pack_sections(sections: List[Section], budget: int = 0) -> Tuple[str, List[PackNote]]:
    """Assemble sections under `budget` chars. Returns (text, notes).

    Two passes over real, re-measured sizes (the clip markers themselves cost
    chars — a packer that ignores its own marker would overshoot the budget it
    exists to enforce):
      1. CLIP least-important truncatable sections to KEEP_FLOOR + marker.
      2. While still over budget, DROP the least-important section to a marker —
         or remove it outright when even the marker does not fit.
    High-priority sections are affected only when nothing else is left to give,
    and the caller's base instructions are never part of what gets packed.
    """
    items = [{"s": s, "text": s.text, "state": "kept"}
             for s in sections if s.text and s.text.strip()]
    notes: List[PackNote] = []

    def size(its) -> int:
        return sum(len(it["text"]) for it in its) + 2 * max(0, len(its) - 1)

    if not budget or budget <= 0 or size(items) <= budget:
        return ("\n\n".join(it["text"] for it in items),
                [PackNote(it["s"].name, "kept") for it in items])

    # Pass 1 — clip truncatable sections to the floor, least important first.
    for it in sorted(items, key=lambda d: -d["s"].priority):
        if size(items) <= budget:
            break
        s = it["s"]
        if not s.truncatable or len(it["text"]) <= KEEP_FLOOR:
            continue
        orig = len(it["text"])
        it["text"] = it["text"][:KEEP_FLOOR].rstrip() + _CLIP_MARK.format(
            name=s.name, orig=orig, kept=KEEP_FLOOR)
        it["state"] = "clipped"
        notes.append(PackNote(s.name, "clipped", f"{orig} -> {KEEP_FLOOR} chars"))

    # Pass 2 — drop sections (to a marker, else entirely) until the budget fits.
    while size(items) > budget and items:
        it = max(items, key=lambda d: d["s"].priority)
        s = it["s"]
        marker = _DROP_MARK.format(name=s.name)
        idx = items.index(it)
        trial = [dict(d) for d in items]
        trial[idx]["text"] = marker
        if size(trial) <= budget or len(items) == 1:
            it["text"] = marker
            it["state"] = "dropped"
            notes.append(PackNote(s.name, "dropped",
                                  f"{len(s.text)} chars removed"))
        else:
            orig = len(it["text"])
            items.remove(it)
            notes.append(PackNote(s.name, "dropped",
                                  f"{orig} chars removed (no room even for a marker)"))

    notes.extend(PackNote(it["s"].name, "kept") for it in items
                 if it["state"] == "kept")
    return "\n\n".join(it["text"] for it in items), notes
