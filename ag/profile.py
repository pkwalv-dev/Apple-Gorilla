"""Loads the user's 'standard of intelligence' as evaluation principles.

Key design choice (per the project brief): AG must NOT mimic the user's style.
It reads these files as *principles and standards* the critic holds the output to,
so quality converges on the user's bar by underlying design, not imitation.
"""
from __future__ import annotations

from .config import PROFILE_DIR


def load_principles() -> str:
    """Concatenate the profile markdown files into one principles block."""
    parts = []
    for name in ("principles.md", "about_me.md"):
        p = PROFILE_DIR / name
        if p.exists():
            content = p.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"# {name}\n{content}")
    if not parts:
        return "(No profile provided. Use general standards of correctness and clarity.)"
    return "\n\n".join(parts)


def load_user_context() -> str:
    """Return only the *substantive* facts about the user, for priming the model.

    Used to tailor the initial prompt's depth/framing to the user. Returns "" when
    about_me.md is still an unfilled template, so we never inject scaffolding or
    "fill this in" placeholders into the model. This is FACTS for tailoring content —
    not a voice to imitate (see load_principles for the quality bar).
    """
    p = PROFILE_DIR / "about_me.md"
    if not p.exists():
        return ""
    substantive = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(">"):
            continue
        # Only filled bullet values count as user facts. Non-bullet prose is
        # treated as template guidance and ignored, so an unfilled template
        # deterministically yields no context.
        if s.startswith("-"):
            body = s.lstrip("-").strip()
            if not body or body.endswith(":"):  # empty template field
                continue
            substantive.append(body)
    return "\n".join(substantive).strip()
