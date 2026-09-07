"""Ingest a claude.ai data export and distill it into profile/about_me.md.

The claude.ai export is a .zip containing conversations.json: a list of
conversations, each with chat_messages that carry a sender ("human"/"assistant")
and text. We extract only the user's OWN messages and hand them to the model to
distill durable, non-sensitive facts. Everything stays local; nothing is uploaded
beyond the normal model call the user initiates.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import List

from . import prompts
from .config import Config, PROFILE_DIR


def _messages_from_obj(obj) -> List[str]:
    """Pull human-authored message text from a parsed conversations.json object.

    Tolerates a few shape variants seen across export versions.
    """
    convs = obj if isinstance(obj, list) else obj.get("conversations", [obj])
    out: List[str] = []
    for c in convs:
        if not isinstance(c, dict):
            continue
        entries = c.get("chat_messages") or c.get("messages") or []
        for m in entries:
            if not isinstance(m, dict):
                continue
            sender = m.get("sender") or m.get("role")
            if sender not in ("human", "user"):
                continue
            text = m.get("text")
            if not text:  # newer exports use a content-parts list
                parts = m.get("content", [])
                if isinstance(parts, list):
                    text = " ".join(
                        p.get("text", "") for p in parts if isinstance(p, dict)
                    )
            if isinstance(text, str) and text.strip():
                out.append(text.strip())
    return out


def extract_human_messages(path) -> List[str]:
    """Accept a .zip, a .json, or a directory containing conversations.json."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No export at: {path}")

    if p.is_dir():
        f = p / "conversations.json"
        if f.exists():
            return _messages_from_obj(json.loads(f.read_text()))
        out: List[str] = []
        for j in p.glob("*.json"):
            try:
                out += _messages_from_obj(json.loads(j.read_text()))
            except Exception:
                continue
        return out

    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            name = next(
                (n for n in z.namelist() if n.endswith("conversations.json")), None
            )
            names = [name] if name else [n for n in z.namelist() if n.endswith(".json")]
            out = []
            for n in names:
                try:
                    out += _messages_from_obj(json.loads(z.read(n).decode("utf-8")))
                except Exception:
                    continue
            return out

    if p.suffix == ".json":
        return _messages_from_obj(json.loads(p.read_text()))

    raise ValueError(f"Unsupported export format: {path} (want .zip, .json, or dir)")


def build_corpus(messages: List[str], max_chars: int = 40000) -> str:
    """Keep the most recent messages within a character budget (recency wins)."""
    picked, total = [], 0
    for m in reversed(messages):
        if total + len(m) > max_chars:
            break
        picked.append(m)
        total += len(m)
    picked.reverse()
    return "\n\n---\n\n".join(picked)


def distill(client, cfg: Config, messages: List[str]) -> str:
    corpus = build_corpus(messages)
    res = client.complete(
        system=prompts.INGEST_SYSTEM,
        user="Here are the user's own past messages. Distill the profile.\n\n" + corpus,
        cfg=cfg,
        max_tokens=cfg.meta_output_tokens,
    )
    return res.text.strip()


def write_profile(markdown: str) -> Path:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    dest = PROFILE_DIR / "about_me.md"
    dest.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    return dest
