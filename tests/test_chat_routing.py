"""Conversational routing + style-meta filtering.

Covers the classifier that sends plain chat to the instruct model one-shot (leaving
tools/web for real tasks) and the backstop that keeps presentation feedback out of
durable memory. Hermetic: no Ollama, no network.
"""
from types import SimpleNamespace

import pytest

from ag import pipeline
from ag.pipeline import _conversational, _is_style_meta


# --- the conversational classifier -----------------------------------------
@pytest.mark.parametrize("prompt", [
    "How's it going?",
    "Explain why you format your messages the way you do.",
    "what do you think about jazz?",
    "who are you?",
    "thanks, that was helpful",
])
def test_plain_chat_is_conversational(prompt):
    assert _conversational(prompt) is True


@pytest.mark.parametrize("prompt", [
    "read the file ag/reason.py and summarize it",
    "run the test suite",
    "generate an image of a red bicycle",
    "fetch https://example.com and tell me the title",
    "calculate 4823 * 917",
    "debug why the server crashes on startup",
    "commit and push the changes",
])
def test_task_prompts_are_not_conversational(prompt):
    assert _conversational(prompt) is False


def test_a_long_prompt_is_treated_as_a_task():
    assert _conversational("please " + "explain " * 100) is False


# --- style-meta detection ---------------------------------------------------
@pytest.mark.parametrize("text", [
    "User prefers less asterisk-heavy formatting.",
    "User wants AG to stop overusing bold formatting in outputs.",
    "The user prefers fewer formatting markers (bold and asterisks).",
    "User wants your replies to be more concise in tone.",
])
def test_style_meta_is_detected(text):
    assert _is_style_meta(text) is True


@pytest.mark.parametrize("text", [
    "User prefers metric units.",
    "The user is building a Rust game engine called Bolt.",
    "The AG launcher for WSL training is ag-train.sh under ~/ag-lora.",
    "User has an RTX 4060 with 8GB of VRAM.",
])
def test_real_facts_are_not_style_meta(text):
    assert _is_style_meta(text) is False


# --- capture drops style-meta before it can be stored -----------------------
def test_capture_memory_does_not_store_formatting_feedback(tmp_path, monkeypatch):
    from ag import config as cfgmod
    from ag.memory import manager as mgrmod
    from ag.memory.embed import HashingEmbedder
    from ag import memory

    monkeypatch.setattr(cfgmod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(mgrmod, "get_embedder", lambda cfg=None, **k: HashingEmbedder())
    memory.reset()

    class _Client:                       # distiller "extracts" a formatting preference
        def complete(self, **kw):
            return SimpleNamespace(text='{"facts": [{"fact": "User prefers less bold '
                                        'formatting.", "basis": "stated", '
                                        '"volatile": false}]}')

    cfg = cfgmod.Config()
    cfg.working_memory = False           # isolate the durable path
    saved = pipeline.capture_memory(_Client(), cfg, "stop using so much bold", "Okay.")
    assert saved == []                   # the style note was dropped, not stored
    # No durable SEMANTIC fact about formatting was written (the raw episode may quote
    # the word, but that is a transcript, not a recalled fact).
    from ag.memory import MemoryKind
    sem = memory.get_manager("root").store.all("root", [MemoryKind.SEMANTIC])
    assert not any("bold" in m.text.lower() for m in sem)
    memory.reset()
