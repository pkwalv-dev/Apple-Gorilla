"""Per-session working memory (A3): the buffer that keeps one conversation coherent.

Hermetic: every test redirects the sessions dir to a tmp path and uses a scripted
summarizer client (or none), so nothing touches the real state dir, Ollama, or network.
"""
import json
from types import SimpleNamespace

import pytest

from ag.memory import working


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    d = tmp_path / "sessions"
    monkeypatch.setattr(working, "SESSIONS_DIR", d)
    monkeypatch.setattr(working, "_CURRENT_POINTER", d / "current.json")
    yield


class _Summarizer:
    """Stands in for the model: returns a fixed digest for any summarize call."""
    def __init__(self, text="DIGEST: earlier turns condensed."):
        self.text = text
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(text=self.text)


# --- verbatim recall --------------------------------------------------------
def test_recent_turns_are_kept_verbatim(tmp_path):
    wm = working.load("s1")
    working.update(wm, "what's the capital of France?", "Paris.", client=None)
    block = working.render(wm)
    assert "Paris." in block and "capital of France" in block
    assert "Recent turns" in block


def test_empty_buffer_renders_nothing():
    assert working.render(working.load("blank")) == ""


# --- eviction into the rolling summary --------------------------------------
def test_overflow_is_folded_into_summary_via_model(tmp_path):
    wm = working.load("s2")
    client = _Summarizer()
    for i in range(4):                       # recent_turns=2 keeps only the last 2 pairs
        working.update(wm, f"question {i}", f"answer {i}", client=client,
                       recent_turns=2)
    assert client.calls >= 1                 # older turns were summarized, not dropped
    assert "DIGEST" in wm.summary
    assert len(wm.recent) == 4               # exactly the last 2 exchanges, verbatim
    assert "question 0" not in working.render(wm)   # scrolled off the tail
    assert "DIGEST" in working.render(wm)           # ...but preserved in the summary


def test_overflow_without_a_model_degrades_to_extractive(tmp_path):
    wm = working.load("s3")
    for i in range(4):
        working.update(wm, f"topic {i}", f"reply {i}", client=None, recent_turns=2)
    # No model: coverage is kept extractively rather than lost.
    assert "topic 0" in wm.summary or "reply 0" in wm.summary


# --- the decision ledger survives the tail ----------------------------------
def test_a_pinned_decision_survives_scrolling_off_the_tail(tmp_path):
    wm = working.load("s4")
    working.update(wm, "let's go with session ids for this", "Sounds good.",
                   client=_Summarizer(), recent_turns=2)
    for i in range(4):                        # push the decision far past the tail
        working.update(wm, f"unrelated {i}", f"ok {i}", client=_Summarizer(),
                       recent_turns=2)
    block = working.render(wm)
    assert "session ids" in block
    assert "Key facts & decisions" in block
    assert "unrelated 0" not in block         # ordinary turns do scroll off


# --- isolation: no cross-conversation bleed ---------------------------------
def test_sessions_are_isolated(tmp_path):
    a = working.load("A")
    working.update(a, "my project is called Bolt", "Noted.", client=None)
    working.save(a)
    b = working.load("B")                      # a different conversation
    assert "Bolt" not in working.render(b)
    assert working.render(b) == ""


def test_save_and_reload_roundtrips(tmp_path):
    wm = working.load("rt")
    working.update(wm, "hello", "hi there", client=None)
    working.save(wm)
    again = working.load("rt")
    assert "hi there" in working.render(again)
    assert again.turn_count == 1


# --- seeding from client history (page reload lands mid-conversation) --------
def test_seed_only_populates_an_empty_buffer_and_never_duplicates(tmp_path):
    wm = working.load("seed")
    history = [{"role": "user", "text": "earlier q"},
               {"role": "ai", "text": "earlier a"}]
    working.seed_from_history(wm, history)
    assert "earlier q" in working.render(wm)
    # A second seed on the now-non-empty buffer is a no-op (no duplicate turns).
    working.seed_from_history(wm, history)
    assert working.render(wm).count("earlier q") == 1


# --- session resolution -----------------------------------------------------
def test_explicit_session_id_is_used_as_is(tmp_path):
    assert working.resolve_session("web-abc") == "web-abc"


def test_cli_continues_a_recent_session_and_rotates_after_idle(tmp_path):
    first = working.resolve_session(None, idle_reset_min=45)
    second = working.resolve_session(None, idle_reset_min=45)
    assert first == second                     # consecutive runs chain

    # Backdate the pointer beyond the idle window: the next run starts fresh.
    ptr = json.loads(working._CURRENT_POINTER.read_text())
    ptr["at"] = ptr["at"] - 46 * 60
    working._CURRENT_POINTER.write_text(json.dumps(ptr))
    third = working.resolve_session(None, idle_reset_min=45)
    assert third != first


def test_dry_run_client_does_not_summarize(tmp_path):
    from ag.model import DryRunClient
    wm = working.load("dry")
    for i in range(4):
        working.update(wm, f"q{i}", f"a{i}", client=DryRunClient(), recent_turns=2)
    # DryRun answers are stubs, so the model summary path is skipped (extractive only).
    assert "q0" in wm.summary or "a0" in wm.summary


# --- pipeline wiring: capture_memory advances the buffer --------------------
def test_capture_memory_advances_the_working_buffer(tmp_path, monkeypatch):
    """The pipeline's post-answer hook must fold the finished turn into the session
    buffer — and pin a decision — independently of durable auto-memory."""
    from types import SimpleNamespace
    from ag import config as cfgmod
    from ag.memory import manager as mgrmod
    from ag.memory.embed import HashingEmbedder
    from ag import memory, pipeline

    monkeypatch.setattr(cfgmod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(mgrmod, "get_embedder", lambda cfg=None, **k: HashingEmbedder())
    memory.reset()

    class _Client:
        def complete(self, **kw):            # distiller: nothing durable this turn
            return SimpleNamespace(text='{"facts": []}')

    cfg = cfgmod.Config()
    pipeline.capture_memory(_Client(), cfg, "let's go with plan A for the layout",
                            "Sounds good — plan A it is.", session_id="cap1")
    wm = working.load("cap1")
    assert "plan A" in working.render(wm)
    assert any("plan A" in e["text"] for e in wm.ledger)
    memory.reset()


def test_capture_memory_without_session_id_is_a_noop_for_working(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ag import config as cfgmod
    from ag.memory import manager as mgrmod
    from ag.memory.embed import HashingEmbedder
    from ag import memory, pipeline

    monkeypatch.setattr(cfgmod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(mgrmod, "get_embedder", lambda cfg=None, **k: HashingEmbedder())
    memory.reset()

    class _Client:
        def complete(self, **kw):
            return SimpleNamespace(text='{"facts": []}')

    pipeline.capture_memory(_Client(), cfgmod.Config(), "hi", "hello", session_id="")
    assert not (working.SESSIONS_DIR.exists() and any(working.SESSIONS_DIR.iterdir()))
    memory.reset()
