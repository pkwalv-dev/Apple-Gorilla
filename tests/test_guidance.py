"""The ask-for-help channel.

The governing rule: escalation must never stall the run. A system that blocks
waiting for a human is worse than one that proceeds with a declared assumption,
because the human is usually not there.
"""
from __future__ import annotations

import pytest

from ag import guidance


@pytest.fixture(autouse=True)
def _clean_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(guidance, "GUIDANCE_DIR", tmp_path / "guidance")
    monkeypatch.setattr(guidance, "PENDING_FILE", tmp_path / "guidance" / "pending.jsonl")
    monkeypatch.setattr(guidance, "ANSWERED_FILE", tmp_path / "guidance" / "answered.jsonl")
    yield


def test_ask_returns_immediately_with_a_usable_fallback():
    """The single most important property in this module."""
    req = guidance.ask("Which database should I target?",
                       options=[{"label": "sqlite", "detail": "local file"},
                                {"label": "postgres", "detail": "server"}],
                       recommendation="sqlite",
                       context="user asked to store results")
    assert req.id
    note = req.fallback_note()
    assert note and "sqlite" in note, "the fallback must state what AG will do"


def test_a_request_with_no_recommendation_still_yields_a_note():
    req = guidance.ask("Ambiguous thing?")
    assert req.fallback_note()


def test_pending_roundtrip_and_answer():
    req = guidance.ask("Proceed with deletion?", options=[{"label": "yes"},
                                                          {"label": "no"}])
    assert [r.id for r in guidance.pending()] == [req.id]
    got = guidance.get(req.id)
    assert got is not None and got.question == "Proceed with deletion?"

    answered = guidance.answer(req.id, "no")
    assert answered is not None and answered.answer == "no"
    assert guidance.pending() == [], "answering must remove it from pending"
    assert any(r.id == req.id for r in guidance.answered())


def test_answer_accepts_an_id_prefix():
    req = guidance.ask("Long id?")
    assert guidance.answer(req.id[:8], "sure") is not None


def test_answering_an_unknown_id_returns_none():
    assert guidance.answer("nonexistent", "x") is None


def test_withdraw_removes_without_answering():
    req = guidance.ask("Never mind?")
    assert guidance.withdraw(req.id)
    assert guidance.pending() == []
    assert not any(r.id == req.id for r in guidance.answered())


def test_queue_is_bounded():
    """An unattended run must not fill the disk with questions nobody reads."""
    for i in range(guidance.MAX_PENDING + 25):
        guidance.ask(f"question {i}")
    assert len(guidance.pending()) <= guidance.MAX_PENDING


def test_duplicate_questions_are_not_asked_twice():
    a = guidance.ask("Same question?", context="ctx")
    b = guidance.ask("Same question?", context="ctx")
    assert a.id == b.id, "re-asking an open question must reuse the open request"
    assert len(guidance.pending()) == 1


def test_should_escalate_on_low_confidence():
    yes, why = guidance.should_escalate(confidence=0.2, action="summarise a file")
    assert yes and why, "escalating must come with a stated reason"
    no, _ = guidance.should_escalate(confidence=0.95, action="summarise a file")
    assert not no


def test_should_escalate_on_irreversibility_even_when_confident():
    """Confidence is not permission. A high-confidence wrong answer that deletes
    data is the worst case, so irreversible actions escalate regardless."""
    yes, why = guidance.should_escalate(confidence=0.99, action="delete the archive",
                                        reversible=False)
    assert yes
    assert "undone" in why.lower() or "revers" in why.lower()


def test_explicit_irreversibility_beats_keyword_matching():
    """A caller who declares `reversible=False` must be believed even when the
    phrasing contains no recognised danger word. Keyword lists only catch harm
    they have vocabulary for; the caller knows what it is about to do."""
    yes, why = guidance.should_escalate(confidence=0.99,
                                        action="apply the migration",
                                        reversible=False)
    assert yes and why


def test_reversible_high_confidence_actions_do_not_ask():
    """The counterweight: escalation must stay rare, or it becomes noise the user
    learns to ignore."""
    no, _ = guidance.should_escalate(confidence=0.97, action="read a file")
    assert not no


def test_context_for_prompt_is_bounded_and_useful():
    for i in range(10):
        r = guidance.ask(f"q{i}")
        guidance.answer(r.id, f"answer number {i}")
    ctx = guidance.context_for_prompt(limit=3)
    assert "answer number 9" in ctx, "the most recent decisions must be included"
    assert "answer number 0" not in ctx, "the limit must be honoured"


def test_context_for_prompt_is_empty_when_nothing_is_settled():
    assert guidance.context_for_prompt() == ""


def test_render_is_human_readable():
    req = guidance.ask("Pick one?", options=[{"label": "a", "detail": "first"},
                                             {"label": "b"}],
                       recommendation="a")
    out = req.render(index=1)
    assert "Pick one?" in out and "a" in out and "b" in out


def test_stats_reports_the_queue():
    r = guidance.ask("q1")
    guidance.ask("q2")
    guidance.answer(r.id, "done")
    st = guidance.stats()
    assert st["pending"] == 1 and st["answered"] >= 1


def test_corrupt_queue_file_does_not_crash_the_reader():
    """State on disk is untrusted input like any other."""
    guidance.GUIDANCE_DIR.mkdir(parents=True, exist_ok=True)
    guidance.PENDING_FILE.write_text('{"broken": \nnot json at all\n{"id": "x"}\n')
    assert isinstance(guidance.pending(), list)


def test_as_dict_is_json_serialisable():
    import json
    json.dumps(guidance.ask("q?", options=[{"label": "x"}]).as_dict())
