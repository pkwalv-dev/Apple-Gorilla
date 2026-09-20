"""Instruction-following compensation in the model layer.

Abliterated / merged models are run here on purpose (the specialist coder), and
their known failure mode is discipline, not knowledge. The harness cannot fix the
weights, so it must verify outputs and repair prompts rather than assume
compliance. These tests pin that compensation.
"""
from __future__ import annotations

import dataclasses

from ag.config import Config
from ag.model import ModelResult, complete_json, looks_degenerate


def _cfg():
    return dataclasses.replace(Config(), backend="dry")


class StubClient:
    """Answers from a scripted queue; records every prompt it was given."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, *, system, user, cfg, **kw):
        self.prompts.append(user)
        text = self.replies.pop(0) if self.replies else "{}"
        return ModelResult(text=text, input_tokens=10, output_tokens=5,
                           model="stub")


# --- looks_degenerate --------------------------------------------------------

def test_clean_answers_are_never_flagged():
    for ok in ("42", "Paris", "The migration adds one column.",
               '```json\n{"a": 1}\n```', "SHORT"):
        assert looks_degenerate(ok) is None, ok


def test_empty_is_degenerate():
    assert looks_degenerate("") == "empty"
    assert looks_degenerate("   \n ") == "empty"
    assert looks_degenerate(None) == "empty"


def test_repetition_loop_is_caught_and_sized():
    chunk = "Result: the value is 42 and the task is complete. "
    text = chunk * 12
    reason = looks_degenerate(text)
    assert reason and reason.startswith("repetition loop")


def test_repetition_requires_three_tail_copies():
    chunk = "a moderately long sentence that is not tiny, " * 2
    assert looks_degenerate(chunk) is None


# --- complete_json -----------------------------------------------------------

def test_clean_json_passes_first_try():
    c = StubClient(['{"patches": [], "rationale": "nothing to do"}'])
    obj, raw, attempts = complete_json(c, system="s", user="u", cfg=_cfg())
    assert obj == {"patches": [], "rationale": "nothing to do"}
    assert attempts == 1


def test_prose_wrapped_json_is_recovered():
    c = StubClient(['Sure! Here is the JSON: {"ok": true} — hope that helps.'])
    obj, _, attempts = complete_json(c, system="s", user="u", cfg=_cfg())
    assert obj == {"ok": True} and attempts == 1


def test_unparseable_reply_triggers_a_repair_prompt():
    """The retry must NAME the failure and restate the contract — that is the
    repair, not just asking again and hoping."""
    c = StubClient(["I think the answer might be something like...",
                    '{"fixed": true}'])
    obj, _, attempts = complete_json(c, system="s", user="u", cfg=_cfg())
    assert obj == {"fixed": True} and attempts == 2
    assert len(c.prompts) == 2
    repair = c.prompts[1]
    assert "could not be used" in repair and "no markdown fences" in repair.lower()


def test_persistent_garbage_returns_none_after_the_attempt_cap():
    c = StubClient(["garbage", "still garbage", "never json"])
    obj, raw, attempts = complete_json(c, system="s", user="u", cfg=_cfg(),
                                       attempts=2)
    assert obj is None and attempts == 2 and raw == "still garbage"
    assert len(c.prompts) == 2      # the cap held


def test_degenerate_json_also_repairs():
    """A repetition loop that HAPPENS to contain JSON is still degenerate — the
    loop guard must win over the parser."""
    loop = '{"tool": "calc"} ' * 30
    c = StubClient([loop, '{"clean": 1}'])
    obj, _, attempts = complete_json(c, system="s", user="u", cfg=_cfg())
    assert obj == {"clean": 1} and attempts == 2


def test_budget_is_charged_per_call():
    from ag.budget import Budget, BudgetExceeded
    import pytest
    b = Budget(max_model_calls=1)
    c = StubClient(["no json here", '{"too": "late"}'])
    with pytest.raises(BudgetExceeded):
        complete_json(c, system="s", user="u", cfg=_cfg(), budget=b, attempts=3)
