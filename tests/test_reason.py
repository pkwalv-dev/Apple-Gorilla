"""Tests for the reason->act->observe tool-use loop (scripted client, no model)."""
from ag import reason
from ag.config import Config
from ag.model import ModelResult
from ag.permissions import PermissionBroker


class ScriptedClient:
    """Returns queued replies in order, so the loop is deterministic."""
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def complete(self, *, system, user, cfg, max_tokens=None):
        self.calls += 1
        text = self._replies.pop(0) if self._replies else "done."
        return ModelResult(text=text)


def _broker(*caps):
    b = PermissionBroker(allow_external_tools=True)
    for c in caps:
        b.grant(c)
    return b


def test_available_tools_respect_grants():
    b = _broker()  # no grants: only ungated tools (calc/recall/remember)
    names = {t.name for t in reason.available_tools(b)}
    assert {"calc", "recall", "remember"} <= names
    assert "read_file" not in names and "python_exec" not in names
    b2 = _broker("filesystem_read", "code_exec")
    names2 = {t.name for t in reason.available_tools(b2)}
    assert "read_file" in names2 and "python_exec" in names2


def test_parse_action_finds_tool_call():
    tools = reason.available_tools(_broker())
    a = reason._parse_action('sure: {"tool":"calc","args":{"expr":"2+2"}}', tools)
    assert a == {"tool": "calc", "args": {"expr": "2+2"}}
    assert reason._parse_action("just prose, no tool", tools) is None


def test_solve_dispatches_calc_then_answers():
    client = ScriptedClient([
        '{"tool":"calc","args":{"expr":"6*7"}}',   # step 1: call calc
        "The product is 42.",                        # step 2: final answer
    ])
    r = reason.solve(client, Config(), system="You are precise.",
                     user="What is 6*7?", broker=_broker())
    assert r.answer == "The product is 42."
    assert len(r.steps) == 1
    assert r.steps[0]["tool"] == "calc" and r.steps[0]["observation"] == "42"


def test_solve_without_tools_is_single_shot():
    client = ScriptedClient(["direct answer"])
    r = reason.solve(client, Config(), system="s", user="u", broker=None)
    assert r.answer == "direct answer"
    assert r.steps == []
    assert client.calls == 1


def test_solve_forces_answer_when_steps_exhausted():
    cfg = Config()
    cfg.max_tool_steps = 2
    # Model keeps calling a tool forever; loop must stop and force a final answer.
    client = ScriptedClient([
        '{"tool":"calc","args":{"expr":"1+1"}}',
        '{"tool":"calc","args":{"expr":"2+2"}}',
        "final synthesized answer",
    ])
    r = reason.solve(client, cfg, system="s", user="u", broker=_broker())
    assert r.answer == "final synthesized answer"
    assert len(r.steps) == 2  # bounded by max_tool_steps
