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


# --- a non-answer is not an answer ------------------------------------------
def test_a_bare_tool_name_is_not_mistaken_for_an_answer():
    from ag import reason
    names = {"read_file", "calc"}
    assert reason._is_not_an_answer("read_file", names)
    assert reason._is_not_an_answer("  'read_file' ", names)
    assert reason._is_not_an_answer("", names)
    # A terse but real answer is left alone — the guard must not eat these.
    assert not reason._is_not_an_answer("10063", names)
    assert not reason._is_not_an_answer("Paris.", names)


def test_the_loop_reasks_once_when_it_ends_with_a_non_answer(monkeypatch):
    """After a good observation the model sometimes emits a bare tool name — a tool
    call whose JSON never got written. Returning that discards the work just done."""
    from ag import reason
    from ag.config import Config
    from ag.model import ModelResult
    from ag.permissions import PermissionBroker

    replies = iter([
        '{"tool":"calc","args":{"expr":"2+2"}}',   # a real tool call
        "calc",                                    # ...then a non-answer
        "The answer is 4.",                        # the re-ask
    ])

    class _C:
        def __init__(self): self.n = 0
        def complete(self, **kw):
            self.n += 1
            return ModelResult(text=next(replies))

    c = _C()
    rr = reason.solve(c, Config(), system="s", user="what is 2+2?",
                      broker=PermissionBroker(allow_external_tools=True))
    assert rr.answer == "The answer is 4."
    assert c.n == 3                       # asked again exactly once
    assert [s["tool"] for s in rr.steps] == ["calc"]


def test_the_reask_happens_only_once(monkeypatch):
    """If the second attempt is also useless, AG returns what it has rather than
    looping — an unhelpful answer beats an unbounded one."""
    from ag import reason
    from ag.config import Config
    from ag.model import ModelResult
    from ag.permissions import PermissionBroker

    replies = iter(['{"tool":"calc","args":{"expr":"2+2"}}', "calc", "calc", "calc"])

    class _C:
        def __init__(self): self.n = 0
        def complete(self, **kw):
            self.n += 1
            return ModelResult(text=next(replies))

    c = _C()
    rr = reason.solve(c, Config(), system="s", user="q",
                      broker=PermissionBroker(allow_external_tools=True))
    assert c.n == 3 and rr.answer == "calc"
