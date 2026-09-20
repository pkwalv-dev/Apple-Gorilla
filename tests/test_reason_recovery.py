"""The reason loop's compensation for weak instruction-following.

When the specialist (or any small model) drives the tool loop, its characteristic
failure is a tool call that *almost* parses. The loop must repair rather than hand
the user protocol garbage — and stop cleanly when the model simply cannot comply.
"""
from __future__ import annotations

import dataclasses

from ag import agents, reason
from ag.config import Config
from ag.model import ModelResult


def _cfg(**kw):
    return dataclasses.replace(Config(), backend="dry", **kw)


class StubClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, *, system, user, cfg, **kw):
        self.prompts.append(user)
        text = self.replies.pop(0) if self.replies else "the answer is 42"
        return ModelResult(text=text, input_tokens=10, output_tokens=5,
                           model="stub")


def _tools_on_cfg():
    # allow_local_tools so available_tools is non-empty; broker grants nothing,
    # which is fine: the tests exercise parsing, not gated tools.
    return _cfg(allow_local_tools=True)


# --- malformed tool-call recovery -------------------------------------------

def test_prose_wrapped_tool_call_is_used_normally():
    """extract_json already salvages prose-wrapped JSON — the common mild case."""
    c = StubClient(['Let me use the calc tool: {"tool": "calc", "args": {"expr": "2+2"}}',
                    "The result is 4."])
    rr = reason.solve(c, _tools_on_cfg(), system="s", user="what is 2+2?")
    assert rr.answer == "The result is 4."
    assert rr.steps and rr.steps[0]["tool"] == "calc"


def test_broken_tool_call_gets_a_repair_observation():
    """`tool: calc, expr 2+2` cannot parse; the loop must name the problem and
    continue from the same transcript instead of answering with garbage."""
    c = StubClient(["tool: calc with expr being 2+2 {json}",
                    '{"tool": "calc", "args": {"expr": "2+2"}}',
                    "2+2 is 4."])
    rr = reason.solve(c, _tools_on_cfg(), system="s", user="what is 2+2?")
    assert rr.answer == "2+2 is 4."
    # The second model call must carry the system's repair observation.
    assert "OBSERVATION (system)" in c.prompts[1]
    assert "valid JSON" in c.prompts[1]


def test_a_model_that_cannot_comply_is_not_looped_forever():
    """The counterweight: repairs are capped, so a hopelessly sloppy model still
    terminates — with its prose, not an infinite loop or an exception."""
    replies = ["tool: calc {json attempt %d" % i for i in range(10)]
    c = StubClient(replies)
    rr = reason.solve(c, _tools_on_cfg(), system="s", user="hi", max_steps=4)
    assert rr.answer                      # something came back
    assert "OBSERVATION (system)" in c.prompts[1]
    # repairs happen twice (the cap), then the output is accepted as an answer
    assert c.prompts.count(c.prompts[0]) == 1
    assert len(c.prompts) <= 5


def test_plain_prose_mentioning_tools_is_not_mistaken_for_an_attempt():
    """'I used the tool of logic' must never trigger repair — the detector is
    for protocol-shaped output only."""
    c = StubClient(["The tool of plain reasoning says the capital is Paris."])
    rr = reason.solve(c, _tools_on_cfg(), system="s", user="capital of France?")
    assert rr.answer.startswith("The tool of plain reasoning")
    assert len(c.prompts) == 1            # no repair was triggered


# --- budget enforcement ------------------------------------------------------

def test_model_call_budget_stops_the_loop_with_a_declaration():
    from ag.budget import Budget
    c = StubClient(['{"tool": "calc", "args": {"expr": "1"}}',
                    '{"tool": "calc", "args": {"expr": "2"}}',
                    '{"tool": "calc", "args": {"expr": "3"}}'])
    rr = reason.solve(c, _tools_on_cfg(), system="s", user="t",
                      budget=Budget(max_model_calls=2))
    assert "(stopped:" in rr.answer
    assert "budget" in rr.answer
    assert len(c.prompts) <= 3            # the loop did not run away


def test_tool_call_budget_blocks_further_tools():
    from ag.budget import Budget
    c = StubClient(['{"tool": "calc", "args": {"expr": "1"}}',
                    '{"tool": "calc", "args": {"expr": "2"}}'])
    rr = reason.solve(c, _tools_on_cfg(), system="s", user="t",
                      budget=Budget(max_tool_calls=1))
    assert "(stopped: tool-call budget exhausted" in rr.answer
    assert len(rr.steps) == 1             # only the first tool ran


def test_unlimited_by_default_preserves_existing_behaviour():
    c = StubClient(['{"tool": "calc", "args": {"expr": "1"}}', "done"])
    rr = reason.solve(c, _cfg(allow_local_tools=True), system="s", user="t")
    assert rr.answer == "done" and "(stopped:" not in rr.answer


# --- consult_specialist robustness -------------------------------------------

def _run_consult(outer_replies, specialist_replies, monkeypatch, spec_name="spec-abl:latest"):
    """Drive the consult_specialist tool with a stubbed specialist backend."""
    from ag import model
    monkeypatch.setattr(model, "ollama_has_model", lambda cfg, name, **kw: True)
    spec_client = StubClient(specialist_replies)
    monkeypatch.setattr(model, "make_client", lambda c, **kw: spec_client)
    cfg = _cfg(allow_local_tools=True, specialist_model=spec_name)
    c = StubClient(outer_replies)
    rr = reason.solve(c, cfg, system="s", user="write the exploit notes")
    return rr, spec_client


def test_specialist_answer_is_used_directly_when_clean(monkeypatch):
    rr, spec = _run_consult(
        ['{"tool": "consult_specialist", "args": {"task": "decompile x"}}',
         "Final answer."],
        ["decompiled output"], monkeypatch)
    assert rr.answer == "Final answer."
    assert "decompiled output" in str(rr.steps[0]["observation"])
    assert len(spec.prompts) == 1


def test_degenerate_specialist_output_is_retried_under_a_tighter_contract(monkeypatch):
    loop = "obj dump section .text at 0x401000 " * 30
    rr, spec = _run_consult(
        ['{"tool": "consult_specialist", "args": {"task": "decompile x"}}',
         "Final answer."],
        [loop, "concise decompile notes"], monkeypatch)
    assert len(spec.prompts) == 2
    assert "under 150 words" in spec.prompts[1]
    assert "concise decompile notes" in str(rr.steps[0]["observation"])


def test_persistent_specialist_loop_is_truncated_and_LABELLED(monkeypatch):
    loop = "same same same token soup never ends " * 30
    rr, spec = _run_consult(
        ['{"tool": "consult_specialist", "args": {"task": "x"}}', "Final."],
        [loop, loop], monkeypatch)
    obs = str(rr.steps[0]["observation"])
    assert "truncated" in obs and "use with care" in obs


# --- sub-agent model override ------------------------------------------------

def test_delegate_passes_the_model_override_through(monkeypatch):
    from ag.permissions import PermissionBroker
    seen = {}

    def fake_spawn(client, cfg, broker, *, role, task, parent_agent, depth,
                   model=""):
        seen.update(role=role, model=model)
        return agents.SubAgentResult(role=role, task=task, output="sub done")

    monkeypatch.setattr(agents, "spawn", fake_spawn)
    # spawn_agent is GATED, so it needs BOTH: the external-tools master switch
    # AND the explicit grant — the broker's two-key rule for dangerous tools.
    broker = PermissionBroker(allow_external_tools=True)
    broker.grant("spawn_agent")
    c = StubClient([
        '{"tool": "delegate", "args": {"role": "recon", "task": "map the lab", '
        '"model": "specialist"}}',
        "Scout finished."])
    rr = reason.solve(c, _tools_on_cfg(), system="s", user="recon the lab",
                      broker=broker)
    assert seen == {"role": "recon", "model": "specialist"}
    assert rr.answer == "Scout finished."


def test_model_override_never_loosens_the_grant():
    """delegate without the spawn_agent grant must stay unavailable regardless of
    any model argument — model choice is who THINKS, not what is PERMITTED."""
    from ag.permissions import PermissionBroker
    # Even with the master switch on, no spawn_agent grant means no delegate tool —
    # a model= argument changes nothing about that (it isn't even parsed here:
    # authority is decided structurally, before any model output is consulted).
    broker = PermissionBroker(allow_external_tools=True)
    tools = reason.available_tools(broker, StubClient([]), _tools_on_cfg(),
                                   agent="root", parents=(), depth=0, task="t")
    names = {t.name for t in tools}
    assert "delegate" not in names       # spawn_agent not granted → not offered


def test_model_client_resolution():
    cfg = _cfg(specialist_model="spec-abl:latest", ollama_model="qwen3:8b")
    default = object()
    # dry backend: an override would silently change the run's cost, so it is
    # declined and the default client stands.
    client, note = agents._model_client(cfg, "specialist", default)
    assert client is default
    # empty / same-as-primary resolve to the default without ceremony
    client, _ = agents._model_client(cfg, "", default)
    assert client is default
    client, _ = agents._model_client(cfg, "qwen3:8b", default)
    assert client is default
