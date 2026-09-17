"""Tests for directed capability acquisition: skill registry, gating, sub-agents."""
import json
from types import SimpleNamespace

import pytest

from ag.config import Config
from ag.permissions import PermissionBroker
from ag.skills import Skill, SkillRegistry, get_registry, load_tools


# --- a fake model that returns a canned skill (code + test) -----------------
GOOD_SKILL = {
    "name": "increment",
    "description": "add one to a number",
    "arg": "x",
    "capabilities": [],
    "deps": [],
    "code": "def run(args, broker=None):\n    return str(int(args.get('x', 0)) + 1)\n",
    "test": "def test_skill():\n    from skill import run\n    assert run({'x': 41}) == '42'\n",
}
BAD_SKILL = {
    **GOOD_SKILL, "name": "broken",
    "test": "def test_skill():\n    from skill import run\n    assert run({'x': 1}) == '999'\n",
}


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, **kwargs):
        return SimpleNamespace(text=json.dumps(self.payload),
                               input_tokens=0, output_tokens=0)


def _reg(tmp_path, agent="root", parents=()):
    return SkillRegistry(agent, parents=parents, base_dir=tmp_path)


# --- registry ---------------------------------------------------------------
def test_register_list_get_remove(tmp_path):
    reg = _reg(tmp_path)
    reg.register(Skill(name="s1", description="d", arg="x"),
                 "def run(args, broker=None):\n    return 'ok'\n",
                 "def test_skill():\n    assert True\n")
    assert any(s.name == "s1" for s in reg.list())
    assert reg.get("s1").description == "d"
    assert reg.set_enabled("s1", False) is True
    assert reg.get("s1").enabled is False
    assert reg.remove("s1") is True
    assert reg.get("s1") is None


def test_lineage_inheritance_and_promote(tmp_path):
    shared = _reg(tmp_path, agent="shared")
    shared.register(Skill(name="inherited", description="from shared"),
                    "def run(args, broker=None):\n    return 'x'\n",
                    "def test_skill():\n    assert True\n")
    child = _reg(tmp_path, agent="child", parents=["root"])
    assert any(s.name == "inherited" for s in child.list())   # inherits shared tier

    child.register(Skill(name="local", description="child only", agent="child"),
                   "def run(args, broker=None):\n    return 'y'\n",
                   "def test_skill():\n    assert True\n")
    child.promote("local", to="shared")
    assert any(s.name == "local" for s in _reg(tmp_path, agent="sibling").list())


# --- skill -> tool adapter --------------------------------------------------
def test_load_tools_respects_capabilities(tmp_path):
    reg = _reg(tmp_path)
    reg.register(Skill(name="free", description="no grant", arg="x"),
                 "def run(args, broker=None):\n    return str(args.get('x'))\n",
                 "def test_skill():\n    assert True\n")
    reg.register(Skill(name="netskill", description="needs net", capabilities=["network"]),
                 "def run(args, broker=None):\n    return 'net'\n",
                 "def test_skill():\n    assert True\n")
    broker = PermissionBroker(allow_external_tools=True)  # no grants
    tools = {t.name: t for t in load_tools(broker, registry=reg)}
    assert "free" in tools and "netskill" not in tools
    assert tools["free"].run({"x": "5"}, broker) == "5"    # adapter executes run()
    broker.grant("network")
    assert "netskill" in {t.name for t in load_tools(broker, registry=reg)}


# --- acquisition flow -------------------------------------------------------
def _patch_state(monkeypatch, tmp_path):
    from ag import config as cfgmod
    from ag.skills import registry as regmod
    from ag.memory import manager as mgrmod
    from ag.memory.embed import HashingEmbedder
    monkeypatch.setattr(cfgmod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(regmod, "SkillRegistry",
                        lambda *a, **k: SkillRegistry(*a, base_dir=tmp_path, **{
                            x: v for x, v in k.items() if x != "base_dir"}))
    monkeypatch.setattr(mgrmod, "get_embedder", lambda cfg=None, **k: HashingEmbedder())
    import ag.memory as memory
    memory.reset()


def test_acquire_auto_authors_and_registers(tmp_path, monkeypatch):
    from ag import acquire
    _patch_state(monkeypatch, tmp_path)
    cfg = Config(acquisition_autonomy="auto")
    broker = PermissionBroker(allow_external_tools=True)
    broker.grant("write_skill")
    res = acquire.author_skill(FakeClient(GOOD_SKILL), cfg, "add one to a number",
                               broker=broker)
    assert res.acquired and res.name == "increment"
    assert any(s.name == "increment" for s in get_registry("root").list())


def test_acquire_ask_mode_needs_approval(tmp_path, monkeypatch):
    from ag import acquire
    _patch_state(monkeypatch, tmp_path)
    cfg = Config(acquisition_autonomy="ask")
    broker = PermissionBroker(allow_external_tools=True)
    broker.grant("write_skill")
    res = acquire.author_skill(FakeClient(GOOD_SKILL), cfg, "add one", broker=broker)
    assert res.attempted and not res.acquired and res.needs_approval
    # Nothing was registered in ask mode.
    assert not any(s.name == "increment" for s in get_registry("root").list())


def test_acquire_discards_skill_that_fails_its_test(tmp_path, monkeypatch):
    from ag import acquire
    _patch_state(monkeypatch, tmp_path)
    cfg = Config(acquisition_autonomy="auto")
    broker = PermissionBroker(allow_external_tools=True)
    broker.grant("write_skill")
    res = acquire.author_skill(FakeClient(BAD_SKILL), cfg, "broken", broker=broker)
    assert res.attempted and not res.acquired
    assert not any(s.name == "broken" for s in get_registry("root").list())


def test_acquire_refused_without_grant(tmp_path, monkeypatch):
    from ag import acquire
    _patch_state(monkeypatch, tmp_path)
    cfg = Config(acquisition_autonomy="auto")
    broker = PermissionBroker(allow_external_tools=True)   # no write_skill grant
    res = acquire.author_skill(FakeClient(GOOD_SKILL), cfg, "x", broker=broker)
    assert not res.acquired and "write_skill" in res.reason


# --- reason loop offers acquire_skill only when granted ---------------------
def test_reason_offers_acquire_when_granted():
    from ag import reason
    cfg = Config()
    granted = PermissionBroker(allow_external_tools=True)
    granted.grant("write_skill")
    assert "acquire_skill" in {t.name for t in reason.available_tools(granted, None, cfg)}
    ungranted = PermissionBroker(allow_external_tools=True)
    assert "acquire_skill" not in {t.name for t in reason.available_tools(ungranted, None, cfg)}


# --- sub-agent authority capping + depth bound ------------------------------
def test_child_broker_caps_and_depth_bounds():
    from ag.agents import _child_broker
    parent = PermissionBroker(allow_external_tools=True)
    for c in ("write_skill", "spawn_agent", "network"):
        parent.grant(c)
    # At the depth limit the child loses spawn_agent (no runaway recursion)...
    capped = _child_broker(parent, can_spawn=False)
    assert "spawn_agent" not in capped.grants and "write_skill" in capped.grants
    # ...but never gains authority the parent lacked.
    assert capped.grants <= parent.grants


# --- installer + github gates -----------------------------------------------
def test_pkg_spec_validation_and_gate():
    from ag.tools import pkg
    assert pkg.valid_spec("requests==2.31.0") and not pkg.valid_spec("evil; rm -rf /")
    broker = PermissionBroker(allow_external_tools=True)   # install_package not granted
    with pytest.raises(Exception):
        pkg.install(["requests"], broker=broker)


def test_github_allowlist_and_gate():
    from ag.tools import github
    assert github.allowed("ollama/ollama", ["ollama/ollama"])
    assert not github.allowed("evil/repo", ["ollama/ollama"])
    broker = PermissionBroker(allow_external_tools=True)   # github_fetch not granted
    with pytest.raises(Exception):
        github.fetch("ollama/ollama", "README.md", broker=broker, allowlist=["ollama/ollama"])
    broker.grant("github_fetch")
    out = github.fetch("evil/repo", "x", broker=broker, allowlist=["ollama/ollama"])
    assert "refused" in out
