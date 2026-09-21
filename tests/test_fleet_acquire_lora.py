"""Phase 3 trio: parallel sub-agents, reuse-first acquisition, verified LoRA rows.

Each of these is a *process* guarantee — easy to break silently, invisible in a
single-threaded happy path:
- parallel fan-out must not reorder results, must not lose fleet records, and one
  child's crash must not take down the batch;
- acquisition must prefer an existing skill over authoring a new near-duplicate;
- a LoRA dataset must never contain a refusal or a repetition loop — those train
  the adapter to do exactly that.
"""
from __future__ import annotations

import dataclasses
import json
import threading
import time
from types import SimpleNamespace

import pytest

from ag import agents, fleet, lora
from ag.config import Config
from ag.permissions import PermissionBroker


def _cfg(**kw):
    return dataclasses.replace(Config(), backend="dry", **kw)


def _broker():
    b = PermissionBroker(allow_external_tools=True)
    b.grant("spawn_agent")
    return b


# ---------------------------------------------------------------------------
# parallel spawn_many
# ---------------------------------------------------------------------------

def test_spawn_many_preserves_input_order_under_staggered_completion(monkeypatch):
    """Completion order (slow→mid→fast) differs from input order; the result list
    must track INPUT order anyway."""
    def fake_spawn(client, cfg, broker, *, role, task, parent_agent, depth,
                   model=""):
        time.sleep(0.02 if task == "slow" else 0.25 if task == "fast" else 0.05)
        return agents.SubAgentResult(role=role, task=task, output=f"did {task}")
    monkeypatch.setattr(agents, "spawn", fake_spawn)
    tasks = [{"role": "a", "task": "fast"}, {"role": "b", "task": "slow"},
             {"role": "c", "task": "mid"}]
    out = agents.spawn_many(None, _cfg(), _broker(), tasks, workers=3)
    assert [r.task for r in out] == ["fast", "slow", "mid"]   # input order, always
    assert [r.output for r in out] == ["did fast", "did slow", "did mid"]


def test_spawn_many_is_actually_parallel(monkeypatch):
    def fake_spawn(client, cfg, broker, *, role, task, parent_agent, depth,
                   model=""):
        time.sleep(0.15)
        return agents.SubAgentResult(role=role, task=task, output="ok")
    monkeypatch.setattr(agents, "spawn", fake_spawn)
    tasks = [{"role": "w", "task": f"t{i}"} for i in range(4)]
    t0 = time.monotonic()
    agents.spawn_many(None, _cfg(), _broker(), tasks, workers=4)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.45, f"4 x 0.15s took {elapsed:.2f}s — ran serially"


def test_one_crash_fails_one_child_not_the_batch(monkeypatch):
    def fake_spawn(client, cfg, broker, *, role, task, parent_agent, depth,
                   model=""):
        if task == "boom":
            raise RuntimeError("backend died")
        return agents.SubAgentResult(role=role, task=task, output=f"fine {task}")
    monkeypatch.setattr(agents, "spawn", fake_spawn)
    tasks = [{"role": "a", "task": "t1"}, {"role": "b", "task": "boom"},
             {"role": "c", "task": "t3"}]
    out = agents.spawn_many(None, _cfg(), _broker(), tasks, workers=3)
    assert len(out) == 3
    assert out[0].output == "fine t1" and out[2].output == "fine t3"
    assert "sub-agent failed" in out[1].output and "backend died" in out[1].output


def test_spawn_many_passes_per_task_model_overrides(monkeypatch):
    seen = {}
    def fake_spawn(client, cfg, broker, *, role, task, parent_agent, depth,
                   model=""):
        seen[task] = model
        return agents.SubAgentResult(role=role, task=task, output="ok")
    monkeypatch.setattr(agents, "spawn", fake_spawn)
    tasks = [{"role": "scout", "task": "recon", "model": "specialist"},
             {"role": "writer", "task": "report"}]
    agents.spawn_many(None, _cfg(), _broker(), tasks, workers=2)
    assert seen == {"recon": "specialist", "report": ""}


def test_spawn_many_still_requires_the_grant():
    with pytest.raises(Exception):
        agents.spawn_many(None, _cfg(), PermissionBroker(allow_external_tools=True),
                          [{"task": "x"}])


def test_fleet_records_survive_a_concurrent_spawn_storm(tmp_path, monkeypatch):
    """Without the lock, load-modify-write on agents.jsonl loses records."""
    monkeypatch.setattr(fleet, "FLEET_DIR", tmp_path / "fleet")
    monkeypatch.setattr(fleet, "AGENTS_FILE", tmp_path / "fleet" / "agents.jsonl")
    monkeypatch.setattr(fleet, "STOP_FILE", tmp_path / "fleet" / "STOP")
    n = 16
    with __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: fleet.record_spawn(f"root.w{i}", role="w"),
                      range(n)))
    names = {r.agent for r in fleet.list_agents()}
    assert names == {f"root.w{i}" for i in range(n)}, \
        f"lost records: {n - len(names)}"


# ---------------------------------------------------------------------------
# reuse-first acquisition
# ---------------------------------------------------------------------------
from ag import acquire
from ag.skills import Skill


class _FakeRegistry:
    def __init__(self, skills):
        self._skills = skills

    def list(self, *, include_disabled=True):
        return list(self._skills)


def _skill(name, desc):
    return Skill(name=name, description=desc, arg="x", args=["x"],
                 agent="root", source="authored")


def test_find_reusable_matches_a_covering_skill():
    reg = _FakeRegistry([_skill("csv_to_json", "convert a CSV file to JSON")])
    hit = acquire.find_reusable(reg, "convert my CSV file into JSON")
    assert hit and hit.name == "csv_to_json"


@pytest.mark.parametrize("spec", [
    "turn a PDF into a podcast",                 # unrelated domain
    "resize an image to thumbnail size",
    "two",                                       # too few distinctive tokens
])
def test_find_reusable_rejects_non_covering_requests(spec):
    reg = _FakeRegistry([_skill("csv_to_json", "convert a CSV file to JSON")])
    assert acquire.find_reusable(reg, spec) is None


def test_author_reuses_instead_of_calling_the_model(tmp_path, monkeypatch):
    """When a covering skill exists, NO model call may happen — reuse is free."""
    reg = _FakeRegistry([_skill("csv_to_json", "convert a CSV file to JSON")])
    monkeypatch.setattr(acquire, "get_registry", lambda *a, **k: reg)

    class ExplodingClient:
        def complete(self, **kw):
            raise AssertionError("the model must not be called when reuse hits")

    b = PermissionBroker(allow_external_tools=True)
    b.grant("write_skill")
    r = acquire.author_skill(ExplodingClient(), _cfg(),
                             "convert my CSV export into JSON", broker=b)
    assert r.acquired and r.name == "csv_to_json" and "reusing" in r.reason


def test_author_still_authors_when_nothing_covers(tmp_path, monkeypatch):
    """The counterweight: reuse must not become a veto on all new skills."""
    reg = _FakeRegistry([_skill("csv_to_json", "convert a CSV file to JSON")])
    monkeypatch.setattr(acquire, "get_registry", lambda *a, **k: reg)
    called = {"n": 0}

    class NoSkillClient:
        def complete(self, **kw):
            called["n"] += 1
            return SimpleNamespace(text='{"name": "x", "code": "", "test": ""}',
                                   input_tokens=1, output_tokens=1)

    b = PermissionBroker(allow_external_tools=True)
    b.grant("write_skill")
    r = acquire.author_skill(NoSkillClient(), _cfg(),
                             "reverse engineer this binary protocol", broker=b)
    assert called["n"] == 1                    # the model WAS asked
    assert not r.acquired                      # (authoring failed later; fine)


# ---------------------------------------------------------------------------
# verified LoRA teacher rows
# ---------------------------------------------------------------------------

def test_clean_teacher_rows_are_kept(monkeypatch, tmp_path):
    monkeypatch.setattr(lora, "LORA_DIR", tmp_path / "lora")
    monkeypatch.setattr(lora, "DATASET_FILE", tmp_path / "lora" / "dataset.jsonl")
    monkeypatch.setattr(lora, "ADAPTERS_DIR", tmp_path / "lora" / "adapters")
    assert lora._teacher_row_ok("A precise answer.") is None
    st = lora.build_dataset(Config(), use_memory=False, use_teacher=True,
                            client=SimpleNamespace(
                                complete=lambda **k: SimpleNamespace(
                                    text="A precise answer.", input_tokens=0,
                                    output_tokens=0)))
    assert st.from_teacher > 0 and st.total == st.from_teacher


@pytest.mark.parametrize("bad", [
    "I'm sorry, but I can't help with that request.",
    "loop unit seven nine delta " * 40,
    "",
])
def test_unverified_teacher_rows_never_enter_the_dataset(bad):
    assert lora._teacher_row_ok(bad) is not None


def test_refusing_teacher_yields_zero_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(lora, "LORA_DIR", tmp_path / "lora")
    monkeypatch.setattr(lora, "DATASET_FILE", tmp_path / "lora" / "dataset.jsonl")
    monkeypatch.setattr(lora, "ADAPTERS_DIR", tmp_path / "lora" / "adapters")
    st = lora.build_dataset(Config(), use_memory=False, use_teacher=True,
                            client=SimpleNamespace(
                                complete=lambda **k: SimpleNamespace(
                                    text="I'm sorry, but I can't do that.",
                                    input_tokens=0, output_tokens=0)))
    assert st.total == 0


def test_old_poisoned_rows_are_filtered_on_reuse(tmp_path, monkeypatch):
    """A row written before verification existed must not survive by being old."""
    monkeypatch.setattr(lora, "LORA_DIR", tmp_path / "lora")
    monkeypatch.setattr(lora, "DATASET_FILE", tmp_path / "lora" / "dataset.jsonl")
    monkeypatch.setattr(lora, "ADAPTERS_DIR", tmp_path / "lora" / "adapters")
    lora._ensure()
    rows = [
        {"instruction": "a", "output": "I'm sorry, but I can't help.", "source": "teacher"},
        {"instruction": "b", "output": "A precise answer.", "source": "teacher"},
    ]
    lora.DATASET_FILE.write_text("".join(json.dumps(r) + "\n" for r in rows))
    kept = lora._existing_teacher_pairs()
    assert len(kept) == 1 and kept[0]["instruction"] == "b"


# ---------------------------------------------------------------------------
# capability statement: evidence, or an honest gap
# ---------------------------------------------------------------------------

def test_capability_file_says_unmeasured_when_unmeasured(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from ag import routing
    from ag.cli import cmd_capability
    monkeypatch.setattr(routing, "ROUTING_DIR", tmp_path / "routing")
    monkeypatch.setattr(routing, "PROFILES_FILE", tmp_path / "routing" / "p.json")
    out = tmp_path / "CAP.md"
    assert cmd_capability(NS(out=str(out))) == 0
    text = out.read_text(encoding="utf-8")
    assert "## Models" in text and "## Authority" in text
    assert "none yet" in text          # no bench --record has run here
    assert "overblown" not in text     # control: it does not praise itself


def test_capability_file_cites_measured_scores(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from ag import routing
    from ag.cli import cmd_capability
    monkeypatch.setattr(routing, "ROUTING_DIR", tmp_path / "routing")
    monkeypatch.setattr(routing, "PROFILES_FILE", tmp_path / "routing" / "p.json")
    routing.record_bench(Config(), "qwen3:8b", fitness=8.25, by_category={})
    out = tmp_path / "CAP.md"
    cmd_capability(NS(out=str(out)))
    assert "8.25/10" in out.read_text(encoding="utf-8")
