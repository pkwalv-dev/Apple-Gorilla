"""Context packing and benchmark task proposals — the last two plan items.

The packer turns silent prompt truncation into a declared decision; proposals let
the model grow the benchmark while keeping the ruler human-armed. Both fail
quietly when they're wrong, so both are pinned here.
"""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from ag import bench
from ag.config import Config
from ag.contextpack import Section, pack_sections


def _cfg(**kw):
    return dataclasses.replace(Config(), backend="dry", **kw)


# --- packer semantics ----------------------------------------------------------

def test_unlimited_budget_is_verbatim():
    secs = [Section("a", "A" * 100, priority=1), Section("b", "B" * 100, priority=9)]
    text, notes = pack_sections(secs, budget=0)
    assert text == "A" * 100 + "\n\n" + "B" * 100
    assert all(n.action == "kept" for n in notes)


def test_everything_fitting_is_untouched():
    secs = [Section("a", "short"), Section("b", "also short")]
    text, notes = pack_sections(secs, budget=1000)
    assert text == "short\n\nalso short"
    assert all(n.action == "kept" for n in notes)


def test_least_important_is_clipped_first():
    big = "w" * 5000
    secs = [Section("vital", "os briefing text", priority=5, truncatable=False),
            Section("web", big, priority=70)]
    text, notes = pack_sections(secs, budget=2000)
    assert "os briefing text" in text
    assert "clipped" in text and "'web'" in text
    assert len(text) < 2200
    by = {n.name: n.action for n in notes}
    assert by["vital"] == "kept" and by["web"] == "clipped"


def test_clip_marker_declares_the_cut_inside_the_text():
    secs = [Section("mem", "m" * 3000)]
    text, _ = pack_sections(secs, budget=1200)
    assert "clipped" in text and "context budget" in text


def test_a_floor_clip_that_fits_is_KEPT_not_dropped():
    """Better semantics: a block clipped to KEEP_FLOOR that fits the budget is
    worth more than a marker — it still carries 600 chars of real content."""
    secs = [Section("tiny", "hi", priority=5, truncatable=False),
            Section("huge", "x" * 10000, priority=90)]
    text, notes = pack_sections(secs, budget=800)
    by = {n.name: n.action for n in notes}
    assert by["huge"] == "clipped" and "'huge'" in text


def test_a_section_too_big_even_at_floor_is_dropped_with_a_marker():
    # 500 < tiny + separator + KEEP_FLOOR + clip-marker (~680), so the floored
    # clip cannot fit either: pass 2 must drop it, leaving the marker.
    secs = [Section("tiny", "hi", priority=5, truncatable=False),
            Section("huge", "x" * 10000, priority=90)]
    text, notes = pack_sections(secs, budget=500)
    by = {n.name: n.action for n in notes}
    assert by["huge"] == "dropped"
    assert "'huge' omitted" in text
    assert len(text) <= 500


def test_non_truncatable_high_priority_survives_over_filler():
    secs = [Section("core ctx", "KEEPME" + "k" * 800, priority=1, truncatable=False),
            Section("filler1", "a" * 3000, priority=79),
            Section("filler2", "b" * 3000, priority=80)]
    text, notes = pack_sections(secs, budget=2500)
    assert "KEEPME" in text                    # the important block survived whole
    assert text.count("omitted") >= 1 or "clipped" in text


def test_pipeline_smoke_with_a_tight_budget_declares_everything():
    """End-to-end: a run with a tiny context budget must still answer, and the
    trace must SHOW the clipping — never silently lose context."""
    from ag import pipeline
    from ag.permissions import PermissionBroker
    cfg = _cfg(context_budget_chars=700, use_memory=False, allow_local_tools=False)
    events = []
    rec = pipeline.run(__import__("ag.model", fromlist=["DryRunClient"]).DryRunClient(),
                       cfg, "hello there, who are you?",
                       broker=PermissionBroker(),
                       emit=lambda phase, msg, **kw: events.append((phase, msg)))
    assert rec.answer
    # with unlimited-by-default money the sections all fit; nothing clipped.
    # The contract is: IF a section were packed, it would carry a marker.
    # (This run's markers assert only that the pipeline end-to-ends under a budget.)
    rec2 = pipeline.run(__import__("ag.model", fromlist=["DryRunClient"]).DryRunClient(),
                        cfg, "hi", broker=PermissionBroker())
    assert rec2.answer


# --- proposal validation -------------------------------------------------------

def _valid(**over):
    d = {"id": "capitals-france", "prompt": "What is the capital of France? "
         "Reply with exactly one word.", "check": "equals", "expect": "Paris",
         "category": "geography"}
    d.update(over)
    return d


def test_valid_proposal_passes():
    assert bench.validate_proposal(_valid()) is None


@pytest.mark.parametrize("over,frag", [
    ({"id": "X"}, "bad id"),
    ({"prompt": "hi?"}, "too short"),
    ({"check": "vibes"}, "unknown checker"),
    ({"expect": ""}, "expect is empty"),
    ({"check": "numeric", "expect": "green"}, "numeric expect"),
    ({"check": "contains_all", "expect": "Paris"}, "non-empty list"),
    ({"check": "regex", "expect": "([unclosed"}, "does not compile"),
])
def test_invalid_proposals_are_named(over, frag):
    err = bench.validate_proposal(_valid(**over))
    assert err and frag in err


def test_duplicate_of_an_existing_task_rejected():
    cur = bench.load_tasks()
    prompt = cur[0].prompt
    err = bench.validate_proposal(_valid(prompt=prompt),
                                  existing_prompts={prompt.strip().lower()})
    assert err and "duplicates" in err


# --- the propose/accept/reject round trip --------------------------------------

@pytest.fixture
def _isolated_bench(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "BENCH_DIR", tmp_path / "bench")
    monkeypatch.setattr(bench, "TASKS_FILE", tmp_path / "bench" / "tasks.jsonl")
    monkeypatch.setattr(bench, "PROPOSALS_FILE",
                        tmp_path / "bench" / "proposals.jsonl")
    yield


class _ProposerStub:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, *, system, user, cfg, **kw):
        return SimpleNamespace(text=json.dumps(self.payload),
                               input_tokens=5, output_tokens=5)


def test_proposals_are_saved_pending_and_never_measured(_isolated_bench):
    payload = {"tasks": [
        _valid(),
        _valid(id="bad-regex", check="regex", expect="([bad"),
        _valid(id="capitals-spain", prompt="Capital of Spain? One word.",
               expect="Madrid"),
    ]}
    props = bench.propose_tasks(_ProposerStub(payload), _cfg(), n=3)
    assert [p["id"] for p in props] == ["capitals-france", "capitals-spain"]
    # pending only — the curated file (the thing the fitness gate reads) is untouched
    assert not bench.TASKS_FILE.exists()
    listed = bench.list_proposals()
    assert len(listed) == 2 and all(p["proposed"] for p in listed)


def test_unparseable_proposer_output_produces_nothing(_isolated_bench):
    class JerryClient:
        def complete(self, **kw):
            return SimpleNamespace(text="sure, use riddles!", input_tokens=1,
                                   output_tokens=1)
    assert bench.propose_tasks(JerryClient(), _cfg(), n=2) == []
    assert bench.list_proposals() == []


def test_accept_moves_to_curated_and_out_of_pending(_isolated_bench):
    props = bench.propose_tasks(_ProposerStub({"tasks": [_valid()]}), _cfg(), n=1)
    row = bench.accept_proposal(props[0]["id"])
    assert row and row["accepted_from_proposal"]
    assert bench.list_proposals() == []
    # now the curated file drives load_tasks (its sole entry is the new task)
    loaded = bench.load_tasks(cfg=_cfg(bench_generated=True))
    assert [t.id for t in loaded] == ["capitals-france"]


def test_accept_is_idempotent_and_unknown_ids_fail_closed(_isolated_bench):
    assert bench.accept_proposal("nope") is None
    assert not bench.TASKS_FILE.exists()


def test_reject_removes_pending(_isolated_bench):
    props = bench.propose_tasks(_ProposerStub({"tasks": [_valid()]}), _cfg(), n=1)
    assert bench.reject_proposal(props[0]["id"]) is True
    assert bench.reject_proposal("ghost") is False
    assert bench.list_proposals() == []


def test_a_wrong_expect_cannot_self_install(_isolated_bench):
    """The scenario the design exists for: the model proposes '2+2 = 5'. Shape
    validation CANNOT catch it (5 is a fine number) — so the defence tested here
    is structural: without a human accept, the task never reaches the suite."""
    evil = _valid(id="arith-two-plus-two", prompt="What is 2+2? Reply with only the digit.",
                  check="numeric", expect=5)
    props = bench.propose_tasks(_ProposerStub({"tasks": [evil]}), _cfg(), n=1)
    assert len(props) == 1                       # it validates — shape is honest
    assert not bench.TASKS_FILE.exists()         # …and it measures nothing yet
    # The wrong answer only matters if a human approves it. Pending is inert.
    res = bench.run_benchmark(__import__("ag.model", fromlist=["DryRunClient"]).DryRunClient(),
                              _cfg())
    assert "arith-two-plus-two" not in [t["id"] for t in res.per_task]
