"""Model routing for the primary/specialist pair: security tagging, measured
benchmark evidence in the capability doc, and guidance that cites measurements.

The specialist here is the abliterated coder. The design rule these tests pin:
a low measured instruction score must change what the primary is TOLD — because
delegating an instruction-bound step to a model measured bad at instructions is
how confident wrong-format answers get made.
"""
from __future__ import annotations

import pytest

from ag import routing
from ag.config import Config


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(routing, "ROUTING_DIR", tmp_path / "routing")
    monkeypatch.setattr(routing, "PROFILES_FILE", tmp_path / "routing" / "profiles.json")
    yield


def _cfg():
    return Config(specialist_model="spec-abl:latest", ollama_model="qwen3:8b")


# --- the security tag ---------------------------------------------------------

@pytest.mark.parametrize("prompt", [
    "run a pentest on my lab network",
    "plan the recon phase for an authorised engagement",
    "is this website vulnerable to sql injection?",
    "help me solve this CTF buffer overflow challenge",
    "scan the target subnet and enumerate open ports",
    "write a threat model for this service",
])
def test_security_prompts_are_tagged(prompt):
    assert "security" in routing.tags_for(prompt)


@pytest.mark.parametrize("prompt", [
    "how do I keep my dog healthy",
    "write a poem about rain",
    "what is 17 times 23",
    "debug this python traceback",
])
def test_everyday_prompts_are_not_security(prompt):
    assert "security" not in routing.tags_for(prompt)


def test_seed_prefers_the_specialist_for_security_and_code():
    """The abliterated coder is the right brain for authorised security work: the
    instruct primary's fine-tune refuses refuse-bait phrasing, the specialist just
    does the task. (Refusal is what's routed around — tool grants are unchanged.)"""
    doc = routing.load(_cfg())
    assert doc["seed_preference"]["security"] == "specialist"
    assert doc["seed_preference"]["code"] == "specialist"
    assert routing.specialist_preferred(doc, {"security", "general"})


def test_old_saved_docs_learn_the_new_tag_without_losing_tallies():
    """A profiles file written before the 'security' tag existed must gain its
    seed preference on load while keeping the tallies it earned."""
    import json
    stale = {
        "roles": routing._SEED["roles"],
        "seed_preference": {"code": "specialist"},
        "stats": {"code": {"specialist": {"uses": 9, "wins": 8, "losses": 1}}},
        "updated": "yesterday",
    }
    routing.PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    routing.PROFILES_FILE.write_text(json.dumps(stale))
    doc = routing.load(_cfg())
    assert doc["seed_preference"]["security"] == "specialist"
    assert doc["stats"]["code"]["specialist"]["wins"] == 8


# --- measured evidence --------------------------------------------------------

def test_record_bench_credits_the_right_role():
    cfg = _cfg()
    role = routing.record_bench(cfg, "spec-abl:latest", fitness=7.5,
                                by_category={"instruction": {"passed": 4, "n": 10,
                                                             "pass_rate": 0.4}})
    assert role == "specialist"
    role = routing.record_bench(cfg, "qwen3:8b", fitness=8.0,
                                by_category={"instruction": {"passed": 9, "n": 10,
                                                             "pass_rate": 0.9}})
    assert role == "primary"
    doc = routing.load(cfg)
    assert doc["bench"]["specialist"]["fitness"] == 7.5
    assert doc["bench"]["primary"]["fitness"] == 8.0


def test_unmeasured_is_none_not_zero():
    """No measurement must never read as a 0% score — the guidance cannot assert
    a weakness it has no evidence for."""
    doc = routing.load(_cfg())
    assert routing.instruction_pass_rate(doc, "specialist") is None
    assert routing.instruction_pass_rate(doc, "primary") is None


def test_tiny_samples_are_not_a_measurement():
    cfg = _cfg()
    routing.record_bench(cfg, "spec-abl:latest", fitness=5.0,
                         by_category={"instruction": {"passed": 0, "n": 2,
                                                      "pass_rate": 0.0}})
    doc = routing.load(cfg)
    assert routing.instruction_pass_rate(doc, "specialist") is None  # n < 4


def test_low_measured_instruction_rate_enters_the_guidance():
    cfg = _cfg()
    routing.record_bench(cfg, "spec-abl:latest", fitness=6.0,
                         by_category={"instruction": {"passed": 8, "n": 20,
                                                      "pass_rate": 0.4},
                                      "arithmetic": {"passed": 20, "n": 20,
                                                     "pass_rate": 1.0}})
    text = routing.guidance(cfg, "spec-abl:latest")
    assert "Measured caution" in text and "40%" in text
    assert "strict" in text.lower() or "tightly-formatted" in text


def test_good_measured_instruction_rate_adds_no_warning():
    cfg = _cfg()
    routing.record_bench(cfg, "spec-abl:latest", fitness=9.0,
                         by_category={"instruction": {"passed": 19, "n": 20,
                                                      "pass_rate": 0.95}})
    text = routing.guidance(cfg, "spec-abl:latest")
    assert "Measured caution" not in text


def test_bench_evidence_does_not_blend_with_live_tallies():
    """The two evidence kinds stay separate: live outcomes in `stats`, measured
    runs in `bench`. Blending them would make neither auditable."""
    cfg = _cfg()
    routing.record(cfg, "specialist", {"code"}, "success")
    routing.record_bench(cfg, "spec-abl:latest", fitness=7.0,
                         by_category={"instruction": {"passed": 5, "n": 10,
                                                      "pass_rate": 0.5}})
    doc = routing.load(cfg)
    assert "bench" not in doc["stats"] and "specialist" not in str(
        doc["stats"].get("instruction", {}))
    assert doc["stats"]["code"]["specialist"]["uses"] == 1
