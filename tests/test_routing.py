"""The model-capability doc: task tagging, seeded preferences, and learning.

Hermetic: the profiles file is redirected to a tmp path, so nothing touches real state.
"""
import pytest

from ag import routing
from ag.config import Config


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(routing, "ROUTING_DIR", tmp_path / "routing")
    monkeypatch.setattr(routing, "PROFILES_FILE", tmp_path / "routing" / "profiles.json")
    yield


def _cfg():
    return Config(specialist_model="ag-coder-abliterated:latest", ollama_model="qwen3:8b")


# --- task tagging -----------------------------------------------------------
def test_tags_pick_up_task_kind():
    assert "code" in routing.tags_for("debug this python function")
    assert "creative" in routing.tags_for("write me a poem about the sea")
    assert "math" in routing.tags_for("calculate the integral of x^2")
    assert routing.tags_for("how's it going?") == {"general"}   # floor only


# --- seeding + persistence --------------------------------------------------
def test_load_seeds_and_persists():
    doc = routing.load(_cfg())
    assert "roles" in doc and "primary" in doc["roles"]
    assert routing.PROFILES_FILE.exists()          # first load wrote the seed


# --- seeded preference ------------------------------------------------------
def test_seed_prefers_specialist_for_code_only():
    doc = routing.load(_cfg())
    assert routing.specialist_preferred(doc, {"code"}) is True
    assert routing.specialist_preferred(doc, {"general"}) is False
    assert routing.specialist_preferred(doc, {"creative"}) is False


# --- guidance ---------------------------------------------------------------
def test_guidance_names_the_specialist_and_tool():
    g = routing.guidance(_cfg(), "ag-coder-abliterated:latest")
    assert "ag-coder-abliterated:latest" in g
    assert "consult_specialist" in g
    assert "code" in g                              # seeded specialist tag


# --- learning ---------------------------------------------------------------
def test_learning_flips_preference_with_evidence():
    cfg = _cfg()
    # 'creative' is seeded to the primary. Give the specialist a winning record and the
    # primary a losing one until the learned preference overrides the seed.
    for _ in range(4):
        routing.record(cfg, "specialist", {"creative"}, "rating_up")
        routing.record(cfg, "primary", {"creative"}, "rating_down")
    doc = routing.load(cfg)
    assert routing._learned_preference(doc, "creative") == "specialist"
    assert routing.specialist_preferred(doc, {"creative"}) is True


def test_success_and_failure_are_tallied():
    cfg = _cfg()
    routing.record(cfg, "primary", {"code"}, "success")
    routing.record(cfg, "primary", {"code"}, "failure")
    cell = routing.load(cfg)["stats"]["code"]["primary"]
    assert cell["uses"] == 2 and cell["wins"] == 1 and cell["losses"] == 1


def test_role_for_model_maps_specialist():
    cfg = _cfg()
    assert routing.role_for_model(cfg, cfg.specialist_model) == "specialist"
    assert routing.role_for_model(cfg, cfg.ollama_model) == "primary"


def test_unknown_signal_and_role_are_ignored():
    cfg = _cfg()
    routing.record(cfg, "bogus", {"code"}, "success")   # bad role
    routing.record(cfg, "primary", {"code"}, "bogus")   # bad signal
    assert routing.load(cfg)["stats"] == {}
