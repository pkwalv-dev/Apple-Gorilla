"""Tests for on-demand Ollama model updates (no real network)."""
from ag import update
from ag.config import Config


def test_split_model_variants():
    assert update._split_model("qwen2.5:7b") == ("library", "qwen2.5", "7b")
    assert update._split_model("llama3.1") == ("library", "llama3.1", "latest")
    assert update._split_model("user/custom:q4") == ("user", "custom", "q4")


def test_check_unknown_when_not_installed(monkeypatch):
    monkeypatch.setattr(update, "local_manifest_digest", lambda m: None)
    s = update.check_model_update(Config(), "ghost:1b")
    assert s.state == "unknown"
    assert s.available is False
    assert "not installed" in s.reason


def test_check_unknown_when_registry_unreachable(monkeypatch):
    monkeypatch.setattr(update, "local_manifest_digest", lambda m: "sha256:aaa")
    monkeypatch.setattr(update, "remote_manifest_digest", lambda m, **k: None)
    s = update.check_model_update(Config(), "qwen2.5:7b")
    # Fails SAFE: unknown, never a false "up-to-date".
    assert s.state == "unknown"
    assert s.available is False


def test_check_up_to_date(monkeypatch):
    monkeypatch.setattr(update, "local_manifest_digest", lambda m: "sha256:same")
    monkeypatch.setattr(update, "remote_manifest_digest", lambda m, **k: "sha256:same")
    s = update.check_model_update(Config(), "qwen2.5:7b")
    assert s.state == "up-to-date"
    assert s.available is False


def test_check_update_available(monkeypatch):
    monkeypatch.setattr(update, "local_manifest_digest", lambda m: "sha256:old")
    monkeypatch.setattr(update, "remote_manifest_digest", lambda m, **k: "sha256:new")
    s = update.check_model_update(Config(), "qwen2.5:7b")
    assert s.state == "update-available"
    assert s.available is True
    d = s.as_dict()
    assert d["available"] is True and d["remote_digest"] == "sha256:new"


def test_status_dict_is_json_shaped():
    s = update.ModelUpdateStatus("m", "up-to-date", "a", "a", "ok")
    d = s.as_dict()
    for k in ("model", "state", "local_digest", "remote_digest", "reason", "available"):
        assert k in d
