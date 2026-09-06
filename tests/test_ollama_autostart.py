"""Tests for the Ollama auto-start guard (ag.model.ensure_ollama_running).

These are hermetic: reachability, exe discovery, and process spawning are all
monkeypatched, so nothing here touches a real Ollama server or launches a process.
"""
from ag.config import Config
from ag import model


def test_returns_true_without_spawning_when_already_reachable(monkeypatch):
    calls = {"spawn": 0}
    monkeypatch.setattr(model, "_ollama_reachable", lambda host, timeout=1.5: True)
    monkeypatch.setattr(model, "_find_ollama_exe", lambda: (_ for _ in ()).throw(
        AssertionError("must not look for the exe when already reachable")))
    monkeypatch.setattr(model, "_spawn_ollama_serve",
                        lambda exe: calls.__setitem__("spawn", calls["spawn"] + 1))
    assert model.ensure_ollama_running(Config()) is True
    assert calls["spawn"] == 0


def test_spawns_then_becomes_reachable(monkeypatch):
    # Not reachable on the first probe, reachable after we "start" the server.
    seq = iter([False, True])
    spawned = {"n": 0}
    monkeypatch.setattr(model, "_ollama_reachable",
                        lambda host, timeout=1.5: next(seq, True))
    monkeypatch.setattr(model, "_find_ollama_exe", lambda: "ollama")

    def fake_spawn(exe):
        spawned["n"] += 1
        return True

    monkeypatch.setattr(model, "_spawn_ollama_serve", fake_spawn)
    # First loop probe returns True (from `seq`), so we return before any sleep.
    assert model.ensure_ollama_running(Config(), timeout=5) is True
    assert spawned["n"] == 1


def test_disabled_by_config(monkeypatch):
    monkeypatch.setattr(model, "_ollama_reachable", lambda host, timeout=1.5: False)
    monkeypatch.setattr(model, "_find_ollama_exe", lambda: (_ for _ in ()).throw(
        AssertionError("must not spawn when autostart is disabled")))
    cfg = Config()
    cfg.ollama_autostart = False
    assert model.ensure_ollama_running(cfg) is False


def test_skips_remote_host(monkeypatch):
    monkeypatch.setattr(model, "_ollama_reachable", lambda host, timeout=1.5: False)
    monkeypatch.setattr(model, "_find_ollama_exe", lambda: (_ for _ in ()).throw(
        AssertionError("must not start a server for a remote host")))
    cfg = Config()
    cfg.ollama_host = "http://192.168.1.50:11434"
    assert model.ensure_ollama_running(cfg) is False


def test_returns_false_when_exe_missing(monkeypatch):
    monkeypatch.setattr(model, "_ollama_reachable", lambda host, timeout=1.5: False)
    monkeypatch.setattr(model, "_find_ollama_exe", lambda: None)
    assert model.ensure_ollama_running(Config()) is False


def test_is_local_host_matrix():
    assert model._is_local_host("http://127.0.0.1:11434")
    assert model._is_local_host("http://localhost:11434")
    assert not model._is_local_host("http://10.0.0.7:11434")


def test_make_client_ollama_invokes_guard(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(model, "ensure_ollama_running",
                        lambda cfg, **kw: called.__setitem__("n", called["n"] + 1) or True)
    cfg = Config()
    client = model.make_client(cfg, backend="ollama")
    assert isinstance(client, model.OllamaClient)
    assert called["n"] == 1
