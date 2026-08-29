"""Tests for the built-in web app (no socket is bound)."""
from ag import server
from ag.config import Config


def test_page_is_valid_html():
    assert server.PAGE.strip().startswith("<!doctype html>")
    assert "Apple-Gorilla" in server.PAGE
    assert "/run" in server.PAGE  # the UI posts prompts here


def test_page_has_realtime_log_and_scorecard_ui():
    # The live log + scorecard + tools UI must be present (the whole point of the GUI).
    for hook in ("id=\"log\"", "getReader", "/tools", "scorecard", "accuracy"):
        assert hook in server.PAGE, f"GUI missing {hook!r}"


def test_page_uses_the_theme():
    # The evolvable theme (professional font + palette) must be wired into the page,
    # with no unsubstituted placeholders.
    from ag import theme
    assert "Inter" in server.PAGE and "fonts.googleapis" in server.PAGE
    assert "--accent" in server.PAGE  # design tokens present
    assert "__THEME__" not in server.PAGE and "__FONTS__" not in server.PAGE
    assert theme.THEME_CSS in server.PAGE


def test_page_has_update_banner_and_check():
    # The one-click yes/no model-update flow must be wired into the page.
    for hook in ("id=\"update\"", "/update/check", "applyUpdate", "/update/apply"):
        assert hook in server.PAGE, f"update UI missing {hook!r}"


def test_page_has_self_improvement_ui():
    # The self-improvement commands (bench/evolve/history/status) must be in the GUI.
    for hook in ("/bench", "/evolve", "doEvolve()", "doBench()", "loadHistory()",
                 "loadDoctor()", "Benchmark", "Evolve", "proposer"):
        assert hook in server.PAGE, f"self-improvement UI missing {hook!r}"


def test_doctor_data_shape():
    d = server._doctor_data(Config())
    for key in ("backend", "effective_backend", "fitness_gate", "bench_tasks",
                "bench_samples", "evolve_gate", "snapshots", "ollama_reachable"):
        assert key in d, f"doctor data missing {key!r}"
    assert d["bench_tasks"] >= 10
    assert isinstance(d["ollama_reachable"], bool)


def test_build_broker_respects_allow_web():
    on = Config(); on.allow_web = True
    off = Config(); off.allow_web = False
    assert server._build_broker(on) is not None
    assert server._build_broker(off) is None


def test_run_prompt_dry_run_produces_answer(monkeypatch):
    # Force the dry-run stub so no backend/network is needed.
    import ag.server as S
    monkeypatch.setattr(S, "make_client", lambda cfg: __import__(
        "ag.model", fromlist=["make_client"]).make_client(dry_run=True))
    cfg = Config()
    cfg.allow_web = False  # skip web retrieval in the test
    answer, meta = server.run_prompt(cfg, "hello there")
    assert answer
    assert "dry_run" in meta
