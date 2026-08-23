"""Tests for the built-in web app (no socket is bound)."""
from ag import server
from ag.config import Config


def test_page_is_valid_html():
    assert server.PAGE.strip().startswith("<!doctype html>")
    assert "Apple-Gorilla" in server.PAGE
    assert "/run" in server.PAGE  # the UI posts prompts here


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
