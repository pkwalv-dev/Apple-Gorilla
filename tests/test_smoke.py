"""Smoke tests. These gate every self-modification: if they fail against a
candidate edit, evolve.py rolls the change back automatically."""
import json

from ag.config import Config
from ag.model import make_client, extract_json
from ag.pipeline import run as run_pipeline
from ag import prompts


def test_config_roundtrip(tmp_path):
    cfg = Config()
    p = tmp_path / "config.json"
    cfg.save(p)
    loaded = Config.load(p)
    assert loaded.model == cfg.model
    assert loaded.autonomy_level in ("manual", "guarded", "never")


def test_prompts_importable_and_tagged():
    # The dry-run router depends on these role tags existing.
    assert "[role:optimizer]" in prompts.OPTIMIZER_SYSTEM
    assert "[role:critic]" in prompts.CRITIC_SYSTEM
    assert "[role:evolver]" in prompts.EVOLVER_SYSTEM


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('noise {"b": 2} tail') == {"b": 2}
    assert extract_json("no json here") is None


def test_pipeline_dry_run_produces_answer():
    cfg = Config()
    client = make_client(dry_run=True)
    rec = run_pipeline(client, cfg, "Explain why the sky is blue.")
    assert rec.answer
    assert rec.dry_run is True
    assert rec.iterations >= 0
    # dry-run critic auto-passes, so no revision loops.
    assert rec.critiques and rec.critiques[0]["verdict"] == "pass"


def test_user_context_empty_for_template(tmp_path, monkeypatch):
    # An UNFILLED template (headings + empty bullet fields) -> no context injected.
    from ag import profile
    monkeypatch.setattr(profile, "PROFILE_DIR", tmp_path)
    (tmp_path / "about_me.md").write_text(
        "# About Me (fill this in)\n"
        "> AG uses this as context\n"
        "## Who I am\n- Role / background:\n- What I'm expert in:\n"
        "## Standing preferences\n- Depth vs. brevity default:\n"
    )
    assert profile.load_user_context() == ""


def test_user_context_extracts_filled_facts(tmp_path, monkeypatch):
    from ag import profile
    monkeypatch.setattr(profile, "PROFILE_DIR", tmp_path)
    (tmp_path / "about_me.md").write_text(
        "# About Me\n"
        "> AG uses this as context\n"
        "- Role / background: senior data scientist\n"
        "- What I'm expert in:\n"          # empty field -> skipped
    )
    ctx = profile.load_user_context()
    assert "senior data scientist" in ctx
    assert "expert in" not in ctx  # empty field dropped


def test_optimizer_receives_user_context():
    # When context is present, it must reach the optimizer's user message.
    cfg = Config()
    seen = {}

    class Spy:
        def complete(self, *, system, user, cfg, max_tokens=None):
            seen["user"] = user
            from ag.model import ModelResult
            return ModelResult(text="SYSTEM:\n(none)\n\nUSER:\nx")

    from ag.pipeline import optimize
    optimize(Spy(), cfg, "do a thing", user_context="I am a marine biologist.")
    assert "marine biologist" in seen["user"]
    assert "do a thing" in seen["user"]


def test_backend_selection(monkeypatch):
    from ag import model
    from ag.model import make_client, DryRunClient, OllamaClient
    cfg = Config()
    assert isinstance(make_client(cfg, dry_run=True), DryRunClient)
    assert isinstance(make_client(cfg, backend="dry"), DryRunClient)
    assert isinstance(make_client(cfg, backend="ollama"), OllamaClient)
    # auto with NO creds/profile -> dry-run stub. Force the no-creds condition so
    # the test is hermetic regardless of whether this machine is logged in.
    monkeypatch.setattr(model, "_has_anthropic_creds", lambda: False)
    monkeypatch.setattr(model, "has_oauth_profile", lambda: False)
    assert isinstance(make_client(cfg, backend="auto"), DryRunClient)


def test_oauth_profile_detection(tmp_path, monkeypatch):
    from ag import model
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert model.has_oauth_profile() is False
    # A bare config dir (from `ant auth status`) must NOT count as logged in.
    cfgd = tmp_path / "anthropic" / "configs"
    cfgd.mkdir(parents=True)
    (cfgd / "default.json").write_text("{}")
    assert model.has_oauth_profile() is False
    # Only a real credential file counts.
    creds = tmp_path / "anthropic" / "credentials"
    creds.mkdir(parents=True)
    (creds / "default.json").write_text("{}")
    assert model.has_oauth_profile() is True


def test_ollama_client_builds_without_server():
    # Constructing the client must not require a running Ollama.
    from ag.model import OllamaClient
    cfg = Config()
    c = OllamaClient(cfg)
    assert c._model == cfg.ollama_model


def test_config_json_on_disk_is_valid():
    from ag.config import CONFIG_PATH
    data = json.loads(CONFIG_PATH.read_text())
    assert data["autonomy_level"] in ("manual", "guarded", "never")
