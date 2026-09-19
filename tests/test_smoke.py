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


def test_run_is_single_call_no_self_review():
    # The self-review loop is removed: a run makes exactly ONE model call (no
    # optimize, no critique/revise). Quality now comes from the model + memory.
    from ag.model import ModelResult

    class _Counter:
        def __init__(self): self.n = 0
        def complete(self, **kw):
            self.n += 1
            return ModelResult(text="Answer.")

    fc = _Counter()
    rec = run_pipeline(fc, Config(), "hi")
    assert fc.n == 1                 # exactly one model call — no review passes
    assert rec.answer
    # Honest scorecard: speed is measured; unjudged axes are None, not a fake 0.
    assert rec.scorecard["speed"] is not None
    assert rec.scorecard["accuracy"] is None
    assert rec.scorecard["overall"] is None


def test_source_and_profile_writes_force_utf8():
    # Regression: AG's self-evolve once wrote source files with the platform default
    # encoding (cp1252 on Windows), corrupting non-ASCII (em-dash -> curly quote) in
    # ag/prompts.py. Every write_text of human text MUST pass encoding="utf-8".
    import re
    from pathlib import Path
    ag = Path(__file__).resolve().parent.parent / "ag"
    mods = [ag / m for m in ("evolve.py", "ingest.py", "backup.py")]
    mods += sorted((ag / "memory").glob("*.py"))  # memory is now a package
    for path in mods:
        src = path.read_text(encoding="utf-8")
        for m in re.finditer(r"\.write_text\(", src):
            window = src[m.start():m.start() + 220]
            assert "encoding=" in window, f"{path.name}: write_text without encoding= near {m.start()}"


def test_delegate_tool_wired_and_spawns_subagent():
    # Sub-agents are wired into the reason loop as the 'delegate' tool: with the
    # spawn_agent grant it is offered, and invoking it runs exactly one sub-agent call.
    from ag import reason
    from ag.permissions import PermissionBroker
    from ag.model import ModelResult

    class _C:
        def __init__(self): self.n = 0
        def complete(self, **kw): self.n += 1; return ModelResult(text="sub result")

    cfg = Config()
    granted = PermissionBroker(allow_external_tools=True); granted.grant("spawn_agent")
    c = _C()
    tools = {t.name: t for t in reason.available_tools(granted, c, cfg)}
    assert "delegate" in tools
    out = tools["delegate"].run({"role": "researcher", "task": "do x"}, granted)
    assert out == "sub result" and c.n == 1

    # Without the grant the tool is not offered.
    ungranted = PermissionBroker(allow_external_tools=True)
    assert "delegate" not in {t.name for t in reason.available_tools(ungranted, c, cfg)}


def test_local_tools_and_agents_on_by_default_code_exec_off():
    cfg = Config()
    assert cfg.allow_local_tools is True
    assert cfg.allow_code_exec is False


def test_saved_api_key_round_trip(tmp_path, monkeypatch):
    import os
    from ag import model
    monkeypatch.setattr(model, "_oauth_profile_dir", lambda: tmp_path)
    assert model.saved_api_key() is None
    model.save_api_key("  sk-ant-TESTKEY-1234567890  ")   # trimmed on save
    assert model.saved_api_key() == "sk-ant-TESTKEY-1234567890"
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        st = model.signin_status()
        assert st["signed_in"] and st["method"] == "api-key"
    assert model.clear_api_key() is True
    assert model.saved_api_key() is None
    try:
        model.save_api_key("")
        assert False, "empty key should raise"
    except ValueError:
        pass


def test_ollama_streaming_forwards_deltas_and_interrupts(monkeypatch):
    import json as _json
    import urllib.request as _u
    import pytest
    from ag import model
    from ag.config import Config as _Cfg

    lines = [
        _json.dumps({"message": {"content": "Hel"}}).encode(),
        _json.dumps({"message": {"content": "lo"}}).encode(),
        _json.dumps({"message": {"content": ""}, "done": True,
                     "eval_count": 2, "prompt_eval_count": 1}).encode(),
    ]

    class _Resp:
        def __init__(self, ls): self._ls = ls
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __iter__(self): return iter(self._ls)

    monkeypatch.setattr(_u, "urlopen", lambda *a, **k: _Resp(lines))
    got = []
    res = model.OllamaClient(_Cfg()).complete(
        system="s", user="u", cfg=_Cfg(), on_delta=lambda t: got.append(t))
    assert "".join(got) == "Hello"          # every chunk was streamed live
    assert res.text == "Hello" and res.output_tokens == 2

    # Interruption: raising in on_delta stops consumption immediately.
    class _Boom(Exception): pass
    monkeypatch.setattr(_u, "urlopen", lambda *a, **k: _Resp(lines))
    with pytest.raises(_Boom):
        model.OllamaClient(_Cfg()).complete(
            system="s", user="u", cfg=_Cfg(),
            on_delta=lambda t: (_ for _ in ()).throw(_Boom()))


def test_ollama_cancel_stops_stream_and_closes(monkeypatch):
    import json as _json
    import urllib.request as _u
    import pytest
    from ag import model
    from ag.config import Config as _Cfg

    canceller = model.Canceller()
    closed = {"v": False}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): closed["v"] = True; return False   # same-thread close
        def __iter__(self):
            yield _json.dumps({"message": {"content": "hi"}}).encode()
            canceller.cancel()                       # user hits Stop mid-stream
            yield _json.dumps({"message": {"content": " more"}}).encode()

    monkeypatch.setattr(_u, "urlopen", lambda *a, **k: _Resp())
    seen = []
    with pytest.raises(model.Cancelled):
        model.OllamaClient(_Cfg()).complete(
            system="s", user="u", cfg=_Cfg(),
            on_delta=lambda t: seen.append(t), cancel=canceller)
    assert seen == ["hi"]        # stopped right after the flag was set
    assert closed["v"]           # connection closed from the reading thread


def test_image_generate_saves_png(tmp_path, monkeypatch):
    import base64 as _b64, json as _json, io as _io
    import urllib.request as _u
    from ag import images
    from ag.config import Config as _Cfg
    # a 1x1 PNG
    png = _b64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return _json.dumps({"images": [_b64.b64encode(png).decode()]}).encode()

    monkeypatch.setattr(images, "IMAGES_DIR", tmp_path)
    monkeypatch.setattr(_u, "urlopen", lambda *a, **k: _Resp())
    # This stubs the A1111 API specifically, so pin that backend: with "auto" the
    # stub also answers ComfyUI's probe and AG would (correctly) prefer ComfyUI.
    import dataclasses as _dc
    res = images.generate("a red bicycle",
                          _dc.replace(_Cfg(), image_backend="a1111"))
    assert res.data_url.startswith("data:image/png;base64,")
    from pathlib import Path
    assert Path(res.path).exists() and Path(res.path).read_bytes() == png


def test_image_generate_unreachable_raises(monkeypatch):
    """No server, either backend: a clear error, never a fabricated image.

    Autostart is off here on purpose. With it on, this test would launch a real
    ComfyUI on a machine that has one installed and then block for the full startup
    window, because the stubbed urlopen never lets the readiness poll succeed.
    """
    import dataclasses as _dc
    import urllib.request as _u, urllib.error as _e
    import pytest
    from ag import images
    from ag.config import Config as _Cfg
    def boom(*a, **k): raise _e.URLError("refused")
    monkeypatch.setattr(_u, "urlopen", boom)
    for backend in ("a1111", "comfy"):
        cfg = _dc.replace(_Cfg(), image_backend=backend, sd_autostart=False,
                          comfy_autostart=False)
        with pytest.raises(RuntimeError):
            images.generate("x", cfg)


def test_ensure_sd_running(monkeypatch):
    from ag import images
    from ag.config import Config as _Cfg
    # already reachable -> no launch attempt
    monkeypatch.setattr(images, "sd_reachable", lambda cfg, timeout=1.5: True)
    monkeypatch.setattr(images, "_launch_sd", lambda cfg: (_ for _ in ()).throw(
        AssertionError("must not launch when already reachable")))
    assert images.ensure_sd_running(_Cfg()) is True

    # not reachable, autostart off -> False, no launch
    monkeypatch.setattr(images, "sd_reachable", lambda cfg, timeout=1.5: False)
    cfg = _Cfg(); cfg.sd_autostart = False
    assert images.ensure_sd_running(cfg) is False

    # remote host -> never auto-start
    cfg2 = _Cfg(); cfg2.sd_host = "http://10.0.0.5:7860"
    monkeypatch.setattr(images, "_launch_sd", lambda cfg: True)
    assert images.ensure_sd_running(cfg2) is False

    # local, launch succeeds, becomes reachable after start
    seq = iter([False, True])
    monkeypatch.setattr(images, "sd_reachable", lambda cfg, timeout=1.5: next(seq, True))
    monkeypatch.setattr(images, "_launch_sd", lambda cfg: True)
    monkeypatch.setattr(images.time, "sleep", lambda *_: None)
    assert images.ensure_sd_running(_Cfg(), timeout=5) is True


def test_generate_image_tool_offered_when_enabled():
    from ag import reason
    from ag.permissions import PermissionBroker
    cfg = Config()
    names = {t.name for t in reason.available_tools(PermissionBroker(), None, cfg)}
    assert "generate_image" in names
    cfg2 = Config(); cfg2.allow_image_gen = False
    names2 = {t.name for t in reason.available_tools(PermissionBroker(), None, cfg2)}
    assert "generate_image" not in names2


def test_ollama_empty_answer_salvaged(monkeypatch):
    # If suppression/stripping leaves nothing but the model DID emit content
    # (e.g. a truncated <think> with no answer after), don't return a blank answer.
    import json as _json
    from ag import model
    from ag.config import Config as _Cfg

    class _Resp:
        def __init__(self, payload): self._p = payload
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return _json.dumps(self._p).encode("utf-8")

    payload = {"message": {"content": "<think>partial reasoning, truncated"}}
    # complete() does `import urllib.request` internally, so patch the stdlib module.
    import urllib.request as _u
    monkeypatch.setattr(_u, "urlopen", lambda *a, **k: _Resp(payload))
    res = model.OllamaClient(_Cfg()).complete(system="s", user="u", cfg=_Cfg())
    assert res.text.strip() != ""            # not blank
    assert "<think>" not in res.text          # tags removed


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


def test_thinking_toggle_maps_per_backend():
    from ag import model
    from ag.config import Config
    off = Config(); off.think = "off"
    on = Config(); on.think = "on"
    auto = Config(); auto.think = "auto"
    # Ollama: only "on" forces the think flag; "off" uses a /no_think prompt suffix.
    # "auto" leaves the model to its own default (we never silently override it).
    assert model._ollama_think(on) is True
    assert model._ollama_think(off) is None and model._no_think(off) is True
    assert model._ollama_think(auto) is None and model._no_think(auto) is False
    # Claude: off -> disabled, otherwise adaptive
    assert model._anthropic_thinking(off) == {"type": "disabled"}
    assert model._anthropic_thinking(on) == {"type": "adaptive"}
    assert model._anthropic_thinking(auto) == {"type": "adaptive"}


def test_strip_thinking_cleans_leaked_reasoning():
    from ag.model import _strip_thinking
    # paired block removed
    assert _strip_thinking("<think>reasoning here</think>\n\n408") == "408"
    # orphaned block (opener lost, as with Ollama think=false) removed
    assert _strip_thinking("let me work it out ...\n</think>\n408") == "408"
    # ordinary answers pass through untouched
    assert _strip_thinking("just the answer") == "just the answer"


def test_backend_selection(monkeypatch):
    from ag import model
    from ag.model import make_client, DryRunClient, OllamaClient
    cfg = Config()
    assert isinstance(make_client(cfg, dry_run=True), DryRunClient)
    assert isinstance(make_client(cfg, backend="dry"), DryRunClient)
    assert isinstance(make_client(cfg, backend="ollama"), OllamaClient)
    # auto with NO creds/profile falls back to the configured offline backend, so AG
    # still answers with a real local model instead of the stub. Force the no-creds
    # condition so the test is hermetic regardless of whether this machine is logged in.
    monkeypatch.setattr(model, "_has_anthropic_creds", lambda: False)
    monkeypatch.setattr(model, "has_oauth_profile", lambda: False)
    off = Config(); off.offline_backend = "ollama"
    assert isinstance(make_client(off, backend="auto"), OllamaClient)
    dry = Config(); dry.offline_backend = "dry"
    assert isinstance(make_client(dry, backend="auto"), DryRunClient)


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


def test_ollama_payload_carries_keep_alive_and_options(monkeypatch):
    # keep_alive + tuned options must reach the Ollama request (efficiency knobs).
    import json as _json
    from ag.model import OllamaClient
    cfg = Config()
    cfg.ollama_keep_alive = "45m"
    cfg.ollama_options = {"num_ctx": 8192}
    captured = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"message":{"content":"ok"},"eval_count":1}'

    def fake_urlopen(req, timeout=0):
        captured["body"] = _json.loads(req.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    OllamaClient(cfg).complete(system="s", user="u", cfg=cfg, max_tokens=123)
    assert captured["body"]["keep_alive"] == "45m"
    assert captured["body"]["options"]["num_ctx"] == 8192
    assert captured["body"]["options"]["num_predict"] == 123


def test_config_json_on_disk_is_valid():
    from ag.config import CONFIG_PATH
    data = json.loads(CONFIG_PATH.read_text())
    assert data["autonomy_level"] in ("manual", "guarded", "never")
