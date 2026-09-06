"""Tests for permission-gated web access (no live network — all mocked)."""
import pytest

from ag.permissions import PermissionBroker, PermissionError_
from ag.tools import web


def _granted_broker():
    b = PermissionBroker(allow_external_tools=True)
    b.grant("network")
    return b


def test_web_module_stays_egress_only_and_gated():
    # This is an INVARIANT the evolve gate enforces: web.py may self-improve its
    # search/fetch, but must never open an inbound listener or drop the broker check.
    import inspect
    src = inspect.getsource(web)
    for banned in ("socketserver", "HTTPServer", ".listen(", ".bind(", "bind(("):
        assert banned not in src, f"web.py must stay egress-only; found {banned!r}"
    # Every network entry point must still require a broker grant.
    assert "broker.require(" in inspect.getsource(web.web_fetch)
    assert "broker.require(" in inspect.getsource(web.web_search)


def test_network_is_default_denied():
    b = PermissionBroker()  # no grant, external tools off
    with pytest.raises(PermissionError_):
        web.web_fetch("https://example.com", broker=b)
    with pytest.raises(PermissionError_):
        web.web_search("anything", broker=b)


def test_fetch_requires_http_scheme():
    b = _granted_broker()
    with pytest.raises(ValueError):
        web.web_fetch("file:///etc/passwd", broker=b)


def test_strip_html_removes_tags_and_scripts():
    raw = "<html><body><script>evil()</script><p>Hello <b>world</b></p></body></html>"
    out = web.strip_html(raw)
    assert "Hello world" in out
    assert "evil" not in out
    assert "<" not in out


def test_parse_ddg_lite_extracts_results():
    page = '''
      <a rel="nofollow" class="result-link"
         href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=x">
         Example Title</a>
      <td class="result-snippet">A short snippet about the page.</td>
    '''
    results = web.parse_ddg_lite(page, max_results=5)
    assert len(results) == 1
    assert results[0].url == "https://example.com/page"
    assert "Example Title" in results[0].title
    assert "snippet" in results[0].snippet


def test_web_fetch_uses_get_when_granted(monkeypatch):
    b = _granted_broker()
    monkeypatch.setattr(web, "_get", lambda url, timeout=15.0:
                        "<p>Fetched <i>content</i></p>")
    out = web.web_fetch("https://example.com", broker=b, max_chars=100)
    assert "Fetched content" in out


def test_pipeline_injects_web_sources(monkeypatch):
    # gather_web_context is exercised; executor system must carry the sources.
    from ag import pipeline
    from ag.config import Config
    from ag.model import make_client

    monkeypatch.setattr(pipeline, "gather_web_context",
                        lambda q, broker, **kw: "SOURCE: T\nURL: https://x\nBODY")
    seen = {}
    real_complete = make_client(dry_run=True).complete

    class Spy:
        def complete(self, *, system, user, cfg, max_tokens=None):
            # capture the executor call (the one with the engineered user prompt)
            if "web" not in seen and "SOURCE:" in system:
                seen["web"] = system
            return real_complete(system=system, user=user, cfg=cfg,
                                  max_tokens=max_tokens)

    b = _granted_broker()
    pipeline.run(Spy(), Config(), "current news?", web=True, broker=b)
    assert "web" in seen and "SOURCE: T" in seen["web"]


def test_format_history_bounds_and_labels():
    from ag import pipeline
    hist = [{"role": "user", "text": "hi"}, {"role": "ai", "text": "hello"},
            {"role": "user", "text": ""}]  # blanks dropped
    out = pipeline.format_history(hist, max_turns=12)
    assert "User: hi" in out and "Apple-Gorilla: hello" in out
    assert pipeline.format_history([], max_turns=12) == ""
    # only the most recent max_turns survive
    many = [{"role": "user", "text": f"m{i}"} for i in range(20)]
    trimmed = pipeline.format_history(many, max_turns=3)
    assert "m19" in trimmed and "m0" not in trimmed


def test_pipeline_threads_conversation_history():
    # Prior turns must reach BOTH the optimizer user prompt and the executor system,
    # so AG can resolve follow-ups — the working-memory fix.
    from ag import pipeline
    from ag.config import Config
    seen = {"opt_user": "", "exec_sys": ""}

    class Spy:
        def complete(self, *, system, user, cfg, max_tokens=None):
            if "optimizer" in system.lower():
                seen["opt_user"] = user
                return type("R", (), {"text": "SYSTEM:\n(none)\n\nUSER:\ngo",
                                      "input_tokens": 0, "output_tokens": 0,
                                      "dry_run": True})()
            if "critic" in system.lower():
                return type("R", (), {"text": '{"score":9,"verdict":"pass"}',
                                      "input_tokens": 0, "output_tokens": 0})()
            seen["exec_sys"] = system
            return type("R", (), {"text": "answer", "input_tokens": 0,
                                  "output_tokens": 0, "dry_run": True})()

    hist = [{"role": "user", "text": "My name is Sam"},
            {"role": "ai", "text": "Nice to meet you Sam"}]
    cfg = Config(); cfg.auto_memory = False
    pipeline.run(Spy(), cfg, "what's my name?", web=False, history=hist)
    assert "Sam" in seen["opt_user"], "history missing from optimizer prompt"
    assert "Sam" in seen["exec_sys"], "history missing from executor system"


def test_evolve_directive_reaches_proposer():
    # A user directive must be injected into the proposer's prompt and briefing.
    from ag import evolve as evolve_mod
    from ag.config import Config
    captured = {"user": ""}

    class Spy:
        def complete(self, *, system, user, cfg, max_tokens=None):
            captured["user"] = user
            return type("R", (), {"text": '{"rationale":"x","patches":[]}',
                                  "input_tokens": 0, "output_tokens": 0})()

    cfg = Config(); cfg.fitness_gate = False
    evolve_mod.evolve(Spy(), cfg, dry_run=True,
                      directive="make answers more concise")
    assert "make answers more concise" in captured["user"]
    assert "USER DIRECTIVE" in captured["user"]


def test_propose_lists_candidates_without_applying(tmp_path, monkeypatch):
    # propose() must surface candidate patches (valid + invalid) and touch NOTHING.
    from ag import evolve as evolve_mod
    from ag.config import Config

    class Spy:
        def complete(self, *, system, user, cfg, max_tokens=None):
            payload = {"rationale": "tighten the optimizer",
                       "patches": [
                           {"path": "ag/prompts.py",
                            "new_content": "# ok\nX = 1\n"},          # valid evolvable
                           {"path": "ag/server.py",
                            "new_content": "whatever"},               # NOT evolvable
                           {"path": "config.json",
                            "new_content": "{not json"},              # invalid content
                       ]}
            import json as _j
            return type("R", (), {"text": _j.dumps(payload),
                                  "input_tokens": 0, "output_tokens": 0})()

    cfg = Config()
    res = evolve_mod.propose(Spy(), cfg, directive="tighten prompts")
    assert res.attempted and res.directive == "tighten prompts"
    by_path = {p.path: p for p in res.patches}
    assert by_path["ag/prompts.py"].valid is True
    assert by_path["ag/server.py"].valid is False        # not in evolvable set
    assert by_path["config.json"].valid is False         # invalid JSON
    # nothing was written / adopted
    assert by_path["ag/prompts.py"].diff                 # a diff preview is offered
    d = res.as_dict()
    assert d["n_valid"] == 1 and len(d["patches"]) == 3
    # summaries must NOT leak full file bodies to the client view
    assert "new_content" not in d["patches"][0]


def test_apply_selected_rejects_out_of_scope_without_writes():
    # A selection that isn't valid/evolvable must apply nothing (no disk mutation).
    from ag import evolve as evolve_mod
    from ag.config import Config
    res = evolve_mod.apply_selected(
        None, Config(),
        [{"path": "ag/server.py", "new_content": "x"},   # not evolvable
         {"path": "config.json", "new_content": "{bad"}],  # invalid
    )
    assert res.adopted is False and res.rolled_back is False
    assert "no valid selected" in res.reason
