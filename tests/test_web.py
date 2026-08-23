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
