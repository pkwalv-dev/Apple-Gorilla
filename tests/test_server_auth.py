"""Web UI access control.

The threat is mundane and real: `ag serve --host 0.0.0.0` on a cafe or hotel
network publishes an endpoint that runs a model, reads files through the tool
layer, and can start an evolve cycle. The design rule is that the SAFE path must
also be the DEFAULT path — the token is generated automatically rather than
being an option a user has to remember.

Loopback stays unauthenticated on purpose: a personal tool on your own machine
should not make you log in, and adding friction there would push users to
`--no-auth` habits that hurt them later.
"""
from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from ag import server
from ag.config import Config

TOKEN = "test-token-abc123"


@pytest.fixture
def running(request):
    """Start a real HTTP server with the given token (param), on a free port."""
    token = getattr(request, "param", TOKEN)
    server._Handler.cfg = Config.load()
    server._Handler.auth_token = token
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server._Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.15)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        server._Handler.auth_token = ""


def fetch(url, headers=None, method="GET", data=None):
    req = urllib.request.Request(url, headers=headers or {}, method=method, data=data)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode("utf-8", "replace"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), dict(e.headers)


# --------------------------------------------------------------------------
# is the bind address reachable?
# --------------------------------------------------------------------------

def test_loopback_addresses_are_recognised():
    for h in ("127.0.0.1", "::1", "localhost", ""):
        assert server._is_loopback(h)


def test_network_addresses_are_recognised():
    for h in ("0.0.0.0", "192.168.1.50", "10.0.0.2", "::"):
        assert not server._is_loopback(h)


# --------------------------------------------------------------------------
# enforcement
# --------------------------------------------------------------------------

def test_unauthenticated_request_is_refused(running):
    assert fetch(running + "/")[0] == 401


def test_wrong_token_is_refused(running):
    assert fetch(running + "/?token=wrong")[0] == 401


def test_api_endpoints_are_guarded_too(running):
    """Guarding only the HTML page would be theatre — the JSON endpoints are the
    ones that actually do the work."""
    for path in ("/tools", "/doctor", "/evolve/history"):
        assert fetch(running + path)[0] == 401, path


def test_post_endpoints_are_guarded(running):
    status, _, _ = fetch(running + "/bench", method="POST", data=b"{}")
    assert status == 401


def test_query_token_is_accepted(running):
    assert fetch(running + "/?token=" + TOKEN)[0] == 200


def test_bearer_header_is_accepted(running):
    status, _, _ = fetch(running + "/tools",
                         headers={"Authorization": f"Bearer {TOKEN}"})
    assert status == 200


def test_cookie_is_accepted(running):
    status, _, _ = fetch(running + "/tools", headers={"Cookie": f"ag_token={TOKEN}"})
    assert status == 200


def test_first_page_load_sets_the_cookie(running):
    """The page's own fetch() calls cannot carry the query string, so the token has
    to become a cookie on first load or the UI breaks immediately after opening."""
    status, _, headers = fetch(running + "/?token=" + TOKEN)
    assert status == 200
    assert TOKEN in headers.get("Set-Cookie", "")


def test_denial_explains_how_to_authenticate(running):
    """A bare 401 leaves the user stuck; the body has to say what to do."""
    _, body, _ = fetch(running + "/")
    assert "token" in body.lower()


@pytest.mark.parametrize("running", [""], indirect=True)
def test_empty_token_means_no_authentication(running):
    """Loopback keeps its current frictionless behaviour."""
    assert fetch(running + "/")[0] == 200
    assert fetch(running + "/tools")[0] == 200


@pytest.mark.parametrize("running", [""], indirect=True)
def test_no_cookie_is_set_when_auth_is_off(running):
    _, _, headers = fetch(running + "/")
    assert "Set-Cookie" not in headers


def test_token_comparison_is_constant_time():
    """Byte-by-byte `==` leaks the token prefix through response timing, which
    turns a 24-byte secret into a short guessing game."""
    import inspect
    src = inspect.getsource(server._Handler._authorized)
    assert "compare_digest" in src
    # The secret itself must never be compared with ==. Comparing non-secret values
    # (a cookie's NAME, say) is fine, so the check targets `auth_token` specifically.
    for line in src.splitlines():
        code = line.split("#", 1)[0]
        if "auth_token" in code and ("==" in code or "!=" in code):
            raise AssertionError(f"plain equality on the secret: {line.strip()}")


def test_page_still_renders_its_pinned_markup(running):
    """The auth change must not disturb the UI contract other tests rely on."""
    status, body, _ = fetch(running + "/?token=" + TOKEN)
    assert status == 200
    for needle in ('id="log"', "getReader", "doEvolve()", 'id="directive"'):
        assert needle in body, needle
