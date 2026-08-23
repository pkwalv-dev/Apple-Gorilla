"""Permission-gated web access for AG (stdlib only, no API key).

Gives the model internet reach via retrieval: `web_search` + `web_fetch`, both
gated through the PermissionBroker's 'network' capability. This is how the local
Ollama model "interacts with the internet" — AG searches/fetches on its behalf and
injects the results as reference context.

SECURITY: fetched web content is UNTRUSTED. It is injected as reference *data*,
explicitly labeled, and must never be treated as instructions (prompt-injection
defense). Network is default-deny; a caller must hold a 'network' grant.
"""
from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List

from ..permissions import PermissionBroker

_UA = "Mozilla/5.0 (compatible; Apple-Gorilla/0.1)"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


def _get(url: str, timeout: float = 15.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        charset = r.headers.get_content_charset() or "utf-8"
        return r.read().decode(charset, errors="replace")


def strip_html(raw: str, max_chars: int = 5000) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def parse_ddg_lite(page: str, max_results: int = 5) -> List[SearchResult]:
    """Tolerant parser for DuckDuckGo's lite/HTML results.

    Result anchors carry a `//duckduckgo.com/l/?uddg=<encoded-real-url>` href.
    We decode the real URL and pair it with the anchor text and any snippet.
    """
    results: List[SearchResult] = []
    # Anchor tags whose href routes through DDG's /l/?uddg= redirect.
    for m in re.finditer(
        r'<a[^>]+href="([^"]*uddg=[^"]+)"[^>]*>(.*?)</a>', page, re.IGNORECASE | re.DOTALL
    ):
        href, title = m.group(1), strip_html(m.group(2), 200)
        qs = urllib.parse.urlparse(href).query
        params = urllib.parse.parse_qs(qs)
        real = params.get("uddg", [None])[0]
        if real and title:
            results.append(SearchResult(title=title, url=real))
        if len(results) >= max_results:
            break
    # Attach snippets in document order when present.
    snippets = [strip_html(s, 300) for s in re.findall(
        r'class="result-snippet"[^>]*>(.*?)</td>', page, re.IGNORECASE | re.DOTALL)]
    for i, snip in enumerate(snippets[:len(results)]):
        results[i].snippet = snip
    return results


def web_search(query: str, *, broker: PermissionBroker,
               max_results: int = 5) -> List[SearchResult]:
    broker.require("network")
    q = urllib.parse.quote(query)
    page = _get(f"https://lite.duckduckgo.com/lite/?q={q}")
    return parse_ddg_lite(page, max_results)


def web_fetch(url: str, *, broker: PermissionBroker, max_chars: int = 5000) -> str:
    broker.require("network")
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("only http(s) URLs are allowed")
    return strip_html(_get(url), max_chars)
