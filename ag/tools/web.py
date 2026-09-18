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


# --------------------------------------------------------------------------- #
# Relevance: distill a query, rerank results, extract the on-topic passage.
# Pure lexical (stdlib) — no model call, no deps — so it stays portable and cheap.
# --------------------------------------------------------------------------- #
_STOPWORDS = frozenset(
    "a an and are as at be but by can could do does for from had has have how i if in "
    "into is it its me my no not of on or our so than that the their them then there "
    "these they this to us was we were what when where which who why will with would you "
    "your please tell give show find get about latest current news".split()
)


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def extract_query(prompt: str, max_terms: int = 8) -> str:
    """Distill a focused search query from a natural-language prompt: keep salient,
    non-stopword terms in order, deduped. Falls back to the raw prompt head."""
    out, seen = [], set()
    for t in _tokens(prompt):
        if len(t) < 2 or t in _STOPWORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return " ".join(out) or prompt.strip()[:120]


def relevance(query: str, text: str) -> float:
    """0..1 lexical relevance: fraction of distinct query terms present in text,
    plus a small density bonus. Cheap proxy for 'is this on-topic'."""
    q = {t for t in _tokens(query) if len(t) >= 2 and t not in _STOPWORDS}
    if not q:
        return 0.0
    toks = _tokens(text)
    if not toks:
        return 0.0
    tset = set(toks)
    covered = sum(1 for t in q if t in tset) / len(q)
    hits = sum(1 for t in toks if t in q)
    density = min(hits / len(toks) * 20, 0.3)
    return round(min(covered + density, 1.0), 4)


def best_passages(text: str, query: str, *, window: int = 600, top: int = 3,
                  max_chars: int = 1800) -> str:
    """Return the most query-relevant slices of a page in reading order, joined and
    capped at max_chars — instead of blindly taking the first N chars."""
    text = text.strip()
    if len(text) <= window:
        return text[:max_chars]
    step = max(window // 2, 1)
    windows = [(i, text[i:i + window]) for i in range(0, len(text), step)
               if text[i:i + window].strip()]
    scored = sorted(windows, key=lambda w: relevance(query, w[1]), reverse=True)
    picked = sorted(scored[:top], key=lambda w: w[0])  # restore reading order
    joined = " … ".join(c for _, c in picked).strip()
    return (joined or text)[:max_chars]


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
