"""Embeddings — recall by *meaning*, with graceful degradation.

Semantic recall is what lets "car" find a memory about a "vehicle" — the property a
keyword index can never have, and the precondition for generalizing across episodes.

This module is engine-agnostic behind a tiny `Embedder` interface:

- OllamaEmbedder  — local, keyless, offline. Talks to the Ollama server AG already
                    uses (`/api/embeddings`, default model `nomic-embed-text`). This
                    is the default and needs no API key or cloud call.
- HashingEmbedder — pure-stdlib fallback (feature-hashing + TF weighting, L2-normed).
                    Weaker than a real model but always available, so recall never
                    hard-fails when Ollama is down or absent.

`get_embedder(cfg)` picks one per policy and *probes* the live one; if it can't be
reached it falls back automatically. Swap in any other backend (a cloud embedder, a
local sentence-transformer) by implementing `Embedder` — AG never calls a provider
directly.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from typing import List, Optional, Sequence

_WORD = re.compile(r"[a-z0-9]+")
_HASH_DIM = 256


def cosine(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    """Cosine similarity in [-1, 1]; 0.0 if either vector is missing/degenerate."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class Embedder:
    """Interface: turn texts into fixed-length vectors. Implementations must never
    raise on a single bad input — return a zero vector instead."""

    name: str = "embedder"
    dim: int = 0

    def embed(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover
        raise NotImplementedError

    def embed_one(self, text: str) -> List[float]:
        out = self.embed([text])
        return out[0] if out else []


class HashingEmbedder(Embedder):
    """Deterministic feature-hashing embedder. No dependencies, no network, no model.

    Each token is hashed into one of `dim` buckets with a signed contribution; the
    vector is TF-weighted and L2-normalized. It captures lexical overlap robustly
    (a decent keyword-like signal in the same vector space as real embeddings), so
    the recall math is identical whether or not a semantic model is present.
    """

    name = "hash"

    def __init__(self, dim: int = _HASH_DIM):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        toks = _WORD.findall((text or "").lower())
        if not toks:
            return vec
        for tok in toks:
            h = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class OllamaEmbedder(Embedder):
    """Local embeddings via the Ollama HTTP API. Keyless and offline.

    Uses stdlib urllib only. A failed call raises so `get_embedder` can fall back at
    construction time; once constructed and probed, per-text failures degrade to a
    zero vector rather than breaking a whole recall.
    """

    name = "ollama"

    def __init__(self, host: str, model: str = "nomic-embed-text", timeout: float = 20.0):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.dim = 0  # discovered on first successful embed

    def _post(self, text: str) -> List[float]:
        body = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/embeddings", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        vec = [float(x) for x in (data.get("embedding") or [])]
        if vec and not self.dim:
            self.dim = len(vec)
        return vec

    def probe(self) -> bool:
        """Confirm the model actually returns a vector; sets self.dim. Raises on failure."""
        v = self._post("probe")
        return bool(v)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed each text, memoising on the exact string.

        Every miss is an HTTP round trip to Ollama, and a single recall embeds the
        same query more than once (scoring, then duplicate detection, then
        contradiction detection all embed it). Those calls pass *identical* strings,
        so an exact-match LRU converts them into one network call. A failed embed is
        deliberately NOT cached: the next call should retry rather than inherit a
        zero vector from a transient outage for the rest of the process's life.
        """
        from ..cache import EMBEDDINGS
        out: List[List[float]] = []
        for t in texts:
            key = f"{self.name}:{self.model}:{t}"
            hit = EMBEDDINGS.get(key)
            if hit is not None:
                out.append(list(hit))
                continue
            try:
                vec = self._post(t)
            except Exception:
                out.append([0.0] * (self.dim or _HASH_DIM))  # don't sink the batch
                continue
            if vec:
                EMBEDDINGS.put(key, list(vec))
            out.append(vec)
        return out


_CACHE: dict = {}


def get_embedder(cfg=None, *, force: bool = False) -> Embedder:
    """Return the embedder for this config, cached per (backend, model, host).

    Policy from `cfg.memory_embed_backend`:
      - "auto" (default): try Ollama, fall back to hashing if unreachable.
      - "ollama": require Ollama (still falls back if truly unreachable — recall must
        never hard-fail — but logs the intent by choosing it first).
      - "hash" / "none": the stdlib fallback, no network.
    """
    backend = getattr(cfg, "memory_embed_backend", "auto") if cfg else "auto"
    model = getattr(cfg, "memory_embed_model", "nomic-embed-text") if cfg else "nomic-embed-text"
    host = getattr(cfg, "ollama_host", "http://127.0.0.1:11434") if cfg else "http://127.0.0.1:11434"
    key = (backend, model, host)
    if not force and key in _CACHE:
        return _CACHE[key]

    emb: Embedder
    if backend in ("hash", "none"):
        emb = HashingEmbedder()
    else:
        candidate = OllamaEmbedder(host, model)
        try:
            candidate.probe()
            emb = candidate
        except Exception:
            emb = HashingEmbedder()  # Ollama down/absent — semantic degrades to lexical
    _CACHE[key] = emb
    return emb


def reset_cache() -> None:
    _CACHE.clear()
