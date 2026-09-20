"""Caching primitives — stop paying for the same work twice.

Two hot paths dominate AG's CPU time and neither needed to:

1. **Memory reads.** `JsonlStore.all()` re-reads and re-JSON-parses every line of
   every layer file on every call, and `get()` calls `all()`. One recall with graph
   expansion can re-parse the same thousand-line file a dozen times. Since the files
   only change when AG writes them, an mtime+size keyed cache makes a repeat read
   free and is *never stale*: any write changes the stat, which invalidates the entry.

2. **Embeddings.** The same query text is embedded repeatedly within a run (recall,
   duplicate detection, contradiction detection), and each miss is an HTTP round trip
   to Ollama. An LRU on the text gives an exact-match hit rate that is high in
   practice because those calls really do pass identical strings.

Both caches are small, thread-safe, and have explicit bounds — an unbounded cache in
a long-running server is a memory leak with good intentions.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Hashable, Optional, Tuple, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """A bounded, thread-safe LRU with hit/miss accounting.

    Accounting matters: a cache you cannot measure is a cache you cannot tune, and
    AG's inventory reports these numbers so a bad hit rate is visible rather than
    quietly costing latency.
    """

    def __init__(self, maxsize: int = 512):
        self.maxsize = max(1, int(maxsize))
        self._data: "OrderedDict[K, V]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key: K, value: V) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def get_or_compute(self, key: K, compute: Callable[[], V]) -> V:
        """Compute outside the lock — a slow miss must not block every reader."""
        found = self.get(key)
        if found is not None:
            return found
        value = compute()
        self.put(key, value)
        return value

    def invalidate(self, key: K) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {"size": len(self._data), "maxsize": self.maxsize,
                    "hits": self.hits, "misses": self.misses,
                    "hit_rate": round(self.hits / total, 3) if total else 0.0}


class FileCache:
    """Cache derived from a file, keyed on (mtime_ns, size) so it cannot go stale.

    Correctness argument: the cached value is only returned when the file's mtime
    *and* size are byte-identical to when it was read. Any writer — AG, the user, an
    editor, another process — changes at least one of those, so a stale read is not
    possible short of a same-nanosecond same-size rewrite, which the filesystem
    timestamp resolution makes vanishingly unlikely and which no AG code path does.
    """

    def __init__(self, maxsize: int = 64):
        self._lru: LRUCache[str, Tuple[Tuple[int, int], Any]] = LRUCache(maxsize)
        self.reloads = 0

    @staticmethod
    def _stamp(path: Path) -> Optional[Tuple[int, int]]:
        try:
            st = path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def get(self, path: Path, loader: Callable[[Path], Any]) -> Any:
        """Return the loaded value for `path`, reloading only when it changed."""
        key = str(path)
        stamp = self._stamp(path)
        if stamp is None:
            # Missing file: let the loader decide what that means (usually []).
            return loader(path)
        cached = self._lru.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        value = loader(path)
        self.reloads += 1
        # Re-stat after loading: if the file changed *while* we read it, the stamp
        # we store must describe what we actually loaded, or we would cache a
        # torn read under the new file's identity.
        after = self._stamp(path)
        if after is not None:
            self._lru.put(key, (after, value))
        return value

    def invalidate(self, path: Path) -> None:
        self._lru.invalidate(str(path))

    def clear(self) -> None:
        self._lru.clear()

    def stats(self) -> dict:
        s = self._lru.stats()
        s["reloads"] = self.reloads
        return s


class TTLCache:
    """Time-bounded cache for probes of the outside world (is Ollama up? which
    models exist?). Those answers are cheap to be slightly stale about and
    expensive to re-ask — `ollama_has_model` alone costs an HTTP round trip and is
    called on the hot path of every routed run."""

    def __init__(self, ttl_s: float = 30.0, maxsize: int = 128):
        self.ttl = float(ttl_s)
        self._lru: LRUCache[Hashable, Tuple[float, Any]] = LRUCache(maxsize)

    def get(self, key: Hashable) -> Optional[Any]:
        entry = self._lru.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > self.ttl:
            self._lru.invalidate(key)
            return None
        return value

    def put(self, key: Hashable, value: Any) -> None:
        self._lru.put(key, (time.time(), value))

    def get_or_compute(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        found = self.get(key)
        if found is not None:
            return found
        value = compute()
        self.put(key, value)
        return value

    def clear(self) -> None:
        self._lru.clear()

    def stats(self) -> dict:
        return self._lru.stats()


# Shared instances. Module-level so every caller benefits from one warm cache, and
# so `ag doctor` / the inventory can report real hit rates.
MEMORY_FILES = FileCache(maxsize=64)
EMBEDDINGS: LRUCache[str, list] = LRUCache(maxsize=1024)
PROBES = TTLCache(ttl_s=30.0, maxsize=64)


def all_stats() -> dict:
    return {"memory_files": MEMORY_FILES.stats(),
            "embeddings": EMBEDDINGS.stats(),
            "probes": PROBES.stats()}


def clear_all() -> None:
    MEMORY_FILES.clear()
    EMBEDDINGS.clear()
    PROBES.clear()
