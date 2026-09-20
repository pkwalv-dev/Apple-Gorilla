"""Caching. A cache that can serve stale data is a correctness bug wearing a
performance costume, so most of these tests are about invalidation, not speed."""
from __future__ import annotations

import time

from ag import cache


def test_lru_basic_get_put():
    c = cache.LRUCache(maxsize=4)
    assert c.get("missing") is None
    c.put("a", 1)
    assert c.get("a") == 1


def test_lru_evicts_least_recently_used():
    c = cache.LRUCache(maxsize=2)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")            # 'a' becomes most-recently-used, so 'b' should go
    c.put("c", 3)
    assert c.get("a") == 1
    assert c.get("b") is None
    assert c.get("c") == 3


def test_lru_reports_hit_rate():
    c = cache.LRUCache(maxsize=4)
    c.put("a", 1)
    c.get("a")
    c.get("nope")
    st = c.stats()
    assert st["hits"] == 1 and st["misses"] == 1


def test_file_cache_refetches_when_the_file_changes(tmp_path):
    """The central invariant. If this fails, AG reads memories that no longer exist."""
    f = tmp_path / "data.txt"
    f.write_text("one")
    fc = cache.FileCache()
    calls = []

    def parse(path):
        calls.append(path)
        return path.read_text()

    assert fc.get(f, parse) == "one"
    assert fc.get(f, parse) == "one"
    assert len(calls) == 1, "unchanged file must be served from cache"

    time.sleep(0.01)
    f.write_text("two")
    assert fc.get(f, parse) == "two", "a changed file MUST NOT serve stale data"
    assert len(calls) == 2


def test_file_cache_notices_same_size_edits(tmp_path):
    """mtime alone is not enough on coarse clocks; size alone misses in-place edits
    of equal length. The key uses both, which is why this case is covered."""
    f = tmp_path / "d.txt"
    f.write_text("aaaa")
    fc = cache.FileCache()
    assert fc.get(f, lambda q: q.read_text()) == "aaaa"
    time.sleep(0.01)
    f.write_text("bbbb")          # identical length, different content
    assert fc.get(f, lambda q: q.read_text()) == "bbbb"


def test_file_cache_invalidate_is_explicit(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("x")
    fc = cache.FileCache()
    fc.get(f, lambda q: q.read_text())
    fc.invalidate(f)
    calls = []
    fc.get(f, lambda q: (calls.append(1), q.read_text())[1])
    assert calls, "invalidate must force a re-read"


def test_file_cache_does_not_cache_a_missing_file(tmp_path):
    """A file that does not exist yet is the normal case on first run. Caching the
    empty result would hide the file forever once it is created."""
    fc = cache.FileCache()
    missing = tmp_path / "gone.txt"
    assert fc.get(missing, lambda q: q.read_text() if q.exists() else "") == ""
    missing.write_text("here now")
    assert fc.get(missing, lambda q: q.read_text()) == "here now"


def test_ttl_cache_expires(monkeypatch):
    c = cache.TTLCache(ttl_s=10.0)
    now = [1000.0]
    monkeypatch.setattr(cache.time, "time", lambda: now[0])
    c.put("k", "v")
    assert c.get("k") == "v"
    now[0] += 5
    assert c.get("k") == "v", "must still be valid inside the TTL"
    now[0] += 6
    assert c.get("k") is None, "must expire once the TTL passes"


def test_ttl_get_or_compute_computes_once(monkeypatch):
    c = cache.TTLCache(ttl_s=100.0)
    calls = []
    assert c.get_or_compute("k", lambda: (calls.append(1), "v")[1]) == "v"
    assert c.get_or_compute("k", lambda: (calls.append(1), "v")[1]) == "v"
    assert len(calls) == 1


def test_shared_instances_exist_and_report_stats():
    for shared in (cache.MEMORY_FILES, cache.EMBEDDINGS, cache.PROBES):
        assert shared.stats() is not None
    assert isinstance(cache.all_stats(), dict)


def test_clear_all_empties_every_shared_cache():
    cache.EMBEDDINGS.put("k", [0.1])
    cache.PROBES.put("p", ["x"])
    cache.clear_all()
    assert cache.EMBEDDINGS.get("k") is None
    assert cache.PROBES.get("p") is None


def _mem(mid, text):
    from ag.memory.types import Memory, MemoryKind
    return Memory(id=mid, text=text, kind=MemoryKind.SEMANTIC, agent="root")


def test_memory_store_invalidates_on_write(tmp_path):
    """Integration: the store's own writes must not be readable as stale."""
    from ag.memory.store import JsonlStore

    store = JsonlStore(tmp_path)
    store.add(_mem("m1", "first"))
    assert len(store.all("root", ["semantic"])) == 1
    store.add(_mem("m2", "second"))
    assert len(store.all("root", ["semantic"])) == 2, \
        "a write must invalidate the cached parse"


def test_cached_parse_results_are_not_shared_mutably(tmp_path):
    """A cache that hands out the same mutable object to two callers turns one
    caller's edit into the other's corruption. `recall` mutates the records it
    scores, so the cache must hand out fresh objects each time."""
    from ag.memory.store import JsonlStore

    store = JsonlStore(tmp_path)
    store.add(_mem("m1", "original"))
    first = store.all("root", ["semantic"])
    first[0].text = "MUTATED BY CALLER"
    second = store.all("root", ["semantic"])
    assert second[0].text == "original", \
        "mutating one read must not corrupt the next"
