"""Tests for the evolution archive (lineage + fitness cache)."""
from ag import archive
from ag.archive import Entry


def _redirect(monkeypatch, tmp_path):
    monkeypatch.setattr(archive, "ARCHIVE_DIR", tmp_path)
    monkeypatch.setattr(archive, "LINEAGE_FILE", tmp_path / "lineage.jsonl")
    monkeypatch.setattr(archive, "CACHE_FILE", tmp_path / "fitness_cache.json")


def _entry(**kw):
    base = dict(ts="2026-01-01T00:00:00", parent_hash="p", candidate_hash="c",
                incumbent_fitness=5.0, candidate_fitness=6.0, delta=1.0,
                tests_passed=True, adopted=True, verdict="improved",
                rationale="r", changed=["ag/prompts.py"], snapshot_id="s")
    base.update(kw)
    return Entry(**base)


def test_record_and_history_roundtrip(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    archive.record(_entry(ts="2026-01-01T00:00:01", adopted=False, verdict="regressed"))
    archive.record(_entry(ts="2026-01-01T00:00:02", adopted=True))
    hist = archive.history(limit=10)
    assert len(hist) == 2
    assert hist[0]["ts"] == "2026-01-01T00:00:02"   # newest first
    assert hist[0]["adopted"] is True


def test_adopted_history_filters(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    archive.record(_entry(adopted=False, verdict="regressed"))
    archive.record(_entry(adopted=True))
    adopted = archive.adopted_history()
    assert len(adopted) == 1 and adopted[0]["adopted"] is True


def test_history_skips_corrupt_lines(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    archive.record(_entry())
    with archive.LINEAGE_FILE.open("a") as f:
        f.write("{ not json\n")
    assert len(archive.history()) == 1  # corrupt line ignored, not fatal


def test_fitness_cache_set_get(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    assert archive.get_cached_fitness("abc") is None
    archive.set_cached_fitness("abc", 7.25)
    assert archive.get_cached_fitness("abc") == 7.25


def test_fitness_cache_is_bounded(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    for i in range(10):
        archive.set_cached_fitness(f"h{i}", float(i), cap=3)
    cache = archive._load_cache()
    assert len(cache) == 3
    assert "h9" in cache  # newest kept


def test_evolvable_hash_is_deterministic_and_sensitive():
    a = archive.evolvable_hash({"f": "x"})
    assert a == archive.evolvable_hash({"f": "x"})
    assert a != archive.evolvable_hash({"f": "y"})
