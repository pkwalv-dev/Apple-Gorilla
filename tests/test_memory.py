"""Tests for AG's persistent memory."""
from ag import memory


def _redirect(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(memory, "MEMORY_FILE", tmp_path / "memories.jsonl")


def test_remember_and_recall(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    memory.remember("The user lives in the Pacific timezone.")
    memory.remember("The user's favorite language is Python.")
    hits = memory.recall("what timezone is the user in?")
    assert hits and "Pacific" in hits[0].text


def test_recall_ranks_by_overlap(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    memory.remember("Deploys happen on Fridays.")
    memory.remember("The mascot is a gorilla.")
    hits = memory.recall("when do deploys happen", k=1)
    assert len(hits) == 1 and "Deploys" in hits[0].text


def test_empty_and_dedup(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    assert memory.remember("   ") is None
    a = memory.remember("same fact")
    b = memory.remember("same fact")
    assert a.id == b.id  # exact dup returns the existing memory, no growth
    assert len(memory.all_memories()) == 1


def test_recall_empty_query_returns_nothing(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    memory.remember("something")
    assert memory.recall("") == []
    assert memory.recall("   ") == []


def test_forget_and_clear(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    m = memory.remember("temporary note")
    assert memory.forget(m.id) is True
    assert memory.forget("nonexistent") is False
    memory.remember("a"); memory.remember("b")
    assert memory.clear() == 2
    assert memory.all_memories() == []


def test_prune_keeps_newest(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    for i in range(6):
        memory.remember(f"fact number {i}", max_memories=3)
    mems = memory.all_memories()
    assert len(mems) == 3
    assert mems[-1].text == "fact number 5"  # newest survived


def test_memory_context_formats_block(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    memory.remember("Uses metric units.")
    ctx = memory.memory_context("what units?")
    assert ctx.startswith("- ") and "metric" in ctx


def test_corrupt_line_is_skipped(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    memory.remember("valid fact")
    memory.MEMORY_FILE.write_text('{"bad json\nnot json at all\n'
                                  + memory.MEMORY_FILE.read_text())
    # Should not raise; valid entries still load.
    assert any("valid fact" in m.text for m in memory.all_memories())
