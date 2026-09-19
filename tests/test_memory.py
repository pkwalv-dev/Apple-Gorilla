"""Tests for AG's layered memory system.

Hermetic and offline: every manager uses the stdlib HashingEmbedder, so no Ollama or
network is required and semantic recall degrades to a deterministic lexical signal.
"""
import pytest

from ag import memory
from ag.memory import MemoryKind, MemoryManager
from ag.memory.embed import HashingEmbedder, cosine, get_embedder


def mgr(tmp_path, agent="root", parents=()):
    return MemoryManager(None, agent=agent, parents=parents,
                         base_dir=tmp_path, embedder=HashingEmbedder())


# --- core write/recall ------------------------------------------------------
def test_remember_and_recall(tmp_path):
    m = mgr(tmp_path)
    m.remember("The user lives in the Pacific timezone.")
    m.remember("The user's favorite language is Python.")
    hits = m.recall("what timezone is the user in?")
    assert hits and "Pacific" in hits[0].text


def test_recall_empty_query_returns_nothing(tmp_path):
    m = mgr(tmp_path)
    m.remember("something")
    assert m.recall("") == []
    assert m.recall("   ") == []


def test_empty_text_not_stored(tmp_path):
    m = mgr(tmp_path)
    assert m.remember("   ") is None
    assert m.store.all("root") == []


# --- layers -----------------------------------------------------------------
def test_layers_are_separate(tmp_path):
    m = mgr(tmp_path)
    m.remember("User uses metric units.", kind=MemoryKind.SEMANTIC)
    m.record_episode("what is 2+2", "4", score=8.0)
    m.remember("For arithmetic, call calc first.", kind=MemoryKind.PROCEDURAL)
    assert len(m.store.all("root", [MemoryKind.SEMANTIC])) == 1
    assert len(m.store.all("root", [MemoryKind.EPISODIC])) == 1
    assert len(m.store.all("root", [MemoryKind.PROCEDURAL])) == 1
    # Recall can be scoped to a layer.
    hits = m.recall("how should I handle arithmetic", kinds=[MemoryKind.PROCEDURAL])
    assert hits and hits[0].kind == MemoryKind.PROCEDURAL


def test_context_does_not_auto_inject_episodes(tmp_path):
    """An episode is one past conversation's transcript. The current conversation is
    threaded into the prompt as history, so an episode surfacing in context() only ever
    comes from a DIFFERENT conversation — the cross-conversation bleed that had AG
    volunteering a stale, hallucinated 'rush b' answer in unrelated chats."""
    m = mgr(tmp_path)
    m.record_episode("rush b",
                     "The latest Rush tour dates include Houston and Pittsburgh...",
                     score=5.0)
    m.remember("User prefers metric units.", kind=MemoryKind.SEMANTIC)

    ctx = m.context("rush")
    blob = ctx["text"].lower()
    assert "rush" not in blob and "houston" not in blob     # the episode stays out
    assert not any(h.kind == MemoryKind.EPISODIC for h in ctx["hits"])

    # But the episode is still stored, and still explicitly recallable — not nerfed.
    assert len(m.store.all("root", [MemoryKind.EPISODIC])) == 1
    assert m.recall("rush", kinds=[MemoryKind.EPISODIC])
    # A caller that explicitly asks for episodes in context() still gets them.
    assert any(h.kind == MemoryKind.EPISODIC
               for h in m.context("rush", kinds=[MemoryKind.EPISODIC])["hits"])


# --- dedup / merge ----------------------------------------------------------
def test_exact_duplicate_merges(tmp_path):
    m = mgr(tmp_path)
    a = m.remember("same fact")
    b = m.remember("same fact")
    assert a.id == b.id
    assert len(m.store.all("root", [MemoryKind.SEMANTIC])) == 1
    assert b.use_count >= 1  # merge bumps usage instead of duplicating


# --- namespacing + lineage inheritance --------------------------------------
def test_child_inherits_parent_and_shared(tmp_path):
    parent = mgr(tmp_path, agent="root")
    parent.remember("Parent knows the deploy key rotates on Fridays.")
    shared = mgr(tmp_path, agent="shared")
    shared.remember("Shared: the mascot is a gorilla.")

    child = mgr(tmp_path, agent="child-01", parents=["root"])
    child.remember("Child learned the staging URL.")
    # Child recalls its own, its parent's, and the shared tier.
    assert child.recall("when does the deploy key rotate")[0].text.startswith("Parent")
    assert child.recall("what is the mascot")[0].text.startswith("Shared")
    assert child.recall("staging url")[0].text.startswith("Child")
    # Parent does NOT see the child's private memory.
    assert not any("staging" in h.text.lower() for h in parent.recall("staging url"))


def test_promote_to_shared(tmp_path):
    child = mgr(tmp_path, agent="child-02", parents=["root"])
    m = child.remember("A genuinely useful discovery.")
    child.promote(m.id, to="shared")
    shared = mgr(tmp_path, agent="shared")
    assert any("useful discovery" in x.text for x in shared.store.all("shared"))


# --- graph links ------------------------------------------------------------
def test_links_expand_recall(tmp_path):
    m = mgr(tmp_path)
    a = m.remember("Project Bolt is a Rust game engine.")
    b = m.remember("The renderer uses Vulkan.")
    m.link(a.id, b.id)
    # A query hitting 'a' should pull in linked 'b' even without lexical overlap.
    hits = m.recall("tell me about project bolt", k=1, expand=True)
    ids = {h.id for h in hits}
    assert a.id in ids and b.id in ids


# --- retention cap ----------------------------------------------------------
def test_cap_evicts_lowest_value(tmp_path):
    m = mgr(tmp_path)
    m.caps[MemoryKind.SEMANTIC] = 3
    for i in range(6):
        m.remember(f"low value trivia number {i}", importance=0.1)
    important = m.remember("critical durable fact", importance=1.0)
    kept = m.store.all("root", [MemoryKind.SEMANTIC])
    assert len(kept) == 3
    assert any(x.id == important.id for x in kept)  # high-importance survives


# --- reflection (learning loop) --------------------------------------------
class _FakeClient:
    """Stands in for a model backend, returning canned reflector JSON."""
    def __init__(self, payload):
        self.payload = payload

    def complete(self, **kwargs):
        import json
        from types import SimpleNamespace
        return SimpleNamespace(text=json.dumps(self.payload))


def test_reflect_distills_facts_and_procedures(tmp_path):
    from ag.memory.reflect import reflect
    m = mgr(tmp_path)
    m.record_episode("compute 12*13", "156", score=9.0)
    m.record_episode("compute 99*99", "9801", score=9.0)
    client = _FakeClient({
        "facts": ["User frequently asks for exact arithmetic."],
        "procedures": [{"name": "arith", "when": "math prompts",
                        "steps": ["use calc", "return exact integer"]}],
    })
    out = reflect(m, client, cfg=None)
    assert out["facts"] == 1 and out["procedures"] == 1
    assert len(m.store.all("root", [MemoryKind.SEMANTIC])) == 1
    assert len(m.store.all("root", [MemoryKind.PROCEDURAL])) == 1
    # Episodes are flagged so a second reflect doesn't re-distill them.
    out2 = reflect(m, client, cfg=None)
    assert out2["episodes"] == 0


def test_reflect_skips_without_backend(tmp_path):
    from ag.memory.reflect import reflect
    m = mgr(tmp_path)
    m.record_episode("q", "a")
    out = reflect(m, client=None, cfg=None)
    assert "skipped" in out


def test_consolidate_merges_duplicates(tmp_path):
    from ag.memory.reflect import consolidate
    m = mgr(tmp_path)
    # Same text stored across a cap reset would duplicate; force two near-identical.
    m.store.add(_bare("d1", "The API base url is example.com", m.agent))
    m.store.add(_bare("d2", "The API base url is example.com", m.agent))
    out = consolidate(m)
    assert out["merged"] == 1
    assert len(m.store.all("root", [MemoryKind.SEMANTIC])) == 1


def _bare(mid, text, agent):
    from ag.memory.types import Memory
    return Memory(id=mid, text=text, kind=MemoryKind.SEMANTIC, agent=agent)


# --- embedder ---------------------------------------------------------------
def test_hashing_embedder_is_deterministic_and_normed(tmp_path):
    e = HashingEmbedder()
    v1 = e.embed_one("the quick brown fox")
    v2 = e.embed_one("the quick brown fox")
    assert v1 == v2
    assert cosine(v1, v2) == pytest.approx(1.0, abs=1e-9)
    assert cosine(e.embed_one("cat"), e.embed_one("xylophone thunder")) < 0.5


def test_get_embedder_hash_backend_never_touches_network():
    class C:
        memory_embed_backend = "hash"
    e = get_embedder(C(), force=True)
    assert isinstance(e, HashingEmbedder)


# --- back-compat surface (root namespace) -----------------------------------
def test_backcompat_functions(tmp_path, monkeypatch):
    from ag import config as cfgmod
    from ag.memory import manager as mgrmod
    monkeypatch.setattr(cfgmod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(mgrmod, "get_embedder", lambda cfg=None, **k: HashingEmbedder())
    memory.reset()
    memory.remember("The user prefers concise answers.")
    assert any("concise" in x.text for x in memory.all_memories())
    ctx = memory.memory_context("how should answers be written?")
    assert ctx.startswith("- ") and "concise" in ctx
    hit = memory.recall("answer style", k=1)
    assert hit
    assert memory.forget(hit[0].id) is True
    memory.remember("a"); memory.remember("b")
    assert memory.clear() >= 2
    assert memory.all_memories() == []
    memory.reset()
